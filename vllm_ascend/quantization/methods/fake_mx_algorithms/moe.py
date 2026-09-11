#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Split MoE adapters: RTN, RHT, LHT (sidecar), OmniQuant and FlatQuant."""

from collections.abc import Callable
from typing import Any

import torch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.fake_mx import FakeMXFormat, fake_mx_quantize, randomized_hadamard_transform
from vllm_ascend.utils import maybe_trans_nz

from ..base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .common import (
    DEFAULT_TRANSFORM_MATRIX_SIZE,
    _build_flatquant_sidecar_key,
    _copy_expert_transform_param,
    _layer_prefix,
    _load_transform_params,
    _make_rademacher_signs,
    _physical_to_logical_experts,
    _quant_description,
    _validate_weight_state,
)
from .expert_transforms import (
    _apply_expert_flatquant,
    _apply_expert_learned_hadamard,
    _apply_expert_omniquant,
)
from .flatquant import _get_decompose_dim, transform_flatquant_weight
from .lht import transform_lht_weight
from .rht import _validated_rht_signs

# ---- Post-dispatch transform factories ----
# Each factory binds the algorithm state into a callable with the shared
# signature (hidden_states, group_list, group_list_type); the execution path
# in moe_mlp.py only sees "transform present or not" and never branches on
# the algorithm name. New algorithms only add a factory here.


def _lht_expert_transform(transform_weight: torch.Tensor):
    def transform(x: torch.Tensor, group_list: torch.Tensor, group_list_type: int) -> torch.Tensor:
        return _apply_expert_learned_hadamard(x, transform_weight, group_list, group_list_type)

    return transform


def _flatquant_expert_transform(fc_state: dict[str, torch.Tensor]):
    def transform(x: torch.Tensor, group_list: torch.Tensor, group_list_type: int) -> torch.Tensor:
        return _apply_expert_flatquant(x, fc_state, group_list, group_list_type)

    return transform


def _omniquant_expert_transform(fc_scale: torch.Tensor):
    def transform(x: torch.Tensor, group_list: torch.Tensor, group_list_type: int) -> torch.Tensor:
        return _apply_expert_omniquant(x, fc_scale, group_list, group_list_type)

    return transform


def _rht_fc2_transform(signs: torch.Tensor, matrix_size: int):
    def transform(x: torch.Tensor, group_list: torch.Tensor, group_list_type: int) -> torch.Tensor:
        # RHT's transform is expert-independent; the group arguments are unused.
        return randomized_hadamard_transform(x, signs, matrix_size)

    return transform


class FakeMXMoEMethod(AscendMoEScheme):
    mx_format: FakeMXFormat
    quant_type: QuantType = QuantType.NONE
    algorithm = "rtn"
    required_weight_state: str | None = None

    def __init__(self):
        quant_description = self.config = _quant_description()
        self.group_size = int(quant_description.get("group_size", 32))
        _validate_weight_state(self.algorithm, self.required_weight_state, quant_description)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        if self.dynamic_eplb:
            raise NotImplementedError("Fake-MX MoE validation does not support dynamic EPLB.")

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes,
                dtype=params_dtype,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
        }
        return weights

    @staticmethod
    def get_dynamic_quant_param(
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_weight_processed", False):
            return
        self.prepare_weight(layer)
        layer.w13_weight.data.copy_(fake_mx_quantize(layer.w13_weight.data, self.mx_format, self.group_size))
        layer.w2_weight.data.copy_(fake_mx_quantize(layer.w2_weight.data, self.mx_format, self.group_size))
        # Checkpoints are loaded as [experts, N, K], while the Ascend split
        # grouped-matmul path consumes [experts, K, N].  Keep fake QDQ above
        # in checkpoint layout so MX blocks are formed along the logical K
        # dimension, then match AscendUnquantizedFusedMoEMethod's runtime
        # layout before GMM1/GMM2 execution.
        w13_data = layer.w13_weight.data.transpose(1, 2).contiguous()
        w2_data = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(maybe_trans_nz(w13_data), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(maybe_trans_nz(w2_data), requires_grad=False)
        layer._fake_mx_weight_processed = True

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        """Prepare algorithm parameters before shared weight QDQ and layout conversion."""

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize before dispatch unless the algorithm needs expert-specific state."""
        return fake_mx_quantize(x, self.mx_format, self.group_size)

    @staticmethod
    def _validate_execution_path() -> None:
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            raise NotImplementedError(
                "Fake-MX MoE requires a split dispatch -> GMM1 -> activation QDQ -> "
                "GMM2 -> combine path; FUSED_MC2 is monolithic and has no insertion "
                "point between GMM1 and GMM2. Disable fused MC2 for fake-MX validation."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input is not supported by fake-MX MoE: the router "
                "weight would be applied on opposite sides of the activation QDQ depending "
                "on the algorithm (pre-dispatch QDQ for RTN/RHT vs post-dispatch transform+QDQ "
                "for LHT/OmniQuant/FlatQuant), and QDQ is nonlinear, so the results would "
                "differ across algorithms and from AMCT. Disable the combination for "
                "validation."
            )
        self._validate_execution_path()
        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        if router_logits.shape[1] != num_logical_experts:
            raise AssertionError("Number of global experts mismatch (excluding redundancy)")

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)
        quantized_x = self.quantize_input(x)
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        if moe_comm_method is None:
            raise RuntimeError("Missing MoE communication context.")
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight
        w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight
        has_bias = bool(getattr(getattr(layer, "moe", None), "has_bias", False))
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=quantized_x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                fake_mx_format=self.mx_format,
                fake_mx_group_size=self.group_size,
                fake_mx_fc1_transform=getattr(layer, "_fake_mx_fc1_transform", None),
                fake_mx_fc2_transform=getattr(layer, "_fake_mx_fc2_transform", None),
                w1_bias=layer.w13_bias if has_bias else None,
                w2_bias=layer.w2_bias if has_bias else None,
                w1_scale=None,
                w2_scale=None,
                w1_scale_bias=None,
                w2_scale_bias=None,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )


class LHTMoEMethod(FakeMXMoEMethod):
    """Per-expert LHT: loads transform matrices from sidecar and applies
    ``W @ Q`` to each expert's weight, matching AMCT (no pre-transformed
    checkpoint required)."""

    algorithm = "hadamard_learning"

    def __init__(self):
        super().__init__()
        self.hadamard_learning_matrix_size = int(
            self.config.get("hadamard_learning_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE)
        )
        if self.hadamard_learning_matrix_size <= 0:
            raise ValueError("hadamard_learning_matrix_size must be positive.")
        self.params_path = self.config.get("lht_params_path")
        if not self.params_path:
            raise ValueError("Hadamard Learning MoE requires lht_params_path.")

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = super().get_weight(num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype)
        matrix_size = self.hadamard_learning_matrix_size
        eye = torch.eye(matrix_size, dtype=torch.float32)
        weights.update(
            {
                # Identity placeholders: prepare_weight overwrites every slot
                # from the sidecar and raises if a key is missing.
                "w13_transform_weight": eye.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
                "w2_transform_weight": eye.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
            }
        )
        return weights

    def _load_per_expert_transforms(self, layer: torch.nn.Module) -> None:
        ext_params = _load_transform_params(self.params_path)
        layer_prefix = _layer_prefix(layer)
        phy_to_logical = _physical_to_logical_experts(getattr(layer, "_expert_map", None))
        for comp_name, fc_short in [
            ("w13_transform_weight", "w13"),
            ("w2_transform_weight", "w2"),
        ]:
            param = getattr(layer, comp_name)
            for slot in range(param.shape[0]):
                logical_e = slot if phy_to_logical is None else phy_to_logical.get(slot, slot)
                key = _build_flatquant_sidecar_key(layer_prefix, logical_e, fc_short, "transform_weight")
                if key not in ext_params:
                    raise KeyError(
                        f"LHT sidecar missing required key {key!r} for local expert slot {slot}. "
                        "Refusing to silently fall back to an identity transform: the benchmark "
                        "label would no longer match the executed math. Check that the sidecar "
                        "covers every expert hosted on this rank."
                    )
                param.data[slot].copy_(ext_params[key].to(device=param.device, dtype=param.dtype))

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        matrix_size = self.hadamard_learning_matrix_size
        if layer.w13_weight.shape[-1] % matrix_size or layer.w2_weight.shape[-1] % matrix_size:
            raise ValueError(
                f"Hadamard Learning MoE input dimensions must be divisible by matrix_size ({matrix_size})."
            )
        self._load_per_expert_transforms(layer)
        for expert_idx in range(layer.w13_weight.shape[0]):
            layer.w13_weight.data[expert_idx].copy_(
                transform_lht_weight(
                    layer.w13_weight.data[expert_idx],
                    layer.w13_transform_weight.data[expert_idx],
                    matrix_size,
                ).to(layer.w13_weight.data.dtype)
            )
            layer.w2_weight.data[expert_idx].copy_(
                transform_lht_weight(
                    layer.w2_weight.data[expert_idx],
                    layer.w2_transform_weight.data[expert_idx],
                    matrix_size,
                ).to(layer.w2_weight.data.dtype)
            )
        layer.w13_transform_weight = torch.nn.Parameter(
            layer.w13_transform_weight.data.contiguous(), requires_grad=False
        )
        layer.w2_transform_weight = torch.nn.Parameter(layer.w2_transform_weight.data.contiguous(), requires_grad=False)
        layer._fake_mx_fc1_transform = _lht_expert_transform(layer.w13_transform_weight)
        layer._fake_mx_fc2_transform = _lht_expert_transform(layer.w2_transform_weight)

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Expert matrices are selected after dispatch; the shared path applies
        # the per-expert transform + QDQ after dispatch.
        return x


class RHTMoEMethod(FakeMXMoEMethod):
    """RHT for MoE: deterministic Rademacher signs (seed=0) + normalized FWHT.

    Rotates w13/w2 with the same signs at load time and transforms the
    dispatch input before QDQ. The FC2 activation transform runs in
    moe_mlp with the signs passed through the runtime args.
    """

    algorithm = "rht"

    def __init__(self):
        super().__init__()
        self.rht_matrix_size = int(
            self.config.get("rht_matrix_size", self.config.get("rht_group_size", self.group_size))
        )
        self.rht_seed = int(self.config.get("rht_seed", 0))
        self.params_path = self.config.get("rht_params_path")
        self._rht_signs: torch.Tensor | None = None

    def _get_signs(self) -> torch.Tensor:
        if self._rht_signs is None:
            self._rht_signs = _make_rademacher_signs(self.rht_matrix_size, self.rht_seed)
        return self._rht_signs

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        signs = _validated_rht_signs(
            self._get_signs(),
            self.rht_seed,
            self.rht_matrix_size,
            self.params_path,
            layer,
        )
        # FC1 runs pre-dispatch in quantize_input; only FC2 needs a
        # post-activation transform slot.
        layer._fake_mx_fc2_transform = _rht_fc2_transform(signs, self.rht_matrix_size)
        layer.w13_weight.data.copy_(
            randomized_hadamard_transform(layer.w13_weight.data, signs, self.rht_matrix_size)
        )
        layer.w2_weight.data.copy_(randomized_hadamard_transform(layer.w2_weight.data, signs, self.rht_matrix_size))

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        transformed = randomized_hadamard_transform(x, self._get_signs(), self.rht_matrix_size)
        return fake_mx_quantize(transformed, self.mx_format, self.group_size)


class OmniQuantMoEMethod(FakeMXMoEMethod):
    """OmniQuant for MoE: per-expert per-dimension log-scale transform.

    Weight is scaled up by ``exp(log_scale)`` at load time and the activation
    is scaled down per expert before QDQ (``(x/s) @ (W*s).T == x @ W.T``),
    preserving the linear output while reducing MX QDQ error.
    """

    algorithm = "omniquant"

    def __init__(self):
        super().__init__()
        self.params_path = self.config.get("omniquant_params_path")
        if not self.params_path:
            raise ValueError("OmniQuant MoE requires omniquant_params_path.")

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = super().get_weight(num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype)
        weights.update(
            {
                # Per-expert per-input-dim log_scale; FC1 input = hidden, FC2 input = intermediate.
                "w13_log_scale": torch.zeros(num_experts, hidden_sizes, dtype=torch.float32),
                "w2_log_scale": torch.zeros(num_experts, intermediate_size_per_partition, dtype=torch.float32),
            }
        )
        return weights

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        params = _load_transform_params(self.params_path)
        _copy_expert_transform_param(layer.w13_log_scale.data, params, layer, "w13_log_scale")
        _copy_expert_transform_param(layer.w2_log_scale.data, params, layer, "w2_log_scale")

        fc1_scale = torch.exp(layer.w13_log_scale.data.to(torch.float32)).clamp(min=1e-4, max=1e4)
        fc2_scale = torch.exp(layer.w2_log_scale.data.to(torch.float32)).clamp(min=1e-4, max=1e4)
        # weight' = weight * scale  (broadcasting: [E, 2*inter, hidden] * [E, 1, hidden])
        layer.w13_weight.data.copy_(
            (layer.w13_weight.data.to(torch.float32) * fc1_scale.unsqueeze(1)).to(layer.w13_weight.data.dtype)
        )
        layer.w2_weight.data.copy_(
            (layer.w2_weight.data.to(torch.float32) * fc2_scale.unsqueeze(1)).to(layer.w2_weight.data.dtype)
        )
        layer._fake_mx_fc1_transform = _omniquant_expert_transform(fc1_scale.to(layer.w13_weight.device))
        layer._fake_mx_fc2_transform = _omniquant_expert_transform(fc2_scale.to(layer.w2_weight.device))
        layer.w13_log_scale = torch.nn.Parameter(layer.w13_log_scale.data.contiguous(), requires_grad=False)
        layer.w2_log_scale = torch.nn.Parameter(layer.w2_log_scale.data.contiguous(), requires_grad=False)

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Per-expert scales are selected after dispatch; moe_mlp performs scale + QDQ.
        return x


class FlatQuantMoEMethod(FakeMXMoEMethod):
    """FlatQuant transform + fake MX QDQ for routed MoE experts.

    Each routed expert has independent FC1 and FC2 FlatQuant state
    (left_trans, right_trans, diag_scale) loaded from sidecar. At load time
    each expert's weight is inverse-transformed (QDQ is applied by the
    shared entry point); at forward time the per-expert activation
    transform and QDQ run in moe_mlp after dispatch.
    """

    algorithm = "flatquant"

    def __init__(self):
        super().__init__()
        self.flatquant_params_path = self.config.get("flatquant_params_path")
        if not self.flatquant_params_path:
            raise ValueError("MoE FlatQuant requires flatquant_params_path.")
        self.matrix_size = int(self.config.get("flatquant_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE))
        self.use_diag_scale = bool(self.config.get("flatquant_use_diag", True))

    def _decompose_dim(self, dim: int) -> tuple[int, int]:
        """Decompose a feature dimension into FlatQuant Kronecker dims.

        Prefer ``matrix_size`` as right_dim when divisible, matching the
        Dense FlatQuant path.  Fall back to the generic decomposition otherwise.
        """
        if dim % self.matrix_size == 0:
            return dim // self.matrix_size, self.matrix_size
        return _get_decompose_dim(dim, 1)

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = super().get_weight(num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype)
        # FC1 K = hidden_sizes, FC2 K = intermediate_size_per_partition.
        # AMCT Kronecker decomposition: left_dim * right_dim = K.
        fc1_left_dim, fc1_right_dim = self._decompose_dim(hidden_sizes)
        fc2_left_dim, fc2_right_dim = self._decompose_dim(intermediate_size_per_partition)
        weights.update(
            {
                "fc1_left_trans": torch.eye(fc1_left_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
                "fc1_right_trans": torch.eye(fc1_right_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
                "fc1_diag_scale": torch.ones(num_experts, hidden_sizes, dtype=torch.float32),
                "fc2_left_trans": torch.eye(fc2_left_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
                "fc2_right_trans": torch.eye(fc2_right_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
                "fc2_diag_scale": torch.ones(num_experts, intermediate_size_per_partition, dtype=torch.float32),
            }
        )
        return weights

    def _load_per_expert_state(
        self,
        layer: torch.nn.Module,
        ext_params: dict[str, torch.Tensor],
        fc_name: str,
        phy_to_logical: dict[int, int] | None,
    ) -> None:
        layer_prefix = _layer_prefix(layer)
        num_experts = getattr(layer, f"{fc_name}_left_trans").shape[0]
        # diag participates on both sides (weight inverse transform and the
        # packed activation state) only when the switch is on, mirroring the
        # Linear FlatQuant contract.
        components = ["left_trans", "right_trans"] + (["diag_scale"] if self.use_diag_scale else [])
        for comp in components:
            param = getattr(layer, f"{fc_name}_{comp}")
            for slot in range(num_experts):
                logical_e = slot if phy_to_logical is None else phy_to_logical.get(slot, slot)
                key = _build_flatquant_sidecar_key(layer_prefix, logical_e, fc_name, comp)
                if key not in ext_params:
                    raise KeyError(
                        f"FlatQuant sidecar missing required key {key!r} for local expert slot {slot}. "
                        "Refusing to silently fall back to an identity transform: the benchmark "
                        "label would no longer match the executed math. Check that the sidecar "
                        "covers every expert hosted on this rank."
                    )
                param.data[slot].copy_(ext_params[key].to(device=param.device, dtype=param.dtype))
        if not self.use_diag_scale:
            self._reject_disabled_diag_conflict(ext_params, layer_prefix, fc_name, phy_to_logical, num_experts)

    def _reject_disabled_diag_conflict(
        self,
        ext_params: dict[str, torch.Tensor],
        layer_prefix: str,
        fc_name: str,
        phy_to_logical: dict[int, int] | None,
        num_experts: int,
    ) -> None:
        """Fail fast when diag is disabled but the sidecar carries a real one.

        With ``flatquant_use_diag=False`` the weight inverse transform skips
        the diag division and the packed activation state omits the diag
        multiplication. A non-identity diag in the sidecar would mean the
        exported transforms were calibrated with diag enabled, so honouring
        the switch silently would break the weight/activation pairing.
        """
        for slot in range(num_experts):
            logical_e = slot if phy_to_logical is None else phy_to_logical.get(slot, slot)
            key = _build_flatquant_sidecar_key(layer_prefix, logical_e, fc_name, "diag_scale")
            value = ext_params.get(key)
            if value is None:
                continue
            value = value.to(torch.float32)
            if not torch.allclose(value, torch.ones_like(value)):
                raise ValueError(
                    f"flatquant_use_diag is disabled but sidecar key {key!r} carries a "
                    "non-identity diag_scale; the weight and activation transforms would "
                    "not be paired. Enable flatquant_use_diag or export the sidecar "
                    "without diag."
                )

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        ext_params = _load_transform_params(self.flatquant_params_path)
        phy_to_logical = _physical_to_logical_experts(getattr(layer, "_expert_map", None))
        self._load_per_expert_state(layer, ext_params, "fc1", phy_to_logical)
        self._load_per_expert_state(layer, ext_params, "fc2", phy_to_logical)

        fc1_diag = layer.fc1_diag_scale.data if self.use_diag_scale else None
        fc2_diag = layer.fc2_diag_scale.data if self.use_diag_scale else None
        for expert_idx in range(layer.w13_weight.shape[0]):
            # FC1: w13[e] shape [2*I, H], K=H
            layer.w13_weight.data[expert_idx] = transform_flatquant_weight(
                layer.w13_weight.data[expert_idx],
                layer.fc1_left_trans.data[expert_idx],
                layer.fc1_right_trans.data[expert_idx],
                fc1_diag[expert_idx] if fc1_diag is not None else None,
                layer.fc1_left_trans.shape[1],
                layer.fc1_right_trans.shape[1],
            ).to(layer.w13_weight.data.dtype)
            # FC2: w2[e] shape [H, I], K=I
            layer.w2_weight.data[expert_idx] = transform_flatquant_weight(
                layer.w2_weight.data[expert_idx],
                layer.fc2_left_trans.data[expert_idx],
                layer.fc2_right_trans.data[expert_idx],
                fc2_diag[expert_idx] if fc2_diag is not None else None,
                layer.fc2_left_trans.shape[1],
                layer.fc2_right_trans.shape[1],
            ).to(layer.w2_weight.data.dtype)

        # Freeze the per-expert state as Parameters, then pack the runtime
        # activation state once at load time.
        for param_name in [
            "fc1_left_trans",
            "fc1_right_trans",
            "fc1_diag_scale",
            "fc2_left_trans",
            "fc2_right_trans",
            "fc2_diag_scale",
        ]:
            param = getattr(layer, param_name)
            setattr(layer, param_name, torch.nn.Parameter(param.data.contiguous(), requires_grad=False))
        for fc_name in ("fc1", "fc2"):
            state = {
                "left_trans": getattr(layer, f"{fc_name}_left_trans"),
                "right_trans": getattr(layer, f"{fc_name}_right_trans"),
            }
            if self.use_diag_scale:
                state["diag_scale"] = getattr(layer, f"{fc_name}_diag_scale")
            setattr(
                layer,
                f"_fake_mx_{fc_name}_transform",
                _flatquant_expert_transform(state),
            )

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Per-expert state is selected after dispatch; moe_mlp performs transform + QDQ.
        return x
