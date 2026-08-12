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
"""Fake MXFP4/MXFP8 schemes backed by ordinary floating-point NPU kernels."""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.fake_mx import (
    FakeMXFormat,
    fake_mx_quantize,
    learned_hadamard_transform,
    randomized_hadamard_transform,
)
from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

MAX_FLATQUANT_TRANSFORM_DIM = 256


def _get_decompose_dim(size: int, tp_size: int) -> tuple[int, int]:
    """Decompose a feature size into FlatQuant Kronecker dimensions."""
    left_candidate = math.isqrt(size)
    if left_candidate * left_candidate < size:
        left_candidate += 1

    while True:
        difference = left_candidate * left_candidate - size
        right_candidate = math.isqrt(difference)
        if right_candidate * right_candidate == difference:
            break
        left_candidate += 1

    left_dim = left_candidate - right_candidate
    right_dim = left_candidate + right_candidate
    if left_dim + right_dim > MAX_FLATQUANT_TRANSFORM_DIM:
        raise ValueError(
            f"FlatQuant left and right transform dimensions must not exceed {MAX_FLATQUANT_TRANSFORM_DIM} in total."
        )
    if left_dim * tp_size > MAX_FLATQUANT_TRANSFORM_DIM:
        return MAX_FLATQUANT_TRANSFORM_DIM, tp_size * size // MAX_FLATQUANT_TRANSFORM_DIM
    return left_dim, right_dim


class _AscendFakeMXLinearMethod(AscendLinearScheme):
    is_fake_mx = True
    mx_format: FakeMXFormat
    algorithm = "rtn"
    prequantized_weight = False
    required_weight_state: str | None = None

    def __init__(self):
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.group_size = int(quant_description.get("group_size", 32))
        if self.required_weight_state is not None:
            weight_state = quant_description.get("fake_mx_weight_state")
            if weight_state != self.required_weight_state:
                raise ValueError(
                    f"{self.algorithm} fake-MX requires fake_mx_weight_state="
                    f"{self.required_weight_state!r}, got {weight_state!r}."
                )

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        x = self.transform_activation(layer, x)
        quantized_x = fake_mx_quantize(x, self.mx_format, self.group_size)
        return F.linear(quantized_x, layer.weight, bias)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_weight_processed", False):
            return
        if not self.prequantized_weight:
            layer.weight.data.copy_(fake_mx_quantize(layer.weight.data, self.mx_format, self.group_size))
        layer._fake_mx_weight_processed = True


@register_scheme("W4A4_MXFP4_FAKE", "linear")
class AscendW4A4MXFP4FakeLinearMethod(_AscendFakeMXLinearMethod):
    """MXFP4 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "linear")
class AscendW8A8MXFP8FakeLinearMethod(_AscendFakeMXLinearMethod):
    """MXFP8 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp8"


class _AscendPrequantizedWeightFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """Consume an FP checkpoint whose weight already contains QDQ error."""

    prequantized_weight = True
    required_weight_state = "prequantized_qdq"


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "linear")
class AscendW4A4MXFP4OmniQuantFakeLinearMethod(_AscendPrequantizedWeightFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "omniquant"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "linear")
class AscendW8A8MXFP8OmniQuantFakeLinearMethod(_AscendPrequantizedWeightFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "omniquant"


@register_scheme("W4A4_MXFP4_AUTOROUND_FAKE", "linear")
class AscendW4A4MXFP4AutoRoundFakeLinearMethod(_AscendPrequantizedWeightFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "autoround"


@register_scheme("W8A8_MXFP8_AUTOROUND_FAKE", "linear")
class AscendW8A8MXFP8AutoRoundFakeLinearMethod(_AscendPrequantizedWeightFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "autoround"


class _AscendRHTFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    algorithm = "rht"
    required_weight_state = "rht_rotated_fp"

    def __init__(self):
        super().__init__()
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.rht_group_size = int(quant_description.get("rht_group_size", self.group_size))
        self.rht_seed = int(quant_description.get("rht_seed", 0))

    @staticmethod
    def _make_signs(size: int, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
        return signs.mul_(2).sub_(1).to(device=device)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        local_input_size = layer.weight.shape[-1]
        if isinstance(layer, RowParallelLinear):
            global_input_size = local_input_size * layer.tp_size
            global_signs = self._make_signs(global_input_size, self.rht_seed, layer.weight.device)
            start = layer.tp_rank * local_input_size
            layer.fake_mx_rht_signs = global_signs[start : start + local_input_size]
        else:
            layer.fake_mx_rht_signs = self._make_signs(local_input_size, self.rht_seed, layer.weight.device)
        super().process_weights_after_loading(layer)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return randomized_hadamard_transform(x, layer.fake_mx_rht_signs, self.rht_group_size)


@register_scheme("W4A4_MXFP4_RHT_FAKE", "linear")
class AscendW4A4MXFP4RHTFakeLinearMethod(_AscendRHTFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "linear")
class AscendW8A8MXFP8RHTFakeLinearMethod(_AscendRHTFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendHadamardLearningFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """AMCT-Q learnable Hadamard-like transform plus fake MX QDQ."""

    algorithm = "hadamard_learning"
    required_weight_state = "hadamard_learning_transformed_fp"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.matrix_size = int(quant_description.get("hadamard_learning_matrix_size", 128))
        if self.matrix_size <= 0:
            raise ValueError("hadamard_learning_matrix_size must be positive.")

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        return {
            "transform_weight": torch.empty(
                self.matrix_size,
                self.matrix_size,
                dtype=params_dtype,
            )
        }

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return learned_hadamard_transform(x, layer.transform_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.shape[-1] % self.matrix_size:
            raise ValueError(
                f"Hadamard Learning input dimension ({layer.weight.shape[-1]}) must be "
                f"divisible by matrix_size ({self.matrix_size})."
            )
        layer.transform_weight = torch.nn.Parameter(
            layer.transform_weight.data.contiguous(),
            requires_grad=False,
        )
        super().process_weights_after_loading(layer)


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "linear")
class AscendW4A4MXFP4HadamardLearningFakeLinearMethod(_AscendHadamardLearningFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "linear")
class AscendW8A8MXFP8HadamardLearningFakeLinearMethod(_AscendHadamardLearningFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendFakeMXFlatQuantLinearMethod(_AscendFakeMXLinearMethod):
    """FlatQuant transform plus fake MX QDQ and floating-point GEMM.

    The checkpoint contract is intentionally floating point: ``weight`` is the
    FlatQuant-transformed FP16/BF16 weight, while ``left_trans``,
    ``right_trans``, and ``clip_ratio`` are calibration outputs.
    """

    supports_pertensor_layer_type = True
    required_weight_state = "flatquant_transformed_fp"

    def __init__(self):
        super().__init__()
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.max_supported_tp = int(quant_description.get("max_supported_tp", 4))
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.max_supported_tp:
            raise ValueError(
                f"Fake FlatQuant TP size ({self.tp_size}) exceeds max_supported_tp ({self.max_supported_tp})."
            )
        self.input_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        layer_type = kwargs.get("layer_type")
        if layer_type == "row":
            origin_size = self.input_size * self.tp_size
            _, right_trans_dim = _get_decompose_dim(origin_size // self.max_supported_tp, self.max_supported_tp)
            left_trans_dim = origin_size // right_trans_dim
        else:
            left_trans_dim, right_trans_dim = _get_decompose_dim(self.input_size, 1)
        return {
            "left_trans": torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype),
            "right_trans": torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype),
            "clip_ratio": torch.empty(1, dtype=torch.float32),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        input_shape = x.shape
        left_dim = layer.left_trans.shape[0]
        right_dim = layer.right_trans.shape[0]
        if left_dim * right_dim != input_shape[-1]:
            raise ValueError(
                "FlatQuant transform matrices dimension mismatch: "
                f"left_dim({left_dim}) * right_dim({right_dim}) != in_features({input_shape[-1]})."
            )

        reshaped = x.reshape(-1, left_dim, right_dim)
        transformed = torch.matmul(layer.left_trans.to(x.dtype), reshaped)
        transformed = torch.matmul(transformed, layer.right_trans.to(x.dtype))
        transformed = transformed.reshape(*input_shape)
        quantized_x = fake_mx_quantize(
            transformed,
            self.mx_format,
            self.group_size,
            clip_ratio=layer.aclnn_clip_ratio,
        )
        return F.linear(quantized_x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if isinstance(layer, RowParallelLinear):
            left_dim = layer.left_trans.data.shape[0]
            left_block_size = left_dim // layer.tp_size
            layer.left_trans.data = layer.left_trans.data[
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
            ]

        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.t().contiguous(), requires_grad=False)
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data.contiguous(), requires_grad=False)
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32), requires_grad=False)
        layer.aclnn_clip_ratio = float(layer.clip_ratio.item())
        super().process_weights_after_loading(layer)


@register_scheme("W4A4_MXFP4_FLATQUANT_FAKE", "linear")
class AscendW4A4MXFP4FakeFlatQuantLinearMethod(_AscendFakeMXFlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FLATQUANT_FAKE", "linear")
class AscendW8A8MXFP8FakeFlatQuantLinearMethod(_AscendFakeMXFlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendFakeMXFusedMoEMethod(AscendMoEScheme):
    mx_format: FakeMXFormat
    quant_type: QuantType = QuantType.NONE
    algorithm = "rtn"
    prequantized_weight = False
    required_weight_state: str | None = None

    def __init__(self):
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.group_size = int(quant_description.get("group_size", 32))
        self.rht_group_size = int(quant_description.get("rht_group_size", self.group_size))
        self.rht_seed = int(quant_description.get("rht_seed", 0))
        self.hadamard_learning_matrix_size = int(quant_description.get("hadamard_learning_matrix_size", 128))
        if self.required_weight_state is not None:
            weight_state = quant_description.get("fake_mx_weight_state")
            if weight_state != self.required_weight_state:
                raise ValueError(
                    f"{self.algorithm} fake-MX requires fake_mx_weight_state="
                    f"{self.required_weight_state!r}, got {weight_state!r}."
                )
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
        if self.algorithm == "hadamard_learning":
            matrix_size = self.hadamard_learning_matrix_size
            weights.update(
                {
                    "w13_transform_weight": torch.empty(
                        num_experts,
                        matrix_size,
                        matrix_size,
                        dtype=params_dtype,
                    ),
                    "w2_transform_weight": torch.empty(
                        num_experts,
                        matrix_size,
                        matrix_size,
                        dtype=params_dtype,
                    ),
                }
            )
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
        if self.algorithm == "rht":
            layer.fake_mx_w13_rht_signs = _AscendRHTFakeMXLinearMethod._make_signs(
                layer.w13_weight.shape[-1],
                self.rht_seed,
                layer.w13_weight.device,
            )
            layer.fake_mx_w2_rht_signs = _AscendRHTFakeMXLinearMethod._make_signs(
                layer.w2_weight.shape[-1],
                self.rht_seed + 1,
                layer.w2_weight.device,
            )
        elif self.algorithm == "hadamard_learning":
            matrix_size = self.hadamard_learning_matrix_size
            if layer.w13_weight.shape[-1] % matrix_size or layer.w2_weight.shape[-1] % matrix_size:
                raise ValueError(
                    f"Hadamard Learning MoE input dimensions must be divisible by matrix_size ({matrix_size})."
                )
            layer.w13_transform_weight = torch.nn.Parameter(
                layer.w13_transform_weight.data.contiguous(), requires_grad=False
            )
            layer.w2_transform_weight = torch.nn.Parameter(
                layer.w2_transform_weight.data.contiguous(), requires_grad=False
            )
        if not self.prequantized_weight:
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
        if self.algorithm == "rht":
            x = randomized_hadamard_transform(
                x,
                layer.fake_mx_w13_rht_signs,
                self.rht_group_size,
            )
        # Per-expert learned matrices can only be selected after token
        # dispatch.  Its FC1 transform and QDQ therefore run in moe_mlp.py.
        quantized_x = (
            x if self.algorithm == "hadamard_learning" else fake_mx_quantize(x, self.mx_format, self.group_size)
        )
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
                fake_mx_algorithm=self.algorithm,
                fake_mx_rht_signs=getattr(layer, "fake_mx_w2_rht_signs", None),
                fake_mx_rht_group_size=self.rht_group_size,
                fake_mx_w13_transform=getattr(layer, "w13_transform_weight", None),
                fake_mx_w2_transform=getattr(layer, "w2_transform_weight", None),
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


@register_scheme("W4A4_MXFP4_FAKE", "moe")
class AscendW4A4MXFP4FakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "moe")
class AscendW8A8MXFP8FakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendPrequantizedWeightFakeMXFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    prequantized_weight = True
    required_weight_state = "prequantized_qdq"


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "moe")
class AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod(_AscendPrequantizedWeightFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "omniquant"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "moe")
class AscendW8A8MXFP8OmniQuantFakeFusedMoEMethod(_AscendPrequantizedWeightFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "omniquant"


@register_scheme("W4A4_MXFP4_AUTOROUND_FAKE", "moe")
class AscendW4A4MXFP4AutoRoundFakeFusedMoEMethod(_AscendPrequantizedWeightFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "autoround"


@register_scheme("W8A8_MXFP8_AUTOROUND_FAKE", "moe")
class AscendW8A8MXFP8AutoRoundFakeFusedMoEMethod(_AscendPrequantizedWeightFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "autoround"


@register_scheme("W4A4_MXFP4_RHT_FAKE", "moe")
class AscendW4A4MXFP4RHTFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "rht"
    required_weight_state = "rht_rotated_fp"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "moe")
class AscendW8A8MXFP8RHTFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "rht"
    required_weight_state = "rht_rotated_fp"


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "moe")
class AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "hadamard_learning"
    required_weight_state = "hadamard_learning_transformed_fp"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "moe")
class AscendW8A8MXFP8HadamardLearningFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "hadamard_learning"
    required_weight_state = "hadamard_learning_transformed_fp"
