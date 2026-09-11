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
"""FlatQuant paired transforms and parameter preparation."""

import math
from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.logger import logger
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.quantization.fake_mx import fake_mx_quantize

from .common import (
    DEFAULT_TRANSFORM_MATRIX_SIZE,
    _copy_transform_param,
    _find_transform_param,
    _inverse_fp32,
    _layer_prefix,
    _load_transform_params,
)
from .linear import FakeMXLinearMethod

MAX_FLATQUANT_TRANSFORM_DIM = 256
FLATQUANT_TP_BLOCK_DIAGONAL_TOLERANCE = 1e-6


def _is_tp_block_diagonal_transform(transform: torch.Tensor, tp_size: int) -> bool:
    """Return whether a row-parallel transform has no cross-rank terms."""
    if tp_size <= 1:
        return True
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1] or transform.shape[0] % tp_size:
        return False
    block_size = transform.shape[0] // tp_size
    off_diagonal = transform.clone()
    for rank in range(tp_size):
        start = rank * block_size
        off_diagonal[start : start + block_size, start : start + block_size] = 0
    return bool(torch.max(torch.abs(off_diagonal)).item() <= FLATQUANT_TP_BLOCK_DIAGONAL_TOLERANCE)


# ---- Pure transform functions shared by Linear scheme and patch sites ----


def transform_flatquant_weight(
    weight: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    diag_scale: torch.Tensor | None,
    left_dim: int,
    right_dim: int,
) -> torch.Tensor:
    """FlatQuant weight inverse transform: W' = inv(left) @ (W / diag) @ inv(right).T.

    Returns the transformed weight in float32 with the original shape.
    """
    inv_left = _inverse_fp32(left_trans)
    inv_right_t = _inverse_fp32(right_trans, transpose=True)
    original_shape = weight.shape
    weight_blocked = weight.to(torch.float32).reshape(-1, left_dim, right_dim)
    if diag_scale is not None:
        diag = diag_scale.to(torch.float32).reshape(left_dim, right_dim)
        if not torch.isfinite(diag).all():
            raise ValueError("FlatQuant diag_scale contains NaN or Inf.")
        if torch.any(diag.abs() < 1e-8):
            raise ValueError("FlatQuant diag_scale contains near-zero values.")
        weight_blocked = weight_blocked / diag.unsqueeze(0)
    rotated = torch.matmul(inv_left, weight_blocked)
    rotated = torch.matmul(rotated, inv_right_t)
    return rotated.reshape(original_shape)


def transform_flatquant_activation(
    x: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    diag_scale: torch.Tensor | None,
    left_dim: int,
    right_dim: int,
) -> torch.Tensor:
    """FlatQuant activation forward transform: x' = left.T @ (reshape(x) * diag) @ right.

    Returns the transformed activation with the original input shape.
    """
    input_shape = x.shape
    reshaped = x.reshape(-1, left_dim, right_dim)
    if diag_scale is not None:
        reshaped = reshaped * diag_scale.to(x.dtype).reshape(1, left_dim, right_dim)
    transformed = torch.matmul(left_trans.to(x.dtype).transpose(0, 1), reshaped)
    transformed = torch.matmul(transformed, right_trans.to(x.dtype))
    return transformed.reshape(*input_shape)


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


class FlatQuantLinearMethod(FakeMXLinearMethod):
    """FlatQuant transform plus fake MX QDQ and floating-point GEMM.

    The checkpoint always supplies original BF16/FP16 ``weight``. Calibration
    supplies the activation transforms, whose inverse is applied to the weight
    online before fake-MX QDQ.
    """

    supports_pertensor_layer_type = True
    algorithm = "flatquant"

    def __init__(self):
        super().__init__()
        quant_description = self.config
        self.max_supported_tp = int(quant_description.get("max_supported_tp", 4))
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.max_supported_tp:
            raise ValueError(
                f"Fake FlatQuant TP size ({self.tp_size}) exceeds max_supported_tp ({self.max_supported_tp})."
            )
        self.flatquant_params_path = quant_description.get("flatquant_params_path", None)
        if not self.flatquant_params_path:
            raise ValueError("FlatQuant requires flatquant_params_path.")
        self.matrix_size = int(quant_description.get("flatquant_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE))
        if self.matrix_size <= 0:
            raise ValueError("flatquant_matrix_size must be positive.")
        self.use_diag_scale = bool(quant_description.get("flatquant_use_diag", True))
        self.input_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        layer_type = kwargs.get("layer_type")
        if self.input_size % self.matrix_size == 0:
            # AMCT decomposition: L = D // K, R = K
            if layer_type == "row":
                origin_size = self.input_size * self.tp_size
                left_trans_dim = origin_size // self.matrix_size
                right_trans_dim = self.matrix_size
            else:
                left_trans_dim = self.input_size // self.matrix_size
                right_trans_dim = self.matrix_size
        elif layer_type == "row":
            origin_size = self.input_size * self.tp_size
            _, right_trans_dim = _get_decompose_dim(origin_size // self.max_supported_tp, self.max_supported_tp)
            left_trans_dim = origin_size // right_trans_dim
        else:
            left_trans_dim, right_trans_dim = _get_decompose_dim(self.input_size, 1)
        return {
            "left_trans": torch.eye(left_trans_dim, dtype=torch.float32),
            "right_trans": torch.eye(right_trans_dim, dtype=torch.float32),
            "clip_ratio": torch.ones(1, dtype=torch.float32),
            "diag_scale": torch.ones(
                self.input_size * self.tp_size if layer_type == "row" else self.input_size,
                dtype=torch.float32,
            ),
        }

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        local_input_size = x.shape[-1]
        if isinstance(layer, RowParallelLinear) and getattr(layer, "_fake_mx_flatquant_tp_full_transform", False):
            # Dense (non block-diagonal) transform: the activation transform
            # mixes the full input dim, so gather the full activation, apply
            # the paired transform, and slice this rank's input shard back.
            full_input = tensor_model_parallel_all_gather(x, dim=-1)
            left_dim = layer.left_trans.shape[0]
            right_dim = layer.right_trans.shape[0]
            transformed = transform_flatquant_activation(
                full_input,
                layer.left_trans,
                layer.right_trans,
                layer.diag_scale if hasattr(layer, "diag_scale") else None,
                left_dim,
                right_dim,
            )
            transformed = transformed.narrow(-1, layer.tp_rank * local_input_size, local_input_size)
        else:
            left_dim = layer.left_trans.shape[0]
            right_dim = layer.right_trans.shape[0]
            if left_dim * right_dim != local_input_size:
                raise ValueError(
                    "FlatQuant transform matrices dimension mismatch: "
                    f"left_dim({left_dim}) * right_dim({right_dim}) != in_features({local_input_size})."
                )
            diag = layer.diag_scale if hasattr(layer, "diag_scale") and layer.diag_scale is not None else None
            transformed = transform_flatquant_activation(
                x, layer.left_trans, layer.right_trans, diag, left_dim, right_dim
            )
        if transformed.shape[-1] != local_input_size:
            raise ValueError(
                "FlatQuant transform matrices dimension mismatch: "
                f"transformed features ({transformed.shape[-1]}) != in_features({local_input_size})."
            )
        return transformed

    def quantize_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return fake_mx_quantize(
            self.transform_activation(layer, x),
            self.mx_format,
            self.group_size,
            clip_ratio=layer.aclnn_clip_ratio,
        )

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        ext_params = _load_transform_params(self.flatquant_params_path)
        layer_prefix = _layer_prefix(layer)
        left_key = _copy_transform_param(layer.left_trans.data, ext_params, layer, "left_trans")
        _copy_transform_param(layer.right_trans.data, ext_params, layer, "right_trans")
        _copy_transform_param(layer.diag_scale.data, ext_params, layer, "diag_scale", required=self.use_diag_scale)
        layer.clip_ratio.data.fill_(1.0)
        logger.debug("FlatQuant: loaded %s for %s", left_key, layer_prefix)

        if isinstance(layer, RowParallelLinear):
            # Slice left_trans and diag_scale *before* the weight transform so
            # the per-partition transform matrices match the per-partition
            # weight.  Block-diagonal transforms slice cleanly per rank;
            # dense transforms need the full-dimension weight.
            left_dim = layer.left_trans.data.shape[0]
            left_block_size = left_dim // layer.tp_size
            left_param = _find_transform_param(ext_params, layer, "left_trans")
            use_full_transform = left_param is not None and not _is_tp_block_diagonal_transform(
                left_param[1], layer.tp_size
            )
            layer._fake_mx_flatquant_tp_full_transform = use_full_transform
            if use_full_transform:
                full_weight = tensor_model_parallel_all_gather(layer.weight.data, dim=-1)
                transformed_weight = transform_flatquant_weight(
                    full_weight,
                    layer.left_trans.data,
                    layer.right_trans.data,
                    layer.diag_scale.data if hasattr(layer, "diag_scale") else None,
                    left_dim,
                    layer.right_trans.data.shape[0],
                )
                local_start = layer.tp_rank * layer.weight.shape[-1]
                layer.weight.data.copy_(
                    transformed_weight.narrow(-1, local_start, layer.weight.shape[-1]).to(layer.weight.data.dtype)
                )
            else:
                layer.left_trans.data = layer.left_trans.data[
                    layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
                    layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
                ]
                if hasattr(layer, "diag_scale") and layer.diag_scale is not None:
                    diag_size = layer.diag_scale.data.shape[0]
                    diag_block_size = diag_size // layer.tp_size
                    layer.diag_scale.data = layer.diag_scale.data[
                        layer.tp_rank * diag_block_size : (layer.tp_rank + 1) * diag_block_size
                    ]
                left_dim = layer.left_trans.data.shape[0]
                right_dim = layer.right_trans.data.shape[0]
                diag = (
                    layer.diag_scale.data if hasattr(layer, "diag_scale") and layer.diag_scale is not None else None
                )
                transformed_weight = transform_flatquant_weight(
                    layer.weight.data, layer.left_trans.data, layer.right_trans.data, diag, left_dim, right_dim
                )
                layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))
        else:
            left_dim = layer.left_trans.data.shape[0]
            right_dim = layer.right_trans.data.shape[0]
            diag = layer.diag_scale.data if hasattr(layer, "diag_scale") and layer.diag_scale is not None else None
            transformed_weight = transform_flatquant_weight(
                layer.weight.data, layer.left_trans.data, layer.right_trans.data, diag, left_dim, right_dim
            )
            layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))

        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.contiguous(), requires_grad=False)
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data.contiguous(), requires_grad=False)
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32), requires_grad=False)
        layer.aclnn_clip_ratio = float(layer.clip_ratio.item())
        if hasattr(layer, "diag_scale") and layer.diag_scale is not None:
            layer.diag_scale = torch.nn.Parameter(
                layer.diag_scale.data.to(torch.float32).contiguous(), requires_grad=False
            )
