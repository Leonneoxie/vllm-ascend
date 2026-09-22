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
"""Learned orthogonal transforms exported by AMCT."""

from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.quantization.fake_mx import learned_hadamard_transform

from .common import DEFAULT_TRANSFORM_MATRIX_SIZE, _copy_transform_param, _load_transform_params
from .linear import FakeMXLinearMethod


def validate_lht_matrix(matrix: torch.Tensor, *, name: str) -> None:
    """Reject incompatible sidecars once at load time, before weight rotation.

    Check the CPU source, not its NPU copy or per-token activation. The
    tolerance allows roundoff in exported FP32/FP16/BF16 orthogonal matrices;
    it is not a license to use an unconstrained learned transform.
    """
    if not matrix.is_floating_point() or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"LHT {name}: expected a floating square matrix.")
    tolerance = max(1e-4, 2 * torch.finfo(matrix.dtype).eps)
    source = matrix.detach().to(device="cpu", dtype=torch.float64)
    residual = source @ source.T - torch.eye(source.shape[0], dtype=torch.float64)
    error = residual.abs().max().item()
    if not torch.isfinite(source).all() or not error <= tolerance:
        raise ValueError(
            f"LHT {name}: non-orthogonal transform (max |Q Q^T - I|={error:.6g}, "
            f"tolerance={tolerance:.6g}). Both activation and weight multiply Q; "
            "use the effective orthogonal matrix exported by AMCT "
            "(Cayley LHT or a validated equivalent export). "
            "Do not reuse unconstrained legacy parameters or silently invert/project them."
        )


def transform_lht_weight(
    weight: torch.Tensor,
    transform_weight: torch.Tensor,
    matrix_size: int,
) -> torch.Tensor:
    """Match AMCT's Cayley LHT: both weights and activations multiply Q.

    Do not invert the exported (possibly rounded) Q: AMCT also uses Q
    directly for inv_t=True, with the matrix cast to the operand dtype.
    """
    if transform_weight.shape != (matrix_size, matrix_size):
        raise ValueError("LHT transform shape must match matrix_size.")
    return learned_hadamard_transform(weight, transform_weight)


class LHTLinearMethod(FakeMXLinearMethod):
    """LHT for Linear: accepts original BF16 checkpoint, loads external transform
    matrix and applies the AMCT orthogonal transform at load time."""

    algorithm = "hadamard_learning"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = self.config
        self.matrix_size = int(quant_description.get("hadamard_learning_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE))
        if self.matrix_size <= 0:
            raise ValueError("hadamard_learning_matrix_size must be positive.")
        self.params_path = quant_description.get("lht_params_path")
        if not self.params_path:
            raise ValueError("Hadamard Learning requires lht_params_path.")

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        return {
            "transform_weight": torch.eye(
                self.matrix_size,
                dtype=torch.float32,
            )
        }

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return learned_hadamard_transform(x, layer.transform_weight)

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        if layer.weight.shape[-1] % self.matrix_size:
            raise ValueError(
                f"Hadamard Learning input dimension ({layer.weight.shape[-1]}) must be "
                f"divisible by matrix_size ({self.matrix_size})."
            )
        params = _load_transform_params(self.params_path)
        key = _copy_transform_param(layer.transform_weight.data, params, layer, "transform_weight")
        validate_lht_matrix(params[key], name=f"{self.params_path}:{key}")
        logger.debug("LHT: loaded %s", key)
        transformed_weight = transform_lht_weight(layer.weight.data, layer.transform_weight.data, self.matrix_size)
        layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))
        layer.transform_weight = torch.nn.Parameter(
            layer.transform_weight.data.contiguous(),
            requires_grad=False,
        )
