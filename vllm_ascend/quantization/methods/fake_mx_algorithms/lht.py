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

from .common import DEFAULT_TRANSFORM_MATRIX_SIZE, _copy_transform_param, _inverse_fp32, _load_transform_params
from .linear import FakeMXLinearMethod


def transform_lht_weight(
    weight: torch.Tensor,
    transform_weight: torch.Tensor,
    matrix_size: int,
) -> torch.Tensor:
    """LHT weight inverse transform: W' = W @ inv(T).T (block-wise).

    The learned transform matrix T is invertible but not necessarily
    orthogonal, so inv(T).T != T in general.  The activation forward
    transform is x' = x @ T; the paired weight transform must be
    W' = W @ inv(T).T so that x' @ W'.T == x @ W.T.
    """
    original_shape = weight.shape
    weight_blocked = weight.to(torch.float32).reshape(-1, matrix_size)
    inv_t_t = _inverse_fp32(transform_weight, transpose=True)
    rotated = weight_blocked @ inv_t_t
    return rotated.reshape(original_shape)


class LHTLinearMethod(FakeMXLinearMethod):
    """LHT for Linear: accepts original BF16 checkpoint, loads external transform
    matrix and applies inverse transform to weight at load time."""

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
        logger.debug("LHT: loaded %s", key)
        transformed_weight = transform_lht_weight(layer.weight.data, layer.transform_weight.data, self.matrix_size)
        layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))
        layer.transform_weight = torch.nn.Parameter(
            layer.transform_weight.data.contiguous(),
            requires_grad=False,
        )
