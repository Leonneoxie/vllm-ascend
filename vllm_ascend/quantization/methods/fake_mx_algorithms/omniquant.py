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
"""AMCT log-scale transform paired with MX QDQ."""

from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import RowParallelLinear

from .common import _copy_transform_param, _load_transform_params
from .linear import FakeMXLinearMethod


class OmniQuantLinearMethod(FakeMXLinearMethod):
    """OmniQuant: per-dimension log-scale transform.

    Loads ``log_scale`` from external params. Weight is scaled up by
    ``exp(log_scale)`` and activation is scaled down, preserving the linear
    transform output while reducing MX QDQ error.
    """

    algorithm = "omniquant"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = self.config
        self.params_path = quant_description.get("omniquant_params_path")
        if not self.params_path:
            raise ValueError("OmniQuant requires omniquant_params_path.")
        self.input_size = 0
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        # Row-parallel layers see only their TP shard of the weight but the
        # AMCT sidecar stores the full input dim; size the placeholder at
        # full width and slice per rank in prepare_weight.
        layer_type = kwargs.get("layer_type")
        input_size = self.input_size * self.tp_size if layer_type == "row" else self.input_size
        return {"log_scale": torch.zeros(1, input_size, dtype=params_dtype)}

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        params = _load_transform_params(self.params_path)
        _copy_transform_param(layer.log_scale.data, params, layer, "log_scale")
        if isinstance(layer, RowParallelLinear):
            full_size = layer.log_scale.data.shape[1]
            block_size = full_size // layer.tp_size
            layer.log_scale.data = layer.log_scale.data[
                :, layer.tp_rank * block_size : (layer.tp_rank + 1) * block_size
            ]
        scale = torch.exp(layer.log_scale.data.to(torch.float32)).clamp(min=1e-4, max=1e4)
        layer.weight.data.copy_((layer.weight.data.to(torch.float32) * scale).to(layer.weight.data.dtype))
        layer._fake_mx_scale = scale.to(layer.weight.device)
        layer.log_scale = torch.nn.Parameter(layer.log_scale.data.contiguous(), requires_grad=False)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) / layer._fake_mx_scale).to(x.dtype)
