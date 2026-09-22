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
"""Shared Linear execution and one-time weight preparation."""

from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.fake_mx import FakeMXFormat, fake_mx_quantize

from ..base import AscendLinearScheme
from .common import _quant_description


class FakeMXLinearMethod(AscendLinearScheme):
    is_fake_mx = True
    mx_format: FakeMXFormat
    algorithm = "rtn"

    def __init__(self):
        quant_description = self.config = _quant_description()
        self.group_size = int(quant_description.get("group_size", 32))

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
        quantized_x = self.quantize_activation(layer, x)
        return F.linear(quantized_x, layer.weight, bias)

    def quantize_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply the algorithm transform, then the shared activation QDQ."""
        return fake_mx_quantize(self.transform_activation(layer, x), self.mx_format, self.group_size)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_weight_processed", False):
            return
        self.prepare_weight(layer)
        layer.weight.data.copy_(fake_mx_quantize(layer.weight.data, self.mx_format, self.group_size))
        layer._fake_mx_weight_processed = True

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        """Load algorithm state and transform the original weight once."""
