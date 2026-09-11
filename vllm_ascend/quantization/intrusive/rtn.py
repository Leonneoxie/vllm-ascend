#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.fake_mx import fake_mx_quantize
from vllm_ascend.quantization.fake_mx_audit import audit_call, audit_enabled, audit_event

from .base import IntrusiveLinearAdapter
from .registry import register_intrusive_adapter


@register_intrusive_adapter("rtn")
class RTNLinearAdapter(IntrusiveLinearAdapter):
    """RTN (Round-To-Nearest) intrusive adapter.

    No transform — QDQ is applied directly to the original weight
    and activation.
    """

    algorithm = "rtn"

    def process_weight(self, layer: torch.nn.Module) -> None:
        ctx = self._make_ctx(layer) if audit_enabled() else None
        call = None
        if ctx is not None:
            call = audit_call(ctx, kind="weight")
            call.__enter__()
            call.capture("weight_raw", layer.weight.data)

        try:
            weight_qdq = fake_mx_quantize(layer.weight.data, self.spec.mx_format, self.spec.group_size)
            layer.weight.data = weight_qdq.to(layer.weight.data.dtype)

            if call is not None:
                call.capture("weight_qdq", layer.weight.data)
                audit_event(ctx, "weight_qdq", backend="intrusive", algorithm=self.algorithm)
        finally:
            if call is not None:
                call.__exit__(None, None, None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audit_enabled():
            ctx = self._make_ctx(layer)
            with audit_call(ctx, kind="act") as call:
                call.capture("act_raw", x)
                audit_event(ctx, "activation_transform", backend="intrusive")
                quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
                call.capture("act_qdq", quantized_x)
                audit_event(ctx, "activation_qdq", backend="intrusive")
                result = F.linear(quantized_x, layer.weight, bias)
                call.capture("output", result)
                audit_event(ctx, "linear_output", backend="intrusive")
            return result

        quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
        return F.linear(quantized_x, layer.weight, bias)
