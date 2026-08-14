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

from vllm_ascend.quantization.fake_mx import (
    fake_mx_quantize,
    learned_hadamard_transform,
)
from vllm_ascend.quantization.fake_mx_audit import audit_call, audit_enabled, audit_event
from vllm_ascend.quantization.methods.fake_mx import (
    _copy_transform_param,
    _load_transform_params,
    transform_lht_weight,
)

from .base import IntrusiveLinearAdapter
from .registry import register_intrusive_adapter


@register_intrusive_adapter("hadamard_learning")
class LHTLinearAdapter(IntrusiveLinearAdapter):
    """LHT (Learnable Hadamard Transform) intrusive adapter.

    Manages the transform_weight matrix and applies the shared
    ``transform_lht_weight`` function for weights and
    ``learned_hadamard_transform`` for activations.
    """

    algorithm = "hadamard_learning"

    def process_weight(self, layer: torch.nn.Module) -> None:
        prefix = getattr(layer, "prefix", "") or ""

        if not self.spec.params_path:
            raise ValueError(f"Intrusive LHT requires params_path but it is missing for prefix {prefix!r}.")
        device = layer.weight.device

        if layer.weight.shape[-1] % self.spec.matrix_size:
            raise ValueError(
                f"Intrusive LHT input dimension ({layer.weight.shape[-1]}) must be "
                f"divisible by matrix_size ({self.spec.matrix_size}) for prefix {prefix!r}."
            )

        params = _load_transform_params(self.spec.params_path)

        class _LayerStub:
            pass

        stub = _LayerStub()
        stub.prefix = prefix
        transform_weight = torch.eye(self.spec.matrix_size, dtype=torch.float32, device=device)
        _copy_transform_param(transform_weight, params, stub, "transform_weight")

        ctx = self._make_ctx(layer) if audit_enabled() else None
        call = None
        if ctx is not None:
            call = audit_call(ctx, kind="weight")
            call.__enter__()
            call.capture("weight_raw", layer.weight.data)

        try:
            transformed_weight = transform_lht_weight(layer.weight.data, transform_weight, self.spec.matrix_size)

            if call is not None:
                call.capture("weight_transformed", transformed_weight)
                audit_event(ctx, "weight_transform", backend="intrusive")

            transformed_weight = transformed_weight.to(layer.weight.data.dtype)
            weight_qdq = fake_mx_quantize(transformed_weight, self.spec.mx_format, self.spec.group_size)

            layer.weight.data = weight_qdq.to(layer.weight.data.dtype)

            if call is not None:
                call.capture("weight_qdq", layer.weight.data)
                audit_event(ctx, "weight_qdq", backend="intrusive", algorithm=self.algorithm)
        finally:
            if call is not None:
                call.__exit__(None, None, None)

        self._transform_weight = transform_weight.to(device=device, dtype=layer.weight.dtype)

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
                x = learned_hadamard_transform(x, self._transform_weight)
                call.capture("act_transformed", x)
                audit_event(ctx, "activation_transform", backend="intrusive")
                quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
                call.capture("act_qdq", quantized_x)
                audit_event(ctx, "activation_qdq", backend="intrusive")
                result = F.linear(quantized_x, layer.weight, bias)
                call.capture("output", result)
                audit_event(ctx, "linear_output", backend="intrusive")
            return result

        x = learned_hadamard_transform(x, self._transform_weight)
        quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
        return F.linear(quantized_x, layer.weight, bias)
