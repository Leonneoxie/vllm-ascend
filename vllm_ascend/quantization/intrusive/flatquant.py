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
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.quantization.fake_mx import fake_mx_quantize
from vllm_ascend.quantization.fake_mx_audit import audit_call, audit_enabled, audit_event
from vllm_ascend.quantization.methods.fake_mx import (
    _copy_transform_param,
    _load_transform_params,
    transform_flatquant_activation,
    transform_flatquant_weight,
)

from .base import IntrusiveLinearAdapter
from .registry import register_intrusive_adapter


@register_intrusive_adapter("flatquant")
class FlatQuantLinearAdapter(IntrusiveLinearAdapter):
    """FlatQuant intrusive adapter.

    Manages left/right/diag transform parameters and applies the
    shared ``transform_flatquant_weight`` / ``transform_flatquant_activation``
    functions at the correct lifecycle points.
    """

    algorithm = "flatquant"

    def process_weight(self, layer: torch.nn.Module) -> None:
        prefix = getattr(layer, "prefix", "") or ""

        if not self.spec.params_path:
            raise ValueError(f"Intrusive FlatQuant requires params_path but it is missing for prefix {prefix!r}.")
        if not self.spec.matrix_size or self.spec.matrix_size <= 0:
            raise ValueError(
                f"Intrusive FlatQuant requires positive matrix_size but got "
                f"{self.spec.matrix_size!r} for prefix {prefix!r}."
            )
        device = layer.weight.device

        input_size = layer.weight.shape[-1]
        if isinstance(layer, RowParallelLinear):
            origin_size = input_size * layer.tp_size
            left_dim = origin_size // self.spec.matrix_size
            right_dim = self.spec.matrix_size
        else:
            left_dim = input_size // self.spec.matrix_size
            right_dim = self.spec.matrix_size

        if left_dim * right_dim != (input_size * layer.tp_size if isinstance(layer, RowParallelLinear) else input_size):
            raise ValueError(
                f"Intrusive FlatQuant input dimension ({input_size}) is not divisible "
                f"by matrix_size ({self.spec.matrix_size}) for prefix {prefix!r}."
            )

        ext_params = _load_transform_params(self.spec.params_path)
        left_trans = torch.eye(left_dim, dtype=torch.float32, device=device)
        right_trans = torch.eye(right_dim, dtype=torch.float32, device=device)
        diag_scale = torch.ones(input_size, dtype=torch.float32, device=device)

        class _LayerStub:
            pass

        stub = _LayerStub()
        stub.prefix = prefix
        _copy_transform_param(left_trans, ext_params, stub, "left_trans")
        _copy_transform_param(right_trans, ext_params, stub, "right_trans")
        _copy_transform_param(diag_scale, ext_params, stub, "diag_scale", required=False)

        ctx = self._make_ctx(layer) if audit_enabled() else None
        call = None
        if ctx is not None:
            call = audit_call(ctx, kind="weight")
            call.__enter__()
            call.capture("weight_raw", layer.weight.data)

        try:
            transformed_weight = transform_flatquant_weight(
                layer.weight.data, left_trans, right_trans, diag_scale, left_dim, right_dim
            )

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

        self._left_trans = left_trans
        self._right_trans = right_trans
        self._diag_scale = diag_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left_trans = self._left_trans.to(device=x.device, dtype=x.dtype)
        right_trans = self._right_trans.to(device=x.device, dtype=x.dtype)
        diag = self._diag_scale.to(device=x.device, dtype=x.dtype)
        left_dim = left_trans.shape[0]
        right_dim = right_trans.shape[0]

        if audit_enabled():
            ctx = self._make_ctx(layer)
            with audit_call(ctx, kind="act") as call:
                call.capture("act_raw", x)
                x = transform_flatquant_activation(x, left_trans, right_trans, diag, left_dim, right_dim)
                call.capture("act_transformed", x)
                audit_event(ctx, "activation_transform", backend="intrusive")
                quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
                call.capture("act_qdq", quantized_x)
                audit_event(ctx, "activation_qdq", backend="intrusive")
                result = F.linear(quantized_x, layer.weight, bias)
                call.capture("output", result)
                audit_event(ctx, "linear_output", backend="intrusive")
            return result

        x = transform_flatquant_activation(x, left_trans, right_trans, diag, left_dim, right_dim)
        quantized_x = fake_mx_quantize(x, self.spec.mx_format, self.spec.group_size)
        return F.linear(quantized_x, layer.weight, bias)
