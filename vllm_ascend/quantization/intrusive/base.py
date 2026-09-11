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

from abc import ABC, abstractmethod

import torch

from vllm_ascend.quantization.fake_mx_audit import (
    FakeMXSpec,
    audit_enabled,
    audit_event,
    get_audit_mode,
    get_intrusive_spec,
    make_context,
)


class IntrusiveLinearAdapter(ABC):
    """Base interface for intrusive Fake-MX injection adapters.

    Each subclass manages one algorithm's weight and activation
    lifecycle.  Adapters call shared math functions
    (``fake_mx_quantize``, ``transform_*``) — they never reimplement
    quantization formulas.

    Lifecycle:

    1. ``maybe_create_intrusive_linear_adapter(layer)`` resolves the
       spec and instantiates the adapter via the registry.
    2. ``adapter.process_weight(layer)`` transforms and QDQs the weight,
       writing the result back to ``layer.weight.data``.  Failure must
       raise — never silently return.
    3. ``adapter.apply(layer, x, bias)`` transforms and QDQs the
       activation, then calls ``F.linear``.
    """

    algorithm: str = ""

    def __init__(self, spec: FakeMXSpec) -> None:
        if spec.algorithm != self.algorithm:
            raise ValueError(
                f"Spec algorithm {spec.algorithm!r} does not match "
                f"adapter algorithm {self.algorithm!r}"
            )
        self.spec = spec

    @abstractmethod
    def process_weight(self, layer: torch.nn.Module) -> None:
        """Transform + QDQ the weight and write back to ``layer.weight.data``.

        Success means the QDQ'd weight has been committed.  Any
        precondition failure (missing params, invalid dims) must raise
        an exception — never silently return.
        """

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform + QDQ the activation, then ``F.linear``."""

    # ---- Shared audit helpers ----

    def _make_ctx(self, layer: torch.nn.Module):
        prefix = getattr(layer, "prefix", "") or ""
        return make_context(
            prefix,
            scheme=f"Intrusive{self.algorithm.capitalize()}",
            algorithm=self.algorithm,
            fmt=self.spec.mx_format,
            group_size=self.spec.group_size,
        )


def maybe_create_intrusive_linear_adapter(layer: torch.nn.Module) -> IntrusiveLinearAdapter | None:
    """Resolve and instantiate an intrusive adapter for *layer*.

    Returns ``None`` when audit is off, mode is not intrusive, or the
    layer's prefix resolves to FLOAT / no quant type.

    On success, emits ``node_selected`` and returns a ready adapter.
    On failure (unsupported algorithm, spec mismatch), raises.
    """
    if not audit_enabled() or get_audit_mode() != "intrusive":
        return None

    prefix = getattr(layer, "prefix", "") or ""
    spec = get_intrusive_spec(prefix)
    if spec is None:
        return None

    from .registry import get_intrusive_adapter_class

    adapter_cls = get_intrusive_adapter_class(spec.algorithm)
    adapter = adapter_cls(spec)

    # Emit node_selected after successful adapter creation.
    ctx = adapter._make_ctx(layer)
    audit_event(
        ctx,
        "node_selected",
        backend="intrusive",
        quant_type=spec.quant_type,
        algorithm=spec.algorithm,
        mx_format=spec.mx_format,
        group_size=spec.group_size,
    )
    return adapter
