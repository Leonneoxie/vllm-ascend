"""Unified Fake-MX boundary QDQ entry point.

Core model code (patch_qwen3_5.py, ops/gdn.py) calls
``apply_fake_mx_boundary`` instead of directly importing audit functions.
This keeps boundary logic in one place and ensures consistent event/tensor
capture across attn-cache and gdn-core boundaries.
"""

from __future__ import annotations

from typing import Literal

import torch

from vllm_ascend.quantization.fake_mx import (
    _resolve_mx_params,
    _resolve_target_decision,
    fake_mx_quantize,
)

BoundaryTarget = Literal["attn-cache", "gdn-core"]

_BOUNDARY_CAPTURE_STAGES: dict[str, list[str]] = {
    "attn-cache": [
        "attn_q_raw",
        "attn_q_qdq",
        "attn_k_raw",
        "attn_k_qdq",
        "attn_v_raw",
        "attn_v_qdq",
    ],
    "gdn-core": [
        "gdn_qkv_raw",
        "gdn_qkv_qdq",
    ],
}

_BOUNDARY_SHAPE_KEYS: dict[str, list[str]] = {
    "attn-cache": ["q_shape", "k_shape", "v_shape"],
    "gdn-core": ["mixed_qkv_shape"],
}


def apply_fake_mx_boundary(
    *,
    target: BoundaryTarget,
    owner: torch.nn.Module,
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Apply fake-MX QDQ at a non-Linear operator boundary.

    Resolves the boundary decision, applies QDQ if the target is enabled,
    and records audit events + tensor bundles.  When the target is
    disabled the original tensors are returned unchanged.

    Args:
        target: Boundary identifier (``"attn-cache"`` or ``"gdn-core"``).
        owner: The projection layer that owns the quant config.
        tensors: Input tensors to quantize (Q, K, V for attn-cache;
            a single mixed_qkv for gdn-core).

    Returns:
        Tensors after boundary QDQ (same count as input).
    """
    from vllm_ascend.quantization.fake_mx_audit import (
        audit_enabled,
    )

    applied, reason = _resolve_target_decision(owner, target)
    raw_tensors = tensors

    if not applied:
        if audit_enabled():
            _emit_boundary_event(owner, target, raw_tensors, raw_tensors, applied, reason)
        return tensors

    fmt, group_size = _resolve_mx_params(owner)
    quantized = tuple(fake_mx_quantize(t, fmt, group_size) for t in tensors)

    if audit_enabled():
        _emit_boundary_event(owner, target, raw_tensors, quantized, applied, reason)

    return quantized


def _emit_boundary_event(
    owner: torch.nn.Module,
    target: BoundaryTarget,
    raw_tensors: tuple[torch.Tensor, ...],
    quantized: tuple[torch.Tensor, ...],
    applied: bool,
    reason: str,
) -> None:
    from vllm_ascend.quantization.fake_mx_audit import (
        audit_call,
        audit_event,
        make_context,
    )

    prefix = getattr(owner, "prefix", "") or ""
    method = getattr(owner, "quant_method", None)
    scheme = getattr(method, "quant_method", None)
    scheme_name = type(scheme).__name__ if scheme else "None"
    ctx = make_context(prefix, scheme=scheme_name, target=target)

    event_name = target.replace("-", "_") + "_qdq"
    shape_keys = _BOUNDARY_SHAPE_KEYS.get(target, [])
    shape_fields = {shape_keys[i]: list(t.shape) for i, t in enumerate(quantized) if i < len(shape_keys)}
    audit_event(ctx, event_name, applied=applied, reason=reason, **shape_fields)

    if applied:
        stages = _BOUNDARY_CAPTURE_STAGES.get(target, [])
        with audit_call(ctx, kind=target) as call:
            for i, (raw, qdq) in enumerate(zip(raw_tensors, quantized)):
                base = i * 2
                if base + 1 < len(stages):
                    call.capture(stages[base], raw)
                    call.capture(stages[base + 1], qdq)


def trace_decoder_layer_forward(
    *,
    prefix: str,
    layer_idx: int,
    layer_type: str,
    hidden_states_shape: list[int],
    flash_comm_v1: bool,
) -> None:
    """Record a diagnostic event for decoder layer forward entry."""
    from vllm_ascend.quantization.fake_mx_audit import audit_enabled, audit_event, make_context

    if audit_enabled():
        ctx = make_context(prefix)
        audit_event(
            ctx,
            "decoder_layer_forward",
            layer_idx=layer_idx,
            layer_type=layer_type,
            flash_comm_v1=flash_comm_v1,
            hidden_states_shape=hidden_states_shape,
        )
