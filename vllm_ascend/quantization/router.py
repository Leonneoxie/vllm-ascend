"""Quantization domain router.

Provides :func:`resolve_quant_route` to centralize the decision of which
quantization path each layer should take, replacing the inline audit
special-case that was previously hardcoded inside
``AscendModelSlimConfig.get_quant_method()``.

The router does **not** execute QDQ, load parameters, or write weights.
It does **not** infer layer type from vLLM class hierarchies — the caller
determines ``domain`` and ``selected`` using its own already-imported
type checks and metadata.
"""

from __future__ import annotations

from typing import Literal

QuantDomain = Literal["linear", "attention", "kv_cache", "moe", "embedding"]
RouteAction = Literal["modelslim", "unquantized", "intrusive", "unsupported"]


def resolve_quant_route(
    *,
    domain: QuantDomain,
    prefix: str,
    selected: bool = False,
) -> RouteAction:
    """Decide which quantization path a layer should take.

    The decision is based on the current audit mode, the layer's domain,
    and whether the layer is selected for quantization.

    Routing matrix:

    +------------+----------+--------------------------+
    | domain     | bf16     | intrusive                |
    +============+==========+==========================+
    | linear     | unquant  | intrusive                |
    | attention  | unquant  | unsupported if selected  |
    | kv_cache   | unquant  | unsupported if selected  |
    | moe        | unquant  | unsupported if selected  |
    | embedding  | unquant  | unsupported if selected  |
    +------------+----------+--------------------------+

    In intrusive mode, non-linear domains that are **not** selected
    return ``unquantized`` (e.g. an embedding layer that has no quant
    target).  Non-linear domains that **are** selected return
    ``unsupported`` because intrusive mode only implements the linear
    domain.

    Args:
        domain: Layer domain, determined by the caller via isinstance.
        prefix: Normalized layer prefix (for logging/diagnostics).
        selected: Whether this layer is selected for quantization in
            the current quant config.  The caller determines this using
            its own metadata (``is_fa_quant_layer``, ``is_c8_quant_layer``,
            ``_has_quant_weight``, ``is_layer_skipped_ascend``, etc.).

    Returns:
        One of ``"modelslim"``, ``"unquantized"``, ``"intrusive"``,
        ``"unsupported"``.
    """
    from vllm_ascend.quantization.fake_mx_audit import audit_enabled, get_audit_mode

    if not audit_enabled():
        return "modelslim"

    mode = get_audit_mode()
    if mode == "bf16":
        return "unquantized"

    if mode == "intrusive":
        if domain == "linear":
            return "intrusive"
        if selected:
            return "unsupported"
        return "unquantized"

    return "modelslim"
