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
"""Lightweight audit harness for Fake-MX A/B verification.

Provides event logging and tensor dump so that the ModelSlim integration
path and an independent intrusive path can be compared tensor-by-tensor.

Audit configuration is embedded in ``quant_model_description.json``
under the ``fake_mx_audit`` key.  When the key is absent, every function
is a no-op with zero tensor copies and zero file IO.
"""

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger

AuditMode = str  # "bf16" | "modelslim" | "intrusive"
AuditStage = str
AuditKind = str  # "weight" | "act"


@dataclass(frozen=True)
class FakeMXSpec:
    """Resolved, immutable quantization spec for a single prefix.

    Both the ModelSlim path and the intrusive path consume the same
    spec so that the only experimental variable is the integration
    mechanism, not the quantization parameters.
    """

    quant_type: str
    algorithm: str  # "rtn" | "rht" | "flatquant" | "hadamard_learning"
    mx_format: str  # "mxfp4" | "mxfp8"
    group_size: int
    targets: frozenset[str]
    matrix_size: int | None = None
    params_path: str | None = None


@dataclass
class AuditContext:
    """Per-call context carried through the audit chain."""

    mode: AuditMode = "modelslim"
    prefix: str = ""
    target: str = ""
    scheme: str = ""
    algorithm: str = ""
    format: str = ""
    group_size: int = 32
    rank: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# Explicit quant_type -> (algorithm, mx_format) mapping.
# Unknown quant types must NOT silently fall back to RTN.
QUANT_TYPE_SPECS: dict[str, tuple[str, str]] = {
    "W4A4_MXFP4_FAKE": ("rtn", "mxfp4"),
    "W8A8_MXFP8_FAKE": ("rtn", "mxfp8"),
    "W4A4_MXFP4_RHT_FAKE": ("rht", "mxfp4"),
    "W8A8_MXFP8_RHT_FAKE": ("rht", "mxfp8"),
    "W4A4_MXFP4_HADAMARD_LEARNING_FAKE": ("hadamard_learning", "mxfp4"),
    "W8A8_MXFP8_HADAMARD_LEARNING_FAKE": ("hadamard_learning", "mxfp8"),
    "W4A4_MXFP4_FLATQUANT_FAKE": ("flatquant", "mxfp4"),
    "W8A8_MXFP8_FLATQUANT_FAKE": ("flatquant", "mxfp8"),
    "W4A4_MXFP4_OMNIQUANT_FAKE": ("omniquant", "mxfp4"),
    "W8A8_MXFP8_OMNIQUANT_FAKE": ("omniquant", "mxfp8"),
    "W4A4_MXFP4_AUTOROUND_FAKE": ("autoround", "mxfp4"),
    "W8A8_MXFP8_AUTOROUND_FAKE": ("autoround", "mxfp8"),
    "W4A4_MXFP4_LWC_FAKE": ("lwc", "mxfp4"),
    "W8A8_MXFP8_LWC_FAKE": ("lwc", "mxfp8"),
    "W4A4_MXFP4_LAC_FAKE": ("lac", "mxfp4"),
    "W8A8_MXFP8_LAC_FAKE": ("lac", "mxfp8"),
}

# Algorithms supported by the Intrusive injection path.
# Algorithms not in this set will fail-fast in intrusive mode
# rather than silently degrading to RTN.
INTRUSIVE_SUPPORTED_ALGORITHMS = frozenset({"rtn", "flatquant", "hadamard_learning"})


def _parse_quant_type(quant_type: str) -> tuple[str, str]:
    """Map a quant_type string to (algorithm, mx_format) via explicit table.

    Raises ValueError for unknown quant types — never silently falls back
    to RTN.
    """
    spec = QUANT_TYPE_SPECS.get(quant_type)
    if spec is None:
        raise ValueError(f"Unknown quant_type {quant_type!r}. Supported: {sorted(QUANT_TYPE_SPECS)}")
    return spec


_audit_config_cache: dict[str, Any] | None | bool = False  # False = not yet loaded
_intrusive_quant_desc_cache: dict[str, Any] | None = None  # full quant_description for intrusive
_selected_prefixes: set[str] = set()  # prefixes that emitted node_selected


def _set_audit_config(cfg: dict[str, Any] | None, quant_description: dict[str, Any] | None = None) -> None:
    """Explicitly inject the audit configuration.

    Called by :class:`AscendModelSlimConfig.__init__` after
    ``quant_description`` is parsed.  Also caches the full
    ``quant_description`` so that intrusive mode can resolve specs
    without calling ``get_current_vllm_config()`` (which is unreliable
    in EngineCore spawn subprocesses and profile_run).

    Args:
        cfg: The ``fake_mx_audit`` sub-dict from quant_description.
        quant_description: The full quant_description dict, needed by
            intrusive mode to resolve per-prefix specs.  Optional —
            only needed when ``cfg["mode"] == "intrusive"``.
    """
    global _audit_config_cache, _intrusive_quant_desc_cache
    if cfg is None:
        _audit_config_cache = None
        _intrusive_quant_desc_cache = None
        return
    if not isinstance(cfg, dict):
        _audit_config_cache = None
        _intrusive_quant_desc_cache = None
        return
    _validate_config(dict(cfg), "injected")
    _audit_config_cache = dict(cfg)
    if quant_description is not None:
        _intrusive_quant_desc_cache = dict(quant_description)
    logger.info(
        "Fake-MX audit enabled (injected): mode=%s output_dir=%s layers=%s capture=%s max_calls=%s",
        _audit_config_cache.get("mode"),
        _audit_config_cache.get("output_dir"),
        _audit_config_cache.get("layers", ["*"]),
        _audit_config_cache.get("capture", []),
        _audit_config_cache.get("max_calls", 1),
    )


def _load_audit_config() -> dict[str, Any] | None:
    """Return the audit config injected by ``_set_audit_config()``.

    The sole source is :func:`_set_audit_config`, called by
    :class:`AscendModelSlimConfig` after parsing
    ``quant_description["fake_mx_audit"]``.  No env-var, no
    ``get_current_vllm_config()`` lookup — one path, zero ambiguity.
    """
    global _audit_config_cache
    if _audit_config_cache is False:
        # Not yet injected — audit is off.
        _audit_config_cache = None
    return _audit_config_cache  # type: ignore[return-value]


def _validate_config(cfg: dict[str, Any], path: str) -> None:
    """Validate the audit config fields at load time."""
    mode = cfg.get("mode", "modelslim")
    if mode not in ("bf16", "modelslim", "intrusive"):
        raise ValueError(f"audit mode must be bf16/modelslim/intrusive, got {mode!r}")
    # intrusive mode reads quant_description from vllm_config.quant_config
    # (the same quant_description that embeds fake_mx_audit), so no
    # separate file path is needed.
    if not cfg.get("output_dir"):
        cfg["output_dir"] = "/tmp/fake_mx_audit"
    cfg.setdefault("trace", False)
    cfg.setdefault("capture", [])
    cfg.setdefault("layers", ["*"])
    cfg.setdefault("max_calls", 1)


def audit_enabled() -> bool:
    """Return True when audit is active."""
    return _load_audit_config() is not None


def get_audit_mode() -> AuditMode:
    """Return the active audit mode."""
    cfg = _load_audit_config()
    if cfg is None:
        return "modelslim"
    return cfg.get("mode", "modelslim")


def _prefix_matches(prefix: str, patterns: list[str]) -> bool:
    """Glob-match *prefix* against *patterns* using fnmatchcase.

    Also tries stripping a ``language_model.`` prefix because vLLM's
    Qwen3.5 wrapper maps ``model.layers.*`` to
    ``language_model.model.layers.*``.
    """
    if any(fnmatchcase(prefix, p) or fnmatchcase(f"{prefix}.weight", p) for p in patterns):
        return True
    if prefix.startswith("language_model."):
        stripped = prefix[len("language_model.") :]
        return any(fnmatchcase(stripped, p) or fnmatchcase(f"{stripped}.weight", p) for p in patterns)
    return False


def _should_trace(prefix: str) -> bool:
    cfg = _load_audit_config()
    if cfg is None or not cfg.get("trace", False):
        return False
    return _prefix_matches(prefix, cfg.get("layers", ["*"]))


def _should_capture(prefix: str, stage: AuditStage) -> bool:
    cfg = _load_audit_config()
    if cfg is None:
        return False
    if stage not in cfg.get("capture", []):
        return False
    return _prefix_matches(prefix, cfg.get("layers", ["*"]))


def _get_output_dir() -> str:
    cfg = _load_audit_config()
    if cfg is None:
        return "/tmp/fake_mx_audit"
    return cfg.get("output_dir", "/tmp/fake_mx_audit")


def _get_max_calls() -> int:
    cfg = _load_audit_config()
    if cfg is None:
        return 1
    return int(cfg.get("max_calls", 1))


# ---- Call index management (per prefix+kind) ----

_call_counters: dict[tuple[str, AuditKind], int] = {}


def _next_call_index(prefix: str, kind: AuditKind) -> int:
    key = (prefix, kind)
    idx = _call_counters.get(key, 0)
    _call_counters[key] = idx + 1
    return idx


def _safe_filename(text: str) -> str:
    return text.replace(".", "_").replace("/", "_").replace(" ", "_")


def make_context(
    prefix: str = "",
    *,
    target: str = "",
    scheme: str = "",
    algorithm: str = "",
    fmt: str = "",
    group_size: int = 32,
) -> AuditContext:
    """Build an :class:`AuditContext` populated with the current mode and rank."""
    return AuditContext(
        mode=get_audit_mode(),
        prefix=prefix,
        target=target,
        scheme=scheme,
        algorithm=algorithm,
        format=fmt,
        group_size=group_size,
        rank=get_tensor_model_parallel_rank(),
    )


def audit_event(ctx: AuditContext, event: str, **fields: Any) -> None:
    """Append a JSON event record to ``events.rank<N>.jsonl``."""
    if not audit_enabled():
        logger.debug("audit_event skipped: audit not enabled (event=%s prefix=%s)", event, ctx.prefix)
        return
    if not _should_trace(ctx.prefix):
        logger.debug("audit_event skipped: prefix not traced (event=%s prefix=%s)", event, ctx.prefix)
        return
    # Track selected prefixes for the 0-node assertion.
    if event == "node_selected":
        _selected_prefixes.add(ctx.prefix)
    output_dir = _get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"events.rank{ctx.rank}.jsonl")
    record: dict[str, Any] = {
        "event": event,
        "mode": ctx.mode,
        "prefix": ctx.prefix,
        "target": ctx.target,
        "scheme": ctx.scheme,
        "algorithm": ctx.algorithm,
        "format": ctx.format,
        "group_size": ctx.group_size,
    }
    record.update(fields)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---- Bundle-based tensor capture (P0-2 + P1-5 fix) ----


class AuditCall:
    """Context manager collecting all tensors for one call into a single bundle.

    Usage::

        with audit_call(ctx, kind="act") as call:
            call.capture("act_raw", x)
            call.capture("act_transformed", tx)
            call.capture("act_qdq", qx)
            call.capture("output", y)

    All captured tensors are saved to one ``.pt`` file at ``__exit__``
    so that a single forward call produces a single bundle, regardless
    of how many stages are captured.
    """

    def __init__(self, ctx: AuditContext, kind: AuditKind = "act"):
        self.ctx = ctx
        self.kind = kind
        self.tensors: dict[str, torch.Tensor] = {}
        self._call_index = -1
        self._active = False

    def capture(self, stage: AuditStage, tensor: torch.Tensor) -> None:
        if not self._active:
            return
        if not _should_capture(self.ctx.prefix, stage):
            return
        self.tensors[stage] = tensor.detach().to("cpu", copy=True)

    def __enter__(self) -> "AuditCall":
        self._call_index = _next_call_index(self.ctx.prefix, self.kind)
        self._active = self._call_index < _get_max_calls()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if not self._active or not self.tensors:
            return
        tensor_dir = os.path.join(_get_output_dir(), "tensors")
        os.makedirs(tensor_dir, exist_ok=True)
        safe_prefix = _safe_filename(self.ctx.prefix)
        filename = f"{safe_prefix}.rank{self.ctx.rank}.{self.kind}.call{self._call_index}.pt"
        path = os.path.join(tensor_dir, filename)
        bundle: dict[str, Any] = {
            "metadata": {
                "prefix": self.ctx.prefix,
                "kind": self.kind,
                "call_index": self._call_index,
                "rank": self.ctx.rank,
                "mode": self.ctx.mode,
                "scheme": self.ctx.scheme,
                "algorithm": self.ctx.algorithm,
                "format": self.ctx.format,
                "group_size": self.ctx.group_size,
            },
            "tensors": {
                stage: {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "data": t,
                }
                for stage, t in self.tensors.items()
            },
        }
        torch.save(bundle, path)
        logger.debug(
            "audit_call: prefix=%s kind=%s call=%d stages=%s -> %s",
            self.ctx.prefix,
            self.kind,
            self._call_index,
            list(self.tensors.keys()),
            path,
        )


def audit_call(ctx: AuditContext, kind: AuditKind = "act") -> AuditCall:
    """Create an :class:`AuditCall` context manager."""
    return AuditCall(ctx, kind)


# ---- Intrusive policy resolution (P0-1 fix) ----


def _load_intrusive_quant_description() -> dict[str, Any] | None:
    """Return the cached quant_description for intrusive mode.

    The quant_description is injected by ``_set_audit_config()`` at
    config initialization time, so no ``get_current_vllm_config()``
    call is needed — this works in all subprocess and profile_run
    contexts.
    """
    cfg = _load_audit_config()
    if cfg is None or cfg.get("mode") != "intrusive":
        return None
    if _intrusive_quant_desc_cache is None:
        raise FileNotFoundError(
            "intrusive audit mode requires quant_description to be "
            "injected alongside fake_mx_audit in AscendModelSlimConfig"
        )
    return _intrusive_quant_desc_cache


def resolve_fake_mx_spec(
    prefix: str,
    quant_description: dict[str, Any],
    packed_modules_mapping: dict[str, Any] | None = None,
) -> FakeMXSpec | None:
    """Resolve the final :class:`FakeMXSpec` for *prefix*.

    Both the ModelSlim path and the intrusive path call this function
    so that the spec is guaranteed identical for the same prefix +
    quant_description.
    """
    from vllm_ascend.quantization.modelslim_config import get_quant_type_for_layer

    if packed_modules_mapping is None:
        packed_modules_mapping = {}
    quant_type = get_quant_type_for_layer(quant_description, prefix, "linear", packed_modules_mapping)
    if quant_type is None or quant_type == "FLOAT":
        return None

    algorithm, mx_format = _parse_quant_type(quant_type)
    group_size = int(quant_description.get("group_size", 32))
    targets = frozenset(quant_description.get("fake_mx_quant_targets", ["attn-linear"]))

    matrix_size = None
    params_path = None
    if algorithm == "flatquant":
        matrix_size = int(quant_description.get("flatquant_matrix_size", 128))
        params_path = quant_description.get("flatquant_params_path")
    elif algorithm == "hadamard_learning":
        matrix_size = int(quant_description.get("hadamard_learning_matrix_size", 128))
        params_path = quant_description.get("lht_params_path")

    return FakeMXSpec(
        quant_type=quant_type,
        algorithm=algorithm,
        mx_format=mx_format,
        group_size=group_size,
        targets=targets,
        matrix_size=matrix_size,
        params_path=params_path,
    )


def get_intrusive_spec(prefix: str) -> FakeMXSpec | None:
    """Resolve the :class:`FakeMXSpec` for *prefix* in intrusive mode.

    Returns ``None`` when audit is off, mode is not intrusive, or the
    prefix resolves to FLOAT / no quant type.
    """
    cfg = _load_audit_config()
    if cfg is None or cfg.get("mode") != "intrusive":
        return None
    quant_description = _load_intrusive_quant_description()
    if quant_description is None:
        return None
    # Resolve packed_modules_mapping from the cached quant_description's
    # model_type, avoiding get_current_vllm_config() which fails in
    # profile_run and spawn subprocesses.
    packed_modules_mapping: dict[str, Any] = {}
    try:
        from vllm_ascend.quantization.modelslim_config import packed_modules_model_mapping

        packed_modules_mapping = packed_modules_model_mapping.get("qwen3_5", {})
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Failed to load packed_modules_mapping for intrusive mode: {exc}") from exc
    spec = resolve_fake_mx_spec(prefix, quant_description, packed_modules_mapping)
    if spec is not None and spec.algorithm not in INTRUSIVE_SUPPORTED_ALGORITHMS:
        raise NotImplementedError(
            f"Intrusive mode does not support algorithm {spec.algorithm!r} "
            f"(quant_type={spec.quant_type!r}) for prefix {prefix!r}. "
            f"Supported algorithms: {sorted(INTRUSIVE_SUPPORTED_ALGORITHMS)}."
        )
    return spec


def get_intrusive_policy(prefix: str) -> FakeMXSpec | None:
    """Backward-compatible alias for :func:`get_intrusive_spec`."""
    return get_intrusive_spec(prefix)


def flush_capture() -> None:
    """Flush any buffered audit data."""
    pass


def assert_selected_count() -> None:
    """Assert that at least one node was selected in intrusive mode.

    Call after model loading is complete.  If 0 nodes were selected,
    the configuration is likely wrong (glob patterns, prefix mapper,
    or packed mapping) and the run would produce meaningless results.
    """
    cfg = _load_audit_config()
    if cfg is None or cfg.get("mode") != "intrusive":
        return
    count = len(_selected_prefixes)
    if count == 0:
        raise RuntimeError(
            f"Intrusive audit mode selected 0 nodes. "
            f"Check layers patterns {cfg.get('layers', [])} and quant_description."
        )
    logger.info(
        "Intrusive audit: selected %d nodes (%s)",
        count,
        sorted(_selected_prefixes),
    )


__all__ = [
    "AuditCall",
    "AuditContext",
    "AuditKind",
    "AuditMode",
    "AuditStage",
    "FakeMXSpec",
    "assert_selected_count",
    "audit_call",
    "audit_enabled",
    "audit_event",
    "flush_capture",
    "get_audit_mode",
    "get_intrusive_policy",
    "get_intrusive_spec",
    "make_context",
    "resolve_fake_mx_spec",
]
