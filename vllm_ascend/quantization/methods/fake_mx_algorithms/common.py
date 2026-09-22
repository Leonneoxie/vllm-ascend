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
"""Artifact loading shared by the five core algorithms."""

import os
from functools import cache
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger

DEFAULT_TRANSFORM_MATRIX_SIZE = 128


def _quant_description() -> dict[str, Any]:
    """Return the active fake-MX experiment configuration."""
    return get_current_vllm_config().quant_config.quant_description


def _resolve_model_artifact(path: str) -> str:
    """Resolve an algorithm artifact relative to the loaded model."""
    if os.path.isabs(path):
        return path
    model_path = get_current_vllm_config().model_config.model
    return os.path.join(model_path, path)


@cache
def _load_safetensors(resolved_path: str) -> dict[str, torch.Tensor]:
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Fake-MX transform artifact not found: {resolved_path}")

    from safetensors.torch import load_file

    params = load_file(resolved_path)
    logger.info("Loaded %d fake-MX transform params from %s", len(params), resolved_path)
    return params


def _load_transform_params(path: str) -> dict[str, torch.Tensor]:
    """Load one immutable transform artifact once per worker process."""
    return _load_safetensors(_resolve_model_artifact(path))


def _layer_prefix(layer: torch.nn.Module) -> str:
    """Get the full module path from a layer.

    vLLM's Linear modules (RowParallelLinear, ColumnParallelLinear) set
    ``self.prefix`` in ``__init__``.  vLLM's FusedMoE module (RoutedExperts)
    stores the same value as ``self.layer_name`` instead.  Return whichever
    is available so sidecar key construction works for both Linear and MoE.
    """
    return getattr(layer, "prefix", "") or getattr(layer, "layer_name", "") or ""


def _layer_prefix_candidates(layer: torch.nn.Module) -> tuple[str, ...]:
    """Map vLLM layer prefixes to the prefixes emitted by AMCT exporters."""
    prefix = _layer_prefix(layer)
    logical_prefixes = [prefix]
    if prefix.endswith(".gate_up_proj"):
        # vLLM physically fuses the logical gate/up projections. AMCT exports
        # the same input transform under both logical names; accept either.
        logical_prefixes.extend(
            [
                prefix.removesuffix("gate_up_proj") + "gate_proj",
                prefix.removesuffix("gate_up_proj") + "up_proj",
            ]
        )
    candidates = []
    for candidate in logical_prefixes:
        candidates.extend(
            [
                candidate,
                f"model.{candidate}",
                candidate.replace("language_model.model.", "model.language_model."),
            ]
        )
    return tuple(dict.fromkeys(candidates))


def _find_transform_param(
    params: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    suffix: str,
) -> tuple[str, torch.Tensor] | None:
    """Find the sidecar parameter for a layer, validating fused projections.

    For a fused ``gate_up_proj`` the AMCT export may carry the same
    transform under the physical name and/or both logical names. All
    present candidates must agree bit-for-bit: vLLM applies ONE transform
    to the shared activation, so diverging per-projection transforms
    cannot be honoured and must fail loudly instead of silently picking
    the first match.
    """
    matches = [
        (key, params[key])
        for key in (f"{prefix}.{suffix}" for prefix in _layer_prefix_candidates(layer))
        if key in params
    ]
    if not matches:
        return None
    if len(matches) > 1:
        first_key, first_value = matches[0]
        first_fp32 = first_value.to(torch.float32)
        for key, value in matches[1:]:
            if value.shape != first_value.shape or not torch.allclose(value.to(torch.float32), first_fp32):
                raise ValueError(
                    f"Fused projection transform mismatch for {suffix!r}: {first_key!r} and {key!r} "
                    "carry different values. vLLM fuses gate/up into a single gate_up_proj with one "
                    "shared input transform, so diverging per-projection transforms cannot be "
                    "honoured. Re-export the sidecar with one shared transform (or align the "
                    "calibration) and retry."
                )
        logger.debug("Fused projection transform %r: %d matching keys, all identical", suffix, len(matches))
    return matches[0]


def _copy_transform_param(
    target: torch.Tensor,
    params: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    suffix: str,
    *,
    required: bool = True,
) -> str | None:
    match = _find_transform_param(params, layer, suffix)
    if match is None:
        if required:
            prefix = getattr(layer, "prefix", "") or "<unknown>"
            raise KeyError(f"Missing {suffix!r} transform parameter for layer {prefix!r}.")
        return None

    key, value = match
    if target.shape != value.shape:
        raise ValueError(f"Transform parameter {key!r} has shape {tuple(value.shape)}, expected {tuple(target.shape)}.")
    target.copy_(value.to(device=target.device, dtype=target.dtype))
    return key


def _copy_expert_transform_param(
    target: torch.Tensor,
    params: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    suffix: str,
) -> str:
    """Copy a full-expert sidecar tensor into the local expert slots.

    ``layer._expert_map`` maps logical expert ID to the local physical slot
    (or ``-1`` when the expert is not hosted on this rank).  When absent
    (EP=1) the sidecar is copied through unchanged.
    """
    expert_map = getattr(layer, "_expert_map", None)
    if expert_map is None:
        return _copy_transform_param(target, params, layer, suffix)

    match = _find_transform_param(params, layer, suffix)
    if match is None:
        prefix = _layer_prefix(layer) or "<unknown>"
        raise KeyError(f"Missing {suffix!r} transform parameter for layer {prefix!r}.")

    key, value = match
    if value.ndim != target.ndim or value.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"Transform parameter {key!r} has shape {tuple(value.shape)}, expected "
            f"(*, {', '.join(str(size) for size in target.shape[1:])})."
        )
    expert_map_values = expert_map.detach().cpu().tolist()
    if value.shape[0] != len(expert_map_values):
        raise ValueError(
            f"Transform parameter {key!r} has {value.shape[0]} experts, expected "
            f"{len(expert_map_values)} from _expert_map."
        )

    local_expert_count = sum(slot >= 0 for slot in expert_map_values)
    if target.shape[0] != local_expert_count:
        raise ValueError(
            f"Transform parameter target for {key!r} has {target.shape[0]} local experts, "
            f"but _expert_map selects {local_expert_count}."
        )
    for logical_expert, physical_slot in enumerate(expert_map_values):
        if physical_slot >= 0:
            if physical_slot >= target.shape[0]:
                raise ValueError(
                    f"_expert_map for {key!r} contains invalid local slot {physical_slot}."
                )
            target[physical_slot].copy_(
                value[logical_expert].to(device=target.device, dtype=target.dtype)
            )
    return key


def _inverse_fp32(matrix: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Compute a stable fp32 inverse while retaining the tested matrix orientation."""
    source = matrix.t() if transpose else matrix
    source = source.to(torch.float32)
    identity = torch.eye(source.shape[0], device=source.device, dtype=torch.float32)
    return torch.linalg.solve(source, identity)


def _make_rademacher_signs(size: int, seed: int) -> torch.Tensor:
    """Generate Rademacher signs (+/-1) with a fixed seed, matching AMCT."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
    return signs.mul_(2).sub_(1)


def _build_flatquant_sidecar_key(layer_prefix: str, expert_idx: int, fc_name: str, comp: str) -> str:
    """Build sidecar tensor key for a per-expert transform state.

    Format: layers.{N}.experts.{E}.{fc}.{comp_short}
    Example: layers.0.experts.17.fc1.left_trans
    """
    parts = layer_prefix.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer_idx = parts[i + 1]
            break
    if layer_idx is None:
        raise ValueError(
            f"Fake-MX MoE sidecar key cannot be built: layer prefix "
            f"{layer_prefix!r} does not contain 'layers.{{N}}'. "
            f"This usually means the FusedMoE layer has no prefix "
            f"attribute set (e.g. MTP layers). Add '\"*mtp*\": \"FLOAT\"' "
            f"to module_quant_overrides to exclude MTP from the transform."
        )

    comp_short = "diag" if comp == "diag_scale" else comp
    return f"layers.{layer_idx}.experts.{expert_idx}.{fc_name}.{comp_short}"


def _physical_to_logical_experts(expert_map: torch.Tensor | None) -> dict[int, int] | None:
    """Reverse ``_expert_map`` (logical -> physical slot or -1) into slot -> logical."""
    if expert_map is None:
        return None
    return {
        logical_id: int(expert_map[logical_id].item())
        for logical_id in range(expert_map.numel())
        if int(expert_map[logical_id].item()) != -1
    }


def _validate_weight_state(algorithm: str, required: str | None, quant_description: dict[str, Any]) -> None:
    """Enforce fake_mx_weight_state for prequantized checkpoint schemes."""
    if required is None:
        return
    weight_state = quant_description.get("fake_mx_weight_state")
    if weight_state != required:
        raise ValueError(f"{algorithm} fake-MX requires fake_mx_weight_state={required!r}, got {weight_state!r}.")
