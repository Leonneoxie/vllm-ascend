#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Algorithm-specific per-expert activation transforms for fake-MX MoE.

The shared MoE execution path (ops/fused_moe/moe_mlp.py) only retains the
grouping primitives (cumsum_group_list, grouped matmuls) and the two
transform invocation slots; the per-algorithm math lives here next to the
algorithm modules. The generic loop driver ``_apply_per_expert_transform``
is the fallback executor for transforms that have no vectorized form yet.
"""

import contextlib
from collections.abc import Callable

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_mlp import cumsum_group_list


def _apply_per_expert_transform(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    num_experts: int,
    transform_name: str,
    transform: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Transform dispatched expert rows and preserve any padded tail."""
    if group_list_type == 2:
        message = (
            "Per-expert fake-MX activation transforms do not support group_list_type=2 because "
            "key-value expert IDs cannot be mapped to local transforms without expert-range metadata. "
            "Use group_list_type 0 or 1."
        )
        logger.error(message)
        raise NotImplementedError(message)

    if hidden_states.shape[0] == 0:
        return hidden_states

    boundaries = cumsum_group_list(
        group_list,
        group_list_type,
        0,
        active_num=hidden_states.shape[0],
        expert_num=num_experts,
    )
    if boundaries.numel() != num_experts:
        raise ValueError(
            f"{transform_name} group_list does not match local transforms: "
            f"{boundaries.numel()} groups vs {num_experts} transforms."
        )

    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end_tensor in enumerate(boundaries):
        end = int(end_tensor.item())
        if end < start or end > hidden_states.shape[0]:
            raise ValueError(f"{transform_name} received invalid expert token boundaries.")
        if end > start:
            outputs.append(transform(hidden_states[start:end], expert_idx))
        start = end

    if start < hidden_states.shape[0]:
        outputs.append(hidden_states[start:])
    return torch.cat(outputs, dim=0) if outputs else hidden_states


# Maximum rows processed per chunk in the vectorized FlatQuant transform.
# Bounds the gathered per-row parameter tables (~90 MB at 2048 rows) so the
# transform stays memory-safe for arbitrarily large batches.
_FLATQUANT_MAX_ROWS_PER_CHUNK = 2048


def _reject_unsupported_group_list_type(group_list_type: int) -> None:
    """Fail fast on group_list layouts that cannot be mapped to local experts."""
    if group_list_type == 2:
        message = (
            "Per-expert fake-MX activation transforms do not support group_list_type=2 because "
            "key-value expert IDs cannot be mapped to local transforms without expert-range metadata. "
            "Use group_list_type 0 or 1."
        )
        logger.error(message)
        raise NotImplementedError(message)


def _validated_expert_boundaries(
    group_list: torch.Tensor,
    group_list_type: int,
    num_experts: int,
    active_rows: int,
    transform_name: str,
) -> tuple[torch.Tensor, list[int]]:
    """Validate group_list and return (cumulative boundaries tensor, host-side ends).

    Performs the same checks as the per-expert loop in
    ``_apply_per_expert_transform`` (group count matches the local experts,
    boundaries monotonic and within ``active_rows``) with a single
    device-to-host transfer instead of one ``.item()`` per expert.
    """
    boundaries = cumsum_group_list(
        group_list,
        group_list_type,
        0,
        active_num=active_rows,
        expert_num=num_experts,
    )
    if boundaries.numel() != num_experts:
        raise ValueError(
            f"{transform_name} group_list does not match local transforms: "
            f"{boundaries.numel()} groups vs {num_experts} transforms."
        )
    boundary_ends = boundaries.tolist()
    start = 0
    for end in boundary_ends:
        if end < start or end > active_rows:
            raise ValueError(f"{transform_name} received invalid expert token boundaries.")
        start = end
    _log_per_expert_transform_contract(transform_name, group_list_type, num_experts, boundary_ends, active_rows)
    return boundaries, boundary_ends


def _log_per_expert_transform_contract(
    transform_name: str,
    group_list_type: int,
    num_experts: int,
    boundary_ends: list[int],
    active_rows: int,
) -> None:
    """Log the dispatcher->transform contract once per stable message.

    Lets serving logs confirm that the communication mode delivered a
    group_list whose length and ordering match the local expert transforms
    (see design-v-14 §3.5). The INFO record only carries fields that are
    constant for a given (transform, comm, group_list_type) combination, so
    ``info_once`` deduplicates it to one line per rank. Per-call row counts
    (dispatched rows, padding rows) go to DEBUG.
    """
    comm_type = None
    # Forward context is not set in unit tests; the contract fields that
    # matter there are still logged below.
    with contextlib.suppress(AssertionError, RuntimeError):
        comm_type = _EXTRA_CTX.moe_comm_type
    comm_name = str(comm_type) if comm_type is not None else "unknown"
    # scope="process": every EP/TP rank must be verifiable, not just the
    # local-first one.
    logger.info_once(
        "fake-MX per-expert transform contract: transform=%s comm=%s group_list_type=%d groups=%d local_experts=%d",
        transform_name,
        comm_name,
        group_list_type,
        len(boundary_ends),
        num_experts,
        scope="process",
    )
    dispatched_rows = boundary_ends[-1] if boundary_ends else 0
    logger.debug(
        "fake-MX per-expert transform rows: transform=%s comm=%s group_list_type=%d dispatched_rows=%d padding_rows=%d",
        transform_name,
        comm_name,
        group_list_type,
        dispatched_rows,
        active_rows - dispatched_rows,
    )


def _row_to_expert(boundaries: torch.Tensor, rows: int) -> torch.Tensor:
    """Map each dispatched row to its local physical expert id.

    Padding rows beyond the dispatched tokens map to ``len(boundaries)``
    (one past the last expert) so they index the identity entry of padded
    parameter tables.
    """
    row_idx = torch.arange(rows, device=boundaries.device)
    return torch.searchsorted(boundaries.to(torch.int64), row_idx, right=True)


def _apply_expert_learned_hadamard(
    hidden_states: torch.Tensor,
    transform_weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    """Apply one learned block matrix to each expert's dispatched rows.

    Vectorized implementation. The per-expert transform multiplies every
    matrix_size-sized block of a row by that expert's matrix
    (``reshape(x, -1, matrix_size) @ T_e``). After a global reshape to
    sub-rows each expert's sub-rows form one contiguous segment, so the
    whole transform is a single ``npu_grouped_matmul`` whose group_list is
    the token group_list scaled by the block count. Rows beyond the
    dispatched tokens are zero-filled by the op and restored from the input
    afterwards. Operands use the input dtype, matching AMCT and Linear LHT.
    """
    if transform_weight.ndim != 3:
        raise ValueError(
            f"MoE Hadamard Learning transform must have shape [experts, K, K], got {tuple(transform_weight.shape)}."
        )
    if not transform_weight.is_floating_point():
        raise TypeError(f"MoE Hadamard Learning transform_weight must be floating point, got {transform_weight.dtype}.")
    if transform_weight.shape[-1] != transform_weight.shape[-2]:
        raise ValueError(
            f"MoE Hadamard Learning transform must have shape [experts, K, K], got {tuple(transform_weight.shape)}."
        )
    _reject_unsupported_group_list_type(group_list_type)

    rows, input_dim = hidden_states.shape
    if rows == 0:
        return hidden_states

    num_experts = transform_weight.shape[0]
    matrix_size = transform_weight.shape[-1]
    if input_dim % matrix_size:
        raise ValueError(
            f"Hadamard Learning input dimension ({input_dim}) must be divisible by matrix_size ({matrix_size})."
        )
    _, boundary_ends = _validated_expert_boundaries(
        group_list, group_list_type, num_experts, rows, "MoE Hadamard Learning"
    )

    blocks = input_dim // matrix_size
    if group_list.dtype != torch.int64:
        group_list = group_list.to(torch.int64)
    x_sub = hidden_states.reshape(-1, matrix_size)
    out_sub = torch_npu.npu_grouped_matmul(
        x=[x_sub],
        weight=[transform_weight.to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous()],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list * blocks,
    )[0]
    expected_sub_rows = rows * blocks
    if out_sub.shape[0] != expected_sub_rows:
        raise RuntimeError(
            "MoE Hadamard Learning grouped matmul returned unexpected row count: "
            f"{out_sub.shape[0]} (expected {expected_sub_rows})."
        )
    # npu_grouped_matmul zero-fills sub-rows beyond the dispatched tokens;
    # restore the original padding rows from the input.
    actual_sub_rows = boundary_ends[-1] * blocks
    keep_mask = torch.arange(expected_sub_rows, device=out_sub.device) < actual_sub_rows
    out_sub = torch.where(keep_mask.unsqueeze(-1), out_sub, x_sub)
    return out_sub.reshape(rows, input_dim).to(hidden_states.dtype)


def _apply_expert_flatquant(
    hidden_states: torch.Tensor,
    fc_state: dict[str, torch.Tensor],
    group_list: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    """Apply per-expert FlatQuant activation transform.

    Vectorized implementation. Every row is transformed by its own expert's
    Kronecker factors (``x' = L.T @ (reshape(x) * diag) @ R``); the transform
    has no cross-row dependency, so rows are processed in chunks of
    ``_FLATQUANT_MAX_ROWS_PER_CHUNK`` with batched matmuls after gathering
    per-row parameters through a row->expert map. Padding rows map to the
    identity entry appended to each parameter table and pass through
    unchanged.
    """
    _reject_unsupported_group_list_type(group_list_type)

    rows = hidden_states.shape[0]
    if rows == 0:
        return hidden_states

    left_trans = fc_state["left_trans"]  # [E, L, L]
    right_trans = fc_state["right_trans"]  # [E, R, R]
    diag_scale = fc_state.get("diag_scale")  # [E, K] or None

    num_experts = left_trans.shape[0]
    left_dim = left_trans.shape[-1]
    right_dim = right_trans.shape[-1]
    input_dim = hidden_states.shape[-1]
    if left_dim * right_dim != input_dim:
        raise ValueError(
            f"FlatQuant transform dimension mismatch: left({left_dim}) * right({right_dim}) != input({input_dim})."
        )
    boundaries, _ = _validated_expert_boundaries(group_list, group_list_type, num_experts, rows, "MoE FlatQuant")
    row_to_expert = _row_to_expert(boundaries, rows)

    dtype = hidden_states.dtype
    device = hidden_states.device
    # Padded parameter tables whose last entry is the identity transform;
    # padding rows map there and pass through unchanged.
    left_table = torch.cat([left_trans.to(dtype), torch.eye(left_dim, dtype=dtype, device=device).unsqueeze(0)])
    right_table = torch.cat([right_trans.to(dtype), torch.eye(right_dim, dtype=dtype, device=device).unsqueeze(0)])
    diag_table = None
    if diag_scale is not None:
        diag_table = torch.cat([diag_scale.to(dtype), torch.ones(1, input_dim, dtype=dtype, device=device)])

    out = torch.empty_like(hidden_states)
    for chunk_start in range(0, rows, _FLATQUANT_MAX_ROWS_PER_CHUNK):
        chunk_end = min(chunk_start + _FLATQUANT_MAX_ROWS_PER_CHUNK, rows)
        idx = row_to_expert[chunk_start:chunk_end]
        x3 = hidden_states[chunk_start:chunk_end].reshape(-1, left_dim, right_dim)
        if diag_table is not None:
            x3 = x3 * diag_table.index_select(0, idx).reshape(-1, left_dim, right_dim)
        mid = torch.bmm(left_table.index_select(0, idx).transpose(1, 2), x3)
        out[chunk_start:chunk_end] = torch.bmm(mid, right_table.index_select(0, idx)).reshape(
            chunk_end - chunk_start, input_dim
        )
    return out


def _apply_expert_omniquant(
    hidden_states: torch.Tensor,
    fc_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    """Apply per-expert OmniQuant activation scale: x' = x / scale[expert].

    Dispatched tokens are arranged by local physical expert. group_list
    describes token boundaries per expert. We split by expert, divide the
    segment by that expert's per-dimension scale, then concatenate back.

    The weight was pre-multiplied by the same scale at load time
    (``process_weights_after_loading``), so ``(x / scale) @ (weight * scale).T
    == x @ weight.T`` and the linear output is preserved while MX QDQ error
    is reduced (same math as Linear OmniQuant, applied per-expert).

    Note: ``end_tensor.item()`` triggers a device-to-host sync, matching the
    pattern in ``_apply_expert_learned_hadamard`` / ``_apply_expert_flatquant``.
    Batched device-side implementation should replace this once validated.
    """
    num_experts = fc_scale.shape[0]

    def transform_expert(segment: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return segment / fc_scale[expert_idx].to(segment.dtype)

    return _apply_per_expert_transform(
        hidden_states,
        group_list,
        group_list_type,
        num_experts,
        "MoE OmniQuant",
        transform_expert,
    )

