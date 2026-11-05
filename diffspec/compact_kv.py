# SPDX-License-Identifier: Apache-2.0
"""Fused raw KV gather and local-position RoPE for DiffSpec proposals."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rebuild_compact_kv_kernel(
    raw_key_ptr,
    raw_value_ptr,
    working_key_ptr,
    working_value_ptr,
    plan_ptr,
    cos_sin_ptr,
    plan_stride: tl.constexpr,
    chunk_size: tl.constexpr,
    max_model_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    working_budget: tl.constexpr,
    cos_sin_stride: tl.constexpr,
    rotary_dim: tl.constexpr,
    is_neox_style: tl.constexpr,
    block_tokens: tl.constexpr,
    padded_half_rope: tl.constexpr,
    padded_head_dim: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    positions = tl.program_id(2).to(tl.int64) * block_tokens + tl.arange(0, block_tokens).to(
        tl.int64
    )
    plan_row = plan_ptr + batch * plan_stride
    slots = tl.load(plan_row).to(tl.int64)
    history_len = tl.load(plan_row + 1).to(tl.int64)
    tail_start = tl.load(plan_row + 2).to(tl.int64)
    prefix_len = tl.load(plan_row + 3).to(tl.int64)
    prefix_drop = tl.load(plan_row + 4).to(tl.int64)
    valid_tokens = positions < history_len
    prefix_positions = positions + prefix_drop
    is_prefix = positions < prefix_len
    chunks = tl.load(
        plan_row + 5 + prefix_positions // chunk_size,
        mask=valid_tokens & is_prefix,
        other=0,
    ).to(tl.int64)
    absolute_positions = tl.where(
        is_prefix,
        chunks * chunk_size + prefix_positions % chunk_size,
        tail_start + positions - prefix_len,
    )
    raw_rows = slots * max_model_len + absolute_positions
    raw_bases = (raw_rows * num_heads + head) * head_dim
    working_bases = ((slots * num_heads + head) * working_budget + positions) * head_dim

    if rotary_dim > 0:
        half = rotary_dim // 2
        offsets = tl.arange(0, padded_half_rope).to(tl.int64)
        rope_mask = valid_tokens[:, None] & (offsets[None, :] < half)
        cos_offsets = positions[:, None] * cos_sin_stride + offsets[None, :]
        cos = tl.load(cos_sin_ptr + cos_offsets, mask=rope_mask, other=0).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + cos_offsets + half,
            mask=rope_mask,
            other=0,
        ).to(tl.float32)
        if is_neox_style:
            first_offsets = offsets
            second_offsets = offsets + half
            first = tl.load(
                raw_key_ptr + raw_bases[:, None] + first_offsets[None, :],
                mask=rope_mask,
                other=0,
            ).to(tl.float32)
            second = tl.load(
                raw_key_ptr + raw_bases[:, None] + second_offsets[None, :],
                mask=rope_mask,
                other=0,
            ).to(tl.float32)
        else:
            pair_offsets = offsets[None, :, None] * 2 + tl.arange(0, 2).to(tl.int64)[None, None, :]
            pairs = tl.load(
                raw_key_ptr + raw_bases[:, None, None] + pair_offsets,
                mask=rope_mask[:, :, None],
                other=0,
            ).to(tl.float32)
            first, second = tl.split(pairs)
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        if is_neox_style:
            tl.store(
                working_key_ptr + working_bases[:, None] + first_offsets[None, :],
                rotated_first,
                mask=rope_mask,
            )
            tl.store(
                working_key_ptr + working_bases[:, None] + second_offsets[None, :],
                rotated_second,
                mask=rope_mask,
            )
        else:
            rotated_pairs = tl.join(rotated_first, rotated_second)
            tl.store(
                working_key_ptr + working_bases[:, None, None] + pair_offsets,
                rotated_pairs,
                mask=rope_mask[:, :, None],
            )

    dimensions = tl.arange(0, padded_head_dim).to(tl.int64)
    value_mask = valid_tokens[:, None] & (dimensions[None, :] < head_dim)
    values = tl.load(
        raw_value_ptr + raw_bases[:, None] + dimensions[None, :],
        mask=value_mask,
        other=0,
    )
    tl.store(
        working_value_ptr + working_bases[:, None] + dimensions[None, :],
        values,
        mask=value_mask,
    )
    if rotary_dim < head_dim:
        tail_mask = value_mask & (dimensions[None, :] >= rotary_dim)
        keys = tl.load(
            raw_key_ptr + raw_bases[:, None] + dimensions[None, :],
            mask=tail_mask,
            other=0,
        )
        tl.store(
            working_key_ptr + working_bases[:, None] + dimensions[None, :],
            keys,
            mask=tail_mask,
        )


def _validate_compact_inputs(
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    working_key: torch.Tensor,
    working_value: torch.Tensor,
    plan: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    max_width: int,
    rotary_dim: int,
) -> None:
    if raw_key.ndim != 4 or raw_key.shape != raw_value.shape:
        raise ValueError("raw key/value must have matching 4-D shapes")
    if working_key.ndim != 4 or working_key.shape != working_value.shape:
        raise ValueError("working key/value must have matching 4-D shapes")
    slots, max_model_len, heads, head_dim = raw_key.shape
    if (
        working_key.shape[0] != slots
        or working_key.shape[1] != heads
        or working_key.shape[3] != head_dim
    ):
        raise ValueError("working cache slot, head and head dimensions must match raw KV")
    if min(max_model_len, heads, head_dim) <= 0:
        raise ValueError("raw sequence, head and head dimensions must be positive")
    if plan.ndim != 2 or plan.dtype != torch.int64:
        raise ValueError("KV plan must be a contiguous 2-D int64 tensor")
    if plan.shape[0] > slots or not 0 <= max_width <= working_key.shape[2]:
        raise ValueError("KV plan exceeds the working cache batch or token capacity")
    if rotary_dim < 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and between zero and head_dim")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] < max_width
        or cos_sin_cache.shape[1] < rotary_dim
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("cos/sin cache must contain all local positions and rotary dimensions")
    caches = (raw_key, raw_value, working_key, working_value)
    if any(not cache.is_contiguous() for cache in (*caches, plan)):
        raise ValueError("raw KV, working KV and plan must be contiguous")
    if any(cache.dtype != raw_key.dtype for cache in caches):
        raise ValueError("all KV caches must have the same dtype")
    if raw_key.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("KV caches must use BF16, FP16 or FP32")
    if any(tensor.device != raw_key.device for tensor in (*caches, plan, cos_sin_cache)):
        raise ValueError("KV caches, plan and cos/sin cache must share a device")


def rebuild_compact_kv_from_plan(
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    working_key: torch.Tensor,
    working_value: torch.Tensor,
    plan: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    chunk_size: int,
    max_width: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rebuild compact KV from fixed-capacity per-request chunk plans.

    A plan row contains ``[slot, history_len, tail_start, prefix_len,
    prefix_drop, chunk_ids...]``. Its chunk capacity is
    ``working_budget // chunk_size``. Slots must be distinct across rows.
    Callers guarantee valid chunk IDs, ``prefix_len <= history_len <=
    max_width``, and raw positions within the model length. Padding after each
    row's ``history_len`` remains untouched. ``max_width`` only sizes the
    launch grid; no plan values are copied back to the host.
    """
    _validate_compact_inputs(
        raw_key,
        raw_value,
        working_key,
        working_value,
        plan,
        cos_sin_cache,
        max_width=max_width,
        rotary_dim=rotary_dim,
    )
    if chunk_size <= 0 or plan.shape[1] != 5 + working_key.shape[2] // chunk_size:
        raise ValueError("plan width must be 5 + working_budget // chunk_size")
    if not plan.shape[0] or not max_width:
        return
    _, max_model_len, heads, head_dim = raw_key.shape
    block_tokens = 16
    _rebuild_compact_kv_kernel[(plan.shape[0], heads, triton.cdiv(max_width, block_tokens))](
        raw_key,
        raw_value,
        working_key,
        working_value,
        plan,
        cos_sin_cache,
        plan_stride=plan.shape[1],
        chunk_size=chunk_size,
        max_model_len=max_model_len,
        num_heads=heads,
        head_dim=head_dim,
        working_budget=working_key.shape[2],
        cos_sin_stride=cos_sin_cache.stride(0),
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
        block_tokens=block_tokens,
        padded_half_rope=triton.next_power_of_2(max(rotary_dim // 2, 1)),
        padded_head_dim=triton.next_power_of_2(head_dim),
    )
