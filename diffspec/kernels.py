# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Triton kernels used by DiffSpec on Ascend."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.rope import rope_forward_triton_siso


@triton.jit
def _paged_chunk_attention_scores_kernel(
    query_ptr,
    key_cache_ptr,
    softmax_lse_ptr,
    block_table_ptr,
    seq_lens_ptr,
    output_ptr,
    block_table_stride: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    chunk_size: tl.constexpr,
    max_chunks: tl.constexpr,
    padded_head_dim: tl.constexpr,
    padded_chunk_size: tl.constexpr,
    padded_query_heads_per_kv: tl.constexpr,
    scale: tl.constexpr,
):
    batch_index = tl.program_id(0).to(tl.int64)
    chunk_index = tl.program_id(1).to(tl.int64)
    kv_head = tl.program_id(2).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + batch_index).to(tl.int64)
    token_offsets = tl.arange(0, padded_chunk_size).to(tl.int64)
    token_positions = chunk_index * chunk_size + token_offsets
    valid_tokens = (token_offsets < chunk_size) & (token_positions < seq_len)
    logical_blocks = token_positions // block_size
    block_offsets = token_positions % block_size
    physical_blocks = tl.load(
        block_table_ptr
        + batch_index * block_table_stride
        + logical_blocks,
        mask=valid_tokens,
        other=0,
    ).to(tl.int64)

    query_heads_per_kv = num_query_heads // num_kv_heads
    query_head_offsets = tl.arange(0, padded_query_heads_per_kv).to(
        tl.int64
    )
    valid_query_heads = query_head_offsets < query_heads_per_kv
    query_heads = kv_head * query_heads_per_kv + query_head_offsets
    head_offsets = tl.arange(0, padded_head_dim).to(tl.int64)
    valid_head = head_offsets < head_dim
    query = tl.load(
        query_ptr
        + (batch_index * num_query_heads + query_heads[:, None])
        * head_dim
        + head_offsets[None, :],
        mask=valid_query_heads[:, None] & valid_head[None, :],
        other=0.0,
    ).to(tl.float32)

    key_offsets = (
        (
            (physical_blocks[:, None] * block_size + block_offsets[:, None])
            * num_kv_heads
            + kv_head
        )
        * head_dim
        + head_offsets[None, :]
    )
    key_mask = valid_tokens[:, None] & valid_head[None, :]
    keys = tl.load(key_cache_ptr + key_offsets, mask=key_mask, other=0.0)
    logits = tl.sum(
        query[:, None, :] * keys.to(tl.float32)[None, :, :],
        axis=2,
    ) * scale
    softmax_lse = tl.load(
        softmax_lse_ptr
        + batch_index * num_query_heads
        + query_heads,
        mask=valid_query_heads,
        other=0.0,
    ).to(tl.float32)
    weights = tl.exp(logits - softmax_lse[:, None])
    weights = tl.where(valid_tokens[None, :], weights, 0.0)
    valid_count = tl.sum(valid_tokens.to(tl.float32), axis=0)
    score = tl.sum(weights, axis=1) / tl.maximum(valid_count, 1.0)

    output_offsets = (
        (batch_index * max_chunks + chunk_index) * num_query_heads
        + query_heads
    )
    tl.store(
        output_ptr + output_offsets,
        score,
        mask=valid_query_heads,
    )


def paged_chunk_attention_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    chunk_size: int,
    max_seq_len: int,
    scale: float,
) -> torch.Tensor:
    """Compute normalized logical-chunk attention scores from paged K."""
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if key_cache.ndim != 4:
        raise ValueError(
            "key_cache must have shape [blocks, block_size, kv_heads, head_dim]"
        )
    batch_size, num_query_heads, head_dim = query.shape
    _, block_size, num_kv_heads, key_head_dim = key_cache.shape
    if head_dim != key_head_dim:
        raise ValueError("query and key head dimensions must match")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if softmax_lse.numel() != batch_size * num_query_heads:
        raise ValueError("softmax_lse must contain one value per query head")

    max_chunks = (max_seq_len + chunk_size - 1) // chunk_size
    per_head_scores = torch.empty(
        (batch_size, max_chunks, num_query_heads),
        dtype=torch.float32,
        device=query.device,
    )

    grid = (batch_size, max_chunks, num_kv_heads)
    _paged_chunk_attention_scores_kernel[grid](
        query,
        key_cache,
        softmax_lse,
        block_table,
        seq_lens,
        per_head_scores,
        block_table_stride=block_table.stride(0),
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        padded_head_dim=triton.next_power_of_2(head_dim),
        padded_chunk_size=triton.next_power_of_2(chunk_size),
        padded_query_heads_per_kv=triton.next_power_of_2(
            num_query_heads // num_kv_heads
        ),
        scale=scale,
    )
    return per_head_scores.mean(dim=-1)


def gather_raw_kv_with_local_rope(
    raw_key_cache: torch.Tensor,
    raw_value_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_token_indices: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    rope_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather raw paged KV and rotate K at compact local positions.

    ``selected_token_indices`` is shaped ``[batch, working_tokens]`` and uses
    ``-1`` for padding. Returned tensors use ``[batch, tokens, heads, dim]``.
    """
    if raw_key_cache.shape != raw_value_cache.shape or raw_key_cache.ndim != 4:
        raise ValueError("raw key/value caches must be matching 4-D tensors")
    if selected_token_indices.ndim != 2:
        raise ValueError("selected_token_indices must be two-dimensional")
    batch_size, working_tokens = selected_token_indices.shape
    _, block_size, num_kv_heads, head_dim = raw_key_cache.shape
    if block_table.shape[0] != batch_size:
        raise ValueError("block table batch size does not match token indices")

    valid = selected_token_indices >= 0
    safe_indices = selected_token_indices.clamp_min(0)
    logical_blocks = torch.div(safe_indices, block_size, rounding_mode="floor")
    block_offsets = safe_indices.remainder(block_size)
    physical_blocks = torch.gather(block_table.long(), 1, logical_blocks)
    flat_cache_indices = physical_blocks * block_size + block_offsets

    flat_key_cache = raw_key_cache.view(-1, num_kv_heads, head_dim)
    flat_value_cache = raw_value_cache.view(-1, num_kv_heads, head_dim)
    gathered_key = flat_key_cache.index_select(0, flat_cache_indices.reshape(-1))
    gathered_value = flat_value_cache.index_select(
        0, flat_cache_indices.reshape(-1)
    )
    gathered_key = gathered_key.view(
        batch_size, working_tokens, num_kv_heads, head_dim
    )
    gathered_value = gathered_value.view_as(gathered_key)
    gathered_key.masked_fill_(~valid[:, :, None, None], 0)
    gathered_value.masked_fill_(~valid[:, :, None, None], 0)

    local_positions = torch.arange(
        working_tokens,
        dtype=torch.long,
        device=selected_token_indices.device,
    ).expand(batch_size, -1)
    flat_key = gathered_key.reshape(-1, num_kv_heads, head_dim)
    rope_forward_triton_siso(
        flat_key,
        cos_sin_cache=cos_sin_cache,
        positions=local_positions.reshape(-1),
        rope_dim=rope_dim,
        is_neox_style=is_neox_style,
    )
    return gathered_key, gathered_value
