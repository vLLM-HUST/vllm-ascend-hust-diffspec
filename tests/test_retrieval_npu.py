# SPDX-License-Identifier: Apache-2.0
"""Opt-in numerical checks for the actual Ascend retrieval kernel."""

import json
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
@pytest.mark.parametrize("length", [2051, 32768, 40959])
def test_device_chunk_selection_preserves_exact_chronological_indices(length):
    import torch_npu  # noqa: F401

    from diffspec.runtime import select_context_chunks

    torch.npu.set_device(0)
    torch.manual_seed(17)
    scores = torch.randn((length + 63) // 64)
    expected = select_context_chunks(scores, length, 64, 2048)
    actual = select_context_chunks(scores.npu(), length, 64, 2048)
    assert torch.equal(actual.chunk_indices.cpu(), expected.chunk_indices)
    assert torch.equal(actual.token_indices.cpu(), expected.token_indices)
    torch.testing.assert_close(actual.chunk_scores.cpu(), expected.chunk_scores)


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
@pytest.mark.parametrize("block_size", [64, 1024, 1536])
@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(4, 1), (8, 2), (6, 1), (12, 2)])
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_paged_retrieval_matches_dense_bf16_gqa(
    block_size,
    num_query_heads,
    num_kv_heads,
    chunk_size,
):
    import torch_npu  # noqa: F401

    from diffspec.kernels import paged_chunk_attention_scores

    torch.manual_seed(123)
    torch.npu.set_device(0)
    head_dim = 256
    lengths = [2 * block_size + 13, block_size + 25]
    query = torch.randn((2, num_query_heads, head_dim), dtype=torch.bfloat16)
    keys = torch.randn((9, block_size, num_kv_heads, head_dim), dtype=torch.bfloat16)
    # Reordered physical pages, unequal lengths and partial final chunks.
    tables = torch.tensor([[4, 1, 7], [8, 2, 5]], dtype=torch.int32)
    scale = head_dim**-0.5
    references, normalizers = [], []
    max_chunks = (max(lengths) + chunk_size - 1) // chunk_size
    for row, length in enumerate(lengths):
        dense_keys = keys[tables[row].long()].reshape(-1, num_kv_heads, head_dim)[:length]
        grouped_query = query[row].float().reshape(num_kv_heads, -1, head_dim)
        logits = torch.einsum("hgd,thd->hgt", grouped_query, dense_keys.float())
        logits = logits.reshape(num_query_heads, length) * scale
        normalizers.append(torch.logsumexp(logits, dim=-1))
        probabilities = logits.softmax(-1)
        reference = torch.zeros(max_chunks)
        for chunk in range((length + chunk_size - 1) // chunk_size):
            reference[chunk] = probabilities[
                :, chunk * chunk_size : (chunk + 1) * chunk_size
            ].mean()
        references.append(reference)

    actual = paged_chunk_attention_scores(
        query.npu(),
        keys.npu(),
        torch.stack(normalizers).npu(),
        tables.npu(),
        torch.tensor(lengths, dtype=torch.int32).npu(),
        chunk_size=chunk_size,
        max_seq_len=max(lengths),
        scale=scale,
    ).cpu()
    torch.testing.assert_close(actual, torch.stack(references), rtol=1e-3, atol=1e-6)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_linear_proposer_runs_real_retrieval_kernel(monkeypatch, batch_size):
    """Synthetic verifier state through the real proposer/cache/Triton path.

    Native sampling is replaced with fixed acceptance counts; no model quality
    or generation performance is measured by this test.
    """
    import torch_npu  # noqa: F401
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    from diffspec.proposer import AscendDiffSpecEagleProposer
    from diffspec.runtime import DiffSpecDraftCache, DiffSpecSettings

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    # Qwen3.8-27B TP4 target has six local Q heads, one KV head and D=256.
    seq_len, head_dim, heads, block_size = 4096, 256, 6, 1024
    cache = DiffSpecDraftCache(
        DiffSpecSettings(64, 2048, 4, 50, 3, 0.75, False),
        max_num_reqs=batch_size,
        max_model_len=seq_len,
        num_kv_heads=1,
        head_dim=head_dim,
        target_num_layers=64,
        dtype=torch.bfloat16,
        device=device,
        rotary_embedding=SimpleNamespace(),
    )
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.diffspec_cache = cache
    monkeypatch.setattr(proposer, "_observe_linear_depth", lambda counts: None)
    counts = torch.tensor([i % 4 + 1 for i in range(batch_size)], device=device)
    result = (torch.zeros(batch_size, dtype=torch.int32, device=device), counts)
    monkeypatch.setattr(
        AscendEagleProposer,
        "prepare_next_token_ids_padded",
        lambda *args, **kwargs: result,
    )
    query = torch.ones((batch_size * 4, heads, head_dim), dtype=torch.bfloat16, device=device)
    # Zero K makes dense causal attention uniform, with exact known LSE.
    keys = torch.zeros(
        (batch_size * 4, block_size, 1, head_dim), dtype=torch.bfloat16, device=device
    )
    prefixes = torch.arange(seq_len - 3, seq_len + 1, device=device).repeat(batch_size)
    lse = prefixes.float().log()[:, None].expand(-1, heads).contiguous()
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[4 * (i + 1) for i in range(batch_size)],
        seq_lens_list=[seq_len] * batch_size,
        num_prefills=0,
        block_tables=torch.arange(batch_size * 4, dtype=torch.int32, device=device).view(
            batch_size, 4
        ),
    )
    for _ in range(4):
        cache.begin_target_forward([f"request-{i}" for i in range(batch_size)])
        cache.capture_target_attention(query, lse, keys, metadata, head_dim**-0.5)
        assert cache.proposed_token_ids is None
        proposer.prepare_next_token_ids_padded()
    torch.npu.synchronize()
    status = cache.retrieval_status()
    assert status["retrieval_count"] == 1
    assert status["retrieval_cycle"] == 4
    assert status["selected_slots"] == batch_size
    for row, count in enumerate(counts.cpu().tolist()):
        selected = cache._selected_token_indices[row].cpu()
        assert selected.numel() == cache.history_budget - (4 - count)
        assert selected.min() >= 0
        assert selected.max() == seq_len - 4 + count - 1
    print(json.dumps(dict(batch_size=batch_size, synthetic=True, **status), sort_keys=True))
