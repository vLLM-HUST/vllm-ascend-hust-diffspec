# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from diffspec.proposer import (
    AscendDiffSpecEagleProposer,
)
from diffspec.runtime import (
    DiffSpecAdaptiveController,
    DiffSpecDraftCache,
    DiffSpecSettings,
    DiffSpecTreeProfile,
    build_tree_ancestry_mask,
    chunk_attention_scores,
    merge_attention_states,
    select_context_chunks,
    select_tree_level,
    verify_greedy_tree,
)


class _IdentityRotary:
    def forward_native(self, positions, query, key):
        return query, key


def _decode_history_plan(plan: list[int], chunk_size: int) -> list[int]:
    _, length, tail_start, prefix_length, prefix_drop, *chunks = plan
    prefix = [
        chunks[(position + prefix_drop) // chunk_size] * chunk_size
        + (position + prefix_drop) % chunk_size
        for position in range(prefix_length)
    ]
    return prefix + list(range(tail_start, tail_start + length - prefix_length))


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_retrieval_selects_same_context_from_all_target_heads(monkeypatch, rank):
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 1, 4, 1, 0.7, False),
        max_num_reqs=1, max_model_len=10, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(), target_tensor_parallel_size=2,
    )
    scores = [torch.tensor([[0., 100., 0., 0., 0.]]),
              torch.tensor([[0., 0., 0., 400., 0.]])]
    monkeypatch.setattr("diffspec.kernels.paged_chunk_attention_scores", lambda *a, **k: scores[rank])
    reduced = []

    def all_reduce(local):
        torch.testing.assert_close(local, scores[rank])
        reduced.append(True)
        return scores[0] + scores[1]

    monkeypatch.setattr("vllm.distributed.tensor_model_parallel_all_reduce", all_reduce)
    cache.begin_target_forward(["request"])
    cache.capture_target_attention(
        torch.ones((1, 2, 2)), torch.zeros((1, 2)), torch.zeros((1, 10, 1, 2)),
        SimpleNamespace(num_prefills=0, actual_seq_lengths_q=[1], seq_lens_list=[10],
                        block_tables=torch.zeros((1, 1), dtype=torch.int32)),
        0.5,
    )
    cache.accept_target_queries(torch.ones(1), is_prefill=cache.target_is_prefill)
    assert reduced == [True]
    assert cache._selected_token_indices[0].tolist() == [2, 3, 6, 7, 8, 9]


@pytest.mark.parametrize("counts", [[1, 2], [4, 4], [0, 2], [0, 0]])
@pytest.mark.parametrize("num_prefills", [0, 1])
@pytest.mark.parametrize("adaptive", [False, True])
def test_linear_proposer_retrieves_only_committed_queries(
    monkeypatch, counts, num_prefills, adaptive,
):
    """Linear proposals have no tree IDs, including during decode verification."""
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 2, 4, 3, 0.7, adaptive),
        max_num_reqs=2, max_model_len=16, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.diffspec_cache = cache
    monkeypatch.setattr(proposer, "_observe_linear_depth", lambda counts: None)
    result = (torch.tensor([5, 6]), torch.tensor(counts))
    monkeypatch.setattr(
        AscendEagleProposer, "prepare_next_token_ids_padded",
        lambda *args, **kwargs: result,
    )
    query = torch.arange(32, dtype=torch.float32).view(8, 2, 2)
    lse = torch.arange(16, dtype=torch.float32).view(8, 2)
    key_cache = torch.zeros((4, 8, 1, 2))
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[4, 8], seq_lens_list=[10, 12],
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        num_prefills=num_prefills,
    )
    calls = []
    observed_acceptance = []
    monkeypatch.setattr(
        cache.controller, "observe_retrieval",
        lambda previous, selected, accepted: observed_acceptance.append(accepted),
    )

    def scores(selected_query, selected_key, selected_lse, table, lengths, **kwargs):
        rows = [i for i, count in enumerate(counts) if count > 0]
        indices = [i * 4 + counts[i] - 1 for i in rows]
        expected_lengths = [metadata.seq_lens_list[i] - 4 + counts[i] for i in rows]
        torch.testing.assert_close(selected_query, query[indices])
        torch.testing.assert_close(selected_lse, lse[indices])
        assert selected_key is key_cache
        assert lengths.tolist() == expected_lengths
        assert table.tolist() == metadata.block_tables[rows].tolist()
        calls.append(rows)
        return torch.ones((len(rows), (max(expected_lengths) + 1) // 2))

    monkeypatch.setattr("diffspec.kernels.paged_chunk_attention_scores", scores)
    for _ in range(2):
        cache.begin_target_forward(["a", "b"])
        cache.capture_target_attention(query, lse, key_cache, metadata, 0.5)
        assert cache.proposed_token_ids is None
        assert proposer.prepare_next_token_ids_padded() is result

    retrieved = not num_prefills and any(counts)
    assert cache.metrics.retrieval_count == int(retrieved)
    assert len(calls) == int(retrieved)
    assert observed_acceptance == (
        [next(count - 1 for count in counts if count)] if retrieved and adaptive else []
    )
    assert set(cache._selected_token_indices) == (
        {i for i, count in enumerate(counts) if count} if retrieved else set()
    )


def test_compact_prefill_prepares_attention_before_graph_replay():
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 2, 4, 3, 0.7, False),
        max_num_reqs=1,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.set_active_requests(["request"])
    cache.raw_key[0, :3, 0] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    cache.raw_value[0, :3, 0] = cache.raw_key[0, :3, 0]

    cache.prepare_compact_prefill(SimpleNamespace(seq_lens_list=[3]), 1)

    assert cache.has_pending_attention()
    assert cache._pending_request_indices == [0]
    assert cache._pending_local_positions == [2]
    assert cache._working_lens == [2]


def test_compact_tree_root_normalizes_already_compact_prefill_index():
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.arange = torch.arange(4)
    kwargs = {
        "is_prefill": True,
        "batch_size": 1,
        "num_input_tokens": 1,
        "token_indices_to_sample": torch.tensor([3]),
    }

    proposer._compact_tree_decode_root(kwargs)

    assert kwargs["token_indices_to_sample"].tolist() == [0]


def test_linear_depth_profile_uses_short_and_long_context_depths():
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.num_speculative_tokens = 5
    proposer.speculative_config = SimpleNamespace(
        diffspec_adaptive_profile=True,
        diffspec_verification_mode="auto",
        diffspec_long_context_threshold=49152,
        diffspec_long_context_depth=2,
        num_speculative_tokens=5,
    )
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["request-0"])
    )
    proposer._linear_request_id = None
    proposer._linear_runtime_depth = None
    proposer._linear_depth_floor = 1
    proposer._linear_good_cycles = 0
    proposer._linear_bad_cycles = 0
    proposer._diffspec_tree_enabled = False

    def kwargs(seq_len):
        metadata = SimpleNamespace(seq_lens_list=[seq_len])
        return {"multi_steps_attn_metadata": [{"layer": metadata}]}

    assert proposer._select_linear_depth(kwargs(32768)) == 5
    proposer.runner.input_batch.req_ids = ["request-1"]
    assert proposer._select_linear_depth(kwargs(65536)) == 2

    for _ in range(4):
        proposer._observe_linear_depth(torch.tensor([3]))
    assert proposer._linear_runtime_depth == 3
    for _ in range(3):
        proposer._observe_linear_depth(torch.tensor([1]))
    assert proposer._linear_runtime_depth == 2


@pytest.mark.parametrize("batch_size", [1, 4])
def test_linear_depth_retreats_below_initial_depth_and_ignores_empty_results(batch_size):
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.num_speculative_tokens = 3
    proposer.speculative_config = SimpleNamespace(
        diffspec_adaptive_profile=True,
        diffspec_verification_mode="auto",
        diffspec_long_context_threshold=49152,
        diffspec_long_context_depth=2,
        num_speculative_tokens=3,
    )
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[f"short-{i}" for i in range(batch_size)])
    )
    proposer._linear_request_id = None
    proposer._linear_runtime_depth = None
    proposer._linear_depth_floor = 1
    proposer._linear_good_cycles = 0
    proposer._linear_bad_cycles = 0
    proposer._diffspec_tree_enabled = False
    metadata = SimpleNamespace(seq_lens_list=[128])
    assert proposer._select_linear_depth(
        {"multi_steps_attn_metadata": [{"layer": metadata}]}
    ) == 3
    for _ in range(6):
        proposer._observe_linear_depth(torch.zeros(batch_size, dtype=torch.int32))
    assert proposer._linear_runtime_depth == 3
    assert proposer._linear_bad_cycles == 0
    for _ in range(3):
        proposer._observe_linear_depth(torch.ones(batch_size, dtype=torch.int32))
    assert proposer._linear_runtime_depth == 2
    for _ in range(6):
        proposer._observe_linear_depth(torch.ones(batch_size, dtype=torch.int32))
    assert proposer._linear_runtime_depth == 1

    # Chunked prefills and a changed scheduler batch cannot count as misses
    # or inherit the old batch's reduced depth.
    proposer.diffspec_cache = SimpleNamespace(target_is_prefill=True)
    for _ in range(4):
        proposer._observe_linear_depth(torch.full((batch_size,), 2))
    assert proposer._linear_runtime_depth == 1
    proposer.diffspec_cache.target_is_prefill = False
    proposer.runner.input_batch.req_ids = ["new-request"]
    assert proposer._select_linear_depth(
        {"multi_steps_attn_metadata": [{"layer": metadata}]}
    ) == 3


def test_chunk_attention_scores_matches_dense_gqa_reference():
    torch.manual_seed(7)
    query = torch.randn(4, 8)
    keys = torch.randn(11, 2, 8)
    scale = 8**-0.5

    actual = chunk_attention_scores(query, keys, 4, scale=scale)

    expanded_keys = keys.repeat_interleave(2, dim=1)
    logits = torch.einsum("hd,thd->ht", query, expanded_keys) * scale
    weights = logits.softmax(dim=-1)
    expected = torch.stack(
        (
            weights[:, :4].mean(),
            weights[:, 4:8].mean(),
            weights[:, 8:].mean(),
        )
    )
    torch.testing.assert_close(actual, expected)

    lse = torch.logsumexp(logits, dim=-1)
    with_external_lse = chunk_attention_scores(
        query,
        keys,
        4,
        softmax_lse=lse,
        scale=scale,
    )
    torch.testing.assert_close(with_external_lse, expected)


def test_select_context_chunks_reserves_partial_tail():
    scores = torch.tensor([0.1, 0.9, 0.8, 0.7, 0.01])
    selection = select_context_chunks(
        scores,
        seq_len=18,
        chunk_size=4,
        token_budget=12,
    )
    assert selection.chunk_indices.tolist() == [1, 2, 4]
    assert selection.token_indices.tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 16, 17]


def test_build_tree_ancestry_mask():
    parents = torch.tensor([-1, 0, 0, 1, 2])
    actual = build_tree_ancestry_mask(parents)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(actual, expected)


def test_build_tree_ancestry_mask_rejects_forward_parent():
    with pytest.raises(ValueError, match="parents must precede"):
        build_tree_ancestry_mask(torch.tensor([-1, 2, 0]))


def test_select_tree_level_ranks_global_path_probability():
    logits = torch.tensor([[4.0, 3.0, 0.0], [2.0, 1.0, 0.0]])
    level = select_tree_level(
        logits,
        parent_indices=torch.tensor([4, 7]),
        parent_log_probs=torch.tensor([0.0, -2.0]),
        node_budget=3,
        cumulative_threshold=0.7,
    )
    assert level.token_ids.tolist() == [0, 1, 0]
    assert level.parent_indices.tolist() == [4, 4, 7]
    assert torch.all(level.path_log_probs[:-1] > level.path_log_probs[-1])


def test_merge_attention_states_matches_concatenated_softmax():
    prefix_logits = torch.tensor([[1.0, 2.0]])
    tree_logits = torch.tensor([[0.5, 3.0]])
    prefix_values = torch.tensor([[[2.0], [4.0]]])
    tree_values = torch.tensor([[[8.0], [16.0]]])

    prefix_weights = prefix_logits.softmax(dim=-1)
    tree_weights = tree_logits.softmax(dim=-1)
    prefix_output = torch.sum(prefix_weights[..., None] * prefix_values, dim=1)
    tree_output = torch.sum(tree_weights[..., None] * tree_values, dim=1)
    prefix_lse = torch.logsumexp(prefix_logits, dim=-1)
    tree_lse = torch.logsumexp(tree_logits, dim=-1)
    actual, actual_lse = merge_attention_states(
        prefix_output,
        prefix_lse,
        tree_output,
        tree_lse,
    )

    logits = torch.cat((prefix_logits, tree_logits), dim=-1)
    values = torch.cat((prefix_values, tree_values), dim=1)
    expected = torch.sum(logits.softmax(dim=-1)[..., None] * values, dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, torch.logsumexp(logits, dim=-1))


def test_verify_greedy_tree_follows_matching_branch():
    draft_tokens = torch.tensor([10, 11, 12, 13, 14])
    parents = torch.tensor([-1, 0, 0, 1, 2])
    target_argmax = torch.tensor([10, 11, 12, 99, 98, 97])
    accepted, emitted = verify_greedy_tree(
        draft_tokens,
        parents,
        target_argmax,
    )
    assert accepted.tolist() == [0, 1]
    assert emitted.tolist() == [10, 11, 12]


def test_adaptive_controller_uses_hysteresis_and_stability():
    controller = DiffSpecAdaptiveController(8, adaptive=True)
    slow = DiffSpecTreeProfile(50, 10)
    small = DiffSpecTreeProfile(16, 5)
    fast = DiffSpecTreeProfile(32, 7)
    controller.observe_cycle(slow, emitted_tokens=3, latency_seconds=1.0)
    assert controller.profile == small
    controller.observe_cycle(small, emitted_tokens=2, latency_seconds=1.0)
    assert controller.profile == fast
    controller.observe_cycle(fast, emitted_tokens=4, latency_seconds=1.0)
    assert controller.profile == fast

    controller.observe_retrieval({1, 2, 3}, {1, 2, 3}, 4.0)
    assert controller.retrieval_interval == 16
    controller.observe_retrieval({1, 2, 3}, {4, 5, 6}, 3.0)
    assert controller.retrieval_interval == 8


def test_draft_cache_cpu_attention_matches_dense_reference():
    settings = DiffSpecSettings(
        chunk_size=2,
        token_budget=8,
        retrieval_interval=2,
        max_tree_nodes=4,
        max_tree_depth=2,
        tree_threshold=0.7,
        adaptive_profile=False,
    )
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=1,
        max_model_len=16,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.working_key[0, 0, :2] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cache.working_value[0, 0, :2] = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    cache._pending_request_indices = [0]
    cache._pending_local_positions = [2]
    query = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
    key = torch.tensor([[[1.0, 1.0]]])
    value = torch.tensor([[[6.0, 8.0]]])
    output = torch.empty_like(query)

    actual = cache.forward_attention(query, key, value, 2**-0.5, output)

    expanded_key = cache.working_key[0, :, :3].repeat_interleave(2, dim=0)
    expanded_value = cache.working_value[0, :, :3].repeat_interleave(2, dim=0)
    logits = torch.einsum("hd,hsd->hs", query[0], expanded_key) * 2**-0.5
    expected = torch.einsum(
        "hs,hsd->hd",
        logits.softmax(dim=-1),
        expanded_value,
    )
    torch.testing.assert_close(actual[0], expected)


def test_draft_tree_attention_respects_ancestry():
    settings = DiffSpecSettings(2, 8, 2, 4, 3, 0.7, False)
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=1,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.set_active_requests(["request"])
    cache.working_key[0, 0, :2] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    cache.working_value[0, 0, :2] = torch.tensor(
        [[2.0, 0.0], [0.0, 4.0]]
    )
    cache._working_lens[0] = 2
    node_indices = torch.tensor([0, 1, 2])
    parents = torch.tensor([-1, -1, 0])
    cache.prepare_tree_level(
        node_indices,
        parents,
        torch.tensor([1, 1, 2]),
    )
    query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    tree_key = torch.tensor([[[1.0, 1.0]], [[-1.0, 1.0]], [[2.0, 0.0]]])
    tree_value = torch.tensor([[[6.0, 6.0]], [[8.0, 8.0]], [[10.0, 2.0]]])

    actual = cache.forward_tree_attention(
        query,
        tree_key,
        tree_value,
        2**-0.5,
        torch.empty_like(query),
    )

    allowed_tree = ([0], [1], [0, 2])
    expected = []
    prefix_key = cache.working_key[0, 0, :2]
    prefix_value = cache.working_value[0, 0, :2]
    for node, allowed in enumerate(allowed_tree):
        keys = torch.cat((prefix_key, tree_key[list(allowed), 0]))
        values = torch.cat((prefix_value, tree_value[list(allowed), 0]))
        logits = query[node, 0] @ keys.T * 2**-0.5
        expected.append(logits.softmax(dim=-1) @ values)
    torch.testing.assert_close(actual[:, 0], torch.stack(expected))


def test_full_beam_is_pruned_to_ancestry_closed_verifier_tree():
    settings = DiffSpecSettings(2, 16, 2, 3, 3, 0.7, False)
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=1,
        max_model_len=16,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.set_active_requests(["request"])
    cache.tree_raw_key[0, :6, 0, 0] = torch.arange(6)
    cache.tree_raw_value[0, :6, 0, 0] = torch.arange(10, 16)

    cache.finalize_proposed_tree(
        token_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        parent_indices=torch.tensor([-1, -1, -1, 0, 1, 3]),
        depths=torch.tensor([1, 1, 1, 2, 2, 3]),
        path_log_probs=torch.tensor([-0.1, -0.8, -0.9, -0.2, -1.0, -0.15]),
        profile=DiffSpecTreeProfile(3, 3),
    )

    assert cache.proposed_token_ids.tolist() == [10, 13, 15]
    assert cache.proposed_parent_indices.tolist() == [-1, 0, 1]
    assert cache.proposed_depths_cpu == [1, 2, 3]
    assert cache.tree_raw_key[0, :3, 0, 0].tolist() == [0, 3, 5]
    assert cache.tree_raw_value[0, :3, 0, 0].tolist() == [10, 13, 15]


def test_target_retrieval_skips_prefill_then_uses_decode_queries(monkeypatch):
    settings = DiffSpecSettings(
        chunk_size=2,
        token_budget=8,
        retrieval_interval=2,
        max_tree_nodes=4,
        max_tree_depth=2,
        tree_threshold=0.7,
        adaptive_profile=False,
    )
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=2,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    query = torch.arange(20, dtype=torch.float32).view(5, 2, 2)
    lse = torch.arange(10, dtype=torch.float32).view(5, 2, 1)
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[3, 5],
        seq_lens_list=[6, 4],
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        num_prefills=2,
    )
    key_cache = torch.zeros((1, 2, 1, 2))
    cache.set_active_requests(["request-0", "request-1"])
    cache.capture_target_attention(query, lse, key_cache, metadata, 0.5)
    assert cache._target_query.shape == (2, 2, 2)

    def fake_scores(
        selected_query,
        selected_key_cache,
        selected_lse,
        block_tables,
        seq_lens,
        **kwargs,
    ):
        torch.testing.assert_close(selected_query, query[[2, 4]])
        torch.testing.assert_close(selected_lse, lse[[2, 4], :, 0])
        assert selected_key_cache is key_cache
        assert seq_lens.tolist() == [6, 4]
        return torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.9, 0.0]])

    monkeypatch.setattr(
        "diffspec.kernels.paged_chunk_attention_scores",
        fake_scores,
    )
    cache.accept_target_queries(torch.ones(2), is_prefill=True)
    assert cache._selected_token_indices == {}

    cache.begin_target_forward(["request-0", "request-1"])
    metadata.actual_seq_lengths_q = [1, 2]
    metadata.num_prefills = 0
    cache.capture_target_attention(query[[2, 4]], lse[[2, 4]], key_cache, metadata, 0.5)

    # Retrieval starts only after the configured number of completed
    # verifier cycles; until then _history_indices supplies the recent
    # working window.
    cache.accept_target_queries(torch.ones(2), is_prefill=False)
    assert cache._selected_token_indices == {}
    cache.accept_target_queries(torch.ones(2), is_prefill=False)

    assert cache._selected_token_indices[0].tolist() == [0, 1, 4, 5]
    assert cache._selected_token_indices[1].tolist() == [0, 1, 2, 3]
    status = cache.retrieval_status()
    assert status["retrieval_skipped_prefill"] == 1
    assert status["retrieval_skipped_interval"] == 1
    assert status["retrieval_count"] == 1
    assert status["selected_slots"] == 2

    # No callback on the next forward must not reuse the previous query.
    cache.begin_target_forward(["request-1"])
    cache.accept_target_queries(torch.ones(1), is_prefill=cache.target_is_prefill)
    status = cache.retrieval_status()
    assert status["retrieval_skipped_no_snapshot"] == 1
    assert status["retrieval_count"] == 1
    assert status["selected_slots"] == 1


def test_working_history_rolls_tail_and_excludes_current_root():
    settings = DiffSpecSettings(2, 8, 2, 4, 2, 0.7, False)
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=1,
        max_model_len=16,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache._selected_token_indices[0] = torch.tensor([0, 1, 4, 5])

    # Position 5 is the root being replayed, and must not be in its history.
    assert cache._history_indices(0, history_end=5).tolist() == [0, 1, 3, 4]
    # Without a retrieval refresh, the most recent chunk still rolls forward.
    assert cache._history_indices(0, history_end=9).tolist() == [0, 1, 7, 8]


@pytest.mark.parametrize("history_end", [0, 5, 7, 9, 13, 15])
def test_host_history_plan_matches_committed_sparse_prefix_and_rolling_tail(history_end):
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 2, 4, 2, 0.7, False),
        max_num_reqs=1, max_model_len=16, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    for selection in [None, [], [0, 1, 4, 5], [2, 3, 8, 9]]:
        if selection is not None:
            cache._selected_token_indices[0] = torch.tensor(selection, dtype=torch.long)
        expected = cache._history_indices(0, history_end).tolist()
        assert _decode_history_plan(cache._history_plan_host(0, history_end), cache.settings.chunk_size) == expected
        assert len(expected) <= cache.history_budget
        assert all(token < history_end for token in expected)


@pytest.mark.parametrize("chunk_size", [2, 8, 64])
def test_compact_chunk_plan_matches_full_history_indexing_across_updates(chunk_size):
    import random

    random_state = random.Random(17)
    cache = DiffSpecDraftCache(
        DiffSpecSettings(chunk_size, chunk_size * 8, 2, 4, 2, 0.7, False),
        max_num_reqs=2, max_model_len=4096, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.begin_cycle(["a", "b"])
    for trial in range(200):
        slot = trial % 2
        cache.history_budget = chunk_size * random_state.randint(1, 7)
        retrieval_end = random_state.randint(1, 3072)
        scores = torch.tensor([
            random_state.random()
            for _ in range((retrieval_end + chunk_size - 1) // chunk_size)
        ])
        cache.set_chunk_scores(slot, scores, retrieval_end)
        history_end = max(0, retrieval_end + random_state.randint(-4 * chunk_size, 4 * chunk_size))
        for selection_kind in ("retrieved", "empty", "none"):
            if selection_kind == "empty":
                cache._selected_token_indices[slot] = torch.empty(0, dtype=torch.long)
            elif selection_kind == "none":
                cache._selected_token_indices.pop(slot, None)
            expected = cache._history_indices(slot, history_end).tolist()
            plan = cache._history_plan_host(slot, history_end)
            assert plan[0] == slot and plan[1] == len(expected)
            assert _decode_history_plan(plan, chunk_size) == expected
            assert len(expected) <= cache.history_budget
            assert expected == sorted(set(expected))
            assert all(0 <= token < history_end for token in expected)
    cache.set_active_requests(["b", "new"])
    assert 0 not in cache._host_selection_sources
    assert 0 not in cache._host_selected_chunks


def test_target_tree_attention_and_path_commit():
    settings = DiffSpecSettings(2, 8, 2, 4, 3, 0.7, False)
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=1,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.set_proposed_tree(
        torch.tensor([10, 11, 12]),
        torch.tensor([-1, -1, 0]),
        torch.tensor([1, 1, 2]),
        [1, 1, 2],
    )
    cache.prepare_target_batch(["request"], {"request": [10, 11, 12]})
    cache.begin_target_forward(["request"])
    assert cache.target_tree_active

    key_cache = torch.zeros((2, 4, 1, 2))
    value_cache = torch.zeros_like(key_cache)
    key_cache[0, :2, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    value_cache[0, :2, 0] = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    query = torch.tensor(
        [[[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]], [[2.0, 0.0]]]
    )
    key = torch.tensor(
        [[[1.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]], [[2.0, 2.0]]]
    )
    value = torch.tensor(
        [[[5.0, 5.0]], [[6.0, 1.0]], [[1.0, 7.0]], [[8.0, 3.0]]]
    )
    metadata = SimpleNamespace(
        seq_lens_list=[6],
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        actual_seq_lengths_q=[4],
    )
    output = cache.forward_target_tree_attention(
        1,
        query,
        key,
        value,
        key_cache,
        value_cache,
        metadata,
        2**-0.5,
        torch.empty_like(query),
    )
    assert output.shape == query.shape

    cache.tree_raw_key[0, :3, 0] = key[1:, 0]
    cache.tree_raw_value[0, :3, 0] = value[1:, 0]
    cache.commit_tree_path(torch.tensor([0, 2]))
    assert cache.accepted_target_token_indices().tolist() == [3]
    torch.testing.assert_close(key_cache[0, 2, 0], key[0, 0])
    torch.testing.assert_close(key_cache[0, 3, 0], key[1, 0])
    torch.testing.assert_close(key_cache[1, 0, 0], key[3, 0])
    torch.testing.assert_close(cache.raw_key[0, 3, 0], key[1, 0])
    torch.testing.assert_close(cache.raw_key[0, 4, 0], key[3, 0])


def test_request_slots_survive_reorder_and_reuse_after_cancel():
    settings = DiffSpecSettings(2, 8, 2, 4, 2, 0.7, False)
    cache = DiffSpecDraftCache(
        settings,
        max_num_reqs=2,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.set_active_requests(["a", "b"])
    slots = cache._request_slots.copy()
    cache.set_active_requests(["b", "a"])
    assert cache._active_slots == [slots["b"], slots["a"]]

    cache.set_active_requests(["b", "c"])
    assert "a" not in cache._request_slots
    assert cache._request_slots["c"] == slots["a"]


@pytest.mark.parametrize("selected,history_end,expected", [
    (list(range(65)), 64, list(range(64))),
    (list(range(70)), 64, list(range(64))),
    (list(range(32)) + list(range(96, 128)), 127,
     list(range(32)) + list(range(95, 127))),
    (list(range(12)), 11, list(range(11))),
])
def test_retrieval_and_rolling_tail_form_unique_committed_history(selected, history_end, expected):
    cache = DiffSpecDraftCache(
        DiffSpecSettings(32, 128, 8, 4, 3, 0.7, False),
        max_num_reqs=1, max_model_len=256, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.begin_cycle(["request"])
    cache._selected_token_indices[0] = torch.tensor(selected)
    cache.raw_key[0, :, 0, 0] = torch.arange(256)
    cache.raw_value[0, :, 0, 0] = torch.arange(256) + 1000
    assert cache._history_indices(0, history_end).tolist() == expected
    local = cache._rebuild_working_cache([0], [history_end + 1])
    assert local == [len(expected)]
    assert cache.working_key[0, 0, :len(expected), 0].tolist() == expected
    assert cache.working_value[0, 0, :len(expected), 0].tolist() == [p + 1000 for p in expected]


@pytest.mark.parametrize("graph_tokens", [None, 4])
def test_linear_compact_root_excludes_rejected_target_suffix(monkeypatch, graph_tokens):
    from vllm.config import CUDAGraphMode

    import diffspec.proposer as proposer_module

    monkeypatch.setattr(proposer_module, "is_forward_context_available", lambda: graph_tokens is not None)
    monkeypatch.setattr(proposer_module, "get_forward_context", lambda: SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=SimpleNamespace(num_tokens=graph_tokens),
    ))
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 8, 4, 3, 0.7, False),
        max_num_reqs=1, max_model_len=12, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=_IdentityRotary(),
    )
    cache.begin_cycle(["request"])
    cache.raw_key[0, :4].fill_(1)
    cache.raw_value[0, :4].fill_(2)
    proposer = object.__new__(AscendDiffSpecEagleProposer)
    proposer.diffspec_cache = cache
    proposer._diffspec_tree_enabled = False
    proposer.input_ids = torch.arange(4)
    proposer.hidden_states = torch.zeros(4, 2)
    proposer.arange = torch.arange(4)
    positions = torch.tensor([4, 5, 6, 7])
    proposer._get_positions = lambda n: positions[:n]
    proposer._set_positions = lambda n, p: positions[:n].copy_(p)
    proposer.model = SimpleNamespace(precompute_diffspec_kv=lambda *a: (
        torch.full((4, 2), 100.0), torch.full((4, 2), 200.0),
    ))
    kwargs = {"is_prefill": False, "num_input_tokens": 4, "batch_size": 1,
              "token_indices_to_sample": torch.tensor([0]), "inputs_embeds": None,
              "multi_steps_attn_metadata": [{"layer": SimpleNamespace(
                  seq_lens_list=[8], actual_seq_lengths_q=[4],
              )}], "num_tokens": 4}
    proposer._compact_draft_root((), kwargs)
    assert kwargs["num_input_tokens"] == (graph_tokens or 1)
    assert kwargs["num_tokens"] == 1
    if graph_tokens:
        assert proposer.input_ids[1:].count_nonzero() == 0
        assert positions[1:].count_nonzero() == 0
    assert kwargs["token_indices_to_sample"].tolist() == [0]
    assert cache._pending_local_positions == [4]
    torch.testing.assert_close(cache.working_key[0, :, :4], torch.ones(1, 4, 2))
    torch.testing.assert_close(cache.working_value[0, :, :4], torch.full((1, 4, 2), 2.0))
