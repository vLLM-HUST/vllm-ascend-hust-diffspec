# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

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
from diffspec.proposer import (
    AscendDiffSpecEagleProposer,
)


class _IdentityRotary:
    def forward_native(self, positions, query, key):
        return query, key


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


def test_target_retrieval_uses_last_prefill_query(monkeypatch):
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

    # Retrieval starts only after the configured number of completed
    # verifier cycles; until then _history_indices supplies the recent
    # working window.
    cache.accept_target_queries(torch.ones(2), is_prefill=False)
    assert cache._selected_token_indices == {}
    cache.accept_target_queries(torch.ones(2), is_prefill=False)

    assert cache._selected_token_indices[0].tolist() == [0, 1, 4, 5]
    assert cache._selected_token_indices[1].tolist() == [0, 1, 2, 3]


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
