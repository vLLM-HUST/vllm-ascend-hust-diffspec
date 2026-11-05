# SPDX-License-Identifier: Apache-2.0
"""Real-device check for batched draft attention and scheduler slot changes."""
import os

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
@pytest.mark.parametrize("heads,dim,rotary_dim,neox", [
    (2, 128, 128, True),
    (4, 256, 256, True),
    (4, 256, 128, True),
    (2, 128, 64, False),
])
@pytest.mark.parametrize("slots", [[0, 1], [2, 0]])
def test_batched_history_rebuild_matches_per_request_rope_and_slot_order(
    heads, dim, rotary_dim, neox, slots,
):
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    from diffspec.runtime import DiffSpecDraftCache, DiffSpecSettings

    class Rotary:
        def __init__(self):
            self.rotary_dim = rotary_dim
            self.is_neox_style = neox
            phases = torch.arange(128, dtype=torch.float32)[:, None] * torch.arange(1, rotary_dim // 2 + 1)[None, :] / 1000
            self.cache = torch.cat((phases.cos(), phases.sin()), -1).to(torch.bfloat16).npu()
        def _match_cos_sin_cache_dtype(self, value):
            return self.cache

    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(17)
    cache = DiffSpecDraftCache(
        DiffSpecSettings(8, 64, 4, 4, 4, 0.7, False),
        max_num_reqs=3, max_model_len=128, num_kv_heads=heads, head_dim=dim,
        target_num_layers=2, dtype=torch.bfloat16, device=torch.device("npu:0"),
        rotary_embedding=Rotary(),
    )
    cache.raw_key.copy_(torch.randn(cache.raw_key.shape, dtype=torch.bfloat16))
    cache.raw_value.copy_(torch.randn(cache.raw_value.shape, dtype=torch.bfloat16))
    original_key, original_value = cache.raw_key.clone(), cache.raw_value.clone()
    cache.working_key.fill_(float("nan"));cache.working_value.fill_(float("nan"))
    cache._active_slots = slots
    for end in [0, 1, 7, 62, 66, 73]:
        for slot in slots:
            cache.set_chunk_scores(slot, torch.arange((end + 7) // 8, dtype=torch.float32).npu(), end)
        lengths = [end + 1, end + 6]
        expected=[]
        for row,slot in enumerate(slots):
            indices=cache._history_indices(slot,lengths[row]-1)
            keys=cache.raw_key[slot].index_select(0,indices)
            if indices.numel():
                keys=cache._rotate_local_key(keys)
            values=cache.raw_value[slot].index_select(0,indices)
            expected.append((keys.cpu(),values.cpu()))
        actual_lengths=cache._rebuild_working_cache([0,1],lengths)
        for slot,n,(keys,values) in zip(slots,actual_lengths,expected):
            assert n==len(keys)
            assert torch.equal(cache.working_key[slot,:,:n].transpose(0,1).cpu(),keys)
            assert torch.equal(cache.working_value[slot,:,:n].transpose(0,1).cpu(),values)
        for unused in set(range(3)) - set(slots):
            assert torch.isnan(cache.working_key[unused]).all()
    assert torch.equal(cache.raw_key, original_key)
    assert torch.equal(cache.raw_value, original_value)


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
@pytest.mark.parametrize("slots", [[0], [0, 1, 2], [3, 0, 2]])
@pytest.mark.parametrize("heads,dim,query_heads", [(2, 128, 8), (4, 256, 4)])
def test_batched_linear_attention_matches_ragged_reference(
    monkeypatch, slots, heads, dim, query_heads,
):
    import torch_npu

    from diffspec.runtime import DiffSpecDraftCache, DiffSpecSettings

    class IdentityRotary:
        def forward_native(self, positions, query, key):
            return query, key

    torch.npu.set_device(0)
    torch.manual_seed(31)
    cache = DiffSpecDraftCache(
        DiffSpecSettings(8, 64, 4, 4, 3, 0.7, False),
        max_num_reqs=4, max_model_len=128, num_kv_heads=heads, head_dim=dim,
        target_num_layers=2, dtype=torch.bfloat16, device=torch.device("npu:0"),
        rotary_embedding=IdentityRotary(),
    )
    keys = torch.randn(cache.working_key.shape, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    positions = [3, 11, 6][:len(slots)]
    for slot, position in zip(slots, positions):
        keys[slot, :, position:] = float('nan')
        values[slot, :, position:] = float('nan')
    cache.working_key.copy_(keys)
    cache.working_value.copy_(values)
    calls = []
    original = torch_npu.npu_fused_infer_attention_score

    def measured(**kwargs):
        calls.append((kwargs['query'].shape[0], kwargs['actual_seq_lengths_kv']))
        assert kwargs['key'].shape[2] == cache.settings.token_budget
        assert kwargs['value'].shape[2] == cache.settings.token_budget
        if slots == list(range(slots[0], slots[0] + len(slots))):
            assert kwargs['key'].data_ptr() == cache.working_key[slots[0]].data_ptr()
            assert kwargs['value'].data_ptr() == cache.working_value[slots[0]].data_ptr()
        return original(**kwargs)

    monkeypatch.setattr(torch_npu, 'npu_fused_infer_attention_score', measured)
    for step in range(2):
        query = torch.randn((5, query_heads, dim), dtype=torch.bfloat16)
        new_key = torch.randn((5, heads, dim), dtype=torch.bfloat16)
        new_value = torch.randn_like(new_key)
        output = torch.full_like(query, float('nan')).npu()
        cache._pending_request_indices = slots
        cache._pending_local_positions = [p + step for p in positions]
        actual = cache.forward_attention(
            query.npu(), new_key.npu(), new_value.npu(), dim**-0.5, output,
        ).cpu()
        for row, (slot, position) in enumerate(zip(slots, positions)):
            end = position + step + 1
            keys[slot, :, end - 1] = new_key[row]
            values[slot, :, end - 1] = new_value[row]
            k = keys[slot, :, :end].repeat_interleave(query_heads // heads, dim=0).float()
            v = values[slot, :, :end].repeat_interleave(query_heads // heads, dim=0).float()
            weights = torch.einsum('hd,hsd->hs', query[row].float(), k)
            expected = torch.einsum('hs,hsd->hd', (weights * dim**-0.5).softmax(-1), v)
            torch.testing.assert_close(actual[row].float(), expected, rtol=0.03, atol=0.02)
            assert cache._working_lens[slot] == end
        assert actual[len(slots):].count_nonzero() == 0
        assert calls[-1] == (len(slots), [p + step + 1 for p in positions])
        assert len(calls) == step + 1
        assert cache._pending_request_indices is None
        assert cache._pending_local_positions is None
