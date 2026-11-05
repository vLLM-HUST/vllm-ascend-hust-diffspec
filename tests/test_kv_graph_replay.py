# SPDX-License-Identifier: Apache-2.0
"""Compiled execution must perform every dynamic draft KV state transition."""

from types import SimpleNamespace

import pytest
import torch

from diffspec.ascend_patch import activate_runtime
from diffspec.runtime import DiffSpecDraftCache, DiffSpecSettings
from diffspec.vllm_patch import patch_vllm


class IdentityRotary:
    def forward_native(self, positions, query, key):
        return query, key


@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("prepared_root", [False, True])
def test_compiled_kv_capture_updates_each_draft_step_and_local_position(
    monkeypatch, padding, prepared_root,
):
    patch_vllm()
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 8, 4, 3, 0.7, False),
        max_num_reqs=1,
        max_model_len=12,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=IdentityRotary(),
    )
    cache.begin_cycle(["request-0"])
    cache.raw_key.zero_()
    cache.raw_value.zero_()
    if prepared_root:
        cache.prepare_compact_prefill(SimpleNamespace(seq_lens_list=[6]), 1)
    monkeypatch.setattr(
        cache,
        "_first_attention_metadata",
        lambda: SimpleNamespace(
            actual_seq_lengths_q=[1],
            seq_lens_list=[6],
        ),
    )

    def capture(positions, key, value):
        return torch.ops.vllm.diffspec_capture_raw_kv(positions, key, value, 0)

    compiled = torch.compile(capture, backend="aot_eager", fullgraph=True)
    observed = []
    captured_positions = None
    with activate_runtime(cache):
        for step in range(3):
            key = torch.full((1 + padding, 2), 999.0)
            key[0] = torch.tensor([step + 1.0, step + 2.0])
            value = key + 10
            positions = torch.full((1 + padding,), 11, dtype=torch.int32)
            positions[0] = 5 + step
            result = compiled(positions, key, value)
            if captured_positions is None:
                captured_positions = result
            assert result.data_ptr() == captured_positions.data_ptr()
            assert captured_positions[0].item() == 4 + step
            assert result.shape == positions.shape
            assert result[1:].tolist() == [0] * padding
            observed.append(result[0].item())
            assert cache.has_pending_attention()
            assert cache._pending_local_positions == [4 + step]
            torch.testing.assert_close(cache.raw_key[0, 5 + step, 0], key[0])
            torch.testing.assert_close(cache.raw_value[0, 5 + step, 0], value[0])
            output = torch.full((1 + padding, 1, 2), float("nan"))
            cache.forward_attention(
                torch.ones(1 + padding, 1, 2),
                key.view(1 + padding, 1, 2),
                value.view(1 + padding, 1, 2),
                1.0,
                output,
            )
            assert torch.isfinite(output).all()
            assert output[1:].count_nonzero() == 0
            assert not cache.has_pending_attention()
    assert observed == [4, 5, 6]
    assert cache._draft_step == 3 + int(prepared_root)
    assert cache.raw_key[0, 11].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_rope_graph_keeps_dummy_capture_address_and_zeroes_padding(dtype):
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 8, 4, 3, 0.7, False),
        max_num_reqs=1, max_model_len=12, num_kv_heads=1, head_dim=2,
        target_num_layers=2, dtype=torch.float32, device=torch.device("cpu"),
        rotary_embedding=IdentityRotary(),
    )
    absolute = torch.tensor([7, 8, 9, 10], dtype=dtype)
    captured = cache.graph_safe_rope_positions(absolute, None)
    assert captured.tolist() == [7, 8, 9, 10]
    replay = cache.graph_safe_rope_positions(absolute[:1], torch.tensor([4], dtype=dtype))
    assert replay.data_ptr() == captured.data_ptr()
    assert captured.tolist() == [4, 0, 0, 0]
    cache.graph_safe_rope_positions(absolute, torch.tensor([5, 0, 0, 0], dtype=dtype))
    assert captured.tolist() == [5, 0, 0, 0]
    for count in [8, 2, 7, 1, 0, 4]:
        positions = torch.arange(count, dtype=dtype)
        replay = cache.graph_safe_rope_positions(positions, positions)
        assert replay.untyped_storage().data_ptr() == captured.untyped_storage().data_ptr()
        expected = torch.zeros(12, dtype=dtype)
        expected[:count] = positions
        torch.testing.assert_close(cache._rope_position_buffers[dtype], expected)


def test_kv_capture_is_a_default_piecewise_graph_boundary():
    patch_vllm()
    from vllm.config import CompilationConfig, CompilationMode

    config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE, cudagraph_mode="PIECEWISE")
    config.set_splitting_ops_for_v1("naive")
    assert "vllm::diffspec_capture_raw_kv" in config.splitting_ops
    assert "vllm::unified_attention_with_output" in config.splitting_ops
