# SPDX-License-Identifier: Apache-2.0
"""Device regression for sequence growth without per-length compilation."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("DIFFSPEC_RUN_DEVICE_TESTS") != "1",
    reason="Requires an explicitly selected, idle Ascend device",
)
def test_sequence_growth_reuses_retrieval_binary(tmp_path, monkeypatch):
    import torch_npu  # noqa: F401

    from diffspec.kernels import paged_chunk_attention_scores

    directory = tmp_path / "triton-cache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(directory))
    torch.npu.set_device(0)
    query = torch.ones((2, 4, 256), dtype=torch.bfloat16).npu()
    keys = torch.zeros((9, 1024, 1, 256), dtype=torch.bfloat16).npu()
    table = torch.tensor([[4, 1, 7], [8, 2, 5]], dtype=torch.int32).npu()
    for chunks in [1, *range(9, 19)]:
        lengths = torch.tensor([16 * chunks - 3, min(100, 16 * chunks - 3)], dtype=torch.int32)
        lse = lengths.float().log()[:, None].expand(-1, 4).contiguous()
        actual = paged_chunk_attention_scores(
            query,
            keys,
            lse.npu(),
            table,
            lengths.npu(),
            chunk_size=16,
            max_seq_len=int(lengths.max()),
            scale=256**-0.5,
        ).cpu()
        expected = torch.zeros((2, chunks))
        for row, length in enumerate(lengths.tolist()):
            expected[row, : (length + 15) // 16] = 1.0 / length
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-6)
    variants = list(Path(directory).glob("*/_paged_chunk_attention_scores_kernel.json"))
    # Existing in-memory cache hits need no disk file.
    assert len(variants) <= 1, f"Context growth created {len(variants)} binary variants"
