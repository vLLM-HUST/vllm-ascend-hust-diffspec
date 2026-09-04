from types import SimpleNamespace

import pytest
import torch

import diffspec.ascend_patch as ascend_patch
from diffspec.runtime import (
    DiffSpecDraftCache,
    DiffSpecSettings,
    find_target_rotary_cache,
    validate_diffspec_runtime,
)


def _sage_mate_config(
    *, draft_vocab: int = 248320, tp: int = 4, uses_mrope: bool = True
):
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=1),
        dtype=torch.bfloat16,
        architectures=["LlamaForCausalLMEagle3"],
        get_vocab_size=lambda: draft_vocab,
    )
    speculative = SimpleNamespace(
        use_diffspec=lambda: True,
        method="eagle3",
        draft_model_config=draft,
        disable_padded_drafter_batch=False,
        enforce_eager=False,
    )
    target = SimpleNamespace(
        enforce_eager=False,
        uses_mrope=uses_mrope,
        use_mla=False,
        quantization=None,
        dtype=torch.bfloat16,
        architectures=["Qwen3_5ForConditionalGeneration"],
        get_vocab_size=lambda: 248320,
    )
    return SimpleNamespace(
        speculative_config=speculative,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=target,
    )


def test_sage_mate_contract_requires_tp4_graph_and_matching_draft_vocab():
    # Qwen3.5/Qwen3.8 exposes M-RoPE in the target config.  Text-only Eagle3
    # drafting must not be rejected solely because that capability is present.
    validate_diffspec_runtime(_sage_mate_config())
    validate_diffspec_runtime(_sage_mate_config(uses_mrope=False))

    with pytest.raises(ValueError, match="tensor parallel size 4"):
        validate_diffspec_runtime(_sage_mate_config(tp=1))
    with pytest.raises(ValueError, match="vocabulary does not match"):
        validate_diffspec_runtime(_sage_mate_config(draft_vocab=151936))


def test_hybrid_target_rotary_cache_skips_gdn_layers():
    cache = torch.empty(8, 16)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(),
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        rotary_emb=SimpleNamespace(cos_sin_cache=cache)
                    )
                ),
            ]
        )
    )

    assert find_target_rotary_cache(model) is cache


def test_hybrid_target_rotary_cache_traverses_multimodal_wrapper():
    cache = torch.empty(8, 16)
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            rotary_emb=SimpleNamespace(cos_sin_cache=cache)
                        )
                    )
                ]
            )
        )
    )

    assert find_target_rotary_cache(model) is cache


def test_active_runtime_survives_contextvar_loss_during_graph_execution(monkeypatch):
    runtime = object()
    ascend_patch._ACTIVE_RUNTIME_FALLBACK = runtime
    try:
        monkeypatch.setattr(
            ascend_patch, "_ACTIVE_RUNTIME", SimpleNamespace(get=lambda: None)
        )
        assert ascend_patch._active_runtime() is runtime
    finally:
        ascend_patch._ACTIVE_RUNTIME_FALLBACK = None


def test_pending_cache_state_identifies_draft_attention_without_forward_context():
    pending = SimpleNamespace(has_pending_attention=lambda: True)
    idle = SimpleNamespace(has_pending_attention=lambda: False)

    assert ascend_patch._use_diffspec_draft_attention(pending)
    assert not ascend_patch._use_diffspec_draft_attention(idle)
    assert not ascend_patch._use_diffspec_draft_attention(None)


def test_compact_prefill_state_is_ready_before_compiled_graph_replay():
    rotary = SimpleNamespace(
        forward_native=lambda positions, query, key: (query, key)
    )
    cache = DiffSpecDraftCache(
        DiffSpecSettings(2, 8, 2, 4, 3, 0.7, False),
        max_num_reqs=1,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
        target_num_layers=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rotary_embedding=rotary,
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
