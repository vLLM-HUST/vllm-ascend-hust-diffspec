from types import SimpleNamespace

import pytest
import torch

from diffspec.runtime import find_target_rotary_cache, validate_diffspec_runtime


def _sage_mate_config(*, draft_vocab: int = 248320, tp: int = 4):
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
        uses_mrope=False,
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
    validate_diffspec_runtime(_sage_mate_config())

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
