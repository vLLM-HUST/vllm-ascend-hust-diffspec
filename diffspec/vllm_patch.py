# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Runtime integration with an unmodified vLLM checkout."""

from __future__ import annotations

import json
from typing import Any

import torch

from diffspec.config import DiffSpecPluginConfig

_PATCHED = False


def _enabled(speculative_config: Any) -> bool:
    return (
        speculative_config is not None
        and getattr(speculative_config, "draft_context_policy", "full") == "diffspec"
    )


def _tree_enabled(speculative_config: Any) -> bool:
    return _enabled(speculative_config) and (
        getattr(speculative_config, "diffspec_verification_mode", "auto")
        == "tree"
    )


def _patch_config() -> None:
    import vllm.config.speculative as speculative_module
    import vllm.config.vllm as vllm_config_module
    import vllm.engine.arg_utils as arg_utils

    speculative_cls = speculative_module.SpeculativeConfig
    defaults = DiffSpecPluginConfig()
    for name in defaults.__dataclass_fields__:
        if not hasattr(speculative_cls, name):
            setattr(speculative_cls, name, getattr(defaults, name))

    if not hasattr(speculative_cls, "use_diffspec"):
        setattr(speculative_cls, "use_diffspec", _enabled)
    if not hasattr(speculative_cls, "max_speculative_tokens"):
        setattr(
            speculative_cls,
            "max_speculative_tokens",
            property(
                lambda self: (
                    self.diffspec_max_tree_nodes
                    if _tree_enabled(self)
                    else self.num_speculative_tokens
                )
            ),
        )

    original_create = arg_utils.EngineArgs.create_speculative_config

    def create_speculative_config(self, target_model_config, target_parallel_config):
        raw = self.speculative_config
        if not isinstance(raw, dict) or raw.get("draft_context_policy", "full") != "diffspec":
            return original_create(self, target_model_config, target_parallel_config)

        plugin_config, standard_config = DiffSpecPluginConfig.extract(raw)
        plugin_config.validate(
            method=standard_config.get("method"),
            depth=standard_config.get("num_speculative_tokens"),
            max_model_len=standard_config.get("max_model_len"),
        )
        self.speculative_config = standard_config
        result = original_create(self, target_model_config, target_parallel_config)
        if result is None:
            raise RuntimeError("DiffSpec requires a speculative model")
        plugin_config.attach(result)
        return result

    arg_utils.EngineArgs.create_speculative_config = create_speculative_config

    original_hash = speculative_cls.compute_hash

    def compute_hash(self):
        base = original_hash(self)
        if not _enabled(self):
            return base
        values = {
            name: getattr(self, name)
            for name in defaults.__dataclass_fields__
        }
        return f"{base}:{json.dumps(values, sort_keys=True, separators=(',', ':'))}"

    speculative_cls.compute_hash = compute_hash

    vllm_config_cls = vllm_config_module.VllmConfig
    original_num_tokens = vllm_config_cls.num_speculative_tokens

    def num_speculative_tokens(self):
        spec = self.speculative_config
        if _tree_enabled(spec):
            return spec.diffspec_max_tree_nodes
        return original_num_tokens.fget(self)

    vllm_config_cls.num_speculative_tokens = property(num_speculative_tokens)

    # arg_utils imports these classes by value, so update its binding as well.
    arg_utils.SpeculativeConfig = speculative_cls


def _patch_metrics() -> None:
    from vllm.v1.spec_decode.metrics import SpecDecodingProm

    original_init = SpecDecodingProm.__init__

    def init(self, speculative_config, *args, **kwargs):
        if not _tree_enabled(speculative_config):
            return original_init(self, speculative_config, *args, **kwargs)
        original_depth = speculative_config.num_speculative_tokens
        object.__setattr__(
            speculative_config,
            "num_speculative_tokens",
            speculative_config.diffspec_max_tree_nodes,
        )
        try:
            return original_init(self, speculative_config, *args, **kwargs)
        finally:
            object.__setattr__(
                speculative_config,
                "num_speculative_tokens",
                original_depth,
            )

    SpecDecodingProm.__init__ = init


def _patch_eagle3_model() -> None:
    import types

    import vllm.model_executor.models.llama_eagle3 as eagle3

    def attention_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        sink = getattr(self, "_diffspec_kv_sink", None)
        if sink is not None:
            compact_positions = sink(
                getattr(self, "_diffspec_draft_layer_idx", 0),
                positions,
                k,
                v,
            )
            if compact_positions is not None:
                positions = compact_positions
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    original_decoder_init = eagle3.LlamaDecoderLayer.__init__

    def decoder_init(self, *args, **kwargs):
        original_decoder_init(self, *args, **kwargs)
        self.self_attn._diffspec_draft_layer_idx = self.layer_idx
        self.self_attn.forward = types.MethodType(attention_forward, self.self_attn)

    eagle3.LlamaDecoderLayer.__init__ = decoder_init

    def project_diffspec_kv(self, embeds, hidden_states):
        if self.layer_idx != 0:
            raise ValueError("DiffSpec cache-only prefill requires one draft layer")
        embeds = self.input_layernorm(embeds)
        hidden_states, _ = self._residual_norm(hidden_states=hidden_states)
        qkv_input = torch.cat([embeds, hidden_states], dim=-1)
        qkv, _ = self.self_attn.qkv_proj(qkv_input)
        _, key, value = qkv.split(
            [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
            dim=-1,
        )
        return key, value

    eagle3.LlamaDecoderLayer.project_diffspec_kv = project_diffspec_kv

    def set_diffspec_kv_sink(self, sink):
        for layer in self.layers:
            layer.self_attn._diffspec_kv_sink = sink

    def model_precompute(self, input_ids, hidden_states, input_embeds=None):
        if len(self.layers) != 1:
            raise ValueError("DiffSpec cache-only prefill requires one draft layer")
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        return self.layers[0].project_diffspec_kv(input_embeds, hidden_states)

    eagle3.LlamaModel.set_diffspec_kv_sink = set_diffspec_kv_sink
    eagle3.LlamaModel.precompute_diffspec_kv = model_precompute

    def outer_set_sink(self, sink):
        self.model.set_diffspec_kv_sink(sink)

    def outer_precompute(self, input_ids, hidden_states, inputs_embeds=None):
        return self.model.precompute_diffspec_kv(input_ids, hidden_states, inputs_embeds)

    eagle3.Eagle3LlamaForCausalLM.set_diffspec_kv_sink = outer_set_sink
    eagle3.Eagle3LlamaForCausalLM.precompute_diffspec_kv = outer_precompute


def patch_vllm() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_config()
    _patch_metrics()
    _patch_eagle3_model()
