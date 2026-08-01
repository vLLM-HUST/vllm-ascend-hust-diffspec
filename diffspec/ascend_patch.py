# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Runtime integration with an unmodified vllm-ascend checkout."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

import torch

from diffspec.vllm_patch import _enabled, _tree_enabled

_PATCHED = False
_ACTIVE_RUNTIME: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "diffspec_active_runtime", default=None
)


@contextlib.contextmanager
def activate_runtime(runtime: Any):
    """Expose a draft runtime across vLLM's separate sample phase."""
    token = _ACTIVE_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME.reset(token)


def _runtime_from_runner(runner: Any) -> Any | None:
    return getattr(getattr(runner, "drafter", None), "diffspec_cache", None)


def _patch_forward_context() -> None:
    import vllm_ascend.ascend_forward_context as context_module

    original = context_module.set_ascend_forward_context

    @contextlib.contextmanager
    def set_ascend_forward_context(*args, diffspec_runtime=None, **kwargs):
        runtime = diffspec_runtime or _ACTIVE_RUNTIME.get()
        with original(*args, **kwargs):
            from vllm.forward_context import get_forward_context

            get_forward_context().diffspec_runtime = runtime
            yield

    context_module.set_ascend_forward_context = set_ascend_forward_context
    proxy_cls = context_module._ExtraForwardContextProxy
    if "diffspec_runtime" not in proxy_cls.extra_attrs:
        proxy_cls.extra_attrs += ("diffspec_runtime",)

    # Both modules import the context manager by value.
    import vllm_ascend.spec_decode.llm_base_proposer as proposer_base
    import vllm_ascend.worker.model_runner_v1 as runner_module

    proposer_base.set_ascend_forward_context = set_ascend_forward_context
    runner_module.set_ascend_forward_context = set_ascend_forward_context


def _patch_rotary_cache() -> None:
    import vllm_ascend.ops.rotary_embedding as rotary

    def record_cos_sin_cache(cos_sin_cache):
        current = rotary._cos_sin_cache
        if current is None or current.shape[0] < cos_sin_cache.shape[0]:
            rotary._cos_sin_cache = cos_sin_cache

    def record_interleaved(cos_sin_cache):
        if (
            rotary._cos_cache is not None
            and rotary._sin_cache is not None
            and rotary._cos_cache.shape[0] >= cos_sin_cache.shape[0]
            and rotary._sin_cache.shape[0] >= cos_sin_cache.shape[0]
        ):
            return
        hidden_dim = cos_sin_cache.shape[-1] // 2
        cos_cache, sin_cache = (
            cos_sin_cache.view(-1, 2, hidden_dim).repeat(1, 1, 2).chunk(2, dim=1)
        )
        rotary._cos_cache = cos_cache.squeeze(1)
        rotary._sin_cache = sin_cache.squeeze(1)

    rotary._record_cos_sin_cache = record_cos_sin_cache
    rotary._record_cos_and_sin_cache_interleaved = record_interleaved


def _patch_factory() -> None:
    import vllm_ascend.spec_decode as spec_decode
    import vllm_ascend.worker.model_runner_v1 as runner_module

    original = spec_decode.get_spec_decode_method

    def get_spec_decode_method(method, vllm_config, device, runner):
        if _enabled(vllm_config.speculative_config):
            from diffspec.proposer import AscendDiffSpecEagleProposer

            return AscendDiffSpecEagleProposer(vllm_config, device, runner)
        return original(method, vllm_config, device, runner)

    spec_decode.get_spec_decode_method = get_spec_decode_method
    runner_module.get_spec_decode_method = get_spec_decode_method


def _patch_attention() -> None:
    import vllm_ascend.attention.attention_v1 as attention_module
    from vllm.model_executor.models.utils import extract_layer_index

    cls = attention_module.AscendAttentionBackendImpl
    original_forward = cls.forward
    original_fused = cls.forward_fused_infer_attention

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        runtime = _ACTIVE_RUNTIME.get()
        self.layerIndex = extract_layer_index(layer.layer_name)
        if (
            runtime is not None
            and attention_module._EXTRA_CTX.is_draft_model
            and runtime.has_pending_attention()
        ):
            return runtime.forward_attention(query, key, value, self.scale, output)

        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if (
            runtime is not None
            and not attention_module._EXTRA_CTX.is_draft_model
            and runtime.target_tree_active
        ):
            return runtime.forward_target_tree_attention(
                self.layerIndex,
                query,
                key,
                value,
                self.key_cache,
                self.value_cache,
                attn_metadata,
                self.scale,
                output,
            )
        return original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def forward_fused_infer_attention(
        self, query, key, value, attn_metadata, output, kv_cache=None
    ):
        runtime = _ACTIVE_RUNTIME.get()
        capture = (
            runtime is not None
            and not attention_module._EXTRA_CTX.is_draft_model
            and runtime.is_target_retrieval_layer(self.layerIndex)
        )
        if not capture:
            return original_fused(
                self, query, key, value, attn_metadata, output, kv_cache
            )

        torch_npu = attention_module.torch_npu
        original_fia = torch_npu.npu_fused_infer_attention_score
        original_fia_v2 = torch_npu.npu_fused_infer_attention_score_v2
        captured_lse: list[torch.Tensor] = []

        def fia(*args, **kwargs):
            kwargs["softmax_lse_flag"] = True
            result = original_fia(*args, **kwargs)
            captured_lse.append(result[1])
            return result

        def fia_v2(*args, **kwargs):
            kwargs["return_softmax_lse"] = True
            result = original_fia_v2(*args, **kwargs)
            captured_lse.append(result[1])
            return result

        torch_npu.npu_fused_infer_attention_score = fia
        torch_npu.npu_fused_infer_attention_score_v2 = fia_v2
        try:
            result = original_fused(
                self, query, key, value, attn_metadata, output, kv_cache
            )
        finally:
            torch_npu.npu_fused_infer_attention_score = original_fia
            torch_npu.npu_fused_infer_attention_score_v2 = original_fia_v2
        if captured_lse:
            num_tokens = attn_metadata.actual_seq_lengths_q[-1]
            runtime.capture_target_attention(
                query[:num_tokens],
                captured_lse[-1],
                self.key_cache,
                attn_metadata,
                self.scale,
            )
        return result

    cls.forward = forward
    cls.forward_fused_infer_attention = forward_fused_infer_attention


def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as runner_module
    from vllm.v1.outputs import SamplerOutput

    cls = runner_module.NPUModelRunner
    original_init = cls.__init__
    original_prepare = cls._prepare_inputs
    original_execute = cls.execute_model
    original_sample = cls._sample

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not _tree_enabled(self.speculative_config):
            return
        self.num_spec_tokens = self.speculative_config.diffspec_max_tree_nodes
        self.prev_num_spec_tokens = self.num_spec_tokens
        self.uniform_decode_query_len = self.num_spec_tokens + 1
        self.decode_token_per_req = self.uniform_decode_query_len
        self.draft_token_ids_cpu = torch.empty(
            (self.max_num_reqs, self.num_spec_tokens),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

    def prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        runtime = _runtime_from_runner(self)
        if runtime is not None:
            runtime.prepare_target_batch(
                self.input_batch.req_ids,
                scheduler_output.scheduled_spec_decode_tokens,
            )
        result = original_prepare(self, scheduler_output, num_scheduled_tokens)
        if runtime is not None and runtime.target_tree_active:
            offsets = torch.cat(
                (
                    runtime.proposed_depths.new_zeros(1),
                    runtime.proposed_depths,
                )
            )
            query_count = offsets.numel()
            self.positions[:query_count].copy_(self.positions[:1] + offsets)
        return result

    def execute_model(self, *args, **kwargs):
        runtime = _runtime_from_runner(self)
        token = _ACTIVE_RUNTIME.set(runtime)
        try:
            if runtime is not None:
                import vllm_ascend.ops.rotary_embedding as rotary

                rotary._cos_sin_cache = runtime.target_cos_sin_cache
                runtime.begin_target_forward(self.input_batch.req_ids)
            return original_execute(self, *args, **kwargs)
        finally:
            _ACTIVE_RUNTIME.reset(token)

    def sample(self, logits, spec_decode_metadata):
        runtime = _runtime_from_runner(self)
        if (
            runtime is None
            or not runtime.target_tree_active
            or spec_decode_metadata is None
        ):
            return original_sample(self, logits, spec_decode_metadata)
        self.input_batch.update_async_output_token_ids()
        if not self.input_batch.sampling_metadata.all_greedy:
            raise RuntimeError("DiffSpec tree verification requires greedy sampling")
        if runtime.proposed_token_ids is None or runtime.proposed_parent_indices is None:
            raise RuntimeError("DiffSpec proposed tree metadata is unavailable")
        from diffspec.runtime import verify_greedy_tree

        accepted_nodes, emitted_tokens = verify_greedy_tree(
            runtime.proposed_token_ids,
            runtime.proposed_parent_indices,
            logits.argmax(dim=-1),
        )
        runtime.commit_tree_path(accepted_nodes)
        output_token_ids = torch.full(
            (1, spec_decode_metadata.max_spec_len + 1),
            -1,
            dtype=torch.int32,
            device=logits.device,
        )
        output_token_ids[0, : emitted_tokens.numel()] = emitted_tokens.to(torch.int32)
        return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None)

    def copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        if not self.num_spec_tokens:
            return
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()
        draft_token_ids = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.npu.current_stream()
        num_reqs, num_draft_tokens = draft_token_ids.shape
        with torch.npu.stream(self.draft_token_ids_copy_stream):
            if zeros_only:
                self.draft_token_ids_cpu[:num_reqs, :num_draft_tokens] = 0
            else:
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[
                    :num_reqs, :num_draft_tokens
                ].copy_(draft_token_ids, non_blocking=True)
            self.draft_token_ids_event.record()

    cls.__init__ = init
    cls._prepare_inputs = prepare_inputs
    cls.execute_model = execute_model
    cls._sample = sample
    cls._copy_draft_token_ids_to_cpu = copy_draft_token_ids_to_cpu


def patch_vllm_ascend() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_forward_context()
    _patch_rotary_cache()
    _patch_factory()
    _patch_attention()
    _patch_runner()
    _PATCHED = True
