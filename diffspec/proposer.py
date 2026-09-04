# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from diffspec.runtime import (
    DiffSpecDraftCache,
    DiffSpecSettings,
    find_target_rotary_cache,
    select_tree_level,
    validate_diffspec_runtime,
)
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

logger = init_logger(__name__)


class AscendDiffSpecEagleProposer(AscendEagleProposer):
    """Eagle3 proposer backed by a compact, locally positioned KV cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        validate_diffspec_runtime(vllm_config)
        self.diffspec_settings = DiffSpecSettings.from_vllm_config(vllm_config)
        self.diffspec_cache: DiffSpecDraftCache | None = None
        self._diffspec_tree_enabled = False
        self._linear_request_id: str | None = None
        self._linear_runtime_depth: int | None = None
        self._linear_depth_floor = 1
        self._linear_good_cycles = 0
        self._linear_bad_cycles = 0
        super().__init__(vllm_config, device, runner=runner)

    def load_model(self, model: torch.nn.Module) -> None:
        super().load_model(model)
        if not hasattr(self.model, "set_diffspec_kv_sink"):
            raise TypeError(
                "The Eagle3 model does not expose the DiffSpec pre-RoPE KV hook"
            )
        draft_layers = self.model.model.layers
        if len(draft_layers) != 1:
            raise ValueError("Ascend DiffSpec requires exactly one draft layer")
        draft_attention = draft_layers[0].self_attn
        target_cos_sin_cache = find_target_rotary_cache(self.runner.model)
        self.diffspec_cache = DiffSpecDraftCache(
            self.diffspec_settings,
            max_num_reqs=self.runner.max_num_reqs,
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_kv_heads=draft_attention.num_kv_heads,
            head_dim=draft_attention.head_dim,
            target_num_layers=self.vllm_config.model_config.get_num_layers(
                self.vllm_config.parallel_config
            ),
            dtype=self.dtype,
            device=self.device,
            rotary_embedding=draft_attention.rotary_emb,
            target_cos_sin_cache=target_cos_sin_cache,
        )
        self.model.set_diffspec_kv_sink(self.diffspec_cache.capture_raw_kv)
        cache_bytes = (
            self.diffspec_cache.raw_key.numel()
            + self.diffspec_cache.raw_value.numel()
            + self.diffspec_cache.working_key.numel()
            + self.diffspec_cache.working_value.numel()
        ) * torch.tensor([], dtype=self.dtype).element_size()
        logger.info(
            "DiffSpec draft cache initialized: budget=%d, max_depth=%d, "
            "max_nodes=%d, memory=%.2f GiB",
            self.diffspec_settings.token_budget,
            self.diffspec_settings.max_tree_depth,
            self.diffspec_settings.max_tree_nodes,
            cache_bytes / (1024**3),
        )
        logger.info(
            "DiffSpec RoPE tables: target=%d, draft=%d",
            target_cos_sin_cache.shape[0],
            draft_attention.rotary_emb.cos_sin_cache.shape[0],
        )
        logger.info(
            "DiffSpec verification mode: configured=%s, resolved=%s",
            self.speculative_config.diffspec_verification_mode,
            "tree"
            if self.speculative_config.diffspec_verification_mode == "tree"
            else "linear",
        )

    def _run_merged_draft(self, *args, **kwargs) -> torch.Tensor:
        if self.diffspec_cache is None:
            return super()._run_merged_draft(*args, **kwargs)
        from diffspec.ascend_patch import activate_runtime

        with activate_runtime(self.diffspec_cache):
            self.diffspec_cache.begin_cycle(self.runner.input_batch.req_ids)
            try:
                self._compact_long_prefill(args, kwargs)
                if self._diffspec_tree_enabled:
                    self._compact_tree_decode_root(kwargs)
                    return self._run_tree_draft(**kwargs)
                linear_depth = self._select_linear_depth(kwargs)
                if linear_depth == self.num_speculative_tokens:
                    return super()._run_merged_draft(*args, **kwargs)
                configured_depth = self.num_speculative_tokens
                self.num_speculative_tokens = linear_depth
                try:
                    return super()._run_merged_draft(*args, **kwargs)
                finally:
                    self.num_speculative_tokens = configured_depth
            finally:
                self.diffspec_cache.end_cycle()

    def _select_linear_depth(self, kwargs) -> int:
        """Choose the measured-safe linear depth without resizing buffers."""
        configured_depth = self.num_speculative_tokens
        if (
            not self.speculative_config.diffspec_adaptive_profile
            or self.speculative_config.diffspec_verification_mode == "tree"
        ):
            return configured_depth
        metadata_steps = kwargs.get("multi_steps_attn_metadata") or []
        if not metadata_steps:
            return configured_depth
        first_step = metadata_steps[0]
        metadata = (
            next(iter(first_step.values()), None)
            if isinstance(first_step, dict)
            else first_step
        )
        seq_lens = getattr(metadata, "seq_lens_list", ())
        base_depth = configured_depth
        if (
            seq_lens
            and max(seq_lens)
            >= self.speculative_config.diffspec_long_context_threshold
        ):
            base_depth = min(
                configured_depth,
                self.speculative_config.diffspec_long_context_depth,
            )
        request_ids = self.runner.input_batch.req_ids
        request_id = request_ids[0] if len(request_ids) == 1 else None
        if request_id != self._linear_request_id:
            self._linear_request_id = request_id
            self._linear_runtime_depth = base_depth
            self._linear_depth_floor = base_depth
            self._linear_good_cycles = 0
            self._linear_bad_cycles = 0
        if self._linear_runtime_depth is None:
            self._linear_runtime_depth = base_depth
        return self._linear_runtime_depth

    def _observe_linear_depth(self, valid_counts: torch.Tensor) -> None:
        """Raise depth after sustained acceptance and retreat on misses."""
        depth = self._linear_runtime_depth
        if (
            depth is None
            or self._diffspec_tree_enabled
            or not self.speculative_config.diffspec_adaptive_profile
            or valid_counts.numel() != 1
        ):
            return
        emitted_tokens = int(valid_counts.reshape(-1)[0].item())
        accepted_fraction = max(0, emitted_tokens - 1) / depth
        if accepted_fraction >= 0.8:
            self._linear_good_cycles += 1
            self._linear_bad_cycles = 0
        elif accepted_fraction <= 0.35:
            self._linear_bad_cycles += 1
            self._linear_good_cycles = 0
        else:
            self._linear_good_cycles = 0
            self._linear_bad_cycles = 0

        configured_depth = self.speculative_config.num_speculative_tokens
        if self._linear_good_cycles >= 4 and depth < configured_depth:
            self._linear_runtime_depth = depth + 1
            self._linear_good_cycles = 0
            logger.info(
                "DiffSpec linear depth increased to %d after sustained "
                "acceptance",
                self._linear_runtime_depth,
            )
        elif self._linear_bad_cycles >= 3 and depth > self._linear_depth_floor:
            self._linear_runtime_depth = depth - 1
            self._linear_bad_cycles = 0
            logger.info(
                "DiffSpec linear depth decreased to %d after acceptance drop",
                self._linear_runtime_depth,
            )

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        result = super().prepare_next_token_ids_padded(*args, **kwargs)
        if self.diffspec_cache is not None:
            _, valid_counts = result
            self._observe_linear_depth(valid_counts)
            self.diffspec_cache.accept_target_queries(
                valid_counts,
                is_prefill=self.diffspec_cache.proposed_token_ids is None,
            )
        return result

    def prepare_inputs_padded(self, *args, **kwargs):
        result = super().prepare_inputs_padded(*args, **kwargs)
        if self.diffspec_cache is None or not self.diffspec_cache.target_tree_active:
            return result
        common_metadata, _, _, rejected = result
        accepted = self.diffspec_cache.accepted_target_token_indices()
        return common_metadata, accepted, accepted, rejected

    def _compact_long_prefill(self, args, kwargs) -> None:
        """Replace a long first draft pass with cache-only QKV plus one token."""
        if args:
            raise TypeError("DiffSpec draft execution requires keyword arguments")
        if not kwargs.get("is_prefill"):
            return
        num_input_tokens = kwargs["num_input_tokens"]
        if num_input_tokens == kwargs["batch_size"]:
            return
        assert self.diffspec_cache is not None

        batch_size = kwargs["batch_size"]
        sample_indices = kwargs["token_indices_to_sample"][:batch_size].long()
        input_ids = self.input_ids[:num_input_tokens]
        positions = self._get_positions(num_input_tokens)
        hidden_states = self.hidden_states[:num_input_tokens]
        inputs_embeds = kwargs.get("inputs_embeds")
        raw_key, raw_value = self.model.precompute_diffspec_kv(
            input_ids,
            hidden_states,
            inputs_embeds,
        )
        first_step_metadata = next(
            iter(kwargs["multi_steps_attn_metadata"][0].values())
        )
        self.diffspec_cache.populate_prefill(
            positions,
            raw_key,
            raw_value,
            first_step_metadata,
        )

        compact_ids = input_ids.index_select(0, sample_indices)
        compact_positions = positions.index_select(0, sample_indices)
        compact_hidden_states = hidden_states.index_select(0, sample_indices)
        self.input_ids[:batch_size].copy_(compact_ids)
        self._set_positions(batch_size, compact_positions)
        self.hidden_states[:batch_size].copy_(compact_hidden_states)
        if inputs_embeds is not None:
            compact_embeds = inputs_embeds.index_select(0, sample_indices)
            self.inputs_embeds[:batch_size].copy_(compact_embeds)
            kwargs["inputs_embeds"] = self.inputs_embeds[:batch_size]

        kwargs["num_input_tokens"] = batch_size
        kwargs["num_tokens"] = batch_size
        kwargs["token_indices_to_sample"] = self.arange[:batch_size]
        self.diffspec_cache.prepare_compact_prefill()

    def _compact_tree_decode_root(self, kwargs) -> None:
        batch_size = kwargs["batch_size"]
        num_input_tokens = kwargs["num_input_tokens"]
        if num_input_tokens == batch_size:
            kwargs["token_indices_to_sample"] = self.arange[:batch_size]
            return
        if kwargs.get("is_prefill"):
            return
        sample_indices = kwargs["token_indices_to_sample"][:batch_size].long()
        compact_ids = self.input_ids[:num_input_tokens].index_select(
            0, sample_indices
        )
        compact_positions = self._get_positions(num_input_tokens).index_select(
            0, sample_indices
        )
        compact_hidden_states = self.hidden_states[:num_input_tokens].index_select(
            0, sample_indices
        )
        self.input_ids[:batch_size].copy_(compact_ids)
        self._set_positions(batch_size, compact_positions)
        self.hidden_states[:batch_size].copy_(compact_hidden_states)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            compact_embeds = inputs_embeds.index_select(0, sample_indices)
            self.inputs_embeds[:batch_size].copy_(compact_embeds)
            kwargs["inputs_embeds"] = self.inputs_embeds[:batch_size]
        kwargs["num_input_tokens"] = batch_size
        kwargs["num_tokens"] = batch_size
        kwargs["token_indices_to_sample"] = self.arange[:batch_size]

    def _propose(self, *args, **kwargs) -> torch.Tensor:
        sampling_metadata = kwargs.get("sampling_metadata")
        scheduler_output = kwargs.get("scheduler_output")
        tree_requested = (
            self.speculative_config.diffspec_verification_mode == "tree"
        )
        self._diffspec_tree_enabled = bool(
            tree_requested
            and sampling_metadata is not None
            and scheduler_output is not None
            and not scheduler_output.has_structured_output_requests
            and sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
        )
        if not self._diffspec_tree_enabled and self.diffspec_cache is not None:
            self.diffspec_cache.clear_proposed_tree()
        return super()._propose(*args, **kwargs)

    def _draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.logits_processor(self.model.lm_head, hidden_states)

    def _to_target_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        mapping = getattr(self.model, "draft_id_to_target_id", None)
        if mapping is None:
            return draft_token_ids
        return draft_token_ids + mapping.index_select(
            0, draft_token_ids.reshape(-1)
        ).view_as(draft_token_ids)

    def _run_tree_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        """Generate a breadth-wise Eagle3 candidate tree in eager mode."""
        del target_positions, num_tokens, is_prefill
        if batch_size != 1:
            raise RuntimeError("DiffSpec tree drafting v1 requires batch size 1")
        if self.diffspec_cache is None:
            raise RuntimeError("DiffSpec cache is unavailable")
        if not self.model_returns_tuple():
            raise RuntimeError("DiffSpec Eagle3 must return recurrent hidden state")

        model_positions = self._get_positions(num_input_tokens)
        model_kwargs = {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
        root_last_hidden, root_recurrent_hidden = self.model(**model_kwargs)
        sample_indices = token_indices_to_sample[:batch_size].long()
        root_last_hidden = root_last_hidden.index_select(0, sample_indices)
        frontier_hidden = root_recurrent_hidden.index_select(0, sample_indices)
        frontier_logits = self._draft_logits(root_last_hidden)
        frontier_parent_indices = torch.full(
            (batch_size,), -1, dtype=torch.long, device=self.device
        )
        frontier_log_probs = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

        profile = self.diffspec_cache.controller.profile
        max_nodes = min(profile.max_nodes, self.diffspec_settings.max_tree_nodes)
        max_depth = min(profile.max_depth, self.diffspec_settings.max_tree_depth)
        token_levels: list[torch.Tensor] = []
        parent_levels: list[torch.Tensor] = []
        depth_levels: list[torch.Tensor] = []
        path_log_prob_levels: list[torch.Tensor] = []
        num_nodes = 0

        for depth in range(1, max_depth + 1):
            remaining_capacity = (
                self.diffspec_cache.max_draft_tree_nodes - num_nodes
            )
            if remaining_capacity <= 0:
                break
            level_budget = min(max_nodes, remaining_capacity)
            level = select_tree_level(
                frontier_logits,
                frontier_parent_indices,
                frontier_log_probs,
                node_budget=level_budget,
                cumulative_threshold=self.diffspec_settings.tree_threshold,
            )
            level_size = level.token_ids.numel()
            if level_size == 0:
                break
            node_indices = torch.arange(
                num_nodes,
                num_nodes + level_size,
                dtype=torch.long,
                device=self.device,
            )
            target_token_ids = self._to_target_token_ids(level.token_ids)
            node_hidden_inputs = frontier_hidden.index_select(
                0, level.parent_rows
            )
            depth_tensor = torch.full_like(node_indices, depth)
            token_levels.append(target_token_ids)
            parent_levels.append(level.parent_indices)
            depth_levels.append(depth_tensor)
            path_log_prob_levels.append(level.path_log_probs)
            num_nodes += level_size

            if depth == max_depth:
                raw_key, raw_value = self.model.precompute_diffspec_kv(
                    target_token_ids,
                    node_hidden_inputs,
                )
                self.diffspec_cache.store_tree_raw_kv(
                    node_indices, raw_key, raw_value
                )
                break

            self.diffspec_cache.prepare_tree_level(
                node_indices,
                level.parent_indices,
                depth_tensor,
            )
            node_positions = (
                self.diffspec_cache.pending_tree_local_positions()
            )
            forward_context = get_forward_context()
            if multi_steps_attn_metadata:
                forward_context.attn_metadata = multi_steps_attn_metadata[depth]
            node_last_hidden, node_recurrent_hidden = self.model(
                input_ids=target_token_ids,
                positions=node_positions,
                hidden_states=node_hidden_inputs,
            )
            frontier_logits = self._draft_logits(node_last_hidden)
            frontier_hidden = node_recurrent_hidden
            frontier_parent_indices = node_indices
            frontier_log_probs = level.path_log_probs

        token_ids = torch.cat(token_levels)
        parent_indices = torch.cat(parent_levels)
        depths = torch.cat(depth_levels)
        path_log_probs = torch.cat(path_log_prob_levels)
        self.diffspec_cache.finalize_proposed_tree(
            token_ids,
            parent_indices,
            depths,
            path_log_probs,
            profile,
        )
        if self.diffspec_cache.proposed_token_ids is None:
            raise RuntimeError("DiffSpec tree pruning produced no candidates")
        return self.diffspec_cache.proposed_token_ids.view(1, -1)
