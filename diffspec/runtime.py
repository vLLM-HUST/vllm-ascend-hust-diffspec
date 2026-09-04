# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Target-proxy-guided context selection and tree utilities for DiffSpec."""

import time
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class DiffSpecSettings:
    chunk_size: int
    token_budget: int
    retrieval_interval: int
    max_tree_nodes: int
    max_tree_depth: int
    tree_threshold: float
    adaptive_profile: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "DiffSpecSettings":
        spec_config = vllm_config.speculative_config
        if spec_config is None or not spec_config.use_diffspec():
            raise ValueError("DiffSpec settings require an enabled DiffSpec policy")
        return cls(
            chunk_size=spec_config.diffspec_chunk_size,
            token_budget=spec_config.diffspec_token_budget,
            retrieval_interval=spec_config.diffspec_retrieval_interval,
            max_tree_nodes=spec_config.diffspec_max_tree_nodes,
            max_tree_depth=spec_config.num_speculative_tokens,
            tree_threshold=spec_config.diffspec_tree_threshold,
            adaptive_profile=spec_config.diffspec_adaptive_profile,
        )


@dataclass(frozen=True)
class DiffSpecSelection:
    chunk_indices: torch.Tensor
    token_indices: torch.Tensor
    chunk_scores: torch.Tensor


@dataclass(frozen=True)
class DiffSpecTreeProfile:
    max_nodes: int
    max_depth: int


@dataclass(frozen=True)
class DiffSpecTreeLevel:
    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    parent_rows: torch.Tensor
    path_log_probs: torch.Tensor


@dataclass
class DiffSpecRuntimeMetrics:
    retrieval_count: int = 0
    retrieval_seconds: float = 0.0
    tree_cycles: int = 0
    tree_nodes: int = 0
    accepted_depth: int = 0
    emitted_tokens: int = 0
    working_cache_tokens: int = 0

    def observe_retrieval(self, elapsed_seconds: float) -> None:
        self.retrieval_count += 1
        self.retrieval_seconds += elapsed_seconds

    def observe_tree(
        self,
        *,
        nodes: int,
        accepted_depth: int,
        emitted_tokens: int,
        working_cache_tokens: int,
    ) -> None:
        self.tree_cycles += 1
        self.tree_nodes += nodes
        self.accepted_depth += accepted_depth
        self.emitted_tokens += emitted_tokens
        self.working_cache_tokens = working_cache_tokens

    def snapshot(self) -> dict[str, float | int]:
        tree_cycles = max(self.tree_cycles, 1)
        retrieval_count = max(self.retrieval_count, 1)
        return {
            "retrieval_count": self.retrieval_count,
            "retrieval_ms": (
                self.retrieval_seconds / retrieval_count * 1.0e3
            ),
            "tree_cycles": self.tree_cycles,
            "mean_tree_nodes": self.tree_nodes / tree_cycles,
            "mean_accepted_depth": self.accepted_depth / tree_cycles,
            "mean_emitted_tokens": self.emitted_tokens / tree_cycles,
            "working_cache_tokens": self.working_cache_tokens,
        }


def find_target_rotary_cache(model: torch.nn.Module) -> torch.Tensor:
    """Find the first full-attention RoPE table in a Qwen3.5 wrapper stack."""
    pending = [model]
    visited: set[int] = set()
    while pending:
        module = pending.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        for layer in getattr(module, "layers", ()):
            rotary = getattr(
                getattr(layer, "self_attn", None), "rotary_emb", None
            )
            cache = getattr(rotary, "cos_sin_cache", None)
            if cache is not None:
                return cache
        for name in ("language_model", "model", "module"):
            child = getattr(module, name, None)
            if child is not None:
                pending.append(child)
    raise ValueError("DiffSpec target has no full-attention rotary cache")


def select_tree_level(
    logits: torch.Tensor,
    parent_indices: torch.Tensor,
    parent_log_probs: torch.Tensor,
    *,
    node_budget: int,
    cumulative_threshold: float,
) -> DiffSpecTreeLevel:
    """Select a globally probability-ranked beam level.

    ``cumulative_threshold`` is validated here because it is part of the
    public DiffSpec tree policy.  It is deliberately not applied separately
    to every parent: doing so can rank a very unlikely parent's first child
    ahead of a much more likely parent's second child.  The global path
    probabilities are the sufficient statistic for the optimal-tree beam;
    the threshold is consumed by the adaptive depth policy.
    """
    if logits.ndim != 2:
        raise ValueError("tree logits must be two-dimensional")
    num_parents, vocab_size = logits.shape
    if parent_indices.shape != (num_parents,):
        raise ValueError("one tree parent index is required per logits row")
    if parent_log_probs.shape != (num_parents,):
        raise ValueError("one path probability is required per logits row")
    if node_budget <= 0:
        raise ValueError("tree node budget must be positive")
    if not 0.0 < cumulative_threshold <= 1.0:
        raise ValueError("tree threshold must be in (0, 1]")

    max_children = min(node_budget, vocab_size)
    log_probs = logits.float().log_softmax(dim=-1)
    child_log_probs, child_tokens = log_probs.topk(
        max_children, dim=-1
    )
    candidate_log_probs = parent_log_probs[:, None] + child_log_probs
    keep = min(node_budget, candidate_log_probs.numel())
    _, flat_indices = candidate_log_probs.flatten().topk(keep)
    selected_log_probs = candidate_log_probs.flatten().index_select(
        0, flat_indices
    )
    parent_rows = torch.div(
        flat_indices, max_children, rounding_mode="floor"
    )
    selected_tokens = child_tokens.flatten().index_select(0, flat_indices)
    selected_parents = parent_indices.index_select(0, parent_rows)
    return DiffSpecTreeLevel(
        token_ids=selected_tokens,
        parent_indices=selected_parents,
        parent_rows=parent_rows,
        path_log_probs=selected_log_probs,
    )


def validate_diffspec_runtime(vllm_config: Any) -> None:
    """Validate the Sage Mate TP4 graph execution contract.

    This is a source-admission check, not runtime qualification. In
    particular, a matching checkpoint pair still needs multi-rank graph and
    output evidence before a release can be marked compatible.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_diffspec():
        return
    if spec_config.method != "eagle3":
        raise ValueError("Ascend DiffSpec requires method='eagle3'")
    draft_config = spec_config.draft_model_config
    num_draft_layers = getattr(draft_config.hf_config, "num_hidden_layers", 0)
    if num_draft_layers != 1:
        raise ValueError("Ascend DiffSpec currently requires one draft layer")
    if vllm_config.parallel_config.tensor_parallel_size != 4:
        raise ValueError("Sage Mate DiffSpec requires tensor parallel size 4")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Ascend DiffSpec currently requires pipeline size 1")
    if vllm_config.scheduler_config.max_num_seqs < 2:
        raise ValueError("Sage Mate DiffSpec requires concurrent request capacity")
    if vllm_config.scheduler_config.async_scheduling:
        raise ValueError("Ascend DiffSpec currently requires async scheduling off")
    if spec_config.disable_padded_drafter_batch:
        raise ValueError("Ascend DiffSpec requires the padded Eagle3 drafter")
    if vllm_config.model_config.enforce_eager or spec_config.enforce_eager:
        raise ValueError(
            "Sage Mate DiffSpec requires graph mode for both target and draft models"
        )
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Ascend DiffSpec currently requires prefix caching off")
    # Qwen3.5/Qwen3.8 advertises M-RoPE even for text-only requests.  The
    # target runner owns those positions; Eagle3 consumes target hidden states
    # and keeps its own rotary table for the draft layer.  Rejecting the model
    # here therefore excluded the exact dense Qwen3.8 lane this integration is
    # designed for.  Multimodal request coverage remains a separate capability
    # (the Sage Mate qualification lane is text-only).
    if vllm_config.model_config.use_mla:
        raise ValueError("Ascend DiffSpec does not support MLA models")
    if vllm_config.model_config.quantization is not None:
        raise ValueError("Ascend DiffSpec does not yet support quantized models")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Ascend DiffSpec currently requires target BF16")
    if draft_config.dtype != torch.bfloat16:
        raise ValueError("Ascend DiffSpec currently requires draft BF16")

    target_architectures = set(
        getattr(vllm_config.model_config, "architectures", ()) or ()
    )
    if "Qwen3_5ForConditionalGeneration" not in target_architectures:
        raise ValueError("Sage Mate DiffSpec target must be Qwen3.8/Qwen3.5 dense")
    draft_architectures = set(getattr(draft_config, "architectures", ()) or ())
    supported_drafts = {
        "Eagle3LlamaForCausalLM",
        "LlamaForCausalLMEagle3",
    }
    if not draft_architectures & supported_drafts:
        raise ValueError(
            "DiffSpec requires a one-layer Eagle3 draft checkpoint exposing "
            "the pre-RoPE KV hook"
        )
    target_vocab = vllm_config.model_config.get_vocab_size()
    draft_vocab = draft_config.get_vocab_size()
    if target_vocab != draft_vocab:
        raise ValueError(
            "Eagle3 draft vocabulary does not match the target: "
            f"target={target_vocab}, draft={draft_vocab}"
        )


def chunk_attention_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    chunk_size: int,
    softmax_lse: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Return mean normalized attention mass for each logical token chunk.

    Args:
        query: Query tensor shaped ``[num_query_heads, head_dim]``.
        keys: Key tensor shaped ``[num_tokens, num_kv_heads, head_dim]``.
        chunk_size: Number of logical tokens in a retrieval chunk.
        softmax_lse: Optional per-query-head log-sum-exp from target attention.
        scale: Optional query/key scale. Defaults to ``head_dim**-0.5``.

    Returns:
        FP32 scores shaped ``[ceil(num_tokens / chunk_size)]``.
    """
    if query.ndim != 2 or keys.ndim != 3:
        raise ValueError("query must be 2-D and keys must be 3-D")
    num_query_heads, head_dim = query.shape
    num_tokens, num_kv_heads, key_head_dim = keys.shape
    if head_dim != key_head_dim:
        raise ValueError("query and key head dimensions must match")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.float32, device=keys.device)

    queries_per_kv = num_query_heads // num_kv_heads
    grouped_query = query.float().reshape(
        num_kv_heads, queries_per_kv, head_dim
    )
    logits = torch.einsum("hgd,thd->hgt", grouped_query, keys.float())
    logits = logits.reshape(num_query_heads, num_tokens)
    logits.mul_(scale if scale is not None else 1.0 / sqrt(head_dim))

    if softmax_lse is None:
        softmax_lse = torch.logsumexp(logits, dim=-1)
    else:
        softmax_lse = softmax_lse.float().reshape(-1)
        if softmax_lse.numel() != num_query_heads:
            raise ValueError("softmax_lse must contain one value per query head")
    attention_mass = torch.exp(logits - softmax_lse[:, None])

    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    padded_tokens = num_chunks * chunk_size
    if padded_tokens != num_tokens:
        attention_mass = torch.nn.functional.pad(
            attention_mass, (0, padded_tokens - num_tokens)
        )
    chunk_mass = attention_mass.reshape(
        num_query_heads, num_chunks, chunk_size
    ).sum(dim=-1)
    valid_counts = torch.full(
        (num_chunks,),
        chunk_size,
        dtype=torch.float32,
        device=keys.device,
    )
    valid_counts[-1] = num_tokens - (num_chunks - 1) * chunk_size
    return chunk_mass.mean(dim=0).div_(valid_counts)


def select_context_chunks(
    chunk_scores: torch.Tensor,
    seq_len: int,
    chunk_size: int,
    token_budget: int,
) -> DiffSpecSelection:
    """Select top scoring chunks while always retaining the newest chunk."""
    if chunk_scores.ndim != 1:
        raise ValueError("chunk_scores must be one-dimensional")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if token_budget < chunk_size or token_budget % chunk_size != 0:
        raise ValueError("token_budget must be a positive chunk-size multiple")

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    if num_chunks != chunk_scores.numel():
        raise ValueError("chunk score count does not match sequence length")
    if num_chunks == 0:
        empty = torch.empty(0, dtype=torch.long, device=chunk_scores.device)
        return DiffSpecSelection(empty, empty, chunk_scores)

    max_chunks = token_budget // chunk_size
    if num_chunks <= max_chunks:
        selected_chunks = torch.arange(
            num_chunks, dtype=torch.long, device=chunk_scores.device
        )
    else:
        newest_chunk = num_chunks - 1
        rank_scores = chunk_scores.clone()
        rank_scores[newest_chunk] = float("-inf")
        ranked = rank_scores.topk(max_chunks - 1).indices
        selected_chunks = torch.cat(
            (
                ranked,
                ranked.new_tensor([newest_chunk]),
            )
        ).sort().values

    offsets = torch.arange(
        chunk_size, dtype=torch.long, device=chunk_scores.device
    )
    if seq_len % chunk_size:
        full_chunk_indices = (
            selected_chunks[:-1, None] * chunk_size + offsets[None, :]
        ).reshape(-1)
        tail_indices = torch.arange(
            (num_chunks - 1) * chunk_size,
            seq_len,
            dtype=torch.long,
            device=chunk_scores.device,
        )
        token_indices = torch.cat((full_chunk_indices, tail_indices))
    else:
        token_indices = (
            selected_chunks[:, None] * chunk_size + offsets[None, :]
        ).reshape(-1)
    return DiffSpecSelection(
        chunk_indices=selected_chunks,
        token_indices=token_indices,
        chunk_scores=chunk_scores.index_select(0, selected_chunks),
    )


def build_tree_ancestry_mask(parent_indices: torch.Tensor) -> torch.Tensor:
    """Build a mask where each tree node attends to itself and its ancestors."""
    if parent_indices.ndim != 1:
        raise ValueError("parent_indices must be one-dimensional")
    num_nodes = parent_indices.numel()
    if parent_indices.device.type != "cpu":
        parents = parent_indices.long()
        mask = torch.eye(
            num_nodes, dtype=torch.bool, device=parent_indices.device
        )
        ancestors = parents
        for _ in range(num_nodes):
            valid = ancestors >= 0
            columns = ancestors.clamp_min(0)
            ancestor_updates = torch.zeros_like(mask)
            ancestor_updates.scatter_(
                1, columns[:, None], valid[:, None]
            )
            mask |= ancestor_updates
            parent_values = parents.index_select(0, columns)
            ancestors = torch.where(valid, parent_values, ancestors)
        return mask
    parents = parent_indices.to(device="cpu", dtype=torch.long).tolist()
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for node_index, parent_index in enumerate(parents):
        if parent_index >= node_index or parent_index < -1:
            raise ValueError("tree parents must precede their children")
        mask[node_index, node_index] = True
        while parent_index >= 0:
            mask[node_index, parent_index] = True
            parent_index = parents[parent_index]
    return mask.to(parent_indices.device)


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    tree_output: torch.Tensor,
    tree_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine prefix and tree attention using their log-sum-exp states."""
    prefix_lse = prefix_lse.float()
    tree_lse = tree_lse.float()
    merged_lse = torch.logaddexp(prefix_lse, tree_lse)
    prefix_weight = torch.exp(prefix_lse - merged_lse)
    tree_weight = torch.exp(tree_lse - merged_lse)
    while prefix_weight.ndim < prefix_output.ndim:
        prefix_weight = prefix_weight.unsqueeze(-1)
        tree_weight = tree_weight.unsqueeze(-1)
    output = prefix_output.float() * prefix_weight
    output.add_(tree_output.float() * tree_weight)
    return output.to(prefix_output.dtype), merged_lse


def verify_greedy_tree(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_argmax_by_parent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow target argmax values through a speculative token tree.

    ``target_argmax_by_parent[0]`` is the target token predicted from the
    committed prefix. Entry ``node + 1`` is the target token predicted after
    that node. Returned token IDs include the recovered target token.
    """
    if draft_token_ids.ndim != 1 or parent_indices.shape != draft_token_ids.shape:
        raise ValueError("draft tokens and parent indices must be matching vectors")
    if target_argmax_by_parent.numel() != draft_token_ids.numel() + 1:
        raise ValueError("target argmax tensor must include prefix and every node")

    device = draft_token_ids.device
    accepted_nodes: list[int] = []
    emitted_tokens: list[int] = []
    current_parent = -1
    # One small D2H copy is substantially cheaper than three independently
    # synchronizing copies on Ascend.  Verification itself remains on the CPU
    # because following a variable-length branch is inherently serial and the
    # result is needed by scheduler bookkeeping immediately.
    packed = torch.cat(
        (
            parent_indices.to(torch.long),
            draft_token_ids.to(torch.long),
            target_argmax_by_parent.to(torch.long),
        )
    ).to("cpu").tolist()
    num_nodes = draft_token_ids.numel()
    parent_list = packed[:num_nodes]
    draft_list = packed[num_nodes : 2 * num_nodes]
    target_list = packed[2 * num_nodes :]
    while True:
        target_token = target_list[current_parent + 1]
        matching_child = next(
            (
                node_index
                for node_index, (parent, token) in enumerate(
                    zip(parent_list, draft_list)
                )
                if parent == current_parent and token == target_token
            ),
            None,
        )
        emitted_tokens.append(target_token)
        if matching_child is None:
            break
        accepted_nodes.append(matching_child)
        current_parent = matching_child

    return (
        torch.tensor(accepted_nodes, dtype=torch.long, device=device),
        torch.tensor(emitted_tokens, dtype=draft_token_ids.dtype, device=device),
    )


class DiffSpecAdaptiveController:
    """Select the measured tree profile and adapt retrieval cadence."""

    _PROFILES = (
        DiffSpecTreeProfile(16, 5),
        DiffSpecTreeProfile(32, 7),
        DiffSpecTreeProfile(50, 10),
    )

    def __init__(self, base_retrieval_interval: int, adaptive: bool) -> None:
        self.base_retrieval_interval = base_retrieval_interval
        self.retrieval_interval = base_retrieval_interval
        self.adaptive = adaptive
        self.profile = self._PROFILES[-1]
        self._throughput_ewma: dict[DiffSpecTreeProfile, float] = {}
        self._last_acceptance: float | None = None

    def observe_cycle(
        self,
        profile: DiffSpecTreeProfile,
        emitted_tokens: int,
        latency_seconds: float,
    ) -> None:
        if latency_seconds <= 0:
            raise ValueError("latency_seconds must be positive")
        throughput = emitted_tokens / latency_seconds
        previous = self._throughput_ewma.get(profile, throughput)
        self._throughput_ewma[profile] = 0.8 * previous + 0.2 * throughput
        if not self.adaptive:
            return
        unmeasured = [
            candidate
            for candidate in self._PROFILES
            if candidate not in self._throughput_ewma
        ]
        if unmeasured:
            self.profile = unmeasured[0]
            return
        best_profile = max(
            self._throughput_ewma, key=self._throughput_ewma.__getitem__
        )
        current_throughput = self._throughput_ewma.get(self.profile, 0.0)
        if self._throughput_ewma[best_profile] > current_throughput * 1.05:
            self.profile = best_profile

    def observe_retrieval(
        self,
        previous_chunks: set[int],
        selected_chunks: set[int],
        acceptance_length: float,
    ) -> None:
        union = previous_chunks | selected_chunks
        stability = (
            len(previous_chunks & selected_chunks) / len(union) if union else 1.0
        )
        acceptance_dropped = (
            self._last_acceptance is not None
            and acceptance_length < self._last_acceptance * 0.9
        )
        if self.adaptive and stability >= 0.9 and not acceptance_dropped:
            self.retrieval_interval = min(self.retrieval_interval * 2, 32)
        else:
            self.retrieval_interval = self.base_retrieval_interval
        self._last_acceptance = acceptance_length


class DiffSpecDraftCache:
    """Canonical raw KV and compact attention state for one-layer Eagle3."""

    def __init__(
        self,
        settings: DiffSpecSettings,
        *,
        max_num_reqs: int,
        max_model_len: int,
        num_kv_heads: int,
        head_dim: int,
        target_num_layers: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary_embedding: Any,
        target_cos_sin_cache: torch.Tensor | None = None,
    ) -> None:
        self.settings = settings
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.target_last_layer = target_num_layers - 1
        self.dtype = dtype
        self.device = device
        self.rotary_embedding = rotary_embedding
        self.target_cos_sin_cache = target_cos_sin_cache
        self.history_budget = (
            (
                settings.token_budget
                - settings.max_tree_depth
                - 1
            )
            // settings.chunk_size
            * settings.chunk_size
        )
        if self.history_budget <= 0:
            raise ValueError("DiffSpec token budget is too small for tree tokens")

        cache_shape = (
            max_num_reqs,
            max_model_len,
            num_kv_heads,
            head_dim,
        )
        self.raw_key = torch.empty(cache_shape, dtype=dtype, device=device)
        self.raw_value = torch.empty_like(self.raw_key)
        working_shape = (
            max_num_reqs,
            num_kv_heads,
            settings.token_budget,
            head_dim,
        )
        self.working_key = torch.empty(working_shape, dtype=dtype, device=device)
        self.working_value = torch.empty_like(self.working_key)
        self.max_draft_tree_nodes = (
            settings.max_tree_nodes * settings.max_tree_depth
        )
        tree_shape = (
            max_num_reqs,
            self.max_draft_tree_nodes,
            num_kv_heads,
            head_dim,
        )
        self.tree_raw_key = torch.empty(tree_shape, dtype=dtype, device=device)
        self.tree_raw_value = torch.empty_like(self.tree_raw_key)
        self.tree_key = torch.empty_like(self.tree_raw_key)
        self.tree_value = torch.empty_like(self.tree_raw_key)

        self.controller = DiffSpecAdaptiveController(
            settings.retrieval_interval,
            settings.adaptive_profile,
        )
        self.metrics = DiffSpecRuntimeMetrics()
        self._selected_token_indices: dict[int, torch.Tensor] = {}
        self._working_lens = [0] * max_num_reqs
        self._cycle_open = False
        self._draft_step = 0
        self._pending_request_indices: list[int] | None = None
        self._pending_local_positions: list[int] | None = None
        self._compact_prefill = False
        self._target_query: torch.Tensor | None = None
        self._target_lse: torch.Tensor | None = None
        self._target_key_cache: torch.Tensor | None = None
        self._target_block_tables: torch.Tensor | None = None
        self._target_query_ends: list[int] = []
        self._target_seq_lens: list[int] = []
        self._target_scale = 0.0
        self._retrieval_cycle = 0
        self._request_slots: dict[str, int] = {}
        self._active_slots: list[int] = []
        self._pending_tree_node_indices: torch.Tensor | None = None
        self._pending_tree_parent_indices: torch.Tensor | None = None
        self._pending_tree_depths: torch.Tensor | None = None
        self._tree_parents = torch.full(
            (self.max_draft_tree_nodes,),
            -2,
            dtype=torch.long,
            device=device,
        )
        self._tree_base_len = 0
        self._tree_num_nodes = 0
        self.proposed_token_ids: torch.Tensor | None = None
        self.proposed_parent_indices: torch.Tensor | None = None
        self.proposed_depths: torch.Tensor | None = None
        self.proposed_depths_cpu: list[int] = []
        self.target_tree_active = False
        self._target_tree_layer_kv: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._target_tree_prefix_len = 0
        self._accepted_tree_nodes: torch.Tensor | None = None
        self._accepted_query_offset: torch.Tensor | None = None
        self._draft_logical_seq_len = 0
        self._target_tree_allowed_mask: torch.Tensor | None = None
        self._proposed_profile = self.controller.profile
        self._cycle_started_at: float | None = None
        self._last_accepted_depth = 0

    def begin_cycle(self, request_ids: list[str]) -> None:
        self.set_active_requests(request_ids)
        self._cycle_started_at = time.perf_counter()
        self._cycle_open = True
        self._draft_step = 0
        self._pending_request_indices = None
        self._pending_local_positions = None
        self._pending_tree_node_indices = None
        self._pending_tree_parent_indices = None
        self._pending_tree_depths = None
        self._tree_base_len = 0
        self._tree_num_nodes = 0
        self._tree_parents.fill_(-2)

    def end_cycle(self) -> None:
        self._cycle_open = False
        self._pending_request_indices = None
        self._pending_local_positions = None
        self._compact_prefill = False
        self._pending_tree_node_indices = None
        self._pending_tree_parent_indices = None
        self._pending_tree_depths = None

    def set_active_requests(self, request_ids: list[str]) -> None:
        """Keep canonical rows stable across batch reorder and cancellation."""
        active_request_ids = set(request_ids)
        for request_id in list(self._request_slots):
            if request_id not in active_request_ids:
                slot = self._request_slots.pop(request_id)
                self._selected_token_indices.pop(slot, None)
                self._working_lens[slot] = 0

        used_slots = set(self._request_slots.values())
        free_slots = iter(sorted(set(range(self.max_num_reqs)) - used_slots))
        for request_id in request_ids:
            if request_id not in self._request_slots:
                try:
                    self._request_slots[request_id] = next(free_slots)
                except StopIteration as error:
                    raise RuntimeError(
                        "DiffSpec canonical request slots are exhausted"
                    ) from error
        self._active_slots = [
            self._request_slots[request_id] for request_id in request_ids
        ]

    def begin_target_forward(self, request_ids: list[str]) -> None:
        """Discard a stale target snapshot before a new verifier pass."""
        self.set_active_requests(request_ids)
        self._target_query = None
        self._target_lse = None
        self._target_key_cache = None
        self._target_block_tables = None
        self._target_query_ends = []
        self._target_seq_lens = []
        self._target_tree_layer_kv.clear()
        self._accepted_tree_nodes = None
        self._accepted_query_offset = None
        self._draft_logical_seq_len = 0

    def prepare_target_batch(
        self,
        request_ids: list[str],
        scheduled_spec_token_ids: dict[str, list[int]],
    ) -> None:
        """Activate tree verification when the complete proposal was scheduled."""
        self.set_active_requests(request_ids)
        self.target_tree_active = False
        self._target_tree_allowed_mask = None
        if (
            len(request_ids) != 1
            or self.proposed_token_ids is None
            or self.proposed_parent_indices is None
        ):
            return
        scheduled = scheduled_spec_token_ids.get(request_ids[0])
        if scheduled is None:
            return
        self.target_tree_active = len(scheduled) == self.proposed_token_ids.numel()
        if self.target_tree_active:
            self._target_tree_allowed_mask = self._build_target_tree_mask()

    def rewrite_target_positions(
        self,
        positions,
        cumulative_query_lens,
    ) -> None:
        if not self.target_tree_active:
            return
        query_end = int(cumulative_query_lens[0])
        query_start = 0
        if query_end - query_start != len(self.proposed_depths_cpu) + 1:
            self.target_tree_active = False
            return
        root_position = positions[query_start]
        for node_index, depth in enumerate(self.proposed_depths_cpu):
            positions[query_start + node_index + 1] = root_position + depth

    def accepted_target_token_indices(self) -> torch.Tensor:
        if self._accepted_query_offset is None:
            raise RuntimeError("DiffSpec tree has not been verified")
        return self._accepted_query_offset

    def is_target_retrieval_layer(self, layer_index: int) -> bool:
        return layer_index == self.target_last_layer

    def capture_target_attention(
        self,
        query: torch.Tensor,
        softmax_lse: torch.Tensor | None,
        key_cache: torch.Tensor | None,
        metadata: Any,
        scale: float,
    ) -> None:
        """Save the last target-layer FIA state without materializing weights."""
        if softmax_lse is None or key_cache is None:
            return
        query_ends = list(getattr(metadata, "actual_seq_lengths_q", ()))
        seq_lens = list(getattr(metadata, "seq_lens_list", ()))
        block_tables = getattr(metadata, "block_tables", None)
        if not query_ends or not seq_lens or block_tables is None:
            return
        if not self.target_tree_active:
            query_indices = torch.tensor(
                [end - 1 for end in query_ends],
                dtype=torch.long,
                device=query.device,
            )
            softmax_lse = self._reshape_target_lse(
                query, softmax_lse
            ).index_select(0, query_indices)
            query = query.index_select(0, query_indices)
            query_ends = list(range(1, len(query_ends) + 1))
        self._target_query = query.detach()
        self._target_lse = softmax_lse.detach()
        self._target_key_cache = key_cache
        self._target_block_tables = block_tables[: len(seq_lens)]
        self._target_query_ends = query_ends
        self._target_seq_lens = seq_lens
        self._target_scale = scale

    @staticmethod
    def _reshape_target_lse(
        query: torch.Tensor, softmax_lse: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, num_heads, _ = query.shape
        lse = softmax_lse.squeeze()
        if lse.shape == (num_tokens, num_heads):
            return lse
        if lse.shape == (num_heads, num_tokens):
            return lse.transpose(0, 1)
        if lse.numel() != num_tokens * num_heads:
            raise ValueError(
                "target softmax LSE does not match target query shape"
            )
        return lse.reshape(num_tokens, num_heads)

    def _build_target_tree_mask(self) -> torch.Tensor:
        if self.proposed_parent_indices is None:
            raise RuntimeError("DiffSpec target tree parents are unavailable")
        root_parent = self.proposed_parent_indices.new_tensor([-1])
        node_parents = torch.where(
            self.proposed_parent_indices < 0,
            torch.zeros_like(self.proposed_parent_indices),
            self.proposed_parent_indices + 1,
        )
        return build_tree_ancestry_mask(
            torch.cat((root_parent, node_parents))
        )

    def _target_tree_mask(self) -> torch.Tensor:
        if self._target_tree_allowed_mask is None:
            self._target_tree_allowed_mask = self._build_target_tree_mask()
        return self._target_tree_allowed_mask

    @staticmethod
    def _gather_paged_prefix(
        cache: torch.Tensor,
        block_table: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        block_size = cache.shape[1]
        positions = torch.arange(
            prefix_len, dtype=torch.long, device=cache.device
        )
        physical_blocks = block_table[positions // block_size].long()
        return cache[physical_blocks, positions % block_size]

    def _forward_target_tree_attention_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        prefix_len: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_key = self._gather_paged_prefix(
            key_cache, block_table, prefix_len
        )
        prefix_value = self._gather_paged_prefix(
            value_cache, block_table, prefix_len
        )
        query_heads_per_kv = query.shape[1] // self.num_kv_heads
        prefix_key = prefix_key.repeat_interleave(
            query_heads_per_kv, dim=1
        )
        prefix_value = prefix_value.repeat_interleave(
            query_heads_per_kv, dim=1
        )
        tree_key = key.repeat_interleave(query_heads_per_kv, dim=1)
        tree_value = value.repeat_interleave(query_heads_per_kv, dim=1)
        prefix_logits = torch.einsum(
            "qhd,khd->qhk", query.float(), prefix_key.float()
        )
        tree_logits = torch.einsum(
            "qhd,khd->qhk", query.float(), tree_key.float()
        )
        allowed = self._target_tree_mask()
        tree_logits.masked_fill_(~allowed[:, None, :], float("-inf"))
        logits = torch.cat((prefix_logits, tree_logits), dim=-1).mul_(scale)
        values = torch.cat((prefix_value, tree_value), dim=0)
        probabilities = logits.softmax(dim=-1)
        output = torch.einsum(
            "qhk,khd->qhd", probabilities, values.float()
        ).to(query.dtype)
        return output, torch.logsumexp(logits, dim=-1)

    def _forward_target_tree_attention_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        prefix_len: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        num_queries, num_query_heads, _ = query.shape
        num_blocks, block_size, _, _ = key_cache.shape
        prefix_output, prefix_lse = (
            torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key_cache.view(num_blocks, block_size, -1),
                value=value_cache.view(num_blocks, block_size, -1),
                block_table=block_table[None],
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=[num_queries],
                actual_seq_lengths_kv=[prefix_len],
                num_key_value_heads=self.num_kv_heads,
                num_heads=num_query_heads,
                scale=scale,
                sparse_mode=0,
                softmax_lse_flag=True,
            )
        )
        allowed = self._target_tree_mask()
        tree_output, tree_lse = torch_npu.npu_fused_infer_attention_score(
            query=query.transpose(0, 1)[None],
            key=key.transpose(0, 1)[None],
            value=value.transpose(0, 1)[None],
            atten_mask=~allowed,
            actual_seq_lengths=[num_queries],
            actual_seq_lengths_kv=[num_queries],
            num_heads=num_query_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        prefix_output = prefix_output.view_as(query)
        prefix_lse = prefix_lse[:, :, 0]
        tree_output = tree_output[0].transpose(0, 1)
        tree_lse = tree_lse[0, :, :, 0].transpose(0, 1)
        return merge_attention_states(
            prefix_output,
            prefix_lse,
            tree_output,
            tree_lse,
        )

    def forward_target_tree_attention(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: Any,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Verify all tree nodes against one shared paged target prefix."""
        if not self.target_tree_active:
            raise RuntimeError("DiffSpec target tree is not active")
        expected_queries = len(self.proposed_depths_cpu) + 1
        if query.shape[0] != expected_queries:
            raise RuntimeError("DiffSpec target tree query count mismatch")
        prefix_len = metadata.seq_lens_list[0] - expected_queries
        self._target_tree_prefix_len = prefix_len
        block_table = metadata.block_tables[0]
        if query.device.type == "npu":
            attention_output, merged_lse = (
                self._forward_target_tree_attention_npu(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    block_table,
                    prefix_len,
                    scale,
                )
            )
        else:
            attention_output, merged_lse = (
                self._forward_target_tree_attention_cpu(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    block_table,
                    prefix_len,
                    scale,
                )
            )
        self._target_tree_layer_kv[layer_index] = (
            key.detach(),
            value.detach(),
            key_cache,
            value_cache,
        )
        if self.is_target_retrieval_layer(layer_index):
            self.capture_target_attention(
                query,
                merged_lse,
                key_cache,
                metadata,
                scale,
            )
        output[: query.shape[0]] = attention_output
        return output

    def commit_tree_path(self, accepted_nodes: torch.Tensor) -> None:
        """Commit only the accepted root/path into target and draft caches."""
        if not self.target_tree_active:
            return
        query_indices = torch.cat(
            (
                accepted_nodes.new_tensor([0]),
                accepted_nodes + 1,
            )
        )
        positions = torch.arange(
            self._target_tree_prefix_len,
            self._target_tree_prefix_len + query_indices.numel(),
            dtype=torch.long,
            device=accepted_nodes.device,
        )
        if self._target_block_tables is None:
            raise RuntimeError("DiffSpec target block table is unavailable")
        for key, value, key_cache, value_cache in (
            self._target_tree_layer_kv.values()
        ):
            block_size = key_cache.shape[1]
            block_table = self._target_block_tables[0]
            physical_blocks = block_table[positions // block_size].long()
            block_offsets = positions % block_size
            key_cache[physical_blocks, block_offsets] = key.index_select(
                0, query_indices
            )
            value_cache[physical_blocks, block_offsets] = value.index_select(
                0, query_indices
            )

        slot = self._active_slots[0]
        if accepted_nodes.numel():
            draft_positions = positions[1:]
            self.raw_key[slot, draft_positions] = self.tree_raw_key[
                slot
            ].index_select(0, accepted_nodes)
            self.raw_value[slot, draft_positions] = self.tree_raw_value[
                slot
            ].index_select(0, accepted_nodes)
            self._accepted_query_offset = accepted_nodes[-1:] + 1
        else:
            self._accepted_query_offset = accepted_nodes.new_zeros(1)
        self._accepted_tree_nodes = accepted_nodes
        self._target_seq_lens = [
            self._target_tree_prefix_len + query_indices.numel()
        ]
        self._draft_logical_seq_len = (
            self._target_tree_prefix_len + query_indices.numel() + 1
        )
        accepted_depth = accepted_nodes.numel()
        self._last_accepted_depth = accepted_depth
        emitted_tokens = accepted_depth + 1
        working_cache_tokens = max(self._working_lens, default=0)
        self.metrics.observe_tree(
            nodes=self.proposed_token_ids.numel()
            if self.proposed_token_ids is not None
            else 0,
            accepted_depth=accepted_depth,
            emitted_tokens=emitted_tokens,
            working_cache_tokens=working_cache_tokens,
        )
        if self._cycle_started_at is not None:
            self.controller.observe_cycle(
                self._proposed_profile,
                emitted_tokens,
                time.perf_counter() - self._cycle_started_at,
            )
        if self.metrics.tree_cycles % 32 == 0:
            logger.info("DiffSpec runtime metrics: %s", self.metrics.snapshot())

    def _target_lse_by_token(self) -> torch.Tensor:
        if self._target_query is None or self._target_lse is None:
            raise RuntimeError("target attention state is unavailable")
        return self._reshape_target_lse(
            self._target_query, self._target_lse
        )

    def accept_target_queries(
        self,
        valid_sampled_token_count: torch.Tensor,
        *,
        is_prefill: bool,
    ) -> None:
        """Retrieve context using the last committed target query per request."""
        if (
            self._target_query is None
            or self._target_key_cache is None
            or self._target_block_tables is None
        ):
            return
        if is_prefill:
            return
        self._retrieval_cycle += 1
        interval = self.controller.retrieval_interval
        if self._retrieval_cycle % interval:
            return

        query_ends = self._target_query_ends
        query_starts = [0, *query_ends[:-1]]
        query_lens = [
            end - start for start, end in zip(query_starts, query_ends)
        ]
        batch_size = len(query_lens)
        counts = valid_sampled_token_count[:batch_size].reshape(-1).long()
        starts = torch.tensor(
            query_starts,
            dtype=torch.long,
            device=self._target_query.device,
        )
        lengths = torch.tensor(
            query_lens,
            dtype=torch.long,
            device=self._target_query.device,
        )
        if is_prefill:
            query_offsets = lengths - 1
        elif self.target_tree_active and self._accepted_tree_nodes is not None:
            if self._accepted_query_offset is None:
                raise RuntimeError("DiffSpec accepted query is unavailable")
            query_offsets = self._accepted_query_offset.to(
                device=self._target_query.device
            )
        else:
            query_offsets = counts.sub(1).clamp_min(0)
            query_offsets = torch.minimum(query_offsets, lengths - 1)
        query_indices = starts + query_offsets
        selected_query = self._target_query.index_select(0, query_indices)
        selected_lse = self._target_lse_by_token().index_select(
            0, query_indices
        )

        seq_lens = torch.tensor(
            self._target_seq_lens,
            dtype=torch.int32,
            device=self._target_query.device,
        )
        from diffspec.kernels import (
            paged_chunk_attention_scores,
        )

        retrieval_started_at = time.perf_counter()
        scores = paged_chunk_attention_scores(
            selected_query,
            self._target_key_cache,
            selected_lse,
            self._target_block_tables,
            seq_lens,
            chunk_size=self.settings.chunk_size,
            max_seq_len=max(self._target_seq_lens),
            scale=self._target_scale,
        )
        for request_index, seq_len in enumerate(self._target_seq_lens):
            num_chunks = (
                seq_len + self.settings.chunk_size - 1
            ) // self.settings.chunk_size
            slot = self._active_slots[request_index]
            previous_indices = self._selected_token_indices.get(slot)
            selection = self.set_chunk_scores(
                slot,
                scores[request_index, :num_chunks],
                seq_len,
            )
            if request_index == 0:
                previous_chunks = (
                    set()
                    if previous_indices is None
                    else set(
                        torch.div(
                            previous_indices[:: self.settings.chunk_size],
                            self.settings.chunk_size,
                            rounding_mode="floor",
                        )
                        .to("cpu")
                        .tolist()
                    )
                )
                selected_chunks = set(
                    selection.chunk_indices.to("cpu").tolist()
                )
                self.controller.observe_retrieval(
                    previous_chunks,
                    selected_chunks,
                    float(self._last_accepted_depth),
                )
        self.metrics.observe_retrieval(
            time.perf_counter() - retrieval_started_at
        )

    def populate_prefill(
        self,
        absolute_positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Any,
    ) -> None:
        """Populate canonical raw KV after cache-only draft projection."""
        request_indices = self._canonical_slots(
            self._request_indices(metadata, key.shape[0])
        )
        self._write_raw(request_indices, absolute_positions, key, value)

    def prepare_compact_prefill(self, metadata: Any, num_tokens: int) -> None:
        """Prepare draft attention before entering the compiled Eagle graph.

        Python KV-sink callbacks inside ``torch.compile`` are trace-time side
        effects and are not guaranteed to run during graph replay.  Populate
        the pending request/position state here so the runtime attention hook
        can consume it deterministically on every replay.
        """
        batch_rows = list(range(num_tokens))
        seq_lens = list(getattr(metadata, "seq_lens_list", ()))
        if not seq_lens:
            raise RuntimeError("DiffSpec compact prefill requires sequence lengths")
        request_indices = self._canonical_slots(batch_rows)
        local_positions = self._rebuild_working_cache(batch_rows, seq_lens)
        self._pending_request_indices = request_indices
        self._pending_local_positions = local_positions
        self._draft_step += 1

    def prepare_tree_level(
        self,
        node_indices: torch.Tensor,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
    ) -> None:
        """Prepare one breadth-wise Eagle3 tree expansion level."""
        if len(self._active_slots) != 1:
            raise RuntimeError("DiffSpec tree drafting v1 requires batch size 1")
        if not (
            node_indices.shape == parent_indices.shape == depths.shape
            and node_indices.ndim == 1
        ):
            raise ValueError("tree level metadata must be matching vectors")
        if node_indices.numel() == 0:
            raise ValueError("tree level must contain at least one node")
        if self._tree_base_len == 0:
            self._tree_base_len = self._working_lens[self._active_slots[0]]
        self._tree_parents.index_copy_(0, node_indices, parent_indices)
        self._pending_tree_node_indices = node_indices
        self._pending_tree_parent_indices = parent_indices
        self._pending_tree_depths = depths
        self._tree_num_nodes += node_indices.numel()

    def pending_tree_local_positions(self) -> torch.Tensor:
        """Return compact RoPE positions for the prepared breadth level."""
        if self._pending_tree_depths is None or self._tree_base_len == 0:
            raise RuntimeError("DiffSpec tree level has not been prepared")
        return self._tree_base_len + self._pending_tree_depths - 1

    def set_proposed_tree(
        self,
        token_ids: torch.Tensor,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
        depths_cpu: list[int],
        profile: DiffSpecTreeProfile | None = None,
    ) -> None:
        if not (
            token_ids.shape == parent_indices.shape == depths.shape
            and token_ids.ndim == 1
        ):
            raise ValueError("proposed tree metadata must be matching vectors")
        self.proposed_token_ids = token_ids.detach()
        self.proposed_parent_indices = parent_indices.detach()
        self.proposed_depths = depths.detach()
        self.proposed_depths_cpu = depths_cpu
        self._proposed_profile = profile or self.controller.profile
        self._target_tree_allowed_mask = None

    def finalize_proposed_tree(
        self,
        token_ids: torch.Tensor,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
        path_log_probs: torch.Tensor,
        profile: DiffSpecTreeProfile,
    ) -> None:
        """Prune the draft beam into an ancestry-closed verifier tree.

        Drafting keeps a full ``max_nodes`` beam at every depth, just like the
        SpecExtend optimal-tree construction.  Verification remains bounded by
        ``max_nodes``: candidates are ranked globally by path probability and
        their ancestors are inserted before them.  Keeping the closure here is
        what makes the compact parent indices safe for target tree attention.
        """
        if not (
            token_ids.shape
            == parent_indices.shape
            == depths.shape
            == path_log_probs.shape
            and token_ids.ndim == 1
        ):
            raise ValueError("draft tree candidates must be matching vectors")
        candidate_count = token_ids.numel()
        if candidate_count == 0:
            raise ValueError("draft tree must contain at least one candidate")
        if candidate_count > self.max_draft_tree_nodes:
            raise ValueError("draft tree candidate cache is exhausted")

        target_nodes = min(
            profile.max_nodes,
            self.settings.max_tree_nodes,
            candidate_count,
        )
        order_device = path_log_probs.argsort(descending=True)
        packed = torch.cat(
            (
                order_device,
                parent_indices.to(torch.long),
                depths.to(torch.long),
            )
        ).to("cpu").tolist()
        order = packed[:candidate_count]
        parents_cpu = packed[candidate_count : 2 * candidate_count]
        depths_cpu_all = packed[2 * candidate_count :]
        selected: set[int] = set()
        for candidate in order:
            chain: list[int] = []
            current = candidate
            while current >= 0 and current not in selected:
                chain.append(current)
                current = parents_cpu[current]
            missing = list(reversed(chain))
            if len(selected) + len(missing) <= target_nodes:
                selected.update(missing)
            if len(selected) == target_nodes:
                break
        if len(selected) != target_nodes:
            raise RuntimeError("unable to construct a complete DiffSpec tree")

        selected_cpu = sorted(
            selected,
            key=lambda node: (depths_cpu_all[node], node),
        )
        remap = {
            source_index: compact_index
            for compact_index, source_index in enumerate(selected_cpu)
        }
        compact_parents_cpu = [
            -1 if parents_cpu[node] < 0 else remap[parents_cpu[node]]
            for node in selected_cpu
        ]
        selected_indices = torch.tensor(
            selected_cpu,
            dtype=torch.long,
            device=token_ids.device,
        )
        compact_tokens = token_ids.index_select(0, selected_indices)
        compact_depths = depths.index_select(0, selected_indices)
        compact_parents = torch.tensor(
            compact_parents_cpu,
            dtype=torch.long,
            device=parent_indices.device,
        )

        slot = self._active_slots[0]
        selected_raw_key = self.tree_raw_key[slot].index_select(
            0, selected_indices
        )
        selected_raw_value = self.tree_raw_value[slot].index_select(
            0, selected_indices
        )
        self.tree_raw_key[slot, :target_nodes].copy_(selected_raw_key)
        self.tree_raw_value[slot, :target_nodes].copy_(selected_raw_value)
        self.set_proposed_tree(
            compact_tokens,
            compact_parents,
            compact_depths,
            [depths_cpu_all[node] for node in selected_cpu],
            profile,
        )

    def clear_proposed_tree(self) -> None:
        self.proposed_token_ids = None
        self.proposed_parent_indices = None
        self.proposed_depths = None
        self.proposed_depths_cpu = []
        self._target_tree_allowed_mask = None

    def store_tree_raw_kv(
        self,
        node_indices: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store leaf raw KV when no draft attention pass is required."""
        if len(self._active_slots) != 1:
            raise RuntimeError("DiffSpec tree drafting v1 requires batch size 1")
        slot = self._active_slots[0]
        self.tree_raw_key[slot].index_copy_(
            0,
            node_indices,
            key.reshape(-1, self.num_kv_heads, self.head_dim),
        )
        self.tree_raw_value[slot].index_copy_(
            0,
            node_indices,
            value.reshape(-1, self.num_kv_heads, self.head_dim),
        )

    def set_chunk_scores(
        self,
        request_index: int,
        chunk_scores: torch.Tensor,
        seq_len: int,
    ) -> DiffSpecSelection:
        selection = select_context_chunks(
            chunk_scores,
            seq_len,
            self.settings.chunk_size,
            self.history_budget,
        )
        self._selected_token_indices[request_index] = selection.token_indices
        return selection

    @staticmethod
    def _first_attention_metadata() -> Any | None:
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            return next(iter(metadata.values()), None)
        return metadata

    @staticmethod
    def _request_indices(metadata: Any, num_tokens: int) -> list[int]:
        query_ends = getattr(metadata, "actual_seq_lengths_q", None)
        if query_ends:
            starts = [0, *query_ends]
        else:
            starts = metadata.query_start_loc.to("cpu").tolist()
        request_indices: list[int] = []
        request_index = 0
        for token_index in range(num_tokens):
            while (
                request_index + 1 < len(starts) - 1
                and token_index >= starts[request_index + 1]
            ):
                request_index += 1
            request_indices.append(request_index)
        return request_indices

    def _canonical_slots(self, batch_rows: list[int]) -> list[int]:
        if not self._active_slots:
            raise RuntimeError("DiffSpec request slots have not been initialized")
        return [self._active_slots[batch_row] for batch_row in batch_rows]

    def _write_raw(
        self,
        request_indices: list[int],
        absolute_positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        positions = absolute_positions.flatten().long()
        request_tensor = torch.tensor(
            request_indices,
            dtype=torch.long,
            device=key.device,
        )
        raw_key = key.reshape(-1, self.num_kv_heads, self.head_dim)
        raw_value = value.reshape_as(raw_key)
        self.raw_key[request_tensor, positions] = raw_key
        self.raw_value[request_tensor, positions] = raw_value

    def _rotate_local_key(self, raw_key: torch.Tensor) -> torch.Tensor:
        num_tokens = raw_key.shape[0]
        positions = torch.arange(
            num_tokens,
            dtype=torch.long,
            device=raw_key.device,
        )
        flat_key = raw_key.reshape(num_tokens, -1)
        if raw_key.device.type == "npu":
            from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

            cos_sin_cache = self.rotary_embedding._match_cos_sin_cache_dtype(
                flat_key
            )
            rotated = raw_key.contiguous()
            rope_forward_triton_siso(
                rotated,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                rope_dim=self.rotary_embedding.rotary_dim,
                is_neox_style=self.rotary_embedding.is_neox_style,
            )
            return rotated
        rotated, _ = self.rotary_embedding.forward_native(
            positions,
            flat_key,
            None,
        )
        return rotated.view_as(raw_key)

    def _history_indices(self, request_index: int, history_end: int) -> torch.Tensor:
        selected = self._selected_token_indices.get(request_index)
        if selected is not None:
            if selected.numel() > self.history_budget:
                selected = selected[-self.history_budget :]
            # Replace the previously newest chunk with a rolling tail. This
            # keeps recent accepted tokens visible between retrievals and
            # prevents the current draft root from appearing twice in KV.
            reserved = min(self.settings.chunk_size, selected.numel())
            selected_prefix = selected[:-reserved] if reserved else selected
            tail_start = max(0, history_end - self.settings.chunk_size)
            rolling_tail = torch.arange(
                tail_start,
                history_end,
                dtype=torch.long,
                device=self.device,
            )
            return torch.cat(
                (selected_prefix.to(self.device), rolling_tail)
            )[-self.history_budget :]
        history_start = max(0, history_end - self.history_budget)
        return torch.arange(
            history_start,
            history_end,
            dtype=torch.long,
            device=self.device,
        )

    def _rebuild_working_cache(
        self,
        batch_rows: list[int],
        seq_lens: list[int],
    ) -> list[int]:
        local_positions: list[int] = []
        for batch_row in batch_rows:
            slot = self._active_slots[batch_row]
            history_end = max(0, seq_lens[batch_row] - 1)
            token_indices = self._history_indices(slot, history_end)
            history_len = token_indices.numel()
            raw_key = self.raw_key[slot].index_select(
                0, token_indices
            )
            raw_value = self.raw_value[slot].index_select(
                0, token_indices
            )
            rotated_key = self._rotate_local_key(raw_key)
            self.working_key[
                slot, :, :history_len
            ] = rotated_key.transpose(0, 1)
            self.working_value[
                slot, :, :history_len
            ] = raw_value.transpose(0, 1)
            self._working_lens[slot] = history_len
            local_positions.append(history_len)
        return local_positions

    def capture_raw_kv(
        self,
        layer_index: int,
        absolute_positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor | None:
        if layer_index != 0:
            raise ValueError("DiffSpec draft cache only supports one layer")
        metadata = self._first_attention_metadata()
        if metadata is None:
            return None
        num_tokens = key.shape[0]
        if self._pending_tree_node_indices is not None:
            if self._pending_tree_depths is None:
                raise RuntimeError("DiffSpec tree depths are unavailable")
            slot = self._active_slots[0]
            node_indices = self._pending_tree_node_indices
            raw_key = key.reshape(-1, self.num_kv_heads, self.head_dim)
            raw_value = value.reshape_as(raw_key)
            self.tree_raw_key[slot].index_copy_(0, node_indices, raw_key)
            self.tree_raw_value[slot].index_copy_(0, node_indices, raw_value)
            local_positions = self._tree_base_len + self._pending_tree_depths - 1
            return local_positions.to(
                dtype=absolute_positions.dtype,
                device=absolute_positions.device,
            )
        if self._compact_prefill:
            batch_rows = list(range(num_tokens))
            self._compact_prefill = False
        else:
            batch_rows = self._request_indices(metadata, num_tokens)
        request_indices = self._canonical_slots(batch_rows)
        self._write_raw(
            request_indices,
            absolute_positions,
            key,
            value,
        )
        if not self._cycle_open:
            return None

        seq_lens = list(getattr(metadata, "seq_lens_list", ()))
        if self._draft_logical_seq_len and seq_lens:
            seq_lens[0] = self._draft_logical_seq_len
        if not seq_lens:
            return None
        if self._draft_step == 0:
            local_positions = self._rebuild_working_cache(
                batch_rows,
                seq_lens,
            )
        else:
            local_positions = [
                self._working_lens[request_index]
                for request_index in request_indices
            ]
        if max(local_positions) >= self.settings.token_budget:
            raise RuntimeError("DiffSpec working cache exhausted")

        self._pending_request_indices = request_indices
        self._pending_local_positions = local_positions
        self._draft_step += 1
        return torch.tensor(
            local_positions,
            dtype=absolute_positions.dtype,
            device=absolute_positions.device,
        )

    def has_pending_attention(self) -> bool:
        return (
            self._pending_tree_node_indices is not None
            or self._pending_request_indices is not None
            and self._pending_local_positions is not None
        )

    def _forward_tree_attention_cpu(
        self,
        query: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        slot: int,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = self._tree_base_len
        query_heads_per_kv = query.shape[1] // self.num_kv_heads
        prefix_key = self.working_key[slot, :, :prefix_len]
        prefix_value = self.working_value[slot, :, :prefix_len]
        prefix_key = prefix_key.repeat_interleave(query_heads_per_kv, dim=0)
        prefix_value = prefix_value.repeat_interleave(
            query_heads_per_kv, dim=0
        )
        num_tree_nodes = self._tree_num_nodes
        tree_key = self.tree_key[slot, :num_tree_nodes].transpose(0, 1)
        tree_value = self.tree_value[slot, :num_tree_nodes].transpose(0, 1)
        tree_key = tree_key.repeat_interleave(query_heads_per_kv, dim=0)
        tree_value = tree_value.repeat_interleave(
            query_heads_per_kv, dim=0
        )
        parents = self._tree_parents[:num_tree_nodes]
        allowed = build_tree_ancestry_mask(parents).index_select(
            0, node_indices
        )
        prefix_logits = torch.einsum(
            "nhd,hsd->nhs", query.float(), prefix_key.float()
        )
        tree_logits = torch.einsum(
            "nhd,hsd->nhs", query.float(), tree_key.float()
        )
        tree_logits.masked_fill_(~allowed[:, None, :], float("-inf"))
        logits = torch.cat((prefix_logits, tree_logits), dim=-1).mul_(scale)
        values = torch.cat((prefix_value, tree_value), dim=1)
        probabilities = logits.softmax(dim=-1)
        output.copy_(
            torch.einsum("nhs,hsd->nhd", probabilities, values).to(
                output.dtype
            )
        )
        return output

    def _forward_tree_attention_npu(
        self,
        query: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        slot: int,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        prefix_len = self._tree_base_len
        num_nodes = query.shape[0]
        prefix_output, prefix_lse = (
            torch_npu.npu_fused_infer_attention_score(
                query=query.transpose(0, 1)[None],
                key=self.working_key[
                    slot : slot + 1, :, :prefix_len
                ],
                value=self.working_value[
                    slot : slot + 1, :, :prefix_len
                ],
                actual_seq_lengths=[num_nodes],
                actual_seq_lengths_kv=[prefix_len],
                num_heads=query.shape[1],
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=scale,
                sparse_mode=0,
                softmax_lse_flag=True,
            )
        )
        num_tree_nodes = self._tree_num_nodes
        parents = self._tree_parents[:num_tree_nodes]
        allowed = build_tree_ancestry_mask(parents).index_select(
            0, node_indices
        )
        tree_output, tree_lse = torch_npu.npu_fused_infer_attention_score(
            query=query.transpose(0, 1)[None],
            key=self.tree_key[
                slot, :num_tree_nodes
            ].transpose(0, 1)[None],
            value=self.tree_value[
                slot, :num_tree_nodes
            ].transpose(0, 1)[None],
            atten_mask=~allowed,
            actual_seq_lengths=[num_nodes],
            actual_seq_lengths_kv=[num_tree_nodes],
            num_heads=query.shape[1],
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        prefix_output = prefix_output[0].transpose(0, 1)
        tree_output = tree_output[0].transpose(0, 1)
        prefix_lse = prefix_lse[0, :, :, 0].transpose(0, 1)
        tree_lse = tree_lse[0, :, :, 0].transpose(0, 1)
        merged, _ = merge_attention_states(
            prefix_output,
            prefix_lse,
            tree_output,
            tree_lse,
        )
        output.copy_(merged)
        return output

    def forward_tree_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._pending_tree_node_indices is None:
            raise RuntimeError("DiffSpec tree attention has no pending level")
        slot = self._active_slots[0]
        node_indices = self._pending_tree_node_indices
        self.tree_key[slot].index_copy_(0, node_indices, key)
        self.tree_value[slot].index_copy_(0, node_indices, value)
        if query.device.type == "npu":
            result = self._forward_tree_attention_npu(
                query, scale, output, slot, node_indices
            )
        else:
            result = self._forward_tree_attention_cpu(
                query, scale, output, slot, node_indices
            )
        self._pending_tree_node_indices = None
        self._pending_tree_parent_indices = None
        self._pending_tree_depths = None
        return result

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_pending_attention():
            raise RuntimeError("DiffSpec attention called without prepared state")
        if self._pending_tree_node_indices is not None:
            return self.forward_tree_attention(
                query, key, value, scale, output
            )
        assert self._pending_request_indices is not None
        assert self._pending_local_positions is not None
        for token_index, (request_index, local_position) in enumerate(
            zip(
                self._pending_request_indices,
                self._pending_local_positions,
            )
        ):
            self.working_key[
                request_index, :, local_position
            ] = key[token_index]
            self.working_value[
                request_index, :, local_position
            ] = value[token_index]
            seq_len = local_position + 1
            self._working_lens[request_index] = seq_len
            if query.device.type == "npu":
                import torch_npu

                attention_output, _ = (
                    torch_npu.npu_fused_infer_attention_score(
                        query=query[token_index][None, :, None, :],
                        key=self.working_key[
                            request_index : request_index + 1, :, :seq_len
                        ].contiguous(),
                        value=self.working_value[
                            request_index : request_index + 1, :, :seq_len
                        ].contiguous(),
                        actual_seq_lengths=[1],
                        actual_seq_lengths_kv=[seq_len],
                        num_heads=query.shape[1],
                        num_key_value_heads=self.num_kv_heads,
                        input_layout="BNSD",
                        scale=scale,
                        sparse_mode=0,
                    )
                )
                output[token_index].copy_(attention_output[0, :, 0])
            else:
                query_heads_per_kv = query.shape[1] // self.num_kv_heads
                expanded_key = self.working_key[
                    request_index, :, :seq_len
                ].repeat_interleave(query_heads_per_kv, dim=0)
                expanded_value = self.working_value[
                    request_index, :, :seq_len
                ].repeat_interleave(query_heads_per_kv, dim=0)
                logits = torch.einsum(
                    "hd,hsd->hs",
                    query[token_index].float(),
                    expanded_key.float(),
                )
                probabilities = logits.mul_(scale).softmax(dim=-1)
                output[token_index].copy_(
                    torch.einsum(
                        "hs,hsd->hd", probabilities, expanded_value
                    ).to(output.dtype)
                )
        self._pending_request_indices = None
        self._pending_local_positions = None
        return output
