# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Plugin-owned configuration layered on vLLM's speculative config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiffSpecPluginConfig:
    draft_context_policy: str = "full"
    diffspec_verification_mode: str = "auto"
    diffspec_chunk_size: int = 32
    diffspec_token_budget: int = 2048
    diffspec_retrieval_interval: int = 8
    diffspec_max_tree_nodes: int = 50
    diffspec_tree_threshold: float = 0.7
    diffspec_adaptive_profile: bool = True
    diffspec_long_context_threshold: int = 49152
    diffspec_long_context_depth: int = 2

    @classmethod
    def extract(cls, raw: dict[str, Any]) -> tuple[DiffSpecPluginConfig, dict[str, Any]]:
        values: dict[str, Any] = {}
        remaining = dict(raw)
        for name in cls.__dataclass_fields__:
            if name in remaining:
                values[name] = remaining.pop(name)
        return cls(**values), remaining

    @property
    def enabled(self) -> bool:
        return self.draft_context_policy == "diffspec"

    def validate(self, *, method: str | None, depth: int | None, max_model_len: int | None) -> None:
        if self.draft_context_policy not in ("full", "diffspec"):
            raise ValueError("draft_context_policy must be 'full' or 'diffspec'")
        if not self.enabled:
            return
        if self.diffspec_verification_mode not in ("auto", "linear", "tree"):
            raise ValueError(
                "diffspec_verification_mode must be 'auto', 'linear', or 'tree'"
            )
        if method != "eagle3":
            raise ValueError("draft_context_policy='diffspec' currently requires method='eagle3'")
        positive = {
            "diffspec_chunk_size": self.diffspec_chunk_size,
            "diffspec_token_budget": self.diffspec_token_budget,
            "diffspec_retrieval_interval": self.diffspec_retrieval_interval,
            "diffspec_max_tree_nodes": self.diffspec_max_tree_nodes,
            "diffspec_long_context_threshold": (
                self.diffspec_long_context_threshold
            ),
            "diffspec_long_context_depth": self.diffspec_long_context_depth,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.diffspec_chunk_size > self.diffspec_token_budget:
            raise ValueError("diffspec_chunk_size must not exceed diffspec_token_budget")
        if self.diffspec_token_budget % self.diffspec_chunk_size:
            raise ValueError("diffspec_token_budget must be divisible by diffspec_chunk_size")
        if not 0.0 < self.diffspec_tree_threshold <= 1.0:
            raise ValueError("diffspec_tree_threshold must be in (0, 1]")
        if depth is None or depth <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if (
            self.diffspec_adaptive_profile
            and self.diffspec_verification_mode != "tree"
            and self.diffspec_long_context_depth > depth
        ):
            raise ValueError(
                "diffspec_long_context_depth must not exceed "
                "num_speculative_tokens"
            )
        if self.uses_tree_verification and self.diffspec_max_tree_nodes < depth:
            raise ValueError(
                "diffspec_max_tree_nodes must be greater than or equal to num_speculative_tokens"
            )
        if max_model_len is not None and max_model_len < self.diffspec_token_budget:
            raise ValueError("DiffSpec max_model_len must cover diffspec_token_budget")

    def attach(self, speculative_config: Any) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(speculative_config, name, getattr(self, name))

    @property
    def uses_tree_verification(self) -> bool:
        """Return whether the experimental shared-prefix tree is requested.

        ``auto`` resolves to the measured-safe linear verifier.  This keeps
        greedy requests from silently entering a slower path; an offline
        profile can explicitly select ``tree`` when it wins on the deployed
        model and Ascend software stack.
        """
        return self.enabled and self.diffspec_verification_mode == "tree"
