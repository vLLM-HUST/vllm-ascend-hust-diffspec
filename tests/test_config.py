# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest

from diffspec.config import DiffSpecPluginConfig


def test_extract_keeps_standard_vllm_keys() -> None:
    plugin, standard = DiffSpecPluginConfig.extract(
        {
            "method": "eagle3",
            "model": "draft",
            "num_speculative_tokens": 10,
            "draft_context_policy": "diffspec",
            "diffspec_token_budget": 1536,
        }
    )
    assert plugin.enabled
    assert plugin.diffspec_token_budget == 1536
    assert standard == {
        "method": "eagle3",
        "model": "draft",
        "num_speculative_tokens": 10,
    }


def test_validate_rejects_non_eagle3() -> None:
    config = DiffSpecPluginConfig(draft_context_policy="diffspec")
    with pytest.raises(ValueError, match="requires method='eagle3'"):
        config.validate(method="ngram", depth=10, max_model_len=73728)


def test_validate_rejects_tree_smaller_than_depth() -> None:
    config = DiffSpecPluginConfig(
        draft_context_policy="diffspec",
        diffspec_verification_mode="tree",
        diffspec_max_tree_nodes=5,
    )
    with pytest.raises(ValueError, match="greater than or equal"):
        config.validate(method="eagle3", depth=10, max_model_len=73728)


def test_validate_rejects_unknown_verification_mode() -> None:
    config = DiffSpecPluginConfig(
        draft_context_policy="diffspec",
        diffspec_verification_mode="wide",
    )
    with pytest.raises(ValueError, match="verification_mode"):
        config.validate(method="eagle3", depth=5, max_model_len=73728)


def test_auto_verification_does_not_enable_tree() -> None:
    auto = DiffSpecPluginConfig(draft_context_policy="diffspec")
    tree = DiffSpecPluginConfig(
        draft_context_policy="diffspec",
        diffspec_verification_mode="tree",
    )
    assert not auto.uses_tree_verification
    assert tree.uses_tree_verification


def test_adaptive_linear_depth_must_fit_configured_depth() -> None:
    config = DiffSpecPluginConfig(
        draft_context_policy="diffspec",
        diffspec_long_context_depth=6,
    )
    with pytest.raises(ValueError, match="long_context_depth"):
        config.validate(method="eagle3", depth=5, max_model_len=73728)


def test_full_policy_remains_a_noop() -> None:
    config = DiffSpecPluginConfig()
    config.validate(method="ngram", depth=1, max_model_len=1)
    assert not config.enabled
