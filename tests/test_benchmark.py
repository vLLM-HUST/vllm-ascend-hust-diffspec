# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module(filename: str) -> ModuleType:
    root = Path(__file__).parents[1]
    path = root / "benchmarks" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect their defining module while the class is created.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module("diffspec_long_context.py")
profile_select = _load_benchmark_module("diffspec_select_profile.py")


def _trials(endpoint: str, tps: float) -> list:
    trials = []
    for repeat in range(3):
        for sample in range(5):
            trials.append(
                benchmark.Trial(
                    endpoint=endpoint,
                    prompt_length=32768,
                    output_length=1024,
                    sample=sample,
                    repeat=repeat,
                    prompt_sha256="digest",
                    decode_seconds=1.0,
                    decode_tps=tps,
                    completion_tokens=1024,
                    token_ids=[1, 2],
                )
            )
    return trials


def test_summarize_case_uses_repeat_medians() -> None:
    summary = benchmark.summarize_case(
        _trials("baseline", 10.0), _trials("diffspec", 31.0)
    )
    assert summary["speedup"] == pytest.approx(3.1)
    assert summary["baseline_cv"] == 0.0
    assert summary["diffspec_cv"] == 0.0


def _passing_result(speedup: float = 3.1) -> dict:
    cases = [
        {
            "prompt_length": prompt_length,
            "output_length": output_length,
            "speedup": speedup,
            "baseline_cv": 0.01,
            "diffspec_cv": 0.01,
        }
        for prompt_length in (32768, 65536)
        for output_length in (1024, 2048, 4096)
    ]
    verdict = benchmark.evaluate_result(
        cases, min_speedup=2.7, min_geomean=3.0, max_cv=0.03
    )
    return {
        "passed": verdict["passed"],
        "geometric_mean_speedup": verdict["geometric_mean_speedup"],
        "cases": cases,
        "methodology": {
            "prompt_lengths": [32768, 65536],
            "output_lengths": [1024, 2048, 4096],
            "samples": 5,
            "repeats": 3,
        },
        "profile": {"diffspec_token_budget": 2048},
        "model_hash": "abc",
        "hardware": "Ascend-910B2",
        "dtype": "bfloat16",
    }


def test_evaluate_result_enforces_geometric_mean() -> None:
    result = _passing_result()
    assert result["passed"]
    assert result["geometric_mean_speedup"] == pytest.approx(3.1)

    result = _passing_result(2.9)
    assert not result["passed"]


def test_profile_selector_chooses_fastest_eligible_result() -> None:
    slow = _passing_result(3.1)
    fast = _passing_result(3.4)
    fast["profile"] = {"diffspec_token_budget": 1536}
    selected = profile_select.select_profile([slow, fast])
    assert selected["profile"] == fast["profile"]
    assert selected["geometric_mean_speedup"] == pytest.approx(3.4)


def test_profile_selector_rejects_incomplete_matrix() -> None:
    result = _passing_result()
    result["cases"].pop()
    with pytest.raises(ValueError, match="no profile passes"):
        profile_select.select_profile([result])
