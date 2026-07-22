# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Select and version the fastest passing DiffSpec benchmark profile."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def result_is_eligible(result: dict[str, Any]) -> bool:
    cases = result.get("cases", [])
    methodology = result.get("methodology", {})
    prompt_lengths = methodology.get("prompt_lengths", [])
    output_lengths = methodology.get("output_lengths", [])
    combinations = {
        (case.get("prompt_length"), case.get("output_length")) for case in cases
    }
    expected = {
        (prompt_length, output_length)
        for prompt_length in prompt_lengths
        for output_length in output_lengths
    }
    return (
        bool(expected)
        and combinations == expected
        and int(methodology.get("samples", 0)) >= 5
        and int(methodology.get("repeats", 0)) >= 3
        and bool(result.get("passed"))
        and isinstance(result.get("profile"), dict)
    )


def select_profile(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [result for result in results if result_is_eligible(result)]
    if not eligible:
        raise ValueError(
            "no complete profile satisfies its benchmark thresholds with "
            "at least five samples and three repeats"
        )
    best = max(eligible, key=lambda item: item["geometric_mean_speedup"])
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "method": "diffspec",
        "model_hash": best["model_hash"],
        "hardware": best["hardware"],
        "dtype": best["dtype"],
        "profile": best["profile"],
        "geometric_mean_speedup": best["geometric_mean_speedup"],
        "case_speedups": {
            f"{case['prompt_length']}x{case['output_length']}": case["speedup"]
            for case in best["cases"]
        },
    }


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    loaded = [json.loads(path.read_text()) for path in args.results]
    try:
        selected = select_profile(loaded)
    except ValueError as exc:
        parser.error(str(exc))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = "_".join(
        (
            safe_component(selected["model_hash"]),
            safe_component(selected["hardware"]),
            safe_component(selected["dtype"]),
            timestamp,
        )
    ) + ".json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / filename
    destination.write_text(json.dumps(selected, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
