# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Strict long-context decode benchmark for DiffSpec.

The benchmark alternates requests between a target-only vLLM endpoint and a
DiffSpec endpoint. It uses vLLM's ``return_token_ids`` extension so greedy
correctness is checked on token IDs without enabling logprobs (which would
disable the DiffSpec tree fast path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_LENGTHS = (32768, 65536)
DEFAULT_OUTPUT_LENGTHS = (1024, 2048, 4096)


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key: str | None = None

    @property
    def completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/completions"


@dataclass
class Trial:
    endpoint: str
    prompt_length: int
    output_length: int
    sample: int
    repeat: int
    prompt_sha256: str
    decode_seconds: float
    decode_tps: float
    completion_tokens: int
    token_ids: list[int]


def _extract_record(record: Any) -> list[int] | str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    token_ids = record.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(x, int) for x in token_ids):
        return token_ids
    for field in ("text", "context", "prompt", "input"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def load_corpus(path: Path) -> list[list[int] | str]:
    """Load token-ID or text documents from txt, JSON, or JSONL."""
    raw = path.read_text(encoding="utf-8")
    documents: list[list[int] | str] = []
    if path.suffix.lower() == ".txt":
        return [part for part in raw.split("\n\n") if part.strip()]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    records: Iterable[Any]
    if isinstance(parsed, list):
        records = parsed
    elif parsed is not None:
        records = [parsed]
    else:
        records = (json.loads(line) for line in raw.splitlines() if line.strip())
    for record in records:
        document = _extract_record(record)
        if document:
            documents.append(document)
    if not documents:
        raise ValueError(f"no usable documents found in {path}")
    return documents


def tokenize_corpus(
    documents: list[list[int] | str], tokenizer_name: str
) -> tuple[list[list[int]], int]:
    if all(isinstance(document, list) for document in documents):
        return [list(document) for document in documents], 0  # type: ignore[arg-type]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("transformers is required for a text corpus") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    eos_token_id = tokenizer.eos_token_id
    separator = 0 if eos_token_id is None else int(eos_token_id)
    tokenized: list[list[int]] = []
    for document in documents:
        if isinstance(document, list):
            tokenized.append(document)
        else:
            ids = tokenizer.encode(document, add_special_tokens=False)
            if ids:
                tokenized.append(ids)
    if not tokenized:
        raise ValueError("the corpus produced no tokens")
    return tokenized, separator


def build_prompts(
    documents: list[list[int]],
    separator_token_id: int,
    *,
    prompt_lengths: tuple[int, ...] = DEFAULT_PROMPT_LENGTHS,
    samples: int,
    seed: int,
) -> dict[int, list[list[int]]]:
    stream: list[int] = []
    for document in documents:
        stream.extend(document)
        stream.append(separator_token_id)
    required = max(prompt_lengths)
    if len(stream) < required:
        raise ValueError(
            f"corpus has {len(stream)} tokens, but at least {required} are required"
        )

    prompts: dict[int, list[list[int]]] = {}
    for prompt_length in prompt_lengths:
        rng = random.Random(seed + prompt_length)
        max_start = len(stream) - prompt_length
        starts = [rng.randrange(max_start + 1) for _ in range(samples)]
        prompts[prompt_length] = [
            stream[start : start + prompt_length] for start in starts
        ]
    return prompts


def prompt_digest(token_ids: list[int]) -> str:
    values = array("I", token_ids)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _post_stream(
    endpoint: Endpoint,
    prompt_token_ids: list[int],
    output_length: int,
    timeout: float,
) -> tuple[list[int], list[float], int | None]:
    payload = {
        "model": endpoint.model,
        "prompt": prompt_token_ids,
        "max_tokens": output_length,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = urllib.request.Request(
        endpoint.completions_url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )

    output_ids: list[int] = []
    token_times: list[float] = []
    usage_tokens: int | None = None
    # Benchmark endpoints are explicitly supplied by the operator and are
    # normally local. Do not let cluster-wide HTTP(S)_PROXY settings redirect
    # them through a gateway, which both distorts timing and often returns 502.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            for raw_line in response:
                timestamp = time.perf_counter()
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage")
                if usage:
                    usage_tokens = usage.get("completion_tokens", usage_tokens)
                for choice in event.get("choices", []):
                    delta_ids = choice.get("token_ids") or []
                    output_ids.extend(int(token_id) for token_id in delta_ids)
                    token_times.extend([timestamp] * len(delta_ids))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{endpoint.name} returned HTTP {exc.code}: {body}"
        ) from exc

    return output_ids, token_times, usage_tokens


def run_trial(
    endpoint: Endpoint,
    prompt_token_ids: list[int],
    output_length: int,
    *,
    sample: int,
    repeat: int,
    timeout: float,
) -> Trial:
    output_ids, token_times, usage_tokens = _post_stream(
        endpoint, prompt_token_ids, output_length, timeout
    )
    if len(output_ids) != output_length:
        raise RuntimeError(
            f"{endpoint.name} returned {len(output_ids)} tokens, "
            f"expected {output_length}"
        )
    if usage_tokens is not None and usage_tokens != output_length:
        raise RuntimeError(
            f"{endpoint.name} usage reports {usage_tokens} completion tokens, "
            f"expected {output_length}"
        )
    if len(token_times) < 2 or token_times[-1] <= token_times[0]:
        raise RuntimeError(
            f"{endpoint.name} did not provide enough timed streaming chunks"
        )
    decode_seconds = token_times[-1] - token_times[0]
    return Trial(
        endpoint=endpoint.name,
        prompt_length=len(prompt_token_ids),
        output_length=output_length,
        sample=sample,
        repeat=repeat,
        prompt_sha256=prompt_digest(prompt_token_ids),
        decode_seconds=decode_seconds,
        decode_tps=(output_length - 1) / decode_seconds,
        completion_tokens=len(output_ids),
        token_ids=output_ids,
    )


def coefficient_of_variation(values: list[float]) -> float:
    if not values or statistics.fmean(values) == 0:
        return math.inf
    return statistics.pstdev(values) / statistics.fmean(values)


def summarize_case(
    baseline_trials: list[Trial], diffspec_trials: list[Trial]
) -> dict[str, Any]:
    baseline_repeat_medians = [
        statistics.median(
            trial.decode_tps
            for trial in baseline_trials
            if trial.repeat == repeat
        )
        for repeat in sorted({trial.repeat for trial in baseline_trials})
    ]
    diffspec_repeat_medians = [
        statistics.median(
            trial.decode_tps
            for trial in diffspec_trials
            if trial.repeat == repeat
        )
        for repeat in sorted({trial.repeat for trial in diffspec_trials})
    ]
    baseline_median = statistics.median(
        trial.decode_tps for trial in baseline_trials
    )
    diffspec_median = statistics.median(
        trial.decode_tps for trial in diffspec_trials
    )
    return {
        "baseline_median_decode_tps": baseline_median,
        "diffspec_median_decode_tps": diffspec_median,
        "speedup": diffspec_median / baseline_median,
        "baseline_repeat_medians": baseline_repeat_medians,
        "diffspec_repeat_medians": diffspec_repeat_medians,
        "baseline_cv": coefficient_of_variation(baseline_repeat_medians),
        "diffspec_cv": coefficient_of_variation(diffspec_repeat_medians),
    }


def evaluate_result(
    cases: list[dict[str, Any]], *, min_speedup: float, min_geomean: float,
    max_cv: float
) -> dict[str, Any]:
    speedups = [float(case["speedup"]) for case in cases]
    geometric_mean = math.exp(statistics.fmean(math.log(x) for x in speedups))
    failures: list[str] = []
    for case in cases:
        label = f"{case['prompt_length']}+{case['output_length']}"
        if case["speedup"] < min_speedup:
            failures.append(f"{label} speedup {case['speedup']:.3f} < {min_speedup}")
        if case["baseline_cv"] >= max_cv:
            failures.append(f"{label} baseline CV {case['baseline_cv']:.3%} >= {max_cv:.3%}")
        if case["diffspec_cv"] >= max_cv:
            failures.append(f"{label} DiffSpec CV {case['diffspec_cv']:.3%} >= {max_cv:.3%}")
    if geometric_mean < min_geomean:
        failures.append(
            f"geometric mean {geometric_mean:.3f} < {min_geomean}"
        )
    return {
        "passed": not failures,
        "geometric_mean_speedup": geometric_mean,
        "failures": failures,
    }


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    baseline = Endpoint("baseline", args.baseline_url, args.baseline_model, args.api_key)
    diffspec = Endpoint("diffspec", args.diffspec_url, args.diffspec_model, args.api_key)
    documents = load_corpus(args.corpus)
    tokenized, separator = tokenize_corpus(documents, args.tokenizer)
    prompts = build_prompts(
        tokenized,
        separator,
        prompt_lengths=args.prompt_lengths,
        samples=args.samples,
        seed=args.seed,
    )

    all_trials: list[Trial] = []
    cases: list[dict[str, Any]] = []
    pair_number = 0
    for prompt_length in args.prompt_lengths:
        for output_length in args.output_lengths:
            label = f"input={prompt_length} output={output_length}"
            _progress(f"warming {label}")
            for warmup in range(args.warmups):
                prompt = prompts[prompt_length][warmup % args.samples]
                for endpoint in (baseline, diffspec):
                    run_trial(
                        endpoint,
                        prompt,
                        output_length,
                        sample=warmup % args.samples,
                        repeat=-1,
                        timeout=args.timeout,
                    )

            baseline_trials: list[Trial] = []
            diffspec_trials: list[Trial] = []
            for repeat in range(args.repeats):
                for sample, prompt in enumerate(prompts[prompt_length]):
                    endpoints = (
                        (baseline, diffspec)
                        if pair_number % 2 == 0
                        else (diffspec, baseline)
                    )
                    pair_number += 1
                    pair: dict[str, Trial] = {}
                    for endpoint in endpoints:
                        _progress(
                            f"running {label} repeat={repeat + 1}/{args.repeats} "
                            f"sample={sample + 1}/{args.samples} {endpoint.name}"
                        )
                        pair[endpoint.name] = run_trial(
                            endpoint,
                            prompt,
                            output_length,
                            sample=sample,
                            repeat=repeat,
                            timeout=args.timeout,
                        )
                    if pair["baseline"].token_ids != pair["diffspec"].token_ids:
                        mismatch = next(
                            index
                            for index, (left, right) in enumerate(
                                zip(
                                    pair["baseline"].token_ids,
                                    pair["diffspec"].token_ids,
                                    strict=True,
                                )
                            )
                            if left != right
                        )
                        raise RuntimeError(
                            f"greedy token mismatch for {label}, sample={sample}, "
                            f"repeat={repeat}, output offset={mismatch}"
                        )
                    baseline_trials.append(pair["baseline"])
                    diffspec_trials.append(pair["diffspec"])
                    all_trials.extend(pair.values())

            case = {
                "prompt_length": prompt_length,
                "output_length": output_length,
                **summarize_case(baseline_trials, diffspec_trials),
            }
            cases.append(case)
            _progress(f"completed {label}: speedup={case['speedup']:.3f}x")

    verdict = evaluate_result(
        cases,
        min_speedup=args.min_speedup,
        min_geomean=args.min_geomean,
        max_cv=args.max_cv,
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "diffspec",
        "baseline": asdict(baseline),
        "diffspec": asdict(diffspec),
        "model_hash": args.model_hash,
        "hardware": args.hardware,
        "dtype": args.dtype,
        "profile": json.loads(args.profile.read_text()) if args.profile else None,
        "methodology": {
            "prompt_lengths": list(args.prompt_lengths),
            "output_lengths": list(args.output_lengths),
            "samples": args.samples,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "seed": args.seed,
            "ignore_eos": True,
            "temperature": 0,
            "decode_tps": "(completion_tokens - 1) / (last_token_time - first_token_time)",
            "min_case_speedup": args.min_speedup,
            "min_geometric_mean_speedup": args.min_geomean,
            "max_cv": args.max_cv,
        },
        "cases": cases,
        **verdict,
        "trials": [asdict(trial) for trial in all_trials],
    }
    return result


def _positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("lengths must not contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--diffspec-url", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--diffspec-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompt-lengths",
        type=_positive_int_csv,
        default=DEFAULT_PROMPT_LENGTHS,
        help="comma-separated prompt lengths",
    )
    parser.add_argument(
        "--output-lengths",
        type=_positive_int_csv,
        default=DEFAULT_OUTPUT_LENGTHS,
        help="comma-separated completion lengths",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--min-speedup", type=float, default=2.7)
    parser.add_argument("--min-geomean", type=float, default=3.0)
    parser.add_argument("--max-cv", type=float, default=0.03)
    parser.add_argument("--model-hash", default="unknown")
    parser.add_argument("--hardware", default="Ascend NPU")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = benchmark(args)
    except Exception as exc:
        _progress(f"DiffSpec benchmark failed: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _progress(
        f"wrote {args.output}; geometric mean "
        f"{result['geometric_mean_speedup']:.3f}x; passed={result['passed']}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
