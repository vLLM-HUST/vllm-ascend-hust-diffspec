# Sage Mate Qwen3.8-27B checkpoint gate — 2026-09-04

## Verdict

DiffSpec is **blocked**, not compatible, for Qwen3.8-27B on the Sage Mate
TP4 graph lane. The source port passed its host suite, but no eligible Eagle3
draft checkpoint was available to begin the required device qualification.

## Source candidate

- vLLM-HUST `762f85b311fbab0bcf8921dd216f5093cd58b9b8`
  (`0.28.1rc1.dev319`)
- vLLM-Ascend-HUST `4e57439e58ed3d78e675f9fd7b4614fb183c5394`
  (`0.25.1rc1`)
- DiffSpec `96188b9923928b3d51bbf7f81d38fcd1144e3fb9`
- Host suite: 38 passed

The candidate adapts the current Eagle3, Ascend attention, model runner,
sampler and speculative-metadata surfaces. It rejects async scheduling and
requires TP4 graph mode; neither TP1 nor eager fallback is an accepted result.

## Draft inventory result

The target Qwen3.8/Qwen3.5 tokenizer vocabulary has 248,320 entries. The local
Qwen3-1.7B and Qwen3-8B Eagle3 drafts have 151,936 entries and are rejected.
The inspected VirVen Qwen3.5 draft targets a different model contract and is
packaged for SGLang. The advertised NIM built-in draft was not available as an
independently usable checkpoint.

Because the draft gate did not pass, there is no honest TP4 evidence for
per-rank draft tokens, accept/reject decisions, KV metadata consistency,
capture/replay, cancellation, exception recovery, concurrency, acceptance
rate, P50/P95 latency or throughput. Changing only a dependency declaration
cannot supply any of those results.

Qualification may resume only with a one-layer Eagle3 draft whose architecture,
vocabulary and tokenizer contract match the Qwen3.8-27B target. Until then the
compatible-model list for this exact lane is empty.
