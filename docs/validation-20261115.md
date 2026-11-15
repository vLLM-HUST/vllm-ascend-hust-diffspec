# Pinned TP4 functional evidence — 2026-11-15

This record covers the implementation merged by [PR #11](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/pull/11) at
`48b182803f2b1bc397d61fd8f5a50c8716f2c4ed`. It records the requested committed-query and padding boundary regression, with numerical measurements kept in the benchmark repository.

## Runtime and scope

The run uses Qwen3.5-35B-A3B BF16 with its pinned Specforge Eagle3 draft, four actual Ascend910B2 devices, TP4/draftTP4, fixed K4, PIECEWISE, and APC/async disabled. The model views retain the revisions and dependency patches documented in the supported evaluation configuration. Twenty Core/Ascend source-file hashes match the frozen runtime.

The focused committed-query, graph-padding, rejected-suffix and TP aggregation suite reports **36 passed**. Two actual model requests, with 4096/4352 input tokens and 256 generated tokens each, complete without request corruption flags or preemption.

The first model start observes native metadata without an extra dummy row. A second start explicitly appends one padding query/LSE/table/metadata row to the real two-request target snapshot and restores the original references after retrieval. It preserves real query values, KV data, native valid counts, model weights and the original verification algorithm. This is a controlled regression fixture, not a claim that native scheduling produced the extra row.

There are **21 two-request-plus-dummy retrieval cycles**, all with rejected suffixes. The original implementation passes only the two committed prefixes to the retrieval kernel; selected token positions stay below their committed bounds. All four ranks select identical chunk lists and token digests in **24/24 retrieval cycles**.

## Qualification scope

This is a focused two-request padding regression. Complete numerical equivalence, HTTP serving and full-accuracy qualification remain outside this record.

The associated historical performance pairs remain attributed to the 2026-10-30 frozen implementation; they are not relabelled as new performance measurements of this merge. This record does not close issue #1 or establish HTTP serving/full-accuracy qualification.

## Evidence custody

Reports, commands, environment/source identities, both model attempts, four-rank traces, historical comparisons and an offline verifier are archived in the [fixed benchmark evidence directory](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/fe71138b4973872051d0b4963370c99be8385c5f/reports/tech-outcomes-20261117/diffspec/) through [benchmark PR #680](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/680).

The [published Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/diffspec-tech-outcomes-20261115-v1) preserves the raw functional and frozen-performance archives. Its download URLs, asset IDs and verified SHA256 digests are recorded in the benchmark report's release manifest. The original DiffSpec mechanism retains its external attribution; this work contributes implementation, Ascend compatibility and separate validation.
