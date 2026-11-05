# vllm-diffspec

Ascend integration of **DiffSpec: Accelerating Long Sequence Generation with
Differential Speculative Decoding**. DiffSpec is an external research mechanism.
This repository contributes compatibility fixes, Ascend implementations and
separate evaluation; it does not claim the original method as HUST research.

The mechanism is opt-in: `draft_context_policy` defaults to `full`. Installation
and extension discovery do not establish compatibility or acceleration.

## Runtime design

- Identify prefill from attention metadata, and retrieve with the last committed
  query after native verification. Missing tree IDs do not identify linear prefill.
- Aggregate target chunk scores across TP ranks and combine selected history with
  a rolling recent window. Exclude rejected suffixes and graph padding.
- Store canonical pre-RoPE draft KV at absolute positions. A compact request plan
  describes selected chunks and committed prefix/tail boundaries.
- Gather KV, apply local RoPE and write the working cache in one Triton kernel.
  Draft attention reads capacity views with valid lengths; contiguous slots reuse
  storage and reordered slots retain request order.

Target weights and native rejection-verification mathematics are preserved.
This does not prove complete numerical equivalence. Full raw KV stays allocated,
so a bounded working cache does not establish lower total memory.

## Supported evaluation configuration

| Component | Configuration |
|---|---|
| Target | `Qwen/Qwen3.5-35B-A3B`, native BF16 |
| Target revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| Draft | `jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge`, BF16 |
| Draft revision | `d07a277ececba408f017a0416002d7abb23caf36` |
| Devices | Four Ascend 910B2; TP4 / draft TP4; PP1 |
| Execution | `PIECEWISE`, native linear verification, fixed K4 |
| Working cache | Budget 2048; chunk 64; retrieval interval 4 |
| Scheduling | Prefix caching and async scheduling disabled; text only |

Check weights and original inference files against publisher hashes. The local
configuration records auxiliary boundaries `[4,24,36]` from the publisher serving
path, retaining the draft output head and `d2t` mapping. This layer choice is not
a demonstrated acceptance improvement. Matching vocabularies alone does not
establish checkpoint compatibility.

Required source bases, together with their recorded runtime patches:

| Repository | Base commit | Adaptation |
|---|---|---|
| vLLM Core | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` | KV-cache grouping control |
| vLLM Ascend | `4e57439e58ed3d78e675f9fd7b4614fb183c5394` | Native plane alignment and GDN/cache compatibility |
| DiffSpec | `1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254` | Linear retrieval and compact-cache integration |

This plugin does not install Core/Ascend patches. Record source hashes and actual
loaded worker paths; installed package versions do not identify this combination.

## Compatibility boundaries

| Path | Behavior |
|---|---|
| `auto` / `linear` + `PIECEWISE` | Retrieve on committed decode queries at the configured cadence |
| Prefill or mixed prefill/decode | Conservatively skip retrieval |
| Full-graph replay | Rejected: replay may bypass target snapshot callbacks |
| Qwen3.5 MoE | Unquantized BF16 and linear verification only |
| Tensor parallelism | TP4 required; TP2 is outside this submission |
| Tree verification | Existing experimental dense path; rejected for MoE |
| Native MTP2 + APC + async + `FULL_AND_PIECEWISE` | No supported reusable path |

MLA, pipeline parallelism and quantized checkpoints are unsupported.
Other model pairs need separate compatibility and device checks.
Eagle3 evaluation is a separate external-mechanism workload; offline checks do
not qualify HTTP serving, full accuracy or native-MTP configurations.

## Installation

Use the matching patched Core/Ascend environment:

```bash
python -m pip install -e . --no-deps
```

The package provides a `vllm.general_plugins` entry point and an experimental
extension manifest. Activation requires explicit plugin selection and
`draft_context_policy="diffspec"`; discovery alone does not enable retrieval.

## Quick Start

The model paths refer to verified target and draft views. The Core patch supplies
the cache-grouping option below.

```bash
VLLM_PLUGINS=ascend,diffspec vllm serve /path/to/target-model \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.75 \
  --min-kv-cache-group-layers 10 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,32,48,64]}' \
  --speculative-config '{
    "method": "eagle3",
    "model": "/path/to/eagle3-draft-model",
    "num_speculative_tokens": 4,
    "draft_tensor_parallel_size": 4,
    "enforce_eager": false,
    "draft_context_policy": "diffspec",
    "diffspec_verification_mode": "auto",
    "diffspec_chunk_size": 64,
    "diffspec_token_budget": 2048,
    "diffspec_retrieval_interval": 4,
    "diffspec_adaptive_profile": false
  }'
```

## Configuration

Plugin defaults differ from the explicit preset above.

| Key | Default | Meaning |
|---|---:|---|
| `draft_context_policy` | `full` | Set to `diffspec` to enable retrieval |
| `diffspec_verification_mode` | `auto` | Linear; explicit tree is experimental |
| `diffspec_chunk_size` | `32` | Tokens per retrieval chunk |
| `diffspec_token_budget` | `2048` | Working-cache capacity |
| `diffspec_retrieval_interval` | `8` | Verifier cycles between retrievals |
| `diffspec_max_tree_nodes` | `50` | Tree verifier-node limit |
| `diffspec_tree_threshold` | `0.7` | Tree probability threshold |
| `diffspec_adaptive_profile` | `false` | Opt-in acceptance-based depth adaptation |
| `diffspec_long_context_threshold` | `49152` | Adaptive context threshold |
| `diffspec_long_context_depth` | `2` | Adaptive starting depth above threshold |

Workers log verification mode and retrieval cadence at startup.
`drafter.diffspec_cache.retrieval_status()` reports attempts, retrievals, selected
slots and skips for prefill, cadence, missing snapshots or no committed tokens.
A loaded proposer alone does not prove retrieval.

## Validation and reporting

```bash
python -m pytest tests/ -q
DIFFSPEC_RUN_DEVICE_TESTS=1 python -m pytest \
  tests/test_linear_attention_batch_npu.py tests/test_retrieval_npu.py -q
```

Device tests require an idle, explicitly selected Ascend device. Keep regression
coverage for graph replay, committed boundaries, slot reuse, partial RoPE,
ragged history and unused-capacity masking. Performance scripts are separate.

Reports must identify model revisions, dataset construction and input hashes,
execution settings, independent starts, acceptance, throughput, latency
semantics, failures and output checks. Assembled MMLU-Pro references form a
fixed-length generation workload, not standard answer accuracy. Natural-EOS
probes are separate. Raw measurements and numerical claims belong in
`vllm-hust-benchmark` under the organization's evidence policy.

## Reference and license

DiffSpec: *Accelerating Long Sequence Generation with Differential Speculative
Decoding*. The original method retains its external attribution. Repository code
is Apache-2.0; see [LICENSE](LICENSE). Governance metadata is in
[MOD_METADATA.json](MOD_METADATA.json).
