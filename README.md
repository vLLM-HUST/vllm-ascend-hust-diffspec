# vllm-diffspec

**DiffSpec: Accelerating Long Sequence Generation with Differential Speculative Decoding**

> 📄 **Paper**: **SC26** (Supercomputing 2026)
>
> DiffSpec is a differential speculative decoding framework that aligns
> speculative effort with its position-specific utility during long-sequence
> generation. It combines relevance-aware KV selection, utility-guided draft
> construction, and adaptive execution to concentrate speculation where an
> accepted token provides the greatest system-level benefit. This repository
> provides a vLLM Ascend implementation built around Eagle3.

---

## What DiffSpec Does

In long-sequence generation, the cost of each decoding step rises with the
attention span and KV-cache footprint. Successful speculation therefore has
the highest marginal payoff late in generation. However, lightweight draft
models become less accurate as the sequence grows, so conventional uniform
speculation fails most often precisely where it matters most.

**DiffSpec** replaces this acceptance-centric, uniform policy with
**differential speculation** guided by position-specific utility:

1. **Estimate speculative utility** from the current sequence position and
   verifier feedback, capturing both the potential execution benefit and the
   risk of rejection.
2. **Select relevant target KV chunks** using attention-derived signals so the
   lightweight draft model retains semantically important long-range context
   within a bounded working set.
3. **Construct drafts adaptively** by allocating speculative depth and branching
   to high-utility or high-rejection-risk positions instead of expanding every
   position uniformly.
4. **Preserve KV reuse and locality** across decoding steps, reducing the cost
   of the sparse, non-contiguous cache accesses introduced by context selection.
5. **Verify candidates efficiently** while exploiting shared prefixes in the
   adaptive speculative structure.

In this vLLM Ascend implementation, target-guided chunks are scored directly
from paged target KV, combined with a rolling recent window, and rebuilt as a
compact pre-RoPE Eagle3 cache with local positions. The integration is
**non-invasive**: it changes draft construction and speculative scheduling,
while leaving the target model, OpenAI-compatible API, and native verification
semantics intact.

---

## Key Results

The following engineering measurements use Llama-3.1-8B, a one-layer Eagle3
draft model, BF16, TP1, batch size 1, greedy decoding, and an Ascend 910B2.
Decode TPS excludes time to first token.

| Input | Output | Target-only TPS | DiffSpec TPS | Speedup |
|---:|---:|---:|---:|---:|
| 32K | 1K | 15.49 | 26.17 | **1.69×** |
| 32K | 2K | 15.47 | 27.74 | **1.79×** |
| 32K | 4K | 15.52 | 28.66 | **1.85×** |
| 64K | 1K | 15.62 | 32.51 | **2.08×** |
| 64K | 2K | 15.39 | 38.05 | **2.47×** |
| 64K | 4K | 15.16 | 44.75 | **2.95×** |
| 128K | 1K | 13.75 | 42.50 | **3.09×** |
| 128K | 2K | 14.04 | 48.04 | **3.42×** |
| 128K | 4K | 12.22 | 52.08 | **4.26×** |

---

## Architecture

DiffSpec is organized into four layers:

```text
┌──────────────────────────────────────┐
│  vLLM / vLLM Ascend Adapter Layer    │  ← runtime hooks and config routing
├──────────────────────────────────────┤
│  Adaptive Eagle3 Proposer            │  ← depth policy and verification mode
├──────────────────────────────────────┤
│  Compact Draft KV Layer              │  ← canonical KV and local working KV
├──────────────────────────────────────┤
│  Target-Guided Retrieval Layer       │  ← paged-K scoring and chunk selection
└──────────────────────────────────────┘
```

- **Target-guided retrieval**: scores 32-token chunks from paged target KV
  without materializing the full attention matrix.
- **Dual-view draft cache**: stores canonical pre-RoPE KV at absolute positions
  and rebuilds a compact cache with local draft RoPE positions.
- **Adaptive verification**: starts long contexts at a shallow depth and
  adjusts depth according to sustained acceptance.

---

## Requirements

- vLLM and vLLM Ascend
- Ascend 910B-series NPU with a working CANN/PyTorch NPU environment
- A target model and a compatible one-layer Eagle3 draft model
- BF16, TP1, PP1, eager execution, and batch size 1
- Prefix caching, async scheduling, quantization, MLA, and M-RoPE disabled


---

## Installation

Install vLLM and vLLM Ascend first, then install DiffSpec into the same Python
environment:

```bash
git clone <repository-url> vllm-diffspec
cd vllm-diffspec
python3 -m pip install -e . --no-deps
```

The plugin auto-registers through the `vllm.general_plugins` entry point. If
your deployment filters plugins with `VLLM_PLUGINS`, include both `diffspec`
and `ascend`.

---

## Quick Start

With sibling `vllm-hust-dev-hub`, the repository manifest handles installation,
entry-point validation, safe defaults, and argument injection:

```bash
cd ../vllm-hust-dev-hub
./manage.sh restart --optimization diffspec \
  --draft-model /path/to/eagle3-draft-model
```

The equivalent low-level invocation is documented below for standalone use.

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve /path/to/target-model \
  --dtype bfloat16 \
  --max-num-seqs 1 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --speculative-config '{
    "method": "eagle3",
    "model": "/path/to/eagle3-draft-model",
    "num_speculative_tokens": 5,
    "enforce_eager": true,
    "draft_context_policy": "diffspec",
    "diffspec_verification_mode": "auto",
    "diffspec_chunk_size": 64,
    "diffspec_token_budget": 2048,
    "diffspec_retrieval_interval": 4,
    "diffspec_max_tree_nodes": 50,
    "diffspec_tree_threshold": 0.75,
    "diffspec_adaptive_profile": true,
    "diffspec_long_context_threshold": 49152,
    "diffspec_long_context_depth": 2
  }'
```

## Configuration

The following keys are added to vLLM's `--speculative-config` JSON object:

| Configuration Key | Default | Description |
|---|---:|---|
| `draft_context_policy` | `full` | Set to `diffspec` to enable the plugin |
| `diffspec_verification_mode` | `auto` | Verification mode: `auto` |
| `diffspec_chunk_size` | `64` | Tokens per retrieval chunk |
| `diffspec_token_budget` | `2048` | Maximum compact draft context |
| `diffspec_retrieval_interval` | `4` | Initial verifier cycles between retrievals |
| `diffspec_max_tree_nodes` | `50` | Maximum verifier nodes in tree mode |
| `diffspec_tree_threshold` | `0.75` | Candidate-tree probability threshold |
| `diffspec_adaptive_profile` | `true` | Enable online depth/profile adaptation |
| `diffspec_long_context_threshold` | `49152` | Sequence length that activates the shallow depth floor |
| `diffspec_long_context_depth` | `2` | Initial depth at or above the long-context threshold |


---

## Paper Reference

```bibtex
@inproceedings{diffspec2026,
  title     = {DiffSpec: Accelerating Long Sequence Generation with Differential Speculative Decoding},
  booktitle = {Proceedings of the International Conference for High
               Performance Computing, Networking, Storage, and Analysis (SC)},
  year      = {2026},
}
```

## License

Apache-2.0
