# vLLM Learning Project — Linux / GPU edition

A hands-on, source-grounded path into **vLLM**: what it is, how the engine works, how to run and
tune it, and how to change it. You run everything on a rented Linux GPU (RunPod), because that is
the machine vLLM is actually built for.

This is a learning lab, not a library. Every claim about vLLM in these docs was checked against
the **v0.30.0 source tree** and the current official docs on 2026-10-02. The labs are meant to be
*run by you* and their numbers *recorded by you* — see
[How this project stays honest](#how-this-project-stays-honest).

**New here? Read in this order:** this file → [`docs/03-runpod-setup.md`](docs/03-runpod-setup.md)
(rent the pod) → [`CURRICULUM.md`](CURRICULUM.md) (the staged path) → `labs/00`.

---

## The 20-minute version

```bash
# 1. Rent the pod. The exact console recipe (image, disks, ports) is in docs/03-runpod-setup.md.

# 2. Put this guide on it — rsync over direct SSH, or git clone your own copy:
rsync -avz -e "ssh -p <mapped-port>" ~/Desktop/vlearning/ root@<public-ip>:/workspace/vlearning/

# 3. On the pod. One command: uv + venv + vLLM + the vLLM v0.30.0 source + model prefetch
#    + a GPU sanity check. It also puts the venv, the interpreter and the model cache on
#    /workspace, because RunPod clears the container disk every time a pod stops.
bash /workspace/vlearning/scripts/bootstrap.sh

# 4. Start the server (leave it running)...
bash /workspace/vlearning/scripts/serve.sh

# 5. ...and in a second terminal, prove the whole stack works:
bash /workspace/vlearning/labs/00_verify_install.sh
```

Then work through `CURRICULUM.md` stage by stage. Stage 0 is "get a token out of a GPU". It gets
harder from there, deliberately.

---

## Why a staged curriculum instead of just reading the docs

vLLM's documentation is thorough but organised by *feature*, not by *dependency*. Read it
alphabetically and you meet speculative decoding before you know what a KV block is.

This project orders the material by dependency and grounds each concept three ways:

1. **Docs** — a link to the current official page for the topic.
2. **Source** — the exact file in the vLLM checkout (`$VLLM_SRC/`) that implements it. The code is
   the source of truth; the docs sometimes lag it.
3. **A runnable lab** — because a concept you have not measured is a concept you do not have.

Start at [`CURRICULUM.md`](CURRICULUM.md).

---

## Verified facts (checked 2026-10-02, vLLM v0.30.0)

| Question | Answer |
| --- | --- |
| What is the current release? | **v0.30.0**, published 2026-09-22. Prior: 0.29.0 (Sep 9), 0.28.0 (Aug 26). There is no 0.31/0.32/1.x line yet — "v0.32" appears only as the release targeted to *remove* Model Runner V1. |
| Which wheels exist? | PyPI ships `cp38-abi3 manylinux_2_28` for `x86_64` and `aarch64` — **Linux only**, so `pip install vllm` on macOS has nothing to fetch. GitHub release assets add CUDA 12.8/12.9/13.0, CPU and XPU builds. |
| Which CUDA am I on? | **The PyPI wheel for v0.30.0 is a CUDA 13.0 build** — its metadata requires `nvidia-cutlass-dsl[cu13]`, and installing it with `--torch-backend=cu130` resolves `torch 2.13.0+cu130` + `nvidia-cuda-runtime 13.0.96`. So the pod uses a **cu130 image** (`runpod/pytorch:1.4.0-cu1300-…`) and `--torch-backend=cu130`. CUDA 13 needs a host driver **R580+**, and a consumer 4090 cannot use CUDA forward-compatibility mode, so the driver must be genuinely new enough. vLLM's in-tree docs still say "compiled with CUDA 12.9 by default" — that describes the nightly/source default and is **stale for the 0.30.0 PyPI wheel**. A CUDA 12.9 wheel does exist as a GitHub release asset if you are stuck on an older driver. |
| Does vLLM run on a Mac? | Only via community plugins (`vllm-metal`/MLX) or an experimental CPU build from source. Retired here — see [`legacy/mac-metal/`](legacy/mac-metal/README.md). |
| What is V1? | **The engine architecture generation, not a version number.** V0 is fully deprecated and removed; V1 is the only engine. It lives under `vllm/v1/`, which is why nearly every source path in these docs contains `v1/`. |
| What is MRV2? | **Model Runner V2** — the model-execution layer *inside* V1, default for all models since v0.29.0 (`docs/design/model_runner_v2.md`). It is not a "V2 engine". MRV1 is deprecated and slated for removal in v0.32, still used as a fallback for a few features (sequence parallelism, dual-batch overlap, elastic EP, custom logits processors, some spec-decode methods). |
| Is prefix caching on by default? | **Yes** — `CacheConfig.enable_prefix_caching = True` (`vllm/config/cache.py`). On older versions it was opt-in, so older guides tell you to pass `--enable-prefix-caching`. To *experiment* you now pass `--no-enable-prefix-caching`. |
| Is chunked prefill on by default? | **Yes** — `SchedulerConfig.enable_chunked_prefill = True`. |
| Default `--gpu-memory-utilization`? | **0.92** (`vllm/config/cache.py`). |
| Default optimization level? | **`-O2`** = `torch.compile` + FULL_AND_PIECEWISE CUDA graphs. `-O0` disables compilation, `-O1` is piecewise only, `--enforce-eager` also disables it. |
| Default `--max-num-batched-tokens`? | Depends on the **GPU and the entrypoint**. On a **< 70 GB card like the 4090**: `LLM()` offline → **8192**, but `vllm serve` → **2048**. On H100/H200-class (≥ 70 GB): 16384 offline / 8192 serve. This asymmetry explains a lot of "why is my server slow?" — see Stage 5. |
| Default `--max-num-seqs`? | **256** below 70 GB (and on A100); **1024** at ≥ 70 GB. `--performance-mode throughput` doubles **both** batch budgets — each only if you left it at the default — and then clamps `max_num_seqs` to `max_num_batched_tokens` (`vllm/engine/arg_utils.py`). |
| What does the KV-cache startup line look like? | **One merged line** now: `GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 8,192 tokens per request: 150.70x` (`vllm/v1/core/kv_cache_utils.py`). Older guides quote two separate lines. |
| Which models should I use? | The non-gated **Qwen3** family: `Qwen3-0.6B` (fast loop), `Qwen3-4B` (comfortable), `Qwen3-8B` (realistic for 24 GB), `Qwen3-30B-A3B` (30B MoE — needs 48 GB+). `meta-llama/Llama-3.1-8B-Instruct` is **gated** (license + token), so it is not the teaching default. |

### The machine you are renting

```
GPU            1× RTX 4090, 24 GB GDDR6X           — Ada (SM 8.9), so FP8 W8A8 works
Precision      bf16 / fp16, fp8 weights, fp8 KV cache
Interconnect   PCIe only (no NVLink) — matters for the multi-GPU stage
Storage        /workspace volume (persists) + container disk (wiped when stopped)
Cost           ~$0.34/hr Community · ~$0.74/hr Secure, plus storage — see the setup doc
```

What fits in 24 GB, using bf16 weights (2 bytes/param) and this KV formula:

```
KV bytes/token = 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

| Model | Params | Weights (bf16) | KV / token | Verdict on 24 GB |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | 0.75 B | 1.4 GB | 112 KiB | Instant. Use it for every iteration. |
| `Qwen/Qwen3-4B` | 4.02 B | 7.5 GB | 144 KiB | Comfortable, lots of KV headroom. |
| `Qwen/Qwen3-8B` | 8.19 B | 15.3 GB on disk · ~16.4 GB in VRAM | 144 KiB | Works: ≈4.8 GB left for KV ≈ 35k cached tokens at `--gpu-memory-utilization 0.90`. |
| `Qwen/Qwen3-30B-A3B` | 30.5 B | ~61 GB | 96 KiB (48 fp8) | **No.** Needs 80 GB, or 2×48 GB with `-tp=2`. Its FP8 build (~30 GB) needs a 48 GB card, or 2×24 GB with `-tp=2`. |

(Why two numbers for the 8B: the sum of its `safetensors` files is 15.3 GB, but 8.19 B parameters at
2 bytes each is 16.4 GB — the difference is shared/tied tensors plus GB-vs-GiB. **Size your card
from the VRAM figure**, and remember the engine's own startup line is the final authority.)

(Weights are measured file sizes from the HuggingFace API; KV/token is computed from each model's
`config.json` — 36 layers × 8 KV heads × 128 head_dim × 2 × 2 bytes for the 4B and 8B, 28 layers for
the 0.6B. The whole ladder is 24.2 GB on disk. Note the ratio: 8 KV heads for 32 query heads is
**GQA**, which is why the cache is 4× smaller than a non-grouped model would need — this is the
single biggest reason modern models are servable at all.)

Do not take the table's word for it — Lab 04 makes you derive the real numbers from the startup
log of the card you actually rented.

---

## Layout

```
vlearning/
├── README.md                         ← you are here
├── CURRICULUM.md                     ← the staged learning path (start here after this file)
├── docs/
│   ├── 00-orientation.md             what vLLM is and the problem it solves
│   ├── 01-architecture.md            V1 processes, classes, MRV2, and where each lives in source
│   ├── 02-paged-attention.md         PagedAttention, block pool, KV manager, prefix caching
│   ├── 03-runpod-setup.md            rent the pod: image, storage, SSH, VS Code, cost table
│   ├── 04-tuning-and-compilation.md  memory knobs, batch budgets, -O levels, CUDA graphs
│   ├── 05-metrics-and-benchmarking.md  /metrics, vllm bench, finding the knee
│   ├── 06-scaling.md                 TP/PP/DP/EP, disaggregation, spec decode, KV offloading
│   └── 07-contributing.md            build from source, run the test suite, land a PR
├── labs/                             runnable, numbered experiments (00 → 11)
├── scripts/
│   ├── env.sh                        shared paths + model ladder (sourced by the others)
│   ├── bootstrap.sh                  one-shot pod setup: uv, venv, vLLM, source, prefetch
│   ├── prefetch.sh                   warm the model cache before lab time
│   ├── serve.sh                      launch the OpenAI server with GPU-sane defaults
│   └── check-guide.sh                verify this guide against a real vLLM checkout
├── notes/                            your own scratch space (empty on purpose)
└── legacy/mac-metal/                 the retired Apple Silicon edition, kept for reference
```

---

## Conventions used everywhere in this project

**Paths.** Source references are written relative to `$VLLM_SRC`, which `scripts/env.sh` sets to
`/workspace/src/vllm` on the pod — a shallow clone of upstream at tag `v0.30.0`. Your contribution
fork at `~/Desktop/vllm` is a valid `$VLLM_SRC` too (it tracks `main`, slightly ahead of the tag).
Every `vllm/...` path in these docs was checked against that tree.

**Model ladder.** Labs take a model from the ladder defined in `scripts/env.sh`
(`MODEL_TINY` → `MODEL_MID` → `MODEL_BIG`), so any lab can run in 30 seconds on the tiny model and
be repeated on a realistic one when the numbers matter.

**Running labs.**

```bash
source scripts/env.sh                  # sets $VLLM_SRC, $HF_HOME, $MODEL_*, activates the venv
bash labs/00_verify_install.sh         # shell labs
python labs/01_offline_inference.py    # python labs (venv already active)
bash scripts/serve.sh                  # the server, in its own terminal
```

**Records.** Every lab has a `RECORD:` block. Paste the numbers you measured into `notes/`.
A lab you ran without recording is a lab you will misremember.

---

## How to use the docs

Two URL roots, and the difference matters:

- `https://docs.vllm.ai/en/stable/…` — the latest **stable release**. Use this one.
- `https://docs.vllm.ai/en/latest/…` — **developer preview of `main`**; may describe unreleased code.

vLLM's own design docs also ship inside the source tree you cloned, so read them locally, in the
same checkout as the code they describe:

- `$VLLM_SRC/docs/design/arch_overview.md`
- `$VLLM_SRC/docs/design/paged_attention.md`
- `$VLLM_SRC/docs/design/prefix_caching.md`
- `$VLLM_SRC/docs/design/model_runner_v2.md`
- `$VLLM_SRC/docs/design/optimization_levels.md`
- `$VLLM_SRC/docs/usage/v1_guide.md`

---

## How this project stays honest

- **vLLM facts** (versions, flags, defaults, source paths, log formats) were verified on
  2026-10-02 against a real checkout of vLLM at `v0.30.0`/`main` and against `docs.vllm.ai`.
  Where a fact is version-sensitive, the version is stated inline.
- **RunPod facts** (GPU prices, image tags, storage rules, ports) were verified against RunPod's
  docs and Docker Hub tag lists on the same date. These change faster than vLLM's do — re-check
  the price before a long session.
- **Lab outputs are predictions, not recordings.** The labs were written and reviewed against the
  v0.30.0 source, but they were not executed on a GPU while this guide was written. Where a lab
  says "expect roughly X", that X is arithmetic from a model config or from the engine's own
  source — your run is the real data. If a lab contradicts a doc, **the lab wins** and the doc
  should be fixed.
- vLLM ships roughly every two weeks. If you are reading this more than a month after
  2026-10-02, run the upgrade check in Stage 11 before trusting any number.
- **The guide can check itself.** `bash scripts/check-guide.sh` verifies every `vllm/...` source
  path cited anywhere in these docs against a real checkout, confirms every relative link
  resolves, and cross-checks the labs against `CURRICULUM.md`. Run it after each vLLM upgrade; a
  failure means upstream moved something, which is exactly what you want to know. Add
  `CHECK_URLS=1` to also HTTP-check every `docs.vllm.ai` link.

---

## Ground rules

- One stage at a time. Each stage ends with a checkpoint — if you cannot answer it out loud, the
  stage is not done.
- Every stage has a **measurable experiment**. Record the number so you can compare later.
- Prefer "read the source, then verify by running" over "read a blog post".
- Always write down the **date and version** next to any fact you record about vLLM.
- Stop the pod when you stop working. The meter runs whether or not you are looking at it.

---

## Where to go next

- [`CURRICULUM.md`](CURRICULUM.md) — Stage 0 → Stage 11, easy to hard.
- vLLM docs: <https://docs.vllm.ai/en/stable/>
- The post that makes the whole design click: [Inside vLLM: Anatomy of a High-Throughput LLM
  Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm).
