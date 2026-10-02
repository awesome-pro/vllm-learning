# vLLM Learning Project

A hands-on, source-grounded path into **vLLM** — what it is, how its engine works, and how to
run and tune it — built specifically for an Apple Silicon Mac.

This is a learning lab, not a library. Every claim about vLLM here was checked against the
source tree in `vendor/vllm/` and the current official docs, and every lab in `labs/` was
actually executed on this machine.

---

## Why a staged curriculum instead of just reading the docs

vLLM's documentation is thorough but organised by *feature*, not by *dependency*. Read it
alphabetically and you meet speculative decoding before you understand what a KV block is.

This project orders the material by dependency and grounds each concept three ways:

1. **Docs** — a link to the current official page for the topic.
2. **Source** — the exact file in `vendor/vllm/` that implements it. The code is the source of
   truth; the docs sometimes lag it.
3. **A runnable lab** — because a concept you have not measured is a concept you do not have.

Start at [`CURRICULUM.md`](CURRICULUM.md).

---

## Verified facts (checked 2026, vLLM 0.30.0 released / 0.29.0 installed)

| Question | Answer |
| --- | --- |
| Is vLLM Linux-only? | **The wheels are.** vLLM 0.30.0 on PyPI ships only `manylinux_2_28` for `x86_64` and `aarch64`. `pip install vllm` on macOS has no wheel to fetch. |
| Does vLLM run on macOS at all? | **Yes, two ways.** (a) Native CPU: *"experimental support for macOS with Apple Silicon… users must build from source"* — FP32/FP16 only. (b) **vLLM Metal**: a plugin giving GPU-accelerated inference via MLX. |
| Do I need to rent a GPU? | **No.** The engine, scheduler, paged KV cache and OpenAI API are identical on Metal/CPU. A cloud GPU is only needed for CUDA kernels, multi-GPU parallelism, FP8/most quantization, and production-representative benchmarks. |
| What backend does this project use? | **vLLM Metal** (`vllm-metal`, MLX/Metal), chosen for usable speed with zero compiler setup. |
| What is actually installed here? | `vllm 0.29.0+cpu` + `vllm-metal 0.29.0`, Python 3.12.14, `mlx_lm 0.32.0`, `torch 2.13.0`, platform `MetalPlatform`. The Homebrew formula bundles vLLM **0.29.0** while PyPI's latest is 0.30.0 — the plugin tracks slightly behind. |

### What does "V1" mean? (it is not a version number)

This trips up almost everyone. **`V1` is the name of vLLM's engine architecture generation**,
not a package release. It sits alongside the version, and both are visible in the same startup
log line:

```
Initializing a V1 LLM engine (v0.29.0) with config: ...
                 ^^ engine generation    ^^ package version
```

- **`0.29.0`** — the release you installed. Changes weekly.
- **`V1`** — the engine design. vLLM was rebuilt around a new engine (V1) and the old one (V0)
  was retired.

So "V1" is not something you opted into, and not something your environment got wrong. Since
well before 0.29, V1 is the *only* engine. You can see this in the installed package — the
legacy import path is now literally an alias to the V1 engine:

```python
# site-packages/vllm/engine/llm_engine.py  (vLLM 0.29.0)
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

LLMEngine = V1LLMEngine   # the old name now points at V1
```

And vLLM's own guide states it plainly: *"We have fully deprecated V0."*
(`vendor/vllm/docs/usage/v1_guide.md`)

**Why this project says "V1" so often:** because the architecture doc, and every source path in
it (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/engine/core.py`, …), live under a `v1/`
directory. That is just where the code is. If you ever see "V0", it means the *old* design —
useful only for understanding why V1 removed things like GPU↔CPU KV cache swapping.

### This machine

```
Chip            Apple M4 (10 cores), 24 GB unified memory
OS              macOS 26.6.2 (Tahoe)          — vllm-metal needs macOS 15+
Toolchain       Apple clang 21.0.0, uv 0.11.26, Homebrew 7.0.4
Disk free       ~335 GB
```

---

## Layout

```
vllm-learning/
├── README.md                    ← you are here
├── CURRICULUM.md                ← the staged learning path (start here after this file)
├── docs/
│   ├── 00-orientation.md        what vLLM is and the problem it solves
│   ├── 01-architecture.md       V1 processes, classes, and where each lives in source
│   ├── 02-paged-attention.md    PagedAttention, block pool, KV manager, prefix caching
│   └── 03-apple-silicon-setup.md how the Mac backend works, memory sizing, troubleshooting
├── labs/                        runnable, verified experiments
├── scripts/                     serve + interpreter helpers
├── notes/                       your own scratch space (empty on purpose)
└── vendor/vllm/                 shallow clone of vllm-project/vllm — read only
```

`vendor/vllm/` is a **read-only reference checkout**. It carries its own `AGENTS.md` with
contribution rules; we are not contributing upstream from this project, so nothing here should
be edited. Treat it as documentation you can navigate.

---

## Quickstart

> **Memory warning.** vLLM allocates its whole paged KV cache at startup, sized by
> `--gpu-memory-utilization`. The Metal plugin defaults to `0.92`, which reserves **~15.7 GB**
> on a 24 GB Mac — enough to make it crawl. `scripts/serve.sh` and `labs/01` override this to
> `0.25` (~2.95 GB KV, 25,760 cached tokens, ~3 GB total RSS). Raise it for more concurrency:
> `FRACTION=0.4 bash scripts/serve.sh`. Full breakdown in
> [`docs/03-apple-silicon-setup.md`](docs/03-apple-silicon-setup.md).

> **Interpreter note.** The Homebrew formula installs vLLM into a *private* virtualenv. Your
> shell's `python3` cannot see it. Use `bash scripts/py.sh <script.py>` for the Python labs;
> the `vllm` CLI works directly.

```bash
# 1. Check the backend is installed and see what we are actually running
bash labs/00_verify_install.sh

# 2. Your first generation, offline (no server)
bash scripts/py.sh labs/01_offline_inference.py

# 3. A real OpenAI-compatible server (in another shell, leave it running)
bash scripts/serve.sh

# 4. ...then drive it from a client
bash scripts/py.sh labs/02_openai_client.py
bash scripts/py.sh labs/04_kv_cache_memory.py
```

The lab scripts only need the standard library, but they must run under the vLLM interpreter,
hence `scripts/py.sh`.

Then work through [`CURRICULUM.md`](CURRICULUM.md) in order — it tells you what to read, what
to run, and what to be able to explain before moving on.

---

## How to use the docs

Docs live at two URLs, and knowing which is which matters:

- `https://docs.vllm.ai/en/stable/…` — the latest **stable release**
- `https://docs.vllm.ai/en/latest/…` — **developer preview** (may describe unreleased code)

The nav on those pages is enormous when scraped, so the *readable* copies of vLLM's own design
docs are in `vendor/vllm/docs/`, e.g.:

- `vendor/vllm/docs/design/arch_overview.md`
- `vendor/vllm/docs/design/paged_attention.md`
- `vendor/vllm/docs/design/prefix_caching.md`
- `vendor/vllm/docs/usage/v1_guide.md`

Read those locally; they render cleanly and match the code in the same checkout.

---

## Ground rules

- One stage at a time. Each stage has a checkpoint — if you cannot answer it, the stage is not done.
- Every stage has a **measurable experiment**: record the number so you can compare later.
- Prefer "read the source, then verify by running" over "read a blog post".
- Note the date and version whenever you write down a fact about vLLM. It moves fast.
