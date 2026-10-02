# 03 — Apple Silicon setup: what we installed and how to not melt the Mac

Everything below was verified by running it on this machine, not copied from a guide.

---

## 1. The three ways to run vLLM on a Mac

| Path | Status | Verdict |
| --- | --- | --- |
| **vLLM Metal** (`vllm-metal`) | Community/vLLM-project plugin; upstream vLLM engine + MLX compute + a Metal paged-attention kernel | ✅ **What this project uses.** GPU-accelerated, no compiler, one brew command |
| **Native macOS CPU build** | Official but *experimental*: "vLLM has experimental support for macOS with Apple Silicon… users must build from source." FP32/FP16 only, **no prebuilt wheels** | Viable fallback; slow and needs a full source build |
| **Docker Linux ARM64** | Official prebuilt image `vllm/vllm-openai-cpu:latest-arm64` | Works, but needs colima/Docker first |

**What is *not* true anymore:** "vLLM is Linux-only." The PyPI wheels are indeed
`manylinux`-only (vLLM 0.30.0 publishes `x86_64` and `aarch64` Linux wheels and no macOS
wheel), which is why plain `pip install vllm` fails here. But that is a packaging fact, not a
capability fact.

---

## 2. What is installed

```
vLLM        0.29.0+cpu     (Homebrew formula bundles the vLLM version; PyPI latest is 0.30.0,
                            so the plugin tracks slightly behind)
vllm-metal  0.29.0
mlx_lm      0.32.0
mlx         installed
torch       2.13.0
transformers 5.17.0
Python      3.12.14
platform    MetalPlatform   (reports device_type = cpu; MLX device = gpu; PyTorch device = mps)
```

Install command (one time):

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

No compiler is needed — the formula installs a prebuilt
`vllm-0.29.0+cpu-cp312-cp312-macosx_11_0_arm64.whl` into a private virtualenv.

### The interpreter gotcha

The formula installs vLLM into **Homebrew's private venv** at
`/opt/homebrew/opt/vllm-metal/libexec`. The `vllm` CLI works directly, but your shell's
`python3` **cannot** `import vllm`. That is why this project has `scripts/py.sh`:

```bash
bash scripts/py.sh labs/01_offline_inference.py     # correct
python labs/01_offline_inference.py                 # ModuleNotFoundError: No module named 'vllm'
```

`scripts/py.sh` resolves the interpreter (and honours `VLLM_PY` if you have your own env).

---

## 3. Memory: the thing that will bite you

**vLLM allocates its entire paged KV cache at startup**, sized from
`--gpu-memory-utilization` (a fraction). The Metal plugin's own default is **0.92**.

Measured on this 24 GB M4 Air:

| `--gpu-memory-utilization` | KV budget | blocks | KV tokens | concurrency @4096 |
| --- | --- | --- | --- | --- |
| **0.92** (plugin default) | **15.74 GB** | 8575 | 137,200 | 16.75× |
| 0.45 | ~6.9 GB | ~3780 | ~60,500 | ~14.8× |
| **0.25** (project default) | **2.95 GB** | 1610 | 25,760 | 6.29× |

The startup line to read:

```
Paged attention memory breakdown: metal_limit=19.07GB, fraction=0.25,
  usable_metal=4.77GB, model_memory=1.19GB, overhead=0.62GB,
  kv_budget=2.95GB, per_block_bytes=1835008, num_blocks=1610, max_tokens_cached=25760
```

Key insight: **the model is only 1.19 GB.** Nearly all the memory is the KV cache, and it is
reserved whether or not you use it. At the default 0.92 that is ~15.7 GB on a laptop, which is
how our first server launch made the machine crawl.

`scripts/serve.sh` and `labs/01_offline_inference.py` both default to `FRACTION=0.25`, and both
accept an override:

```bash
FRACTION=0.4 bash scripts/serve.sh          # more concurrency
FRACTION=0.15 bash scripts/serve.sh         # minimal footprint
MAX_MODEL_LEN=2048 bash scripts/serve.sh    # also cuts KV per request
FRACTION=0.25 bash scripts/py.sh labs/01_offline_inference.py
```

Note `--max-model-len` does **not** change total KV capacity in bytes — it changes how many
concurrent full-length requests fit. Lower it to raise concurrency.

### Measured footprint

At `FRACTION=0.25` the whole thing is about **3 GB RSS** (API server ~1.05 GB + EngineCore
~1.98 GB), with ~80 % of system memory free.

---

## 4. Known cosmetic warnings (ignore these)

| Message | Meaning |
| --- | --- |
| `Triton not installed or not compatible; certain GPU-related functions will not be available.` | Expected on Metal — Triton is a CUDA thing. The plugin uses its own Metal kernels. |
| `Set Metal wired_limit to 17.8 GB` | The plugin raising the Metal wired limit. Independent of the KV fraction; it is a ceiling, not an allocation. |
| `Found ulimit of 2048 ... Too many open files` | Harmless at this scale. Raise with `ulimit -n 8192` if you hit fd errors. |
| `Default vLLM sampling parameters have been overridden by the model's generation_config.json` | Qwen3 ships its own defaults. Use `--generation-config vllm` to ignore them. |
| **`!!!!!!! Segfault encountered !!!!!!!` at shutdown** | **Cosmetic.** It happens during interpreter teardown (`at::accelerator::emptyHostCache`) *after* all results are produced and the worker has already logged `Metal worker shutdown complete`. Observed on both `vllm serve` shutdown and offline `LLM()` exit. Results are valid; the process exit code is still 0 for normal runs. |

---

## 5. Process architecture, as observed

On a single-device Mac you see **two** processes, not the GPU topology from
`docs/01-architecture.md`:

```
APIServer  pid=...  (1.05 GB)   HTTP, tokenization, detokenization, streaming
EngineCore pid=...  (1.98 GB)   scheduler, KV cache, MLX/Metal forward pass
```

There is no separate per-GPU worker process here; the plugin runs the model in the engine core.
Compare with the `A + DP + N` table in the vLLM architecture doc — on a single device that
formula degenerates to 1 + 1.

`world_size=1`, `tensor_parallel_size=1`, `enable_chunked_prefill=True`,
`enable_prefix_caching=True` — the last two confirm that V1's zero-config defaults are active
on Metal too.

---

## 6. Measured performance (M4, 24 GB, Qwen3-0.6B, FRACTION=0.25)

| Measurement | Value |
| --- | --- |
| Model load, first run (incl. download) | 54.9 s |
| Model load, cached | 0.7 s |
| Engine init (profile + KV + warmup) | 1.1 s |
| TTFT, short prompt | 57 ms |
| Offline batched generate, 3 prompts | 181.9 tok/s |
| Decode, 14 concurrent requests | 228.1 tok/s |
| Sequential 6 requests | 43.9 tok/s |
| Concurrent 6 requests | 140.7 tok/s (3.33× speedup) |
| Prefix caching TTFT gain | **11.9×** (569 ms → 48 ms) |

These are laptop CPU/GPU-unified numbers, not GPU-server numbers. Treat them as *directions*
(concurrency helps; prefix caching helps enormously), not as benchmarks.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'vllm'` | You used the system Python. Use `bash scripts/py.sh`. |
| Machine becomes unresponsive | KV cache too large. Restart with a smaller `FRACTION` (`0.15`–`0.25`). |
| `No available memory for the cache blocks` | `FRACTION` too low for the model + context. Raise it or lower `MAX_MODEL_LEN`. |
| Port already in use | `PORT=8001 bash scripts/serve.sh`, or kill the old server. |
| Slow first server start | Model download from HF Hub. Set `HF_TOKEN` for higher rate limits. |
| `no kernel image is available` / CUDA errors | You are on the wrong backend; this project uses Metal. |

Confirm the environment any time with:

```bash
bash labs/00_verify_install.sh
```

### Uninstall

```bash
brew uninstall vllm-metal
```

Model weights stay in `~/.cache/huggingface` unless you delete them.

---

## 8. The build-from-source path (optional, separate environment)

The official experimental macOS CPU build. Keep it in a **separate** env so it never clobbers
the Metal install:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv venv --python 3.12
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

Requirements: macOS Sonoma+, Xcode Command Line Tools, Apple Clang ≥ 15.0.0
(this machine has 21.0.0). `VLLM_TARGET_DEVICE` is forced to `cpu`; only FP32/FP16 are
supported; expect a multi-minute compile and much slower inference. vLLM's own `AGENTS.md`
mandates `uv` and a `.venv` for all Python work in that repo.
