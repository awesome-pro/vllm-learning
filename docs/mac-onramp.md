# Your Mac on-ramp

*Do this before you rent anything. Your Mac runs the **same vLLM engine**, the same labs and the
same flags as the pod — with MLX doing the arithmetic on the Metal GPU.*

---

## 1. Why this is real vLLM, not a lookalike

From the [vLLM blog announcement](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0):

> vllm-metal brings vLLM's **scheduler, paged KV cache, and OpenAI-compatible server** to Apple
> Silicon, with **MLX and Metal handling execution**. […] vllm-metal replaces stock attention with a
> paged varlen Metal kernel.

So the V1 scheduler, the paged KV blocks, chunked prefill, prefix caching, the OpenAI server and the
`vllm` CLI are all the same code you will run on the pod. What differs is the compute backend — a
Metal attention kernel instead of CUDA — and therefore every performance number, plus the CUDA-only
features listed in §7.

Verified on this machine (M4, 26 GB, macOS 26.6.2):

| Component | Version |
| --- | --- |
| vLLM | **0.30.0+cpu** — the same release the pod runs |
| vllm-metal | 0.30.0 |
| mlx / mlx-lm | 0.32.1 / 0.32.0 |
| platform | `MetalPlatform` (reports `device_type=cpu`; the work still runs on the Metal GPU) |

---

## 2. Install (once, about 4 minutes)

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash -s -- --stable
```

It bootstraps `uv`, fetches Python 3.12 and prebuilt wheels (nothing is compiled), and installs
everything into `~/.venv-vllm-metal`. No Homebrew formula, no Xcode, no compiler.

Requirements it checks for you: **macOS 15+** and a native **arm64** Python. Uninstall is
`rm -rf ~/.venv-vllm-metal` — one directory, nothing else touched.

> Prefer Homebrew? `brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal`
> then `brew install vllm-project/vllm-metal/vllm-metal` also works. It installs into Homebrew's
> private venv instead, which is why this guide uses the official installer: one directory, and
> `scripts/env.sh` can find it.

---

## 3. There is nothing else to configure

`source scripts/env.sh` detects the platform and picks sane values for each. This is the only place
the two machines differ:

| Variable | Pod | Your Mac | Why |
| --- | --- | --- | --- |
| `VENV` | `/workspace/venv` | `~/.venv-vllm-metal` | where vllm-metal installs |
| `HF_HOME` | `/workspace/hf` | `~/.cache/huggingface` | reuses model weights you already have |
| `UTIL` | `0.90` | **`0.30`** | **unified memory** — see below |
| `MAXLEN` | `8192` | `4096` | KV reserved per request |
| `HOST` | `0.0.0.0` | `127.0.0.1` | no LAN exposure, no macOS firewall prompt |

**Why `UTIL` differs, and why it is the one thing that can hurt.** The pod's 24 GB is dedicated VRAM.
Your Mac's memory is *unified*: vLLM's `--gpu-memory-utilization` is a ceiling on the same pool macOS
and your apps are using, and vLLM reserves the KV cache up front. The pod's 0.90 would ask for about
**23 GB of a 26 GB laptop**. At 0.30 the server took **34,032 KV tokens** and left the machine
responsive — measured, with `memory_pressure` showing 87% free afterwards. Comfortable range is
**0.25–0.40**; go higher only if you are not using the Mac for anything else.

---

## 4. Run the labs — the same ones, in this order

```bash
cd ~/Desktop/vlearning
source scripts/env.sh
python labs/01_offline_inference.py
```

Then, for each server lab, start the server in one terminal and run the lab in another:

```bash
bash scripts/serve.sh                 # terminal 1; Ctrl-C to stop
python labs/02_serve_and_client.py    # terminal 2
```

Measured here, on this Mac, with `Qwen/Qwen3-0.6B` at `UTIL=0.30`, `MAXLEN=4096`:

| Lab | The idea it plants | What it measured on your Mac |
| --- | --- | --- |
| `01_offline_inference.py` | prefill vs decode; the engine batches for you | load 21 s · **214 tok/s** batched · TTFT 0.22 s |
| `02_serve_and_client.py` | TTFT vs inter-token latency | long prompt TTFT **34.6×** the short one, gaps only **1.16×** |
| `03_continuous_batching.py` | why vLLM exists at all | concurrent **6.27×** faster (69 → **433 tok/s**) |
| `04_kv_cache_memory.py` | where the KV arithmetic comes from | predicted **112 KiB/token**, engine reported 34,032 tokens |
| `05_prefix_caching.py` | pay for a prompt once | warm TTFT **2.59×** faster (0.118 s → 0.045 s) |

Those five numbers are the whole of Stages 0–4, and you just measured them on a laptop. The pod
re-measures them on real hardware, which is the point of going there — not the concepts.

**Which models are practical here.** `Qwen/Qwen3-0.6B` is the right default: 1.4 GB of weights leaves
nearly all of a 0.30 budget for KV, and it is a single-file download that the HuggingFace cache may
already hold. `Qwen3-4B` (7.5 GB of weights) is usable if you raise `UTIL` to about 0.5. `Qwen3-8B`
(16.4 GB) is **not** a 0.30 laptop model — it would need `UTIL≈0.75`, leaving macOS and your apps very
little room. Use the pod for the 8B and the MoE; that is precisely what it is for. Weights come from your
normal HF cache (`~/.cache/huggingface`), so check there before assuming a download is needed.

---

## 5. The startup line, and its Mac quirk

```
(EngineCore pid=26105) INFO [kv_cache_utils.py:2395] CPU KV cache size: 34,032 tokens,
  Maximum concurrency for 4,096 tokens per request: 8.31x
```

**It says `CPU`, and that is not a mistake you need to fix.** vllm-metal's platform reports
`device_type=cpu` while MLX executes on the Metal GPU, so the same line that reads `GPU KV cache size`
on the pod reads `CPU KV cache size` here. Everything else about it is identical, including how you
read it: *tokens you can cache*, and the concurrency that buys at your context length.

Also worth seeing, because it matches the pod exactly:

```
Metal: chunked prefill enabled (paged attention), max_num_batched_tokens=2048
Metal memory: 25.8GB total, 14.5GB available
```

`2048` is the same default `vllm serve` uses on the pod's 24 GB card (see `get_batch_defaults()` in
`$VLLM_SRC/vllm/engine/arg_utils.py`) — the same Stage 5 lesson, on your desk.

---

## 6. Warnings to ignore (all observed on this machine)

| You will see | What it means |
| --- | --- |
| **`!!!!!!! Segfault encountered !!!!!!!`** on shutdown | **Cosmetic.** It happens during interpreter teardown *after* the worker logs `Metal worker shutdown complete` and after every result is produced. Observed on both `vllm serve` and offline `LLM()`. Your output is valid. |
| `Triton not installed or not compatible…` | Expected: Triton is a CUDA thing. The plugin uses its own Metal kernels. |
| `objc[…] Class AVFAudioReceiver is implemented in both … ffmpeg … and … cv2/.dylibs …` | Two copies of an ffmpeg dylib (Homebrew's and OpenCV's). Harmless noise from the media dependencies. |
| `Default vLLM sampling parameters have been overridden by the model's generation_config.json` | Qwen3 ships its own sampling defaults. Silence it with `--generation-config vllm`. |
| `Found ulimit of 2048 … Too many open files` | Harmless at this scale; `ulimit -n 8192` in the shell if you ever hit it. |

---

## 7. What does **not** transfer to the pod

Learn these on the Mac, then re-learn them on the pod — they are the reason the pod exists:

- **CUDA graphs and `torch.compile`** (Stages 6): Metal has its own execution path. The `-O2` /
  `--enforce-eager` comparison behaves differently.
- **FP8** (Stage 9): the 4090 is Ada (SM 8.9) and has real FP8 kernels; there is no equivalent here.
- **Tensor parallelism** (Stage 10): one device means `world_size=1`, so `-tp=2` cannot be shown.
- **Process topology** (Stage 1): the pod runs `API server + engine core + one worker per GPU`; on a
  Mac you see **two** processes (`APIServer` and `EngineCore`), because the plugin runs the model
  inside the engine core.
- **Every absolute number.** Treat the table in §4 as *directions* — concurrency helps, prefix caching
  helps enormously — never as a benchmark of the pod.

---

## 8. When to move to the pod

Move when you reach a topic in §7, or when you want to know what the machine actually does. Concretely:
**do Stages 0–5 here, then rent the pod for Stage 6 onward** — and re-run `labs/00` first thing on the
pod, which is what that lab is for.

The point is not to avoid the pod; it is to arrive there already knowing what you expect to see, so
your metered time goes on measuring rather than on reading.
