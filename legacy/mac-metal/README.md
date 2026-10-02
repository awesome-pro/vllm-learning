# Legacy: the Apple Silicon version of this project

This folder is the **original Mac/Metal edition** of the vLLM learning project, kept for
reference. It is not part of the current path — nothing in `labs/` or `docs/` depends on it.

Why it was retired: the whole point of vLLM is a paged KV cache and a scheduler feeding
CUDA kernels. On a Mac those run through a plugin (`vllm-metal`, MLX/Metal) that reimplements
the attention path, and the CUDA-only machinery — CUDA graphs, `torch.compile` with piecewise
CUDA graphs, FP8 kernels, tensor parallelism, `flash-attn`/FlashInfer backends — is simply
absent. You can learn the *engine* on a Mac, but you cannot learn the *system*, and you cannot
measure anything representative.

The project now runs on a rented Linux GPU (RunPod). See the root [`README.md`](../../README.md)
and [`docs/03-runpod-setup.md`](../../docs/03-runpod-setup.md).

## What is in here

> Note: the archived files below are frozen as they were written for the old layout. Relative links
> *inside* them (e.g. `docs/03-apple-silicon-setup.md`, `vendor/vllm/…`) point at paths that no
> longer exist — that is expected in an archive, and `scripts/check-guide.sh` skips this directory
> for exactly that reason.

| File | What it is |
| --- | --- |
| `README-mac-original.md` | The original README: Mac-specific verified-facts table (vLLM 0.29.0 + `vllm-metal`), the "what is V1" explainer, and the machine profile. |
| `CURRICULUM-mac-original.md` | The original 10-stage curriculum, written for a 24 GB M4 with a ~3 GB KV-cache budget. |
| `03-apple-silicon-setup.md` | How the Metal backend worked: plugin boundary, memory sizing, `FRACTION` budgeting, troubleshooting. |
| `labs-mac-original/` | The original lab suite (`00`–`06`) and its `_client.py`, including the metrics/bench lab whose material now lives in lab 08. |
| `serve-mac.sh` | The Metal-era server launcher (`FRACTION=0.25`, `gpu-memory-utilization` as a *memory ceiling* on unified memory). |
| `py-mac.sh` | Wrapper that found Homebrew's private `vllm-metal` interpreter. |

## If you ever want it back

```bash
brew install vllm-project/vllm-metal/vllm-metal   # ~1.9 GB, brings its own Python 3.12
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.25
```

That formula tracks upstream releases but lags them slightly, and it caps you at FP32/FP16 on
the CPU path or the Metal plugin's supported subset. It was removed from this Mac on
2026-10-02 to reclaim 1.9 GB; the HuggingFace cache was deliberately left alone because
`mini-inference-engine`, `miniserve` and the `~/Desktop/vllm` fork's test-suite all share it.
