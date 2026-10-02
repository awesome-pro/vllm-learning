# 00 — Orientation

*Why vLLM exists. Read this before Stage 0 — it is the only doc here with no commands in it.*

---

## 1. The naive server, and why it wastes the machine

The pod is an **RTX 4090, 24 GB** (Ada, SM 8.9). At `--gpu-memory-utilization 0.90` vLLM budgets
21.6 GB for weights, activations and KV. `Qwen/Qwen3-8B` is 8.19 B parameters — **15.3 GB** of bf16
weights — leaving about **4.8 GB** for KV ([README](../README.md); Lab 04 derives it from the startup
log).

**One request at a time.** Every decode pass reads the whole model out of HBM: 15.3 GB. The 4090's
bandwidth is ≈1.0 TB/s *(vendor spec, 384-bit GDDR6X)*, so a lone stream is capped near
15.3 / 1008 ≈ **66 tokens/s** — arithmetic, not a measurement. The tensor cores, capable of orders of
magnitude more work per second, sit idle, because a decode step holds one token's worth of parallel
work. **Single-stream decoding is a bandwidth problem, not a compute problem.**

**Static batching.** Run N requests together and return them together. Weights are read **once per
step regardless of batch size**, so cost per token falls as 1/N:

| Batch | Bytes per decode step | Tokens made | Bytes per token |
| --- | --- | --- | --- |
| 1 | 15.3 GB weights + 0.28 GB KV = **15.6 GB** | 1 | 15.6 GB |
| 32 | 15.3 GB weights + 9.0 GB KV = **24.3 GB** | 32 | **0.76 GB** |

Arithmetic, with every sequence at 2048 tokens of context: 2048 × 144 KiB = 288 MiB of KV each (§3
derives the 144 KiB). That **20×** ratio is the whole argument for batching.

> The column is *bytes moved per decode step*, not *memory required* — and the bottom row is worth
> reading twice. 9.0 GB of KV beside 15.3 GB of weights is **24.3 GB, which does not fit on this
> card**. Batch 32 at 2048 tokens of context is simply not runnable for `Qwen3-8B` on 24 GB. The
> bandwidth argument for batching and the capacity limit on batching bite at different batch sizes,
> and Lab 04 makes you find where the second one lands on the hardware you actually rented.

Two costs remain. **Head-of-line blocking:** the batch runs until its *longest* member finishes, so a
20-token answer waits behind a 2000-token one, holding its KV slots. **Reservation:** a contiguous
`max_model_len` = 8192 buffer costs **1.125 GiB per sequence** at 144 KiB/token, used or not — only
**4 sequences** fit in the 4.8 GB above, even if each needs 300 tokens. The PagedAttention paper
measured 60–80 % of KV memory wasted this way, and KV memory is what bounds concurrency.

## 2. The two founding ideas

**Continuous batching.** Decide **every decode step**, not once per batch: a finished sequence leaves,
a waiting one joins, and its prompt can prefill alongside everyone else's decode. The batch never
drains, so throughput rises; and a new request starts within one step of arriving rather than at a
batch boundary, so latency falls too. Stage 2 makes you measure it.

**PagedAttention.** Store KV in fixed-size **blocks** (16 tokens by default —
`CacheConfig.DEFAULT_BLOCK_SIZE` in `vllm/config/cache.py`) with a per-sequence **block table** mapping
logical block index → physical block id. Non-contiguous blocks mean a sequence grows by attaching one
instead of reallocating and copying, and only the last block is partly wasted: ≤ 15 tokens against
8192. That is virtual memory for attention — and because identical blocks are interchangeable, a
shared prefix can point at the *same* physical blocks (prefix caching) and freeing a sequence is
O(blocks) pointer work (preemption). The two ideas need each other: continuous batching allocates and
frees KV nearly every step.

## 3. Why the KV cache is the binding constraint

Weights are a **fixed** cost paid once at load; compute is a **scheduled** cost you choose per step;
KV is a **growing** cost scaling with concurrency × context, and it runs out first.

```
KV bytes/token = 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

`Qwen/Qwen3-8B`, from its `config.json` — 36 layers, 8 KV heads, head_dim 128, bf16:

```
2 × 36 × 8 × 128 × 2 bytes = 147,456 B = 144 KiB per token
```

A million cached tokens would need ≈ **137 GiB**, more than five 4090s; what fits is ≈ 35k tokens in
the 4.8 GB above, and that is what sizes your deployment. `--kv-cache-dtype fp8` halves bytes/token
(Stage 9). vLLM prints the answer at startup as one merged line (`vllm/v1/core/kv_cache_utils.py`):

```
GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 8,192 tokens per request: 150.70x
```

The first number is the KV you own; the second is that divided by `--max-model-len`. **That line — not
the parameter count — tells you how many concurrent requests you can serve.** Older guides quote two
separate lines; those guides are stale.

## 4. Prefill and decode are different machines

| | **Prefill** | **Decode** |
| --- | --- | --- |
| Does | Processes the prompt, all tokens at once | Generates one token per sequence |
| Work per step | `n` tokens × the whole model | 1 token × the whole model |
| Arithmetic intensity | High — weights reused across `n` tokens | Low — each weight read used once |
| Bound by | **Compute** (FLOPs) | **Memory bandwidth** (HBM) |
| Parallelism | Over the `n` prompt tokens | Over the batch only; the sequence is serial |
| Produces | The KV cache, and the first token | One more token, one more KV entry |

Four things follow:

- **Decode is irreversible.** Token `t+1` needs token `t`, so one sequence's decode cannot be
  parallelised. Batching concurrent sequences is the only lever, and it is a bandwidth lever.
- **Prefill is a purchase** you can split (chunked prefill), cache (prefix caching), or move to
  another machine (disaggregated prefill) — all because prefill is parallel.
- **They compete for the same step**, so a large prefill inflates *everyone's* inter-token latency —
  which is what `--max-num-batched-tokens` resolves (Stage 5).
- **They are two numbers**: TTFT is prefill plus queueing, inter-token latency is decode — hence
  vLLM's separate prompt and generation throughput.

V1 has no separate code path for the two: `Scheduler.schedule()` hands out one token budget across all
requests, prompt tokens and output tokens as the same currency. Its docstring in
`vllm/v1/core/sched/interface.py` says it plainly: "the scheduler produces a dictionary of
`{req_id: num_tokens}`".

## 5. What vLLM is not

- **Not a training framework.** It serves weights produced elsewhere and never updates them.
- **Not a model format.** It loads Hugging Face checkpoints and supports 50+ architectures; the model
  is `Qwen/Qwen3-8B` from the Hub, not a "vLLM model".
- **Not a quantization library.** `--quantization fp8`, `--kv-cache-dtype fp8` and AWQ/GPTQ/GGUF
  checkpoints are precision *knobs*; producing quantized weights is someone else's job.
- **Not a UI or agent framework**, and not CUDA-only: CPU, TPU, Intel XPU, Gaudi and Ascend are also
  supported backends.

## 6. What you will be able to do after this project

By the end of the 12 stages in [CURRICULUM.md](../CURRICULUM.md), without notes and with evidence:

1. Explain and measure continuous batching's throughput/latency tradeoff (Stages 0, 2).
2. Name every process in a running vLLM server and predict the count before starting it (Stage 1).
3. Derive KV bytes/token from a `config.json`, check it against the startup log, and size a deployment
   from it (Stages 3, 4).
4. Say what each scheduler budget bounds, and find the knee past which batching buys nothing (Stage 5).
5. Explain what `-O2` costs at startup (Stage 6); read `/metrics` to tell a saturated engine from a
   starved one (Stage 7).
6. Measure what a structured-output constraint costs (Stage 8) and what FP8 buys and costs (Stage 9).
7. Choose between TP, PP, DP, EP, disaggregation and speculation (Stage 10), then build vLLM from
   source and land a change (Stage 11).

## 7. What this 24 GB card can and cannot show you

**It shows you, for real:** continuous batching; paged KV and prefix caching; the scheduler's two
budgets; `-O2` versus eager; the metrics surface; structured output; FP8 W8A8 and FP8 KV cache (the
4090 is Ada, SM 8.9, so W8A8 is natively supported — an A100 is not); tensor parallelism on a 2×4090
pod.

**It cannot show you:** anything needing more than ≈ 4.8 GB of KV beside its weights
(`Qwen/Qwen3-30B-A3B` is ~61 GB bf16 — out on one card); the ≥ 70 GB batch defaults, so the
16384/1024 figures in the README are *not* observable here; NVLink-scale tensor parallelism (the pod
is PCIe, so `-tp=2` pays full communication cost); expert parallelism; multi-node scale-out.

And **the card's limitation is the lesson**: on a < 70 GB device vLLM gives offline `LLM()` an
8192-token step budget but `vllm serve` only 2048 (`get_batch_defaults()` in
`vllm/engine/arg_utils.py`) — same model, 4× the per-step work, decided by which entrypoint you used
([§5 of the architecture doc](01-architecture.md)). Nothing here was measured on a GPU while it was
written; every number is read out of the v0.30.0 source, computed from a published `config.json`, or
arithmetic shown in full so you can disagree with it.

---

**Next:** [`01-architecture.md`](01-architecture.md) for the V1 processes and classes, then
[`03-runpod-setup.md`](03-runpod-setup.md) to rent the machine.

---

## Checkpoint

✅ Why does a batch of 32 make each token ~20× cheaper in bandwidth than a batch of 1, and what must
hold for that to be true?

✅ What does a block table map, and which founding idea depends on it?

✅ Why is the KV cache — not the weights, not compute — what limits concurrency on a 24 GB card?

✅ Which of prefill and decode is compute-bound and which bandwidth-bound, and what follows for chunked
prefill and for batching?

✅ Name three things vLLM is not.
