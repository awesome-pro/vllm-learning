# 06 — Scale-out: the parts one GPU cannot show you

Stage 10's companion. Everything here was checked against the **v0.30.0** source tree and the current
official docs on **2026-10-02**. Prices are RunPod on-demand Community rates from the same date.

Read this with Stage 10 open. The stage gives you the two-GPU lab; this file tells you what lies
beyond it, what it costs, and which parts your budget will never let you verify. The one-line version:
**parallelism is a memory-for-communication trade.** Every strategy below buys somewhere to put
weights or KV and pays for it in traffic between GPUs, so which one is right is decided by how fast
that traffic is on your hardware. Sections 1–8 are explanations to check against source; §9–11 say
what the 24 GB pod can actually run. Do not rent 2× GPU to read — read here first, then measure.

---

## 1. Tensor parallelism (TP): shard every layer

🔍 `vllm/config/parallel.py` · `vllm/model_executor/layers/linear.py` ·
`vllm/model_executor/models/qwen3.py` · `vllm/distributed/communication_op.py`

TP gives each rank a slice of every layer's weight matrices; all ranks run all layers, on the same
tokens.

| Layer | Class | Split |
| --- | --- | --- |
| QKV projection | `QKVParallelLinear` | Output columns: query heads *and* KV heads divided across ranks |
| Attention output | `RowParallelLinear` | Input rows; an **all-reduce** after the matmul |
| MLP `gate_up_proj` | `MergedColumnParallelLinear` | Output columns |
| MLP `down_proj` | `RowParallelLinear` | Input rows; an **all-reduce** after the matmul |

The two all-reduces per layer are the whole cost of TP: a row-parallel matmul computes only a
*partial* sum, and every rank needs the total before the next layer. vLLM calls
`tensor_model_parallel_all_reduce()` from `RowParallelLinear.forward` when `reduce_results and
tp_size > 1` (`vllm/model_executor/layers/linear.py`).

**The arithmetic, for `Qwen/Qwen3-8B` (36 layers, hidden 4096, 32 query heads, 8 KV heads, head_dim
128, bf16).** Each all-reduce moves `tokens × 4096 × 2 bytes` = 8 KiB per token, and there are 72 of
them per forward pass (two per layer × 36) = **≈ 576 KiB per token**.

| Step | Tokens | All-reduce volume | @ PCIe 4.0 x16 (32 GB/s peak) | @ NVLink (H100, 900 GB/s) |
| --- | --- | --- | --- | --- |
| Single decode | 1 | 0.6 MiB | ~20 µs | ~1 µs |
| Full decode batch | 256 | ~144 MiB | ~4.5 ms | ~0.2 ms |
| One prefill chunk | 2048 | ~1.1 GiB | ~37 ms | ~1.3 ms |

*Prediction, not a measurement* — arithmetic at peak bandwidth with no contention. The ratio is the
lesson: the same collectives cost roughly **25× more over PCIe than over NVLink**. Hence TP stays
inside a node, and hence vLLM's doc prefers pipeline parallelism on cards without NVLink: "if the GPUs
on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of
tensor parallelism for higher throughput and lower communication overhead"
(`$VLLM_SRC/docs/serving/parallelism_scaling.md`). A 2×4090 pod is PCIe only, no NVLink bridge: a
legitimate but not free way to learn TP.

**The KV-head constraint.** TP divides attention *heads*, so the counts must divide: from
`Qwen3Attention.__init__` (`vllm/model_executor/models/qwen3.py`), query heads must divide by
`tp_size`, and KV heads either divide by it (partition) or are a divisor of it (replicate).

For Qwen3-8B that means TP ∈ {1, 2, 4, 8} (32 and 8 both divide); **TP=3 fails** — 32 % 3 ≠ 0 — with
an `AssertionError` in the worker while loading weights, not a friendly CLI error. Above 8, TP is
allowed but each KV head is replicated `tp_size / num_kv_heads` times, so KV memory stops shrinking.
Check any model this way before choosing a TP size. Flag: `--tensor-parallel-size` / `-tp`.

---

## 2. Pipeline parallelism (PP): split the layers into stages

🔍 `vllm/config/parallel.py` · `vllm/distributed/parallel_state.py` (`get_pp_group`) ·
`vllm/v1/worker/gpu/pp_utils.py`

PP cuts the model *along its depth*: rank 0 holds layers 0–17, rank 1 holds 18–35. Only
**activations** cross the link, point-to-point between neighbouring stages — not weight shards, not
partial sums — so traffic per step is `tokens × hidden × 2 bytes` once per stage boundary instead of
72 all-reduces. That is why PP tolerates slower links and extends across nodes:
`$VLLM_SRC/docs/serving/parallelism_scaling.md` recommends TP = GPUs-per-node and PP =
number-of-nodes, and notes PP also handles **uneven splits** (35 layers on 2 GPUs is fine, where TP
must still divide heads).

The cost is the **bubble**. A pipeline has to fill and drain: at the start of a step only the first
stage has work, and the last stage's output arrives later. vLLM makes this concrete — in the MRV2
pipeline handler the sampled-token broadcast from the last rank is consumed **`pp_size` steps later**
(`ring_depth = get_pp_group().world_size`, `vllm/v1/worker/gpu/pp_utils.py`). Micro-batching is what
keeps all stages busy, so with few concurrent requests PP wastes compute exactly when the engine is
idle, and each extra stage adds a serial hop to per-token latency. Hence
`$VLLM_SRC/docs/configuration/optimization.md`: increasing PP "may cause latency penalties".

One PP interaction worth knowing: `--pipeline-parallel-size` with async scheduling disables MRV2 and
falls back to the deprecated MRV1 (`_get_v1_model_runner_unsupported_features`, `vllm/config/vllm.py`),
so check the startup log for that warning if a PP run looks unexpectedly slow
([docs/07-contributing.md](07-contributing.md) has the MRV1/MRV2 tell). Flag: `--pipeline-parallel-size`.

---

## 3. Data parallelism (DP): replicate the engine, shard the requests

🔍 `vllm/v1/engine/coordinator.py` · `vllm/config/parallel.py` · `vllm/v1/engine/core.py`

DP is the simplest idea here and the one to reach for first: run *N complete copies* of the model, each
on its own GPU with its own KV cache, and send different requests to different copies. Nothing is
sharded, so there is no per-layer communication and no bubble, and when the model fits on one card,
throughput scales close to linearly with N until the API server or the network saturates
(`--api-server-count` is the documented escape hatch, `$VLLM_SRC/docs/serving/data_parallel_deployment.md`).

DP's twist is MoE models. Experts are sharded across the DP group (`EP_SIZE = TP_SIZE × DP_SIZE`), so
the ranks are **not** independent: they must run forward passes in lockstep so the all-to-all routing
stays consistent, and a rank with no requests must still run an empty "dummy" forward pass. vLLM
handles that with a separate **`DPCoordinator` process** (`vllm/v1/engine/coordinator.py`), which
collects per-engine queue lengths for API-server load balancing and tracks the global "request wave"
— engines alternate between running and paused, synchronized by an all-reduce in
`DPEngineCoreProc._has_global_unfinished_reqs` (`vllm/v1/engine/core.py`). A dense model needs none of
this, which is the answer to Stage 10's checkpoint question.

Flag: `--data-parallel-size` / `-dp`. `--max-num-seqs` applies *per DP rank*; the admission-control
caps (`--max-num-queued-reqs`) apply to the whole server.

---

## 4. Expert parallelism (EP): shard the experts, pay in all-to-all

🔍 `vllm/distributed/eplb/` · `vllm/v1/worker/gpu/eplb_utils.py` ·
`vllm/model_executor/layers/fused_moe/layer.py`

An MoE layer routes each token to `k` of `E` experts; EP puts different experts on different ranks.
Because routing is data-dependent, every rank must send its tokens to whichever ranks own the chosen
experts and receive results back — an **all-to-all**, not a fixed-pattern collective. Read
`num_experts` and `num_experts_per_tok` from the model's `config.json` first; their ratio is how sparse
the traffic is.

The core problem is **load imbalance**: training-time auxiliary losses only approximately equalise
expert usage and real traffic is skewed, so some ranks get far more tokens than others while the rest
idle. vLLM's answer is **EPLB**, the Expert Parallel Load Balancer: `--enable-eplb` collects per-expert
load statistics every forward pass and periodically re-maps experts across ranks (`--eplb-config`,
defaults `window_size: 1000`, `step_interval: 3000`). It can also place *redundant* copies of popular
experts (`num_redundant_experts`) — the docs give "approximately 2.4 GB for one redundant expert per EP
rank" for DeepSeek-V3, which on a 24 GB card is the whole KV cache.

**Why MoE inference is communication-bound at small batch sizes:** tokens visit only `k` of `E`
experts, so a small batch touches a handful of experts scattered across ranks — tiny payloads with
all-to-all latency attached. More tokens per all-to-all means more work per byte of protocol overhead,
so efficiency rises with batch size. Hence `$VLLM_SRC/docs/serving/expert_parallel_deployment.md`
pairs EP with disaggregation and DeepEP-style kernels. Flags: `--enable-expert-parallel`,
`--all2all-backend`.

---

## 5. Context parallelism (CP): shard the sequence, not the weights

🔍 `vllm/config/parallel.py` (`prefill_context_parallel_size`, `decode_context_parallel_size`) ·
`vllm/v1/worker/gpu/cp_utils.py`

CP splits one request's **token dimension** across ranks: prefill CP splits the `T` new tokens into
`N` chunks so each rank computes a slice of query/key/value, and decode CP splits the stored KV cache
along `T` (interleaved, so future tokens shard naturally). Where TP hits a wall is that attention has
only `H` KV heads: once `tp_size > H`, KV cache is *duplicated* `tp_size / H` times
(`$VLLM_SRC/docs/serving/context_parallel_deployment.md`). CP continues past that wall by sharding
along `T`, with `dcp` bounded by `tp_size / H`.

CP is the only way to serve a very long context when one rank cannot hold the KV, and vLLM's example
is telling: DeepSeek-R1 has **1** KV head with MLA, so `-tp 8` duplicates the KV cache 8×, while
`-dcp 8` removes the duplication. The cost is a ring-style exchange of K/V chunks — more traffic per
step, and it only pays when the context is long enough that the alternative is not serving the request
at all. Flags: `--prefill-context-parallel-size`, `--decode-context-parallel-size` / `-dcp`. Prefill CP
forces MRV2 (MRV1 cannot do it), and `tp_size` must be divisible by `dcp` (`_validate_parallel_config`,
`vllm/config/parallel.py`).

---

## 6. Disaggregated prefill/decode: two engines, one request

📖 <https://docs.vllm.ai/en/stable/features/disagg_prefill/> · 🔍 `vllm/distributed/kv_transfer/` ·
`vllm/config/kv_transfer.py`

Prefill is **compute-bound** (a big matmul over thousands of tokens); decode is **bandwidth-bound**
(one token per sequence per step, dominated by reading weights and KV). In one engine they compete: a
long prefill landing during decodes is the tail-latency spike Stage 5 teaches you about.
Disaggregation puts them on separate workers, with the prefill worker handing its KV cache to the
decode worker through a **KV connector** (`--kv-transfer-config`, e.g. `NixlConnector`).

The trade is explicit: better per-phase utilisation and independently tunable TTFT/ITL, paid for in KV
transfer latency, extra hardware, and moving parts. The doc is blunt: "Disaggregated prefill DOES NOT
improve throughput". If your goal is tokens/second on one box, this is the wrong tool; if it is a
bounded tail latency under mixed traffic, it is the right one.

---

## 7. KV cache offloading: buy context with RAM

📖 <https://docs.vllm.ai/en/stable/features/kv_offloading_usage/> · 🔍 `vllm/v1/worker/gpu/kv_connector.py`

The `OffloadingConnector` extends the prefix cache into slower, larger tiers: completed GPU blocks are
copied to pinned host memory (optionally to filesystem or object-store tiers) and promoted back on a
prefix hit, using `cudaMemcpyAsync` overlapped with compute. This is the one strategy on this page
that **runs on a single 24 GB card**: on the 24 GB pod `$MODEL_BIG` leaves roughly 4.8 GB for KV
(≈ 35k tokens), and a host-RAM tier keeps the rest of a long document cached for a PCIe copy per miss.

When is it a loss? When recomputing is cheap, or when the host tier itself becomes the bottleneck.
Offloading wins on **reuse** — long shared prefixes hit repeatedly — and loses on **streaming**, where
unique prompts miss every time and pay the copy for nothing. Tune with `cpu_bytes_to_use`,
`block_size`/`blocks_per_chunk`, and `max_load_tokens` per request (`0` disables loading for that
request). It is CUDA/ROCm/XPU only, and the host tier is total across workers, not per worker.

---

## 8. Speculative decoding: guess, then verify

📖 <https://docs.vllm.ai/en/stable/features/speculative_decoding/> · 🔍 `vllm/v1/spec_decode/` ·
`vllm/config/speculative.py`

A small **draft** model proposes `k` tokens; the **target** model verifies all `k` in a single forward
pass; the longest accepted prefix is kept, plus one bonus token. Rejection sampling preserves the
target's output distribution, so this is a *latency* trick, not an accuracy trick. vLLM supports draft
models, EAGLE/MTP heads, and n-gram or suffix proposers (`vllm/v1/spec_decode/ngram_proposer.py`,
`eagle.py`, `medusa.py`, `draft_model.py`).

The arithmetic that matters. The first line is vLLM's own definition
(`$VLLM_SRC/docs/features/speculative_decoding/acceptance_metrics.md`); the second is the standard
speedup identity that follows from it — *prediction*, since it ignores sampling overhead:

```
mean_acceptance_length = 1 + num_accepted_draft_tokens / num_spec_steps
speedup ≈ mean_acceptance_length / (verification-step cost / plain-decode-step cost)
```

A verification step costs more than a decode step (the draft ran, and the target processed `k+1`
tokens instead of 1), so the numerator has to beat that ratio. If acceptance is low you pay the draft
cost *plus* a bigger verification batch for ~1 token per step: **slower than no speculation at all**.
That is why vLLM reports the numbers — start the server with `--per-request-spec-decode-metrics
summary` and read `mean_acceptance_length` per request instead of guessing. On the 24 GB pod, run a
`$MODEL_MID` target with a tiny draft model and check the measured win against the formula.

---

## 9. The hardware reality table

Prices are RunPod Community on-demand, 2026-10-02 ([docs/03-runpod-setup.md](03-runpod-setup.md) has
the source table).

| Option | $/hr | Concepts it can actually demonstrate |
| --- | --- | --- |
| 1× RTX 4090 24 GB | $0.34 | KV offloading, speculative decoding, quantisation on `$MODEL_BIG`; **no** parallelism |
| 2× RTX 4090 24 GB (PCIe, no NVLink) | $0.68 | TP=2, PP=2, DP=2 — and their bandwidth cost. The cheapest way to *feel* why TP wants NVLink; also runs `Qwen3-30B-A3B` FP8 at `-tp=2` (~15 GB/GPU) |
| 1× RTX 6000 Ada 48 GB | $0.74 | The 30B MoE FP8 on one card, with FP8 W8A8 — single-GPU work at a size the 4090 cannot reach |
| 1× A100 80 GB (PCIe) | $1.19 | The 30B MoE in **bf16** (~61 GB); Ampere means **no FP8 W8A8**, so Stage 9's labs do not work here |
| 8-GPU node | ≈ 8× a per-GPU rate (8 × $0.74 ≈ $5.92/hr, arithmetic not a quote) | TP=8, EP=8, DP=8 on one node over NVLink-class interconnect — the real versions of EP and EPLB |

**Out of scope, plainly:** multi-node work — RunPod Instant Clusters target 16–64 GPU jobs, far past
this project's budget.

---

## 10. Decision guide

Given a model and a card, ask in this order:

1. **Does it fit on one card?** Use DP (`--data-parallel-size`) or just run separate servers.
2. **Does it fit in one node?** Use TP (`-tp`), sized to the largest value that divides both the query
   heads and the KV heads (8 for Qwen3-8B). Past `num_kv_heads` ranks TP stops buying KV memory.
3. **Across nodes, or on cards without NVLink?** Add PP for the depth
   (`--pipeline-parallel-size`), keeping TP = GPUs per node. Accept the bubble and the extra hop.
4. **Is it MoE at scale?** Add EP (`--enable-expert-parallel`) plus EPLB (`--enable-eplb`), and budget
   memory for redundant experts.
5. **Is the context the problem, not the weights?** Add CP (`-dcp`) once TP has hit the KV-head
   ceiling, or offload KV to host RAM if the win is prefix reuse rather than raw capacity.
6. **Is TTFT or tail ITL the SLO?** Consider disaggregation; if the SLO is per-token latency at low
   QPS, try speculative decoding before adding hardware.

---

## 11. Run these, read those

**Runnable on the 24 GB pod:** KV cache offloading and speculative decoding — the experiments are in
Stage 10 and the reasoning is in §7 and §8.

**Runnable only on a multi-GPU pod:** TP=2 vs TP=1 on one model and workload (per-GPU memory halves,
throughput rises by less than 2×, and the gap is §1's all-reduce cost); PP=2 vs TP=2 on the same two
cards; DP=2 on `$MODEL_TINY` as the near-linear baseline; and the 30B MoE FP8 at `-tp=2` on 2×4090 or
the 48 GB card — the first model in this project that does not fit in 24 GB.

**Read and understand:** multi-node TP+PP with Ray (`--nnodes`,
`--distributed-executor-backend ray`); EP across 8 GPUs with EPLB and redundant experts; decode
context parallel beyond `dcp = tp_size / num_kv_heads`; production disaggregation with NIXL/GDS and
`gdrcopy`. Knowing which side of the §9 line you are on is most of what Stage 10 is for.
