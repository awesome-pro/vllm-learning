# 04 — Tuning and compilation: what the knobs actually bound

Stage 5 is the scheduler's two budgets; Stage 6 is compilation. This doc is the reference for both:
what each flag bounds, what the defaults are and why, what `torch.compile` and CUDA graphs cost you at
startup, and what to reach for when a specific symptom shows up.

Checked against the **v0.30.0** source tree on 2026-10-02. Source paths are relative to `$VLLM_SRC`.
**Nothing here was run on a GPU.** Every number is either a default read out of the source or
arithmetic shown next to it.

---

## 1. The five flags that explain most behaviour differences

| Flag | Default (v0.30.0) | What it actually bounds |
| --- | --- | --- |
| `--gpu-memory-utilization` | `0.92` (`vllm/config/cache.py`) | How big the **KV slab** is. Not the weights. |
| `--max-num-seqs` | `256` under 70 GiB, `1024` at ≥ 70 GiB | How many sequences may be in flight per step |
| `--max-num-batched-tokens` | `2048` for `vllm serve`, `8192` for `LLM()` under 70 GiB | How many tokens of work one step may contain |
| `--max-model-len` | from the model config | The context ceiling, which caps one sequence's KV |
| `--enforce-eager` / `-O{0,1,2,3}` | `-O2` (`vllm/config/vllm.py`) | Whether compile + CUDA graphs are used at all |

**`--gpu-memory-utilization` sizes the KV slab, not the weights.** The arithmetic is literally
`requested_memory = ceil(total_device_memory × gpu_memory_utilization)`
(`vllm/v1/worker/utils.py`), and then
`available_kv = requested_memory − non_kv_cache_memory − cudagraph_estimate`
(`vllm/v1/worker/gpu_worker.py`). Weights are part of `non_kv_cache_memory`: they are subtracted
*from* the requested budget, never scaled by it. Two consequences that trip people up:

- If the weights alone exceed `util × total`, you get a startup OOM, not a smaller KV cache. The OOM
  message prints the exact byte figures and a suggested `--kv-cache-memory-bytes` value — read it.
- The default is `0.92`, but `scripts/env.sh` deliberately uses `0.90` on a 24 GiB card, leaving ~2.4 GB
  of headroom for co-tenants, the CUDA context and fragmentation that the profiler's snapshot misses.

**`--max-num-seqs` bounds sequences, and it costs memory before it is ever used.** The scheduler holds
`max_num_running_reqs = max_num_seqs` and asserts `len(self.running) <= max_num_running_reqs` every step
(`vllm/v1/core/sched/scheduler.py`). But the related `max_num_active_seqs` docstring notes that
`max_num_seqs` "sizes the model runner (per-request buffers and CUDA graph capture)" — so raising it
allocates host and device buffers and captures more graphs whether or not you run that many sequences.

**`--max-num-batched-tokens` is where the entrypoint asymmetry lives.** `EngineArgs.get_batch_defaults()`
(`vllm/engine/arg_utils.py`) picks from device memory *and usage context*:

| Device | `LLM()` offline: batched_tokens / seqs | `vllm serve`: batched_tokens / seqs |
| --- | --- | --- |
| ≥ 160 GiB (B200/B300) | 16384 / 1024 | 16384 / 1024 |
| ≥ 70 GiB, not A100 (H100/H200) | 16384 / 1024 | 8192 / 1024 |
| **everything else, incl. RTX 4090 24 GB and A100** | **8192 / 256** | **2048 / 256** |

Same model, same 4090, same prompt — and `vllm serve` gets a **2048-token** per-step budget while
offline `LLM()` gets **8192**, a 4× difference. That is the single most surprising default in vLLM and
it is deliberate:

- **Offline `LLM()` knows the whole workload up front.** Requests are batched as tightly as memory
  allows and the only objective is total wall-clock time. Fat steps win.
- **`vllm serve` sees an arrival stream.** A fat step means a long prefill can occupy the entire budget
  and every in-flight decode waits for it, so p99 inter-token latency suffers. The smaller budget keeps
  any single step short.

The in-tree `$VLLM_SRC/docs/configuration/optimization.md` (published as the official optimization page)
states the same trade-off plainly: *"Smaller values (e.g., 2048) achieve better ITL because there are
fewer prefills slowing down decodes. Higher values achieve better time to first token (TTFT)."*

Three more rules run *after* the defaults are chosen, in
`_set_default_max_num_seqs_and_batched_tokens_args()` (`vllm/engine/arg_utils.py`):

1. `--performance-mode throughput` doubles **both** `max_num_batched_tokens` and `max_num_seqs` — but
   only for values you did not set yourself.
2. If `--max-num-batched-tokens` is left at default, it is clamped to
   `min(max_num_seqs × max_model_len, value)`.
3. If `--max-num-seqs` is left at default, it is clamped to `min(max_num_seqs, max_num_batched_tokens)`.

Two validations produce hard startup errors (`vllm/config/scheduler.py`): `max_num_batched_tokens <
max_num_seqs` is a **ValueError** (so `--max-num-seqs 256` with a token budget below 256 refuses to
start), and `max_num_batched_tokens > max_num_seqs × max_model_len` warns about "unexpected behavior".

**`--max-model-len` caps one sequence's KV, which makes it a *concurrency* knob in disguise.** A request
may not exceed it, and `get_max_concurrency_for_kv_cache_config()` (`vllm/v1/core/kv_cache_utils.py`)
divides the pool by `cdiv(max_model_len, block_size)` blocks per request. Lowering it does not change
`GPU KV cache size` in tokens at all — it changes how many sequences those tokens hold. It also accepts
`auto`/`-1`: `_auto_fit_max_model_len()` binary-searches the largest length that fits and logs what it
picked, which is often the fastest way out of a startup OOM.

**`--enforce-eager` and `-O` are one dial with two spellings.** `--enforce-eager` sets
`compilation_config.mode = NONE` and `cudagraph_mode = NONE`, and (unless fault tolerance is on) also
turns off JIT kernel warmup (`vllm/config/vllm.py`). `-O0` reaches nearly the same place through the
optimization-level table. `-O` is normalised to `--optimization-level` in `vllm/utils/argparse_utils.py`.

## 2. Chunked prefill

**On by default:** `SchedulerConfig.enable_chunked_prefill: bool = True`
(`vllm/config/scheduler.py`), logged once at startup as
`Chunked prefill is enabled with max_num_batched_tokens=N.`

**What it changes.** With chunked prefill, a prompt too long for the remaining token budget is split:
the scheduler gives it what is left of the step and schedules the rest next step. Decodes are
prioritised, so a long prompt no longer stops every in-flight generation for the duration of its
prefill. The scheduler's own framing of `schedule()` is that there is no prefill phase and no decode
phase — every request just has a `num_computed_tokens` counter trying to catch up with its token count,
and the budget is handed out against that.

**How it interacts with `max_num_batched_tokens`.** Chunk size *is* the leftover budget, so the two
numbers are not independent. Turning chunked prefill off changes the contract: the scheduler refuses to
schedule a request whose prefill exceeds the remaining budget (`break` instead of trimming), and the
config validator then requires `max_num_batched_tokens >= max_model_len` — otherwise it raises
`ValueError: max_num_batched_tokens (...) is smaller than max_model_len (...)`.

> **Doc-versus-source flag.** The official optimization page's chunked-prefill warning says that when
> chunked prefill is disabled, `max_num_batched_tokens` "must be greater than `max_model_len`" and that
> vLLM "may crash at server start-up". The source raises only when it is strictly *less than*
> `max_model_len` (`vllm/config/scheduler.py`), and it raises a clear `ValueError` rather than crashing.

**There is a softer knob than turning it off.** `long_prefill_token_threshold` (default `0`, disabled)
caps how many tokens one prefill may take in a step, specifically so a long prompt cannot starve the
rest of the batch — and the scheduler deliberately ignores the cap when the long request is the *only*
one in the batch, "since there is no other request for it to starve." `long_prefill_token_threshold_adaptive`
floors that cap at a fair share of the budget.

**When to turn it off:** encoder-decoder models (vLLM disables it for you); workloads where every prompt
fits in one step anyway, since chunk boundaries add per-step overhead for no benefit; and experiments
where you specifically want to observe the un-chunked behaviour. For interactive serving, leaving it on
and lowering `max_num_batched_tokens` is usually the better lever.

## 3. The interaction that matters most: KV slab × sequence length

This is where the flags in §1 stop being independent. Take the same predicted card as doc 02 —
`Qwen3-8B` on 24 GiB at `--gpu-memory-utilization 0.90`, KV slab ≈ 4.8 GiB ≈ 34,950 tokens at
**144 KiB/token**, ≈ 2,184 blocks of 16:

| Sequence length (prompt + output) | KV per sequence | Sequences the slab holds |
| --- | --- | --- |
| 8,192 | 1.125 GiB | **4** |
| 4,096 | 576 MiB | **8** |
| 2,048 | 288 MiB | **17** |
| 512 | 72 MiB | **68** |

Now set `--max-num-seqs 256`, the default on this card, and read the last column. To hold 256 concurrent
sequences inside 34,950 tokens, **each sequence would have to average 136 tokens total**. Above that,
the pool empties.

**What happens then is the single most important thing in this doc: nothing errors.**

- A **waiting** request whose `allocate_slots()` returns `None` is simply not admitted this step; the
  scheduler `break`s and the request stays in the waiting queue. No rejection, no exception — just
  latency. TTFT grows roughly linearly with the queue.
- A **running** request that needs another block triggers preemption: the scheduler frees the
  lowest-priority running request, resets its `num_computed_tokens` to 0, and prepends it to the waiting
  queue. That request re-runs its prefill later. Total work per request goes **up**.

So raising `--max-num-seqs` past what the KV cache can hold buys you a longer RUNNING list, not more
throughput — and it converts clean queueing into repeated recomputation. **Latency degrades first; at
high preemption rates throughput eventually degrades too**, because GPU cycles are being spent
re-prefilling sequences that already ran. Watch `vllm:kv_cache_usage_perc` and `vllm:num_preemptions`
in Stage 7 (`/metrics`): preemption is the signal, and it is not subtle.

The correct relationship to hold in your head:

```
sequences that fit ≈ KV_tokens / average_tokens_per_sequence
```

`--max-num-seqs` is an upper bound on the left side. The right side is set by your traffic and by
`--max-model-len`. If the bound is above what the right side allows, it is decoration.

## 4. `--performance-mode` and admission control

`PerformanceMode = Literal["balanced", "interactivity", "throughput"]`, default `"balanced"`
(`vllm/config/vllm.py`), exposed as `--performance-mode`.

| Mode | What it changes (source-verified) |
| --- | --- |
| `balanced` | Defaults as in §1 |
| `throughput` | Doubles `max_num_batched_tokens` **and** `max_num_seqs`, for any you did not set explicitly (`vllm/engine/arg_utils.py`) |
| `interactivity` | Changes CUDA-graph capture sizes to fine-grained small batches, `1..min(max_capture, 32)`, "for minimal padding overhead" (`vllm/config/vllm.py`) |

Worked: on a 4090, `vllm serve --performance-mode throughput` turns the 2048/256 default into
**4096/512**. (An H100-class `vllm serve` defaults to 8192/1024, so even doubled you are at half a
data-center token budget — the doubling is real but it is not parity.) That is still a legitimate
one-flag throughput bump, and it is exactly the change that will start preempting on a 35k-token slab at
long sequence lengths. The two modes are not free upgrades; they move you along the latency/throughput
curve.

`interactivity` is the underrated one. CUDA graphs are captured at a *finite* list of batch sizes, and
a batch is padded up to the nearest captured size, so work is done for padded rows. Fine-grained capture
sizes at small batch sizes mean less wasted padding when your batch is 3 rather than 8.

**Admission control is new and it is the only thing that turns overload into an error.**
`--max-num-queued-reqs` and `--max-num-queued-tokens` (`vllm/config/scheduler.py`, both default `None`)
are enforced in `AsyncLLM.check_admission()` (`vllm/v1/engine/async_llm.py`), which raises
`QueueOverflowError` or `MaxQueuedTokensError`. Both extend `GracefulHTTPError` and carry
**HTTP 503** (`vllm/exceptions.py`) so load balancers and SDKs retry elsewhere.

- `--max-num-queued-reqs` is a coarse capacity valve on in-flight requests (waiting + running), counted
  across every DP rank the API server routes to. Size it as roughly
  `data_parallel_size × max_num_seqs` plus the queue depth you are willing to hold.
- `--max-num-queued-tokens` is a **TTFT QoS** mechanism: set it to `target_TTFT × prefill_throughput` and
  requests get rejected when the prefill backlog would blow your latency target. The count is
  deliberately conservative (a partially-prefilled request still counts its full `prompt_len`), so it
  rejects slightly early — the safe direction.

The decision itself lives in `vllm/v1/engine/async_llm.py`; `vllm/v1/engine/admission_control.py` holds
`SharedAdmissionStats`, the lock-free, cache-line-padded counters that let multiple API-server processes
agree on a global in-flight count without contending.

## 5. Compilation: what `-O` buys, and what it costs

`optimization_level: OptimizationLevel = OptimizationLevel.O2` (`vllm/config/vllm.py`). The four levels
expand through `OPTIMIZATION_LEVEL_00..03` in the same file, and `$VLLM_SRC/docs/design/optimization_levels.md`:

| Level | `compilation_config.mode` | CUDA graphs | Fusions |
| --- | --- | --- | --- |
| `-O0` | `NONE` | `NONE` | all off; `enable_flashinfer_autotune=False` |
| `-O1` | `VLLM_COMPILE` | `PIECEWISE` | norm/act quant fusion where a custom kernel exists |
| `-O2` (default) | `VLLM_COMPILE` | `FULL_AND_PIECEWISE` | + allreduce-rms; + attn-quant on quantized models; + sequence parallelism and GEMM-comms fusion on dense models |
| `-O3` | `VLLM_COMPILE` | `FULL_AND_PIECEWISE` | **currently identical to `-O2`** |

Two things to know about `-O2`: it is the level that adds `FULL` graphs, and it is the level that turns
on `fuse_attn_quant`/`enable_sp`/`fuse_gemm_comms` — all of which are conditional on the model
(`IS_QUANTIZED`, `IS_DENSE`). On a dense bf16 Qwen3 you get the graph-mode change and the allreduce
fusion but not the attention-quant one.

**The CUDA graph modes** are `CUDAGraphMode` in `vllm/config/compilation.py`: `NONE`, `PIECEWISE`,
`FULL`, `FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE`. The last two are *dual-mode*: their values are
tuples, and `decode_mode()` / `mixed_mode()` split them so a runtime dispatcher can pick per batch.

**Why CUDA graphs apply to decode only.** A captured graph is keyed by a `BatchDescriptor`
(`num_tokens`, `num_reqs`, `uniform`, `has_lora`). A pure-decode step is *uniform* — every sequence asks
for exactly one token — so its shape is identical step after step and can be replayed. A prefill or
mixed batch has a shape determined by prompt lengths and chunk boundaries, which changes constantly.
On top of that, attention backends differ in what they can capture: each declares an
`AttentionCGSupport` level (`ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER`), and vLLM
downgrades the requested mode to the best supported one. `$VLLM_SRC/docs/design/cuda_graphs.md` has the
per-backend table; unlisted backends are `NEVER`.

FULL_AND_PIECEWISE means both: full graphs for uniform decode, piecewise graphs (everything except the
graph-incompatible ops, chiefly attention) for everything else.

**Why the first step is slow.** A boot is four distinct phases, and the backend log distinguishes them:
weight load; **memory profiling** (`profile_run` on a dummy max-size batch, plus
`profile_cudagraph_memory()` when graphs are on — this is what turns into `Available KV cache memory`);
**`torch.compile`** (Dynamo tracing plus Inductor codegen, all of which finishes before serving so no
request ever triggers a new compile); and **CUDA graph warmup + capture**, one `dummy_run` plus a capture
per size.

Phase 4's size is set by `cudagraph_capture_sizes`, generated as
`[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`, with `max_cudagraph_capture_size` capped at
**512 by default** (1024 on data-center Blackwell) — "to avoid OOM in tight memory scenarios with small
`max_num_seqs`, and limits capture of large graphs that increase startup time and memory usage." On a
4090 the base pattern is **51 capture sizes** before any extra sizes vLLM appends for uniform decode. A
batch that matches no captured key falls back to PIECEWISE and then to eager.

**The second boot is much faster, and that is the fix for "startup takes 4 minutes".** Compilation
artifacts are cached under `VLLM_CACHE_ROOT` (default `~/.cache/vllm`) in a directory keyed by a hash of
the configs, the PyTorch build and the model's forward function
(`$VLLM_SRC/docs/design/torch_compile.md`). Copy that directory into your container image and the compile
phase largely disappears. `VLLM_DISABLE_COMPILE_CACHE=1` disables the cache; `VLLM_FORCE_AOT_LOAD=1`
makes a cache miss fail loudly instead of silently recompiling.

**CUDA graphs cost KV cache tokens.** `profile_cudagraph_memory()`'s estimate is subtracted from the KV
budget (§1), and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` (default `1`) controls whether it is
applied. The worker logs the arithmetic in `--gpu-memory-utilization` terms when it is. Which means:
**`--enforce-eager` can legitimately print a *larger* `GPU KV cache size` than `-O2` on the same GPU.**
If your goal is "fit more concurrent tokens", eager is not obviously the wrong answer — it just trades
decode latency for cache capacity. Lab 07 measures exactly this.

## 6. Tuning decision table

| Symptom | Likely knob | What to try |
| --- | --- | --- |
| OOM at startup | `--max-model-len`, `--gpu-memory-utilization` | Lower `--max-model-len` first (or set it to `auto`), then drop util to 0.85. Read the OOM message: it prints the exact KV byte value that would fit. Beware — it spells the flag `--kv-cache-memory`, which does not exist (doc 02 §9) |
| OOM at startup, nothing changed | a co-tenant on the GPU | Util is a fraction of *total* memory but profiling measures *free* memory. Isolate the container, or lower util |
| High TTFT | `--max-num-batched-tokens` (up), prefix caching, `--max-num-queued-tokens` | Bigger steps prefill more per step. Confirm prefix caching is on and being hit. Cap the queue so TTFT has a floor you control |
| High inter-token latency | `--max-num-batched-tokens` (down), `--max-num-seqs` (down), `--performance-mode interactivity` | Smaller steps mean fewer prefill tokens competing with your decodes. Prefer this over disabling chunked prefill |
| Low throughput at low concurrency | `--max-num-seqs`, `--max-num-batched-tokens`, `--performance-mode throughput` | The GPU is idle and you are latency-bound. Raise both budgets, then re-check for preemption |
| Throughput plateau | none — you are at the knee | Check `vllm:kv_cache_usage_perc` (near 1) and `vllm:num_preemptions` (rising). More batching from here only adds latency |
| Preemptions in the metrics | util (up), `--max-model-len` (down), `--max-num-seqs` (down) | You admitted more sequences than the pool can hold (§3) |
| Startup takes minutes | `-O1` / `-O0`, `--enforce-eager`, the compile cache | First boot is compile + capture; second boot should be fast. Bake `~/.cache/vllm` into the image |
| Slower with prefix caching on | workload has no shared prefixes | `--no-enable-prefix-caching`; cached blocks are pinning memory nothing will ask for |

## 7. What not to tune

**Load-bearing for correctness — change these only with a reason, and re-validate outputs afterwards:**

| Flag | Why it is not just a performance knob |
| --- | --- |
| `--max-model-len` | Silently changes what a request is allowed to contain. Too low and long prompts are rejected; too high and you may OOM later, not at startup |
| `--kv-cache-dtype`, `--quantization` | Change numerics. `fp8` KV halves bytes *and* changes K/V precision |
| `--block-size` | Sets prefix-cache hash granularity and must satisfy the attention backend's `get_supported_kernel_block_sizes()` |
| `--prefix-caching-hash-algo` | Non-cryptographic hashes are documented as a collision and multi-tenant risk |
| `-tp` / `-pp` | Change how weights are sharded, so reductions happen in a different order; outputs can differ in low-order bits |

**Performance-only — free to move:** `--gpu-memory-utilization`, `--max-num-seqs`,
`--max-num-batched-tokens`, `-O*`, `--enforce-eager`, `--performance-mode`,
`--compilation-config`'s `cudagraph_mode`, `--max-num-queued-reqs`, `--max-num-queued-tokens`.

**Change one thing at a time, and here is the non-obvious reason.** The compile cache key includes
`max_num_batched_tokens` and `max_num_seqs` (`SchedulerConfig.compute_hash()` in
`vllm/config/scheduler.py`, because LoRA allocates static buffers from the former and Inductor decides
32-bit versus 64-bit indexing from it). So every change to those two flags **invalidates the compile
cache and forces a recompile**. An N-variable sweep therefore costs N recompiles, and your "before" and
"after" numbers are measuring compile-time differences as well as steady-state ones. Change one flag,
re-run the identical fixed workload (`temperature=0`, explicit `max_tokens`), and record the number.

The second reason is that the flags are multiplicative, not additive. `--gpu-memory-utilization` sets
the slab; `--max-num-seqs` and `--max-num-batched-tokens` decide how hard you push on it; `-O2` takes a
slice of it back for graphs. A change that helps at 4.8 GiB can hurt at 2.4 GiB. That is precisely why
Stage 5's sweep is a *table* and not a single run.

---

## Read in this order

1. `vllm/engine/arg_utils.py` — `get_batch_defaults()` and
   `_set_default_max_num_seqs_and_batched_tokens_args()`
2. `vllm/config/scheduler.py` — the validators, and `compute_hash()`
3. `vllm/config/vllm.py` — `OPTIMIZATION_LEVEL_00..03` and the `enforce_eager` handling
4. `vllm/config/compilation.py` — `CUDAGraphMode` and the capture-size defaults
5. `vllm/v1/core/sched/scheduler.py` — `schedule()`, then the preemption path
6. `vllm/v1/engine/async_llm.py` — `check_admission()`
7. `$VLLM_SRC/docs/design/optimization_levels.md`, `$VLLM_SRC/docs/design/cuda_graphs.md`,
   `$VLLM_SRC/docs/design/torch_compile.md` — read them locally, next to the code

**Labs.** `labs/06_scheduler_budgets.sh` (Experiment E) and `labs/07_compilation_levels.sh`
(Experiment F). Both need the server restarted between runs when a flag changes — see `CURRICULUM.md`.

---

## ✅ Checkpoint

1. On a 24 GB card, why does `vllm serve` default to `max_num_batched_tokens=2048` while offline
   `LLM()` gets 8192? Which one would you pick for a batch job, and which for a chat endpoint?
2. What exactly happens when `--max-num-seqs` is higher than the KV cache can support? Name the two
   paths (waiting request vs running request) and say which one shows up as a metric.
3. What does chunked prefill change about a long prompt's effect on other in-flight requests, and what
   constraint does disabling it impose on `max_num_batched_tokens`?
4. `-O2` versus `--enforce-eager`: which CUDA graph modes does each select, and why can eager print a
   *larger* `GPU KV cache size`?
5. Why can CUDA graphs only be captured for decode at fixed batch sizes, and what does a batch that
   misses every captured size fall back to?
