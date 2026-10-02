# Curriculum

A staged path through vLLM on a rented Linux GPU. Each stage is:
**concept → read → run → measure → checkpoint.**

Do them in order. Every stage assumes the previous one, and the early stages are deliberately
small: Stage 0 is "get one token out of a GPU", and you should be able to finish it in your first
20 minutes on the pod.

**Pacing.** Stages 0–2 are one sitting (~1 hour). Stages 3–5 are the heart of the project — take
one per sitting. Stages 6–9 are one sitting each. Stage 10 needs a different (more expensive) pod.
Stage 11 is open-ended.

Legend: 📖 docs · 🔍 source (relative to `$VLLM_SRC`) · 🧪 lab · 📊 experiment ·
✅ checkpoint (answer out loud, without notes) · 📝 companion doc · ⏱ time

> **Before Stage 0:** do [`docs/03-runpod-setup.md`](docs/03-runpod-setup.md). It gives you the exact
> pod recipe, the storage decision (which is where the money goes), and the 5-minute bootstrap.
>
> **Every session:** `source scripts/env.sh` first. It sets `$VLLM_SRC`, `$HF_HOME`, the model
> ladder, and activates the venv. Then run labs with `bash labs/…` or `python labs/…`.
>
> **Every session end:** stop the pod. Not "close the tab" — *stop* it. See the shutdown checklist
> in the setup doc.

Model ladder, defined in `scripts/env.sh`:

| Var | Model | Use it for |
| --- | --- | --- |
| `$MODEL_TINY` | `Qwen/Qwen3-0.6B` | every iteration; 30-second answers |
| `$MODEL_MID` | `Qwen/Qwen3-4B` | realistic behaviour without fighting for memory |
| `$MODEL_BIG` | `Qwen/Qwen3-8B` | the actual 24 GB test |
| `$MODEL_MOE` | `Qwen/Qwen3-30B-A3B` | Stage 10 only, on a 48 GB+ card |

---

## Stage 0 — Get a token out of a GPU ⏱ 20 min

**Concept.** vLLM is an inference *engine*: it loads a model's weights, allocates a paged KV cache
up front, and serves many requests concurrently. Before any theory, prove the machine works and
see what the engine tells you about itself at startup. The startup log is not noise — it is the
most information-dense thing vLLM emits, and you will read it in every later stage.

📖 [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) ·
[Installation (GPU)](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

🔍 `vllm/entrypoints/llm.py` (the offline `LLM` class) · `vllm/engine/arg_utils.py`
(where every CLI flag becomes an `EngineConfig`)

🧪 `labs/00_verify_install.sh` — driver, vLLM version, GPU, kernel check, one tiny generation
🧪 `labs/01_offline_inference.py` — the `LLM` class, first generation, model load time

📊 **Experiment.** Record from lab 01: model load time, and vLLM's own
`Avg prompt throughput` / `Avg generation throughput` lines. Notice they are reported
**separately** — that distinction never goes away.

✅ What does `LLM(...)` do that `vllm serve` also does, and what does the server add?
✅ What is in the startup log's KV-cache line, and which number sizes your concurrency?
✅ Why is the *first* generation slower than the second?

📝 [`docs/00-orientation.md`](docs/00-orientation.md)

---

## Stage 1 — Serve it and speak HTTP ⏱ 30 min

**Concept.** Production vLLM is a server. V1 splits the **API server** (HTTP, tokenization,
detokenization, streaming) from the **engine core** (scheduling + KV cache) from the **workers**
(one per GPU), and they talk over ZMQ. A single-process script hides that split; the server makes
it visible, and it is what makes later debugging possible.

You will also measure, by hand, the two numbers every LLM serving discussion is about:
**TTFT** (time to first token — dominated by prefill) and **inter-token latency** (dominated by
decode + batching).

📖 [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) ·
[Architecture overview](https://docs.vllm.ai/en/stable/design/arch_overview/)

🔍 `vllm/entrypoints/launchers/` · `vllm/entrypoints/launchers/api_server/entry.py` ·
`vllm/entrypoints/launchers/api_server/routers.py` · `vllm/v1/engine/async_llm.py` ·
`vllm/v1/engine/core.py` · `vllm/v1/engine/input_processor.py` ·
`vllm/v1/engine/output_processor.py` · `vllm/v1/engine/detokenizer.py` ·
`vllm/v1/executor/multiproc_executor.py`

> **A trap worth seeing on day one:** the path most guides give for the server,
> `vllm/entrypoints/openai/api_server.py`, is now a **59-line deprecation shim** that re-exports
> from `vllm/entrypoints/launchers/api_server/` — true at the v0.30.0 tag, not just on `main`. Read
> the launcher package, and notice how often "the documented location" is no longer where the code
> lives. That habit is worth more than any single fact in this guide.

🧪 `scripts/serve.sh` in one terminal, `labs/02_serve_and_client.py` in another

📊 **Experiment A.** Send one streaming request and timestamp every chunk. Split the timeline into
TTFT and the gaps between tokens. Then send the same request with `"stream": false` and compare.
Also: `nvidia-smi` in a third terminal while generating — watch the memory number *not* move.

✅ Why is tokenization in the API server but scheduling in the engine core?
✅ Why does the engine core run a **busy loop** instead of being event-driven?
✅ How many OS processes are there for one 1-GPU server, and what is each one?

📝 [`docs/01-architecture.md`](docs/01-architecture.md)

---

## Stage 2 — Continuous batching: why vLLM exists ⏱ 45 min

**Concept.** A naive server runs one request at a time: latency is fine, throughput is terrible,
and the GPU idles. Static batching waits for a full batch and runs it to completion: throughput
improves, but short requests wait behind long ones. **Continuous batching** makes a scheduling
decision *every single decode step* — finished sequences leave the batch, waiting ones join it, and
a prefill can share the step with ongoing decodes. This is the single most important idea in vLLM.

📖 [Quickstart / serving basics](https://docs.vllm.ai/en/stable/getting_started/quickstart/) ·
[vLLM V1 guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)

🔍 `vllm/v1/core/sched/scheduler.py` — read `schedule()` first, then the running/waiting queue
handling · `vllm/v1/core/sched/interface.py` (the contract + `SchedulerOutput`) ·
`vllm/v1/core/sched/output.py` · `vllm/v1/core/sched/request_queue.py`

🧪 `labs/03_continuous_batching.py`

📊 **Experiment B (the important one).** Send N requests **sequentially**, time the total wall
clock. Send the same N **concurrently**, time the total. Then compare *per-request* latency in both
cases. You are measuring the throughput/latency tradeoff on a real engine, and you will see total
time collapse while per-request latency rises slightly. Record all four numbers.

✅ Why does continuous batching raise throughput without raising per-request latency as much as
static batching would?
✅ Where in `scheduler.py` does a finished request free its slot for a waiting one?
✅ Why can a prefill and a decode legitimately share one step?

📝 [`docs/01-architecture.md`](docs/01-architecture.md) (scheduler section)

---

## Stage 3 — PagedAttention and the KV cache ⏱ 1 hour

**Concept.** Never trust "the GPU is the limit". The real constraint is **KV cache memory**, and
vLLM's founding idea is to store it in fixed-size **blocks** referenced by a per-sequence **block
table** — virtual memory for attention. Blocks let sequences share memory, grow without copying,
and be freed cheaply, which is what makes high concurrency possible on a fixed card.

This is the stage where the startup log stops being trivia and becomes arithmetic.

📖 [Paged Attention (design)](https://docs.vllm.ai/en/stable/design/paged_attention/) ·
[Hybrid KV cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) ·
[Engine args](https://docs.vllm.ai/en/stable/configuration/engine_args/)

🔍 `vllm/v1/core/block_pool.py` · `vllm/v1/core/kv_cache_manager.py` ·
`vllm/v1/core/kv_cache_coordinator.py` · `vllm/v1/core/single_type_kv_cache_manager.py` ·
`vllm/v1/core/kv_cache_utils.py` · `vllm/v1/worker/block_table.py` ·
`vllm/v1/attention/backends/registry.py`

🧪 `labs/04_kv_cache_memory.py`

📊 **Experiment C.** Hold the model fixed and vary `--max-model-len` and
`--gpu-memory-utilization`. From each startup log, read **`Available KV cache memory`**,
**`GPU KV cache size: N tokens`** and **`Maximum concurrency for M tokens per request: X.xx`**.
Then *derive* KV bytes/token from the model config
(`2 × layers × kv_heads × head_dim × dtype_bytes`) and check it against vLLM's number. They should
agree within a few percent — if they do not, you have misread a config field, and finding out which
one is the actual lesson.

✅ Why 16 tokens per block by default, and when would you change it?
✅ A block table maps *what* to *what*, and who owns the blocks?
✅ What does the KV cache coordinator do that a single manager cannot?
✅ Why does `--gpu-memory-utilization 0.92` on a 24 GB card *not* leave 24 GB for weights?

📝 [`docs/02-paged-attention.md`](docs/02-paged-attention.md)

---

## Stage 4 — Prefix caching: pay for a prompt once ⏱ 45 min

**Concept.** If two requests share a prefix, their KV blocks are bit-identical. Automatic prefix
caching hashes block contents and reuses them, so the second request skips prefill for the shared
part. This is why RAG, long system prompts and multi-turn chat get dramatically cheaper — and why
the default flipped to **on**.

📖 [Automatic prefix caching (feature)](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) ·
[Prefix caching (design)](https://docs.vllm.ai/en/stable/design/prefix_caching/)

🔍 `vllm/v1/core/kv_cache_utils.py` (block hashing) · `vllm/v1/core/block_pool.py`
(eviction + `get_computed_blocks`) · `vllm/config/cache.py` (the default you are overriding)

🧪 `labs/05_prefix_caching.py`

📊 **Experiment D.** Send the same long prompt twice, with prefix caching on and with
`--no-enable-prefix-caching`. Compare **TTFT of the second request**, on a fresh server each time.
Then change *one token at the start* of the prompt and watch the reuse vanish. Record TTFT for all
three cases; explain each in terms of prefill tokens actually computed.

✅ What exactly is hashed to decide a block is reusable, and why content-based rather than
address-based?
✅ What is the eviction policy, and what access pattern does it assume?
✅ Why did prefix caching have to become the default before it was useful in practice?

---

## Stage 5 — The scheduler's two budgets ⏱ 1 hour

**Concept.** The scheduler is governed by two numbers, and confusing them is the most common
misunderstanding in vLLM: `--max-num-seqs` bounds **how many sequences** are in flight,
`--max-num-batched-tokens` bounds **how many tokens of work** one step may contain. Together with
`--gpu-memory-utilization` (KV cache size) they explain almost every real behaviour difference.

There is also a trap worth knowing on day one: on a **< 70 GB card**, `vllm serve` defaults
`max_num_batched_tokens` to **2048** while the offline `LLM()` class defaults to **8192**. The same
model, same GPU, same prompt — different throughput, purely because of the entrypoint. That is not
a bug; it is a deliberate latency/throughput choice, and this stage makes you see it.

📖 [Optimization and tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) ·
[Conserving memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/) ·
[Serve args](https://docs.vllm.ai/en/stable/configuration/serve_args/)

🔍 `vllm/engine/arg_utils.py` — `get_batch_defaults()` is the function that decides those defaults
from your device memory and usage context · `vllm/config/scheduler.py` (chunked prefill) ·
`vllm/config/cache.py` · `vllm/v1/engine/admission_control.py`

🧪 `labs/06_scheduler_budgets.sh`

📊 **Experiment E.** At a fixed request load, sweep `--max-num-batched-tokens`
(2048 → 8192 → 16384) and then `--max-num-seqs` (4 → 32 → 256). For each: throughput, p50 and p99
latency, and the startup log's KV figures. Find the knee — the point past which more batching buys
no more throughput and only adds latency. Record the sweep as a table.

✅ What breaks *first* when `--max-num-seqs` is too high on a fixed KV cache?
✅ What does chunked prefill change about a long prompt's effect on other requests?
✅ Why does raising `--gpu-memory-utilization` sometimes *lower* throughput?

📝 [`docs/04-tuning-and-compilation.md`](docs/04-tuning-and-compilation.md)

---

## Stage 6 — Compilation: torch.compile and CUDA graphs ⏱ 45 min

**Concept.** Two of vLLM's biggest speedups have nothing to do with the model: it compiles the
model with `torch.compile`, and it replays decode steps as **CUDA graphs** to eliminate per-kernel
launch overhead. Both cost startup time, both can be switched off, and knowing the `-O` levels
turns "vLLM is slow to start" from a mystery into a dial.

📖 [Optimization levels](https://docs.vllm.ai/en/stable/design/optimization_levels/) ·
[CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) ·
[torch.compile](https://docs.vllm.ai/en/stable/design/torch_compile/)

🔍 `vllm/config/compilation.py` · `vllm/config/vllm.py` (`optimization_level: OptimizationLevel.O2`) ·
`vllm/v1/worker/gpu/cudagraph_utils.py`

🧪 `labs/07_compilation_levels.sh`

📊 **Experiment F.** Run the same fixed workload four ways: `-O0` (or `--enforce-eager`), `-O1`,
`-O2` (default), and `-O2` with the CUDA graph mode pinned to decode-only:

```bash
vllm serve "$MODEL_MID" --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

There is **no `--cuda-graph-mode` flag** — it has never existed in vLLM, despite circulating in
guides and blog posts, and it is the kind of thing that only a `git log -S` will settle. The graph
mode lives inside `--compilation-config` (JSON, as above, or the dotted `-cc.cudagraph_mode=…`
form). Record **startup time**, **decode throughput** and **p50 latency**. Record the process count
from `nvidia-smi` too. Then explain why a *CPU-bound* small-model decode benefits most from CUDA
graphs.

✅ What does `-O2` enable that `-O1` does not, and what does `--enforce-eager` turn off?
✅ Why can CUDA graphs only be captured for the decode phase at fixed batch sizes?
✅ When would you genuinely want `-O0` in production?

---

## Stage 7 — Metrics and benchmarking ⏱ 1 hour

**Concept.** A served model is a system, and vLLM exposes Prometheus metrics that let you tell
**saturated** (GPU-bound, KV full) from **starved** (scheduler waiting on arrivals). Reading them
turns guesswork into diagnosis. vLLM also ships its own benchmark suite, which is the honest way to
compare two configurations.

📖 [Production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) ·
[Metrics design](https://docs.vllm.ai/en/stable/design/metrics/) ·
[Benchmarking CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)

🔍 `vllm/v1/metrics/` · `vllm/benchmarks/` · `$VLLM_SRC/docs/design/metrics.md`

🧪 `labs/08_metrics_and_bench.sh`

📊 **Experiment G.** Capture `/metrics` before, during and after a load burst. Watch
`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`,
`vllm:prefix_cache_hits`/`_queries`, and the TTFT histogram. Then run
`vllm bench serve` and reconcile its reported numbers with what the metrics said. Which single
metric predicted the latency increase you measured in Stage 5?

✅ Which metric distinguishes "GPU saturated" from "scheduler starved"?
✅ Why is throughput measured offline not comparable to throughput measured online?
✅ What is the difference between `vllm:kv_cache_usage_perc` and the `Maximum concurrency` line?

📝 [`docs/05-metrics-and-benchmarking.md`](docs/05-metrics-and-benchmarking.md)

---

## Stage 8 — Sampling, structured output, and reasoning ⏱ 45 min

**Concept.** Everything after the logits is a *logits processor* pipeline plus a sampler.
Constrained decoding (JSON schemas, regex, grammars) hooks in here, and this is where you can add
behaviour without touching the model. It is also a good first read of a subsystem small enough to
finish in one sitting.

📖 [Structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) ·
[Custom logits processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/) ·
[Reasoning outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) ·
[Tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)

🔍 `vllm/v1/sample/` · `vllm/v1/sample/logits_processor/` · `vllm/v1/structured_output/` ·
`vllm/v1/engine/logprobs.py`

🧪 `labs/09_structured_outputs.py`

📊 **Experiment H.** Ask the tiny model for a specific JSON schema 20 times with
`response_format: {"type": "json_schema", …}` and 20 times free-form. Count valid JSON in each arm.
Then measure the latency cost of the constraint (it is not free — the grammar masks logits every
step).

> **A trap that will bite you if you follow an older guide:** `guided_json`, `guided_regex`,
> `guided_choice` and `guided_grammar` were **removed**, and vLLM now *silently ignores* them — you
> get HTTP 200, one warning, and completely unconstrained output. The modern forms are
> `response_format` (JSON schema) and a `structured_outputs` object (`{"regex": …}`,
> `{"choice": [...]}`, `{"json": …}`). Lab 09 includes a one-request probe that demonstrates the
> silent-ignore behaviour, because "my constraint had no effect" is otherwise a very confusing hour.

✅ Where exactly does a structured-output grammar hook into the sampling loop?
✅ Why must logits processors be applied identically to every sequence in a batch?
✅ Why can a constrained sampler be *faster* in wall-clock terms despite extra per-step work?
✅ What does vLLM do when you pass a field it has removed — and how would you have found out?

---

## Stage 9 — Quantization: fitting more, paying less ⏱ 1 hour

**Concept.** Precision is a resource you trade for memory and speed. Two independent knobs:
**weight quantization** (`--quantization fp8`, which can quantize a bf16 checkpoint on the fly) and
**KV cache dtype** (`--kv-cache-dtype fp8`, which halves KV bytes per token and therefore roughly
doubles the tokens you can cache). The 4090 is Ada (SM 8.9), so FP8 W8A8 is supported natively —
this stage is only possible because you are on real hardware.

📖 [Quantization](https://docs.vllm.ai/en/stable/features/quantization/)

🔍 `vllm/model_executor/layers/quantization/fp8.py` · `vllm/config/cache.py`
(`cache_dtype`) · `vllm/v1/core/kv_cache_utils.py` (where KV dtype meets the block size)

🧪 `labs/10_quantization_fp8.sh`

📊 **Experiment I.** Four runs of the same model and workload:
(1) bf16 baseline, (2) `--quantization fp8`, (3) `--kv-cache-dtype fp8`, (4) both.
For each record **weight memory**, **`GPU KV cache size`**, **maximum concurrency**, **throughput**
and **output text** (compare quality by eye on a reasoning prompt). Explain the KV numbers with the
bytes/token formula from Stage 3 — the fp8 KV run should roughly double it.

Two version-specific details the labs will surface for you:

- `--quantization fp8` is a **deprecated alias** that logs a warning and resolves to
  `fp8_per_tensor`. It still works — but read the warning rather than tuning around a flag that is
  on its way out.
- **Ampere does not hard-fail on fp8.** The CUTLASS W8A8 kernels need SM ≥ 89 (Ada/Hopper) and
  block-FP8 needs SM ≥ 90, but vLLM falls back to a **weight-only** FP8 path (FP8-Marlin). So an
  fp8 run on an A100 or A6000 *starts* and gives you the memory win with none of the compute win.
  That distinction — "it ran" versus "it ran faster" — is the whole reason Stage 10's hardware table
  pays attention to the architecture and not just the VRAM number.

✅ What exactly does `--kv-cache-dtype fp8` halve, and what stays the same?
✅ Why is FP8 *W8A8* unavailable on an A100 but available on a 4090, and what does an A100 do
instead when you ask for fp8?
✅ What did you give up, and how would you detect it other than by reading the output?

---

## Stage 10 — Scale-out: the parts one GPU cannot show you ⏱ 2 hours, needs 2 GPUs

**Concept.** Past one device: tensor parallelism (shard every layer), pipeline parallelism
(split layers into stages), data parallelism (replicate the engine), expert parallelism (shard MoE
experts), plus disaggregated prefill/decode, KV cache offloading and speculative decoding. This is
the stage that justifies renting a bigger pod for one session — and with the FP8 MoE
(`Qwen3-30B-A3B`) you finally run a model that does not fit in 24 GB.

📖 [Parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) ·
[Data parallel deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) ·
[Expert parallel deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) ·
[Context parallel](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) ·
[Disaggregated prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill/) ·
[Speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) ·
[KV offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)

🔍 `vllm/config/parallel.py` · `vllm/v1/executor/` · `vllm/v1/engine/coordinator.py` ·
`vllm/distributed/` · `vllm/v1/spec_decode/` · `vllm/distributed/eplb/` ·
`vllm/v1/worker/gpu/eplb_utils.py`

🧪 `labs/11_tensor_parallel.sh` (2× GPU pod, or 1× 48 GB card for the MoE)

📊 **Experiment J.** Serve a model with `-tp=2` and confirm the weights are sharded (per-GPU memory
roughly halves) and measure throughput against `-tp=1`. Then, on the 48 GB card, serve
`Qwen/Qwen3-30B-A3B` (fp8) and watch expert routing show up in per-step timing. Optionally enable
`--speculative-config` and measure accepted tokens per step.

✅ TP vs PP: which costs more communication, and why does TP stay inside a node?
✅ Why does DP need a coordinator for MoE models but not for dense ones?
✅ What does disaggregated prefill actually separate, and what does it buy?
✅ When is KV offloading a win rather than a slowdown?

📝 [`docs/06-scaling.md`](docs/06-scaling.md)

---

## Stage 11 — The contributor path ⏱ open-ended

**Concept.** vLLM is designed to be extended: custom models, attention backends, logits processors,
and a plugin system for out-of-tree hardware and features. Your fork already exists
(`~/Desktop/vllm`, remote `awesome-pro/vllm`, upstream `vllm-project/vllm`) — this stage turns
reading into contributing: build from source on the pod, run the test suite, reproduce a bug, open
a PR.

📖 [Contributing](https://docs.vllm.ai/en/stable/contributing/) ·
[Model implementation](https://docs.vllm.ai/en/stable/contributing/model/basic/) ·
[Registering a model](https://docs.vllm.ai/en/stable/contributing/model/registration/) ·
[Attention backends](https://docs.vllm.ai/en/stable/design/attention_backends/) ·
[Plugin system](https://docs.vllm.ai/en/stable/design/plugin_system/) ·
[CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) ·
[Deprecation policy](https://docs.vllm.ai/en/stable/contributing/deprecation_policy/)

🔍 `$VLLM_SRC/docs/contributing/` · `vllm/model_executor/models/` ·
`vllm/v1/attention/backends/registry.py` · `$VLLM_SRC/docs/design/plugin_system.md`

🧪 Build from source, run `pytest tests/v1/core/test_scheduler.py`, reproduce one real issue

📊 **Experiment K (the payoff).** Pick one thing this guide made you curious about and answer it
*from the source, with evidence*: a test you ran, a profile you captured, a log line you explained.
Write it into `notes/`. If it turns out upstream is wrong, you have found your first PR.

✅ What is the minimum set of methods a new attention backend must implement?
✅ How does a model get registered and discovered?
✅ Where is the boundary between "core vLLM" and "plugin"?
✅ What does upstream's own `AGENTS.md` require of contributors (uv, `.venv`, formatting)?

📝 [`docs/07-contributing.md`](docs/07-contributing.md)

---

## Staying current

vLLM ships about every two weeks. Cheap monthly hygiene, 10 minutes:

1. **Check the version.** `curl -s https://pypi.org/pypi/vllm/json | jq -r .info.version` and
   compare with what the pod reports. Read the release notes for the gap.
2. **Re-read the release notes for flags you use.**
   <https://github.com/vllm-project/vllm/releases> — defaults *do* change (prefix caching and
   chunked prefill both flipped to on; `max_num_batched_tokens` moved).
3. **Let the guide check itself.** After re-cloning the source at the new tag:

   ```bash
   VLLM_SRC=/workspace/src/vllm bash scripts/check-guide.sh
   ```

   It verifies every source path cited in these docs still exists, that every relative link
   resolves, and that the labs match `CURRICULUM.md`. Anything it flags is a doc that has drifted
   from upstream — fix the path, or note what moved. `CHECK_URLS=1` additionally HTTP-checks every
   `docs.vllm.ai` link, which catches renamed documentation pages.
4. **Re-run Stage 5's sweep** if a default changed — your old numbers are no longer comparable.

---

## How you know you are done

You can explain, without notes, and demonstrate by running something:

1. Why continuous batching exists, and what it changes about latency vs throughput.
2. The V1 process split and what lives in each process.
3. How the scheduler picks a batch each step, and what `max_num_seqs` and
   `max_num_batched_tokens` each bound.
4. How a block table turns a sequence into KV memory addresses, and why paging matters.
5. What prefix caching hashes, why it must be content-based, and how to prove a hit.
6. How to size a deployment from the startup KV-cache log, before sending any traffic.
7. Which metrics distinguish a saturated engine from a starved one.
8. What `-O2` buys you, and what you pay for it at startup.
9. What FP8 weight quantization and FP8 KV cache each cost you, measured.
10. Where you would add a feature: model, attention backend, logits processor, or plugin — and how
    to land that change upstream.
