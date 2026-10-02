# Curriculum

A staged path through vLLM. Each stage is: **concept → read → run → measure → checkpoint.**
Do them in order; each one assumes the previous.

Legend: 📖 docs · 🔍 source (paths are relative to `vendor/vllm/`) · 🧪 lab · 📊 experiment ·
✅ checkpoint (be able to answer out loud, without notes)

Model used throughout: **`Qwen/Qwen3-0.6B`** — small enough to run on this Mac, and listed as
fully supported (GQA paged attention, automatic prefix caching) by vllm-metal.

> **How to run the labs.** The `vllm` CLI works directly, but Python scripts must use the
> interpreter in Homebrew's private venv:
>
> ```bash
> bash scripts/py.sh labs/01_offline_inference.py     # python labs
> bash labs/00_verify_install.sh                       # shell labs
> bash scripts/serve.sh                                # start the server
> ```
>
> Keep the memory budget in mind: `FRACTION=0.25` (the default) means ~3 GB for vLLM. See
> [`docs/03-apple-silicon-setup.md`](docs/03-apple-silicon-setup.md).

---

## Stage 0 — Orientation: what problem does vLLM actually solve?

**Concept.** A naive server runs one request at a time: latency is fine, throughput is terrible,
and the GPU sits idle most of the time. vLLM's answer is *continuous batching* plus a
*block-based, paged KV cache* so that memory — not compute — stops being the limit on batch size.

📖 [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) ·
[vLLM V1 guide](https://docs.vllm.ai/en/latest/usage/v1_guide/) ·
[Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)

🔍 `vendor/vllm/docs/design/arch_overview.md`

🧪 `labs/00_verify_install.sh`, `labs/01_offline_inference.py`

📊 From lab 01: record model load time, prompt throughput, generation throughput from vLLM's own
log lines. Notice they are reported **separately** — that distinction never goes away.

✅ Why does splitting prefill and decode into one continuous loop beat static batching?
✅ What is the difference between the `LLM` class and `vllm serve`?
✅ Name the four jobs `LLMEngine` does (input processing, scheduling, model execution, output processing).

📝 `docs/00-orientation.md`

---

## Stage 1 — The two entrypoints and the process architecture

**Concept.** vLLM is not one process. V1 separates an **API server** (HTTP, tokenization,
detokenization, streaming) from the **engine core** (scheduling + KV cache) from **workers**
(one per accelerator). They talk over ZMQ. Understanding this split is what makes later
debugging possible — and it is exactly what a single-process script hides from you.

📖 [Architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/)

🔍 `vllm/entrypoints/llm.py` · `vllm/entrypoints/cli/main.py` ·
`vllm/entrypoints/openai/api_server.py` · `vllm/v1/engine/core.py` ·
`vllm/v1/engine/llm_engine.py` · `vllm/v1/engine/async_llm.py` ·
`vllm/v1/engine/input_processor.py` · `vllm/v1/engine/output_processor.py` ·
`vllm/v1/engine/detokenizer.py` · `vllm/v1/executor/multiproc_executor.py`

🧪 `labs/02_openai_client.py` with `scripts/serve.sh`

📊 Run `vllm serve` and, in another shell, inspect the process tree. Count the processes and
compare with the table in `arch_overview.md` (`A + DP + N`, +1 if DP > 1).

✅ Why is tokenization in the API server but scheduling in the engine core?
✅ Why does the engine core run a **busy loop** instead of being event-driven?
✅ Where would an async request spend its time between arrival and first token?

📝 `docs/01-architecture.md`

---

## Stage 2 — The scheduler: continuous batching for real

**Concept.** Every step, the scheduler picks a set of requests and a token budget, and decides
prefill vs decode. This is the single most important object in vLLM.

📖 [vLLM V1 guide](https://docs.vllm.ai/en/latest/usage/v1_guide/) ·
[Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

🔍 `vllm/v1/core/sched/scheduler.py` (~155 KB — read `schedule()` first, then
`_try_schedule_prefill`-style helpers and the running/waiting queue handling) ·
`vllm/v1/core/sched/interface.py` (the contract + `SchedulerOutput`) ·
`vllm/v1/core/sched/output.py` · `vllm/v1/core/sched/request_queue.py` ·
`vllm/v1/core/sched/async_scheduler.py` · `vllm/v1/engine/admission_control.py`

🧪 `labs/03_continuous_batching.py`

📊 **Experiment A (the important one).** Send N requests **sequentially** and time them. Then send
the same N **concurrently** and time them. Compare total wall-clock and per-request latency.
This is static batching versus continuous batching, measured on a real production engine.
Record both numbers.

✅ What are `max_num_seqs` and `max_num_batched_tokens` each limiting?
✅ Why can a decode step and a prefill step coexist in one batch?
✅ What is chunked prefill, and which flag controls it?

📝 `docs/01-architecture.md` (scheduler section)

---

## Stage 3 — PagedAttention and the KV cache manager

**Concept.** KV cache memory is the real constraint. vLLM stores it in fixed-size **blocks**
referenced by a per-sequence **block table**, exactly as an OS pages virtual memory. This is
what lets sequences share memory, grow without copying, and be evicted cheaply.

📖 [Paged Attention design doc](https://docs.vllm.ai/en/latest/design/paged_attention/) ·
[Automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) ·
[Hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

🔍 `vendor/vllm/docs/design/paged_attention.md` · `vendor/vllm/docs/design/prefix_caching.md` ·
`vllm/v1/core/block_pool.py` · `vllm/v1/core/kv_cache_manager.py` ·
`vllm/v1/core/kv_cache_coordinator.py` · `vllm/v1/core/single_type_kv_cache_manager.py` ·
`vllm/v1/core/kv_cache_utils.py` · `vllm/v1/worker/block_table.py` ·
`vllm/v1/attention/backends/registry.py`

🧪 `labs/04_kv_cache_memory.py`

📊 **Experiment B.** Hold the model fixed and vary `--max-model-len` and
`--gpu-memory-utilization`; read the reported **"GPU KV cache size"** and **"Maximum concurrency"**
lines at startup. Derive the relationship between KV bytes/token, block size, and concurrency.

✅ Why 16 tokens per block by default, and when would you change it?
✅ A block table maps sequence → what, and who owns the blocks?
✅ What does the coordinator do that a single manager cannot?

📝 `docs/02-paged-attention.md`

---

## Stage 4 — Prefix caching: paying for a prompt once

**Concept.** If two requests share a prefix, their KV blocks are identical. Automatic prefix
caching hashes block contents and reuses them, so the second request skips prefill for the shared
part. This is why RAG and chat-with-system-prompt workloads get dramatically cheaper.

📖 [Automatic prefix caching (feature)](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) ·
[Prefix caching (design)](https://docs.vllm.ai/en/latest/design/prefix_caching/)

🔍 `vendor/vllm/docs/design/prefix_caching.md` · `vllm/v1/core/kv_cache_utils.py` (block hashing) ·
`vllm/v1/core/block_pool.py`

🧪 `labs/05_prefix_caching.py`

📊 **Experiment C.** Send the same long prompt twice with a shared prefix, with prefix caching
**on** and **off** (`--no-enable-prefix-caching`). Compare TTFT for the second request. Explain
the gap in terms of prefill tokens actually computed.

✅ What is hashed to decide a block is reusable, and why must the hash be content-based?
✅ What is the eviction policy, and what does it assume about access patterns?

---

## Stage 5 — Memory, batching and the tuning knobs that matter

**Concept.** vLLM's defaults are chosen for safety, not for your hardware. Five flags explain
most real-world behaviour differences.

📖 [Conserving memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/) ·
[Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) ·
[Engine args](https://docs.vllm.ai/en/latest/configuration/engine_args/) ·
[Serve args](https://docs.vllm.ai/en/latest/configuration/serve_args/)

🔍 `vllm/config/cache.py` · `vllm/config/model.py` · `vllm/config/scheduler.py` ·
`vllm/config/parallel.py`

🧪 `labs/06_metrics_and_bench.py`

📊 **Experiment D.** Sweep `--gpu-memory-utilization` (0.25 → 0.6) and
`--max-num-seqs` (4 → 64) at fixed request load. Plot throughput vs p50/p99 latency. Find the
knee — that is your usable capacity.

✅ Why does raising `--gpu-memory-utilization` not always raise throughput?
✅ What breaks first when `--max-num-seqs` is too high?
✅ What does `--enforce-eager` trade away, and why is it useful on this Mac?

---

## Stage 6 — Sampling, structured output, and reasoning

**Concept.** Everything after the logits is a *logits processor* pipeline plus a sampler.
Constrained decoding (JSON schemas, regex, grammars) is implemented here.

📖 [Structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) ·
[Custom logits processors](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/) ·
[Reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)

🔍 `vllm/v1/sample/` · `vllm/v1/sample/logits_processor/` · `vllm/v1/structured_output/` ·
`vllm/v1/engine/logprobs.py`

🧪 extend `labs/02_openai_client.py` with `response_format: json_schema`

📊 **Experiment E.** Force a JSON schema and count how often output is valid vs free-form
generation. Then measure the latency cost of the constraint.

✅ Where exactly does a structured-output grammar hook into the sampling loop?
✅ Why must logits processors be applied identically for every sequence in a batch?

---

## Stage 7 — Operating it: metrics, benchmarks, and observability

**Concept.** A served model is a system. vLLM exposes Prometheus metrics and a benchmark CLI;
reading them turns guesswork into diagnosis.

📖 [Production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) ·
[Benchmarking CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) ·
[Per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/)

🔍 `vllm/v1/metrics/` · `vllm/benchmarks/` · `vendor/vllm/docs/design/metrics.md`

🧪 `labs/06_metrics_and_bench.py`

📊 **Experiment F.** Capture `/metrics` before and after a load burst. Identify: queue depth,
KV cache utilisation, prefix cache hit rate, TTFT histogram. Which one predicts the latency
increase you measured in Experiment D?

✅ Which metric distinguishes "GPU saturated" from "scheduler starved"?
✅ What is the difference between throughput measured offline vs online?

---

## Stage 8 — Scale-out (read now, run on a cloud GPU later)

**Concept.** Past one device: tensor parallelism (split every layer), pipeline parallelism
(split layers across stages), data parallelism (replicate the engine), expert parallelism
(shard MoE experts), plus disaggregated prefill/decode and speculative decoding. Your Mac
cannot exercise these — read them here, rent a GPU for an hour when you want to run them.

📖 [Parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) ·
[Data parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) ·
[Expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) ·
[Context parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) ·
[Disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) ·
[Speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) ·
[Quantization](https://docs.vllm.ai/en/latest/features/quantization/)

🔍 `vllm/config/parallel.py` · `vllm/v1/executor/` · `vllm/v1/engine/coordinator.py` ·
`vllm/distributed/` · `vllm/v1/spec_decode/`

📊 **Cloud experiment (optional, ~$1–3 for 2 hours).** On a 2×GPU box: serve a 7B model with
`-tp=2`; confirm the weights are sharded (per-GPU memory halves) and measure throughput vs
`-tp=1`. Then enable `--speculative-config` and measure accepted tokens per step.

✅ TP vs PP: which one costs more communication, and why is TP inside a node?
✅ Why does DP need a coordinator for MoE models but not dense ones?
✅ What does disaggregated prefill actually separate, and what does it buy?

---

## Stage 9 — Extending vLLM (the contributor path)

**Concept.** vLLM is designed to be extended: custom models, custom attention backends, custom
logits processors, and a plugin system for out-of-tree hardware and features. This is where the
class-hierarchy design in `arch_overview.md` pays off.

📖 [Model implementation](https://docs.vllm.ai/en/latest/contributing/model/basic/) ·
[Registering a model](https://docs.vllm.ai/en/latest/contributing/model/registration/) ·
[Attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/) ·
[Plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/) ·
[CustomOp](https://docs.vllm.ai/en/latest/design/custom_op/)

🔍 `vendor/vllm/docs/contributing/model/` · `vllm/model_executor/models/` ·
`vllm/v1/attention/backends/registry.py` · `vendor/vllm/docs/design/plugin_system.md`

🧪 Read `vllm-metal` itself as the worked example of an out-of-tree plugin — it replaces the
attention path and the model layers while reusing vLLM's engine. This is the best available case
study for how far the plugin boundary stretches.

✅ What is the minimum set of methods a new attention backend must implement?
✅ How does a model get registered and discovered?
✅ Where is the boundary between "core vLLM" and "plugin"?

---

## Companion track: the build-from-source path

Worth doing once, on a rainy day, because it teaches you how the project is actually assembled:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv venv --python 3.12
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

Notes: macOS target device is forced to `cpu`; the build is **experimental**, supports **FP32/FP16
only**, and has no prebuilt wheels; vLLM's own `AGENTS.md` mandates `uv` and a `.venv` for all
Python work. Keep it in a *separate* environment from the Metal install to avoid clobbering it.

---

## How to know you are done

You can explain, without notes, and demonstrate by running something:

1. Why continuous batching exists and what it changes about latency vs throughput.
2. The V1 process split and what lives in each process.
3. How the scheduler chooses a batch each step, and what the two budget flags bound.
4. How a block table turns a sequence into KV memory addresses, and why paging matters.
5. What prefix caching hashes and why it is content-addressed.
6. How to size a deployment from the startup KV-cache log lines.
7. Which metrics tell you the engine is saturated vs starved.
8. Where you would add a feature: model, attention backend, logits processor, or plugin.
