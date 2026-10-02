# 05 — Metrics, benchmarking, and telling saturated from starved

> Stage 7's companion. The experiment is [`labs/08_metrics_and_bench.sh`](../labs/08_metrics_and_bench.sh);
> read this, run that, then write `notes/stage-07-metrics.md`. Source paths below are relative to
> `$VLLM_SRC`. Everything here was checked against the v0.30.0 tree.

---

## 1. What vLLM exposes, and where to read it

Three surfaces, in increasing order of what they cost you to interpret:

| Surface | Where | What it is |
| --- | --- | --- |
| The 5-second log line | wherever you started the server | A hand-picked summary. `LoggingStatLogger.log()` in `vllm/v1/metrics/loggers.py:275-320` prints `Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: N reqs, Waiting: M reqs, GPU KV cache usage: Z%, Prefix cache hit rate: H%`. Note it drops to `logger.debug` when the engine is idle — on a quiet system those lines *disappear*, which is itself a signal. |
| `/metrics` | `http://127.0.0.1:8000/metrics` | Prometheus text. Mounted by `vllm/entrypoints/serve/instrumentator/metrics.py:56-81`. This is the surface production monitors. |
| `LLM.get_metrics()` | in-process, offline | `vllm/entrypoints/llm.py:860-866` — a snapshot of the same aggregated Prometheus metrics, for code that never starts a server. |

The metric *families* are defined in one file: `vllm/v1/metrics/loggers.py` (`PrometheusStatLogger`).
Histogram bucket boundaries are in `vllm/v1/metrics/buckets.py` — read it before you trust a
percentile, because the bucket layout decides the resolution you get. The design rationale is
`$VLLM_SRC/docs/design/metrics.md`, and the user-facing page is
<https://docs.vllm.ai/en/stable/usage/metrics/> (`/design/metrics/` is the deeper one).

Two conventions that trip people up:

- **Counters are exported with a `_total` suffix; gauges are not.** The exposition literally reads
  `# TYPE vllm:generation_tokens_total counter` / `vllm:generation_tokens_total{...} 27453.0`
  (`docs/design/metrics.md:336-363`). So the series is `vllm:prefix_cache_hits_total`, even though
  the code names it `vllm:prefix_cache_hits`.
- **Every series carries `model_name` and `engine` labels** (`vllm/v1/metrics/loggers.py:484`). Sum
  counters over label sets; take the max for gauges. With data parallelism you must decide whether
  you are looking at a fleet or a member.

And one naming trap worth stating plainly: **`vllm:gpu_cache_usage_perc` does not exist.** Older
guides and older Grafana dashboards use it. In v0.30.0 the name is `vllm:kv_cache_usage_perc`
(`vllm/v1/metrics/loggers.py:613`), where `1` means 100 %.

---

## 2. The metrics that actually matter

Grouped by the question they answer, with the real names.

**Is the engine busy, and is anything waiting?**

| Metric | Type | Reading |
| --- | --- | --- |
| `vllm:num_requests_running` | gauge | sequences in execution batches right now. Compare against `max_num_seqs` from the startup log. |
| `vllm:num_requests_waiting` | gauge | admitted-but-not-scheduled. The single most under-used metric in LLM serving. |
| `vllm:num_requests_waiting_by_reason{reason="capacity"\|"deferred"}` | gauge | *why* they wait: `capacity` = the scheduler had no room this step; `deferred` = a transient constraint (LoRA budget, KV transfer). Set from `num_waiting_reqs` / `num_skipped_waiting_reqs` at `vllm/v1/metrics/loggers.py:1088-1099`. |

**Is memory the constraint?**

| Metric | Type | Reading |
| --- | --- | --- |
| `vllm:kv_cache_usage_perc` | gauge | fraction of KV blocks in use. `1.0` means the next sequence has nowhere to go. |
| `vllm:num_preemptions_total` | counter | requests evicted and later recomputed. Any increase means you are past the memory limit and paying for it twice. |
| `vllm:request_prefill_kv_computed_tokens` | histogram | new KV tokens computed during prefill, *excluding* cached ones. Spikes here mean recomputation after preemption. |
| `vllm:cache_config_info` | gauge = 1 | the resolved `CacheConfig` as labels: `block_size`, `cache_dtype`, `enable_prefix_caching`, `gpu_memory_utilization`. Use it to prove which config you are actually measuring. |

**Is the work getting cheaper?**

| Metric | Type | Reading |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` / `_hits_total` | counters | **in tokens, not blocks** (`loggers.py:636-653`). The design doc is explicit that these are counters rather than a hit-rate gauge so you can pick your own window (`docs/design/metrics.md:420-447`). |
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | counters | prefill work and decode work, counted separately. Throughput = `rate()` of these. |

**Latency — and these are four different things**

| Metric | Measures |
| --- | --- |
| `vllm:time_to_first_token_seconds` | arrival → first token. Prefill + queue + tokenization. |
| `vllm:inter_token_latency_seconds` | gap between *successive streamed outputs*. One sample per output event. |
| `vllm:request_time_per_output_token_seconds` | per finished request: `(e2e − TTFT) / (output tokens − 1)`. |
| `vllm:e2e_request_latency_seconds` | arrival → last token. |
| `vllm:request_queue_time_seconds`, `_prefill_time_seconds`, `_decode_time_seconds` | the decomposition. `e2e ≈ queue + prefill + decode`, so whichever term dominates names your bottleneck. |

ITL and TPOT are not the same number, and the design doc says so (`docs/design/metrics.md:68-80`):
they differ whenever one streamed output bundles several tokens, as under speculative decoding, and
for requests that generate one token or fewer. Quote which one you mean.

Two footnotes. The KV-residency histograms (`vllm:kv_block_lifetime_seconds`,
`_idle_before_evict_seconds`, `_reuse_gap_seconds`) are sampled and only emitted with
`--kv-cache-metrics` (`vllm/config/observability.py:62-67`) — do not alert on them. Deprecated
families hide behind `--show-hidden-metrics-for-version`.

---

## 3. The throughput/latency frontier

**Offline throughput is not comparable to online throughput.** Offline, nothing else exists: you
submit a fixed batch and divide tokens by wall clock, so the engine can sit at maximum batch size
for the whole run. Online, requests arrive and must each be *finished*, which means queueing,
preemption and a latency distribution. Worse, the two entrypoints do not even share defaults: on a
< 70 GB card `vllm serve` gets `--max-num-batched-tokens 2048` while `LLM()` gets `8192`
(`vllm/engine/arg_utils.py`, `get_batch_defaults`). Two numbers produced by those two paths are
different experiments. Say which one you ran.

**p99 matters more than the mean.** Means hide the tail, and the tail is where the SLOs live. A
scheduler that batches aggressively looks excellent at p50 and unacceptable at p99. Report both, and
report them for TTFT and inter-token latency *separately* — they are dominated by different things
(prefill/queue vs decode batch size), and an aggregate "latency" number cannot tell you which one
moved.

**The frontier is a curve, not a point.** As you raise concurrency, throughput rises, flattens, and
eventually falls (preemption and recompute). The knee is the only interesting place on the curve.
Lab 06 finds it by sweeping `--max-num-seqs` and `--max-num-batched-tokens`; Lab 08 shows you the
metrics that move as you cross it.

**Capacity and speed are different axes.** `--kv-cache-dtype fp8` doubles the tokens you can cache
in the same memory (Lab 10). `-tp=2` halves both the per-rank weights and the per-rank bytes per
token, so KV capacity grows much faster than the GPU count — but on PCIe every layer pays an
all-reduce, so decode throughput can fall (Lab 11). Neither trades latency for throughput; they
move the curve.

---

## 4. Designing a benchmark that is not lying to you

The failure mode is not a wrong number. It is a number that measures something other than what you
think.

1. **One variable at a time.** Restart the server between configurations; never change two flags
   and attribute the delta to one.
2. **Freeze the prompt distribution.** "Random prompts" with a fixed seed is reproducible; a real
   dataset is realistic but not repeatable across runs. Use `--random-input-len`,
   `--random-output-len` and `--seed` when you need repeatability, and say which you chose.
3. **Freeze the output length.** Set `ignore_eos` and a token cap so every request generates the
   same number of tokens. Then "generated tokens = requests × tokens" is arithmetic rather than a
   guess, and throughput comparisons are not polluted by one configuration being more verbose.
4. **Warm up, and exclude the warmup.** The first request pays CUDA-graph capture, `torch.compile`
   finalisation, kernel autotune and (for structured output) grammar compilation. Run one throwaway
   request before the clock starts. `vllm bench serve` has `--num-warmups` for exactly this.
5. **Enough requests that percentiles mean something.** A p99 from 20 samples is the second-worst
   sample. Hundreds, not dozens, if you intend to quote a p99.
6. **Beware the prefix cache.** Repeating a benchmark against the same server reuses prompts left in
   the cache and inflates throughput — vLLM's own doc warns about this in
   `$VLLM_SRC/docs/benchmarking/cli.md`. Vary the seed, restart the server, or use
   `vllm bench sweep serve`, which resets caches between runs.
7. **Record the version and the seed next to the number.** vLLM ships about every two weeks, and
   defaults have changed more than once (prefix caching and chunked prefill both flipped to on;
   `max_num_batched_tokens` moved). A number without a version is a rumour, including your own.
8. **State what you held fixed.** `labs/07`, `10` and `11` print the full resolved `vllm serve`
   command into their result files for this reason. Paste it into `notes/`.

---

## 5. `vllm bench` — which subcommand, when

Six subcommands live under `vllm bench` (`vllm/entrypoints/cli/benchmark/`):

| Subcommand | Use it when |
| --- | --- |
| `vllm bench serve` | **the default.** Benchmarks a *running* server over HTTP: arrival patterns, concurrency, TTFT/ITL/TPOT percentiles, throughput, optional goodput SLOs. This is the one Stage 7 uses. |
| `vllm bench latency` | you want a fixed batch's latency with no HTTP in the way. Good for A/B-ing a kernel or a compilation level. |
| `vllm bench throughput` | offline maximum throughput, in-process. The number to quote *only* if you also say it is offline. |
| `vllm bench startup` | cold and warm startup, split into total startup time and compilation time (`vllm/benchmarks/startup.py:47-50`). The honest way to measure what `-O2` costs you at boot. |
| `vllm bench sweep serve` | a parameter sweep: it starts the server, runs `vllm bench serve`, and resets caches between configurations. `sweep` also has `serve_workload`, `startup`, `plot` and `plot_pareto`. |
| `vllm bench mm-processor` | multimodal preprocessing cost, when the bottleneck is the image/audio pipeline rather than the model. |

The flags you will actually touch in `vllm bench serve`
(`vllm/benchmarks/serve.py:1607-2019`): `--backend` (`openai`, `openai-chat`, …),
`--base-url`, `--endpoint` (default `/v1/completions`; the chat backend *requires*
`/v1/chat/completions`), `--model`, `--dataset-name` (default `random`), `--num-prompts`
(default 1000), `--random-input-len` / `--random-output-len` / `--random-range-ratio`,
`--request-rate` (default `inf` = all at once), `--max-concurrency`, `--ignore-eos`, `--seed`
(default 0), `--percentile-metrics` (default `ttft,tpot,itl`) and `--metric-percentiles`
(default `99`), `--num-warmups`, `--save-result`, and `--ready-check-timeout-sec` — which defaults
to `0`, meaning **the readiness check is skipped unless you ask for it**.

The report is measured **at the client**, and the doc says so outright
(`$VLLM_SRC/docs/benchmarking/cli.md`, "Understanding the Latency Metrics"). That is why Lab 08
reconciles it against `/metrics`: the client can tell you the *distribution*, while the engine can
tell you *why*.

---

## 6. Metrics → diagnosis

| Observed | Likely cause | Confirm with |
| --- | --- | --- |
| `kv_cache_usage_perc` ≈ 1.0 **and** `waiting{capacity}` rising | KV cache exhausted; the scheduler cannot admit | `num_preemptions_total` climbing, `request_queue_time_seconds` rising |
| `waiting{capacity}` > 0 **but** `kv_cache_usage_perc` low | the token budget is the binder, not memory | the `iteration_tokens_total` histogram's top bucket; `--max-num-batched-tokens` |
| `num_requests_running` pinned at `max_num_seqs`, `waiting` > 0 | the sequence budget is the binder | startup log's `max_num_seqs`; `--max-num-seqs` |
| `num_preemptions_total` climbing | memory pressure: evict, then recompute | `request_prefill_kv_computed_tokens` (the recomputed work) |
| `prefix_cache_queries_total` rising, `_hits_total` flat | no reuse: unique prompts, or the shared prefix is shorter than one block | `cache_config_info` labels (`enable_prefix_caching`, `block_size`) |
| TTFT rising while `inter_token_latency_seconds` is flat | queue or prefill got slower; decode is healthy | `request_queue_time_seconds` vs `request_decode_time_seconds` |
| `inter_token_latency_seconds` rising while TTFT is flat | decode batch grew; the GPU is the limit | `num_requests_running`, `kv_cache_usage_perc` |
| `num_requests_running` < `max_num_seqs` **and** `waiting` = 0 | arrival-limited (starved): nothing to schedule | request rate; `rate(prompt_tokens_total)` |
| `generation_tokens_total` rate flat while `prompt_tokens_total` rate spikes | long prefills are stealing engine steps from decodes | chunked prefill on; `iteration_tokens_total` |
| every counter flat during "load" | you are not talking to that server | `/v1/models`; `curl /metrics` by hand |

---

## 7. Checkpoint

Answer these out loud, without notes:

- ✅ Which single metric tells you the engine is **saturated**, and which tells you it is **starved**? What does each look like on a chart?
- ✅ Why is throughput measured offline not comparable to throughput measured online — name two separate reasons.
- ✅ What is the difference between `vllm:kv_cache_usage_perc` and the startup log's `Maximum concurrency for N tokens per request`? Which one is a capacity and which is a pressure?
- ✅ Why must TTFT and inter-token latency be reported separately, and which one does a bigger decode batch move?
- ✅ Name three things that will silently inflate a `vllm bench serve` result if you repeat it against the same server.

---

## Read in this order

1. `$VLLM_SRC/docs/design/metrics.md` — the design intent, and why queries/hits are counters
2. `vllm/v1/metrics/loggers.py` for the definitions, `vllm/v1/metrics/buckets.py` for the buckets
3. `$VLLM_SRC/docs/benchmarking/cli.md` — the CLI and the latency-metric definitions
4. `vllm/benchmarks/serve.py` — `add_cli_args`, then `calculate_metrics`, then the report printer

**Lab:** [`labs/08_metrics_and_bench.sh`](../labs/08_metrics_and_bench.sh), then
[`labs/09_structured_outputs.py`](../labs/09_structured_outputs.py) to see what a per-step logits mask
does to the same numbers.
