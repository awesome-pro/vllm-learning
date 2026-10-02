#!/usr/bin/env python3
"""Lab 01 - Stage 0: the offline `LLM` class, and where a token's time goes.

WHAT IT DEMONSTRATES
  The offline entrypoint: no HTTP server, no separate process. This is the same V1
  engine that `vllm serve` drives, so everything here transfers. You see:
    * constructing `LLM`, and how long the model + KV cache take to come up,
    * one greedy generation,
    * a small batch of prompts handed over in ONE `llm.generate()` call -- the
      offline API does the batching for you, which is continuous batching at work,
    * wall-clock tokens/sec *and* the engine's own per-request timing, split into
      prefill (TTFT) and decode.

HOW TO RUN
    source scripts/env.sh
    python labs/01_offline_inference.py
    MODEL=$MODEL_MID python labs/01_offline_inference.py      # slower, realistic
    MAX_TOKENS=32 python labs/01_offline_inference.py

PREREQUISITES
    None - no server needed. A GPU pod with vLLM installed in $VENV. Runtime ~40 s
    on the tiny model (plus a one-off download if $MODEL is not cached).

WHY PREFILL AND DECODE THROUGHPUT ARE REPORTED SEPARATELY
  They are different physical regimes, and one blended number hides which one you
  changed:
    * Prefill processes ALL of a request's prompt tokens in one forward pass. It is
      compute-bound (large GEMMs) and highly parallel, so it reports thousands of
      tokens/sec -- but it happens once per request, and its latency is what you
      feel as TTFT.
    * Decode produces ONE token per sequence per forward pass, and each step reads
      the whole weight set plus that sequence's KV cache. It is memory-bandwidth
      bound, so it reports tens of tokens/sec per sequence -- and batching helps it
      enormously, because one step's weight read serves every sequence in the batch.
  Collapse the two and you cannot tell whether a change helped the phase you care
  about. vLLM's own log line keeps them apart ('Avg prompt throughput' vs
  'Avg generation throughput') for exactly this reason.
"""

from __future__ import annotations

import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
UTIL = float(os.environ.get("UTIL", "0.90"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "48"))

# Deterministic prompts: a repeated filler paragraph plus a short question, so
# every run prefills the same number of tokens (the project's prompt style).
FILLER = (
    "PagedAttention stores the KV cache in fixed-size blocks and indexes them "
    "through a per-sequence block table, which is virtual memory for attention. "
)
SINGLE_PROMPT = FILLER * 8 + "\n\nQuestion: in one sentence, why do blocks matter?"

BATCH_PROMPTS = [
    FILLER * 4 + "\n\nQuestion: what does a block table map, and who owns the blocks?",
    FILLER * 8 + "\n\nQuestion: why is the KV cache the real limit, not the weights?",
    FILLER * 12 + "\n\nQuestion: what is the difference between prefill and decode?",
    FILLER * 16 + "\n\nQuestion: when does prefix caching pay off?",
]


def fmt(seconds: float) -> str:
    """Format a duration, tolerating a measurement that never happened (NaN)."""
    return "n/a" if seconds != seconds else f"{seconds:.3f}s"  # NaN != NaN


def rate(tokens: int, seconds: float) -> str:
    """Format a tokens/sec figure, or n/a when the denominator is unusable."""
    return f"{tokens / seconds:,.0f}" if seconds and seconds > 0 else "n/a"


def request_timings(out) -> dict | None:
    """Per-request timings from `RequestOutput.metrics`, or None if unavailable.

    `RequestOutput.metrics` is a `vllm.v1.metrics.stats.RequestStateStats`
    (v0.30.0), and it is populated ONLY when stat logging is on -- which is why
    this lab passes `disable_log_stats=False`. `LLM.__init__` otherwise forces
    `disable_log_stats=True` (vllm/entrypoints/llm.py), and then `out.metrics` is
    None. Fields verified in `vllm/v1/metrics/stats.py`:

        arrival_time         wall clock at the engine frontend (NOT comparable
                             with the monotonic core timestamps below)
        queued_ts            engine-core monotonic: entered the waiting queue
        scheduled_ts         engine-core monotonic: first scheduled into a step
        first_token_ts       engine-core monotonic: first output token produced
        last_token_ts        engine-core monotonic: final output token produced
        first_token_latency  seconds from arrival to first token (== TTFT)
        num_generation_tokens, num_preemptions

    The differences below are the engine's own definitions, from
    `vllm/v1/engine/output_processor.py`: queued = scheduled - queued,
    prefill = first_token - scheduled, decode = last_token - first_token.
    """
    metrics = getattr(out, "metrics", None)
    if metrics is None or not getattr(metrics, "last_token_ts", 0.0):
        return None  # caller falls back to wall-clock arithmetic
    return {
        "queued_s": max(0.0, metrics.scheduled_ts - metrics.queued_ts),
        "prefill_s": max(0.0, metrics.first_token_ts - metrics.scheduled_ts),
        "decode_s": max(0.0, metrics.last_token_ts - metrics.first_token_ts),
        "ttft_s": metrics.first_token_latency,
        "gen_tokens": metrics.num_generation_tokens,
        "scheduled_ts": metrics.scheduled_ts,
        "last_token_ts": metrics.last_token_ts,
    }


def main() -> None:
    print(f"Loading {MODEL} ...  (the slow part - watch the log lines)")
    print(f"max_model_len={MAXLEN}  gpu_memory_utilization={UTIL}")
    print("Note: disable_log_stats=False is REQUIRED for per-request metrics;")
    print("      LLM() defaults it to True, and then RequestOutput.metrics is None.")
    load_start = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=MAXLEN,
        gpu_memory_utilization=UTIL,
        disable_log_stats=False,  # keep RequestStateStats + the stat loggers on
    )
    load_s = time.perf_counter() - load_start
    print(f"\nModel + KV cache ready in {load_s:.1f}s\n")

    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # --- 1. One request -----------------------------------------------------
    print("=" * 72)
    print(f"1. ONE GREEDY REQUEST   (temperature=0, max_tokens={MAX_TOKENS})")
    print("=" * 72)
    t0 = time.perf_counter()
    single = llm.generate([SINGLE_PROMPT], greedy, use_tqdm=False)[0]
    single_wall = time.perf_counter() - t0
    n_prompt = len(single.prompt_token_ids or [])
    n_single = len(single.outputs[0].token_ids)
    print(f"prompt   : {len(SINGLE_PROMPT):,} chars, {n_prompt:,} prompt tokens")
    print(f"output   : {single.outputs[0].text.strip()[:300]!r}")
    print(f"tokens   : {n_single} generated (finish_reason={single.outputs[0].finish_reason})")
    print(f"wall     : {single_wall:.3f}s -> {(n_prompt + n_single) / single_wall:,.0f} tok/s "
          "blended (the next section unpicks why 'blended' is a trap)")

    timings = request_timings(single)
    print("\nengine-reported breakdown (RequestOutput.metrics -> RequestStateStats):")
    if timings is None:
        print("  unavailable - the engine attached no per-request stats.")
        print("  Fallback: only the wall clock above is usable, and it cannot")
        print("  separate prefill from decode.")
    else:
        print(f"  queueing        : {fmt(timings['queued_s'])}")
        print(f"  prefill (TTFT)  : {fmt(timings['prefill_s'])}  "
              f"({rate(n_prompt, timings['prefill_s'])} prompt tok/s)")
        print(f"  decode          : {fmt(timings['decode_s'])}  "
              f"({rate(timings['gen_tokens'], timings['decode_s'])} output tok/s)")
        print(f"  first_token_latency (arrival -> first token): {fmt(timings['ttft_s'])}")
        print(f"  generated tokens the engine counted          : {timings['gen_tokens']}")

    # --- 2. A batch of prompts in ONE call ----------------------------------
    print("\n" + "=" * 72)
    print(f"2. BATCH OF {len(BATCH_PROMPTS)} PROMPTS IN ONE llm.generate() CALL")
    print("=" * 72)
    print("You never chose a batch size: you handed over a list, the engine admitted")
    print("what fit, and the rest waited for slots to free up. Continuous batching.")
    t0 = time.perf_counter()
    batch = llm.generate(BATCH_PROMPTS, greedy, use_tqdm=False)
    batch_wall = time.perf_counter() - t0

    print(f"\n  {'req':<4}{'prompt tok':>11}{'gen tok':>9}{'prefill':>10}{'decode':>10}"
          f"{'decode tok/s':>14}  finish")
    all_timings = []
    for i, out in enumerate(batch, 1):
        t = request_timings(out)
        all_timings.append(t)
        gen = len(out.outputs[0].token_ids)
        print(f"  {f'#{i}':<4}{len(out.prompt_token_ids or []):>11,}{gen:>9}"
              f"{(fmt(t['prefill_s']) if t else 'n/a'):>10}"
              f"{(fmt(t['decode_s']) if t else 'n/a'):>10}"
              f"{(rate(gen, t['decode_s']) if t else 'n/a'):>14}"
              f"  {out.outputs[0].finish_reason or '?'}")

    total_gen = sum(len(o.outputs[0].token_ids) for o in batch)
    total_prompt = sum(len(o.prompt_token_ids or []) for o in batch)
    print(f"\nwall clock for the whole batch : {batch_wall:.3f}s")
    print(f"prompt tokens                  : {total_prompt:,}")
    print(f"generated tokens               : {total_gen:,}")
    print(f"output tokens/sec (wall clock) : {total_gen / batch_wall:.1f}")

    decode_rate = float("nan")
    if all(t is not None for t in all_timings):
        # The engine-core window: first request scheduled -> last request finished.
        window = max(t["last_token_ts"] for t in all_timings) - min(
            t["scheduled_ts"] for t in all_timings
        )
        decode_rate = sum(t["gen_tokens"] for t in all_timings) / window
        print("\nfrom the engine's own timestamps (whole batch):")
        print(f"  batch window (first scheduled -> last finished) : {window:.3f}s")
        print(f"  aggregate decode rate over that window          : {decode_rate:.1f} output tok/s")
        print(f"  sum of per-request prefill time                 : "
              f"{sum(t['prefill_s'] for t in all_timings):.3f}s")
        print("  Compare the per-request decode rates above with the aggregate: the batch")
        print("  serves N sequences per weight read, so aggregate decode scales roughly")
        print("  with batch size while per-request decode barely moves. That gap IS")
        print("  continuous batching.")

    # --- 3. The engine's counters -------------------------------------------
    print("\n" + "=" * 72)
    print("3. WHAT THE ENGINE COUNTED  (llm.get_metrics(), V1 engine only)")
    print("=" * 72)
    try:
        wanted = {
            "vllm:prompt_tokens",
            "vllm:generation_tokens",
            "vllm:request_success",
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
        }
        found: dict[str, float] = {}
        for metric in llm.get_metrics():
            if metric.name in wanted:
                found[metric.name] = found.get(metric.name, 0.0) + getattr(metric, "value", 0.0)
        if found:
            for name in sorted(found):
                print(f"  {name:<28} {found[name]}")
            print("\n  prompt_tokens are prefill work; generation_tokens are decode work.")
            print("  vLLM's periodic log line reports them as two separate throughputs.")
            print("  (A run this short may not reach a log tick - that is expected.)")
        else:
            print("  no vllm:* counters exposed in this process.")
    except Exception as exc:  # llm.get_metrics asserts that log_stats is enabled
        print(f"  llm.get_metrics() unavailable: {type(exc).__name__}: {exc}")

    # --- record --------------------------------------------------------------
    prefill_avg = (
        sum(t["prefill_s"] for t in all_timings if t) / len(all_timings)
        if all_timings
        else float("nan")
    )
    ttft0 = all_timings[0]["ttft_s"] if all_timings and all_timings[0] else float("nan")

    print("\n" + "=" * 72)
    print(f"RECORD: lab 01 offline LLM  (model={MODEL}, max_model_len={MAXLEN}, "
          f"max_tokens={MAX_TOKENS})")
    print("=" * 72)
    print(f"  model_load_seconds                 {load_s:.2f}")
    print(f"  single_request_wall_seconds        {single_wall:.3f}")
    print(f"  single_prompt_tokens               {n_prompt}")
    print(f"  batch_size                         {len(BATCH_PROMPTS)}")
    print(f"  batch_wall_seconds                 {batch_wall:.3f}")
    print(f"  batch_prompt_tokens                {total_prompt}")
    print(f"  batch_generated_tokens             {total_gen}")
    print(f"  batch_output_tokens_per_second     {total_gen / batch_wall:.1f}")
    print(f"  mean_per_request_prefill_seconds   {prefill_avg:.4f}")
    print(f"  aggregate_decode_tokens_per_second {decode_rate:.1f}")
    print(f"  first_request_ttft_seconds         {ttft0:.4f}")
    print("-" * 72)
    print("Write these into notes/01-offline.md with the vLLM version and the date.")
    print("CHECKPOINT: why does the batch beat the single request on total tokens/sec")
    print("while barely changing any single request's decode rate?")


if __name__ == "__main__":
    main()
