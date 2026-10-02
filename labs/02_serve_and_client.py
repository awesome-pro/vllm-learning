#!/usr/bin/env python3
"""Lab 02 - Stage 1: serve it, speak HTTP, and measure TTFT vs inter-token latency.

WHAT IT DEMONSTRATES
  That production vLLM is a server, and that the two numbers every serving
  discussion is about are measurable by hand:
    * TTFT (time to first token) -- queueing + tokenization + PREFILL. It grows
      with prompt length.
    * inter-token latency (the gap between streamed tokens) -- DECODE. It stays
      roughly flat as the prompt grows, which is the whole reason prefill and
      decode are treated as different problems.
  Same prompt is sent twice: once non-streaming (you get nothing until the end)
  and once streaming (you can timestamp every chunk). Also exercises /v1/models
  and /health, the two endpoints a load balancer uses.

HOW TO RUN
    # terminal 1
    bash scripts/serve.sh
    # terminal 2
    source scripts/env.sh
    python labs/02_serve_and_client.py
    MAX_TOKENS=32 python labs/02_serve_and_client.py

PREREQUISITES
    The server must already be running (`bash scripts/serve.sh` in another
    terminal). Runtime ~20 s on the tiny model.

READING THE OUTPUT
    Everything is measured with `ignore_eos=True`, so both arms generate exactly
    MAX_TOKENS tokens and the comparison is apples to apples. All requests run at
    temperature 0: a measurement you cannot repeat is not a measurement.
"""

from __future__ import annotations

import os
import sys
import time

import _client as c

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
TIMEOUT = float(os.environ.get("TIMEOUT", "180"))

FILLER = (
    "PagedAttention stores the KV cache in fixed-size blocks and indexes them "
    "through a per-sequence block table, which is virtual memory for attention. "
)
QUESTION = "\n\nQuestion: in one sentence, what does the block table map to what?"

# ~40 prompt tokens and ~450 prompt tokens: enough separation that TTFT must move.
SHORT_PROMPT = FILLER * 5 + QUESTION
LONG_PROMPT = FILLER * 60 + QUESTION

# ignore_eos fixes the decode length; temperature=0 removes sampling variance.
FIXED = {"ignore_eos": True, "temperature": 0.0}


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def gap_summary(gaps: list[float]) -> str:
    """One-line min/median/max of the inter-token gaps, in milliseconds."""
    if not gaps:
        return "no gaps measured"
    ordered = sorted(gaps)
    return (f"n={len(ordered)}  min={ordered[0] * 1e3:6.1f} ms  "
            f"median={ordered[len(ordered) // 2] * 1e3:6.1f} ms  "
            f"max={ordered[-1] * 1e3:6.1f} ms")


def main() -> None:
    print(f"Looking for a vLLM server at {c.BASE} ...")
    try:
        served = c.models()
    except c.ServerNotRunning as exc:
        sys.exit(f"\n{exc}")
    model = served[0]
    print(f"OK - serving {served}")
    print(f"Using model id '{model}', max_tokens={MAX_TOKENS}, ignore_eos=True")

    section("1. The two endpoints a load balancer uses")
    print(f"  GET /v1/models -> {len(served)} model(s): {served}")
    try:
        c.get_text("/health", timeout=5)
        print("  GET /health    -> HTTP 200 (engine alive; the body is empty by design)")
    except RuntimeError as exc:
        print(f"  GET /health    -> {exc}")
        print("                   503 means the engine core died; check the server log.")

    section("2. The same request, non-streaming vs streaming")
    print(f"prompt: {len(SHORT_PROMPT):,} chars (~{len(SHORT_PROMPT) // 4} tokens)")

    text, elapsed, usage = c.chat(
        SHORT_PROMPT, model=model, max_tokens=MAX_TOKENS, timeout=TIMEOUT, **FIXED
    )
    print(f"\n  NON-STREAMING (one response at the very end)")
    print(f"    wall clock            : {elapsed:.3f}s")
    print(f"    usage                 : {usage}")
    print(f"    text (first 80 chars) : {text.strip()[:80]!r}")
    print("    There is no TTFT here: the client cannot see when the first token")
    print("    existed, only when the last one did.")

    print("\n  STREAMING (timestamping every chunk as it arrives)")
    ttft, total, gaps, streamed = c.measure_stream(
        SHORT_PROMPT, model=model, max_tokens=MAX_TOKENS, timeout=TIMEOUT, **FIXED
    )
    print(f"    TTFT                  : {ttft:.3f}s   <- queueing + tokenization + prefill")
    print(f"    total                 : {total:.3f}s")
    print(f"    decode after TTFT     : {total - ttft:.3f}s for {len(gaps) + 1} deltas")
    print(f"    inter-token gaps      : {gap_summary(gaps)}")
    print(f"    same text as above    : {text.strip()[:80] == streamed.strip()[:80]}")
    print("\n    Streaming did not make the engine faster. It made the engine")
    print("    *observable*, which is the only reason we can split the timeline.")

    section("3. TTFT scales with prompt length; the gap does not")
    rows = []
    measured = {}
    for label, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
        ttft_s, total_s, gap_list, _ = c.measure_stream(
            prompt, model=model, max_tokens=MAX_TOKENS, timeout=TIMEOUT, **FIXED
        )
        ordered = sorted(gap_list)
        median_gap = ordered[len(ordered) // 2] * 1e3 if ordered else float("nan")
        measured[label] = (ttft_s, total_s, median_gap, len(gap_list) + 1)
        rows.append([
            label,
            f"{len(prompt) // 4:,}",
            f"{ttft_s:.3f}",
            f"{total_s:.3f}",
            f"{median_gap:.1f}",
            f"{len(gap_list) + 1}",
        ])
    print()
    c.print_table(
        ["prompt", "~prompt tok", "TTFT s", "total s", "median gap ms", "deltas"], rows
    )
    short_ttft, _, short_gap, _ = measured["short"]
    long_ttft, _, long_gap, _ = measured["long"]
    print(f"\n  TTFT grew {long_ttft / short_ttft:.1f}x with the prompt "
          f"({short_ttft:.3f}s -> {long_ttft:.3f}s): prefill work is proportional")
    print("  to prompt tokens, and it is the dominant part of TTFT.")
    print(f"  The median gap moved {long_gap / short_gap:.2f}x "
          f"({short_gap:.1f} ms -> {long_gap:.1f} ms): a decode step costs about the")
    print("  same no matter how long the prompt was, because each step is dominated")
    print("  by reading the weights, not by the prompt. (It does creep up slowly: the")
    print("  longer sequence's KV cache must be read too, and attention cost grows")
    print("  with context - but nothing like linearly.)")

    section("4. Watch the GPU while this runs")
    print("  In a third terminal, run:")
    print("      watch -n 1 nvidia-smi --query-gpu=memory.used,utilization.gpu "
          "--format=csv")
    print("  and start this lab again. Two things to look for:")
    print("    * VRAM barely moves during decode. The KV cache was allocated in full")
    print("      at startup, so decode reuses memory instead of growing it - that is")
    print("      the point of paged attention. (Weights + KV are already resident.)")
    print("    * utilisation shows *spiky* bursts: each burst is one decode step")
    print("      serving the whole batch. A fast small model can leave the GPU idle")
    print("      between steps, which is exactly why CUDA graphs matter in Stage 6.")

    print("\nCHECKPOINT")
    print("  - Why can a server be fast (high tokens/sec) and still feel slow (high")
    print("    TTFT)? Which knob fixes each one? Answer in terms of prefill vs decode.")
    print("  - Why is tokenization in the API server but scheduling in the engine core?")

    c.record(
        f"lab 02 serve+client (model={model}, max_tokens={MAX_TOKENS})",
        short_prompt_tokens=len(SHORT_PROMPT) // 4,
        long_prompt_tokens=len(LONG_PROMPT) // 4,
        nonstreaming_wall_seconds=round(elapsed, 3),
        short_ttft_seconds=round(short_ttft, 3),
        long_ttft_seconds=round(long_ttft, 3),
        ttft_growth_factor=round(long_ttft / short_ttft, 2),
        short_median_gap_ms=round(short_gap, 2),
        long_median_gap_ms=round(long_gap, 2),
        gap_growth_factor=round(long_gap / short_gap, 2),
        short_decode_tokens_per_second=round(measured["short"][3] / (measured["short"][1] - short_ttft), 1),
    )


if __name__ == "__main__":
    main()
