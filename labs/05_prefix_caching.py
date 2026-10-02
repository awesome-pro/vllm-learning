#!/usr/bin/env python3
"""Lab 05 - Stage 4: prefix caching, and how to prove a hit.

WHAT IT DEMONSTRATES
  That a shared prefix is paid for once. Three requests, streamed so TTFT is exact:
    1. LONG shared prefix + question A          -> cold: full prefill
    2. the SAME prefix    + question B          -> warm: blocks reused, TTFT drops
    3. a prefix differing in its FIRST CONTENT TOKEN + question A
                                               -> control: reuse collapses
  Then vllm:prefix_cache_queries / vllm:prefix_cache_hits are scraped from /metrics
  around each request, so the engine's accounting and the timings can be compared.

  RUN IT TWICE: once against the default server (prefix caching ON) and once with
  `--no-enable-prefix-caching`. The lab reads vllm:cache_config_info and labels the
  run, so the two tables are distinguishable in your notes.

HOW TO RUN
    # terminal 1 (prefix caching is ON by default in v0.30.0)
    bash scripts/serve.sh
    # terminal 2
    source scripts/env.sh
    python labs/05_prefix_caching.py
    # restart terminal 1 as:  bash scripts/serve.sh $MODEL --no-enable-prefix-caching
    python labs/05_prefix_caching.py

PREREQUISITES
    The server must already be running. Runtime ~10 s on the tiny model.

WHY THE THIRD REQUEST IS THE CONTROL
  A block is keyed on a hash CHAIN: its own tokens plus the hash of the block before
  it (vllm/v1/core/kv_cache_utils.py). Diverging at the first content token changes
  the first block's hash and therefore every descendant's, so none of the shared
  filler can be reused. That is why a one-word edit at the START of a prompt is
  expensive while appending to the END is nearly free.

  The chat template renders some boilerplate before the user content, so the marker
  below is the first token of the *user content* - which is what the experiment
  needs: the divergence lands before the filler, inside the first content block.
"""

from __future__ import annotations

import os
import sys

import _client as c

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
TIMEOUT = float(os.environ.get("TIMEOUT", "180"))
REPEATS = int(os.environ.get("REPEATS", "3"))  # filler multiplier for the shared prefix
# Set PREFIX_NONCE to force a genuinely cold prefix on a server that has already
# served this lab:  PREFIX_NONCE=$(date +%s) python labs/05_prefix_caching.py
NONCE = os.environ.get("PREFIX_NONCE", "")

FILLER = (
    "Retrieval augmented generation grounds a model in documents it was not trained "
    "on, which reduces hallucination but makes the prompt long and repetitive across "
    "requests, so serving it efficiently is mostly a question of not recomputing the "
    "same prefix over and over again. "
)
SHARED = FILLER * REPEATS
SHARED_START = f"OMEGA {NONCE} " + SHARED
OTHER_START = f"ALPHA {NONCE} " + SHARED  # same length, different first content token

# ignore_eos fixes the decode length, so any total-time difference is a prefill
# difference and nothing else.
FIXED = {"temperature": 0.0, "ignore_eos": True, "max_tokens": MAX_TOKENS}


def prefix_counters() -> tuple[float, float]:
    """(hits, queries) token counters from /metrics, or (nan, nan) if absent."""
    try:
        m = c.metrics()
        return (float(m.get("vllm:prefix_cache_hits", float("nan"))),
                float(m.get("vllm:prefix_cache_queries", float("nan"))))
    except Exception:
        return float("nan"), float("nan")


def run_case(label: str, prefix: str, question: str, model: str) -> dict:
    """Stream one request; return its timings plus the cache-counter deltas."""
    hits_before, queries_before = prefix_counters()
    ttft, total, gaps, text = c.measure_stream(
        prefix + f"\n\nQuestion: {question}", model=model, timeout=TIMEOUT, **FIXED
    )
    hits_after, queries_after = prefix_counters()
    return {
        "case": label,
        "ttft": ttft,
        "total": total,
        "decode": total - ttft,
        "deltas": len(gaps) + 1,
        "hits": hits_after - hits_before,
        "queries": queries_after - queries_before,
        "answer": text.strip()[:40],
    }


def main() -> None:
    print(f"Looking for a vLLM server at {c.BASE} ...")
    try:
        served = c.models()
    except c.ServerNotRunning as exc:
        sys.exit(f"\n{exc}")
    model = served[0]
    print(f"OK - serving {served}")

    info: dict = {}
    try:
        info = c.labels("vllm:cache_config_info")
    except Exception:
        pass
    caching = info.get("enable_prefix_caching", "unknown")
    try:
        block_size = int(str(info.get("block_size", "16")))
    except ValueError:
        block_size = 16

    if caching == "False":
        print("\n>>> enable_prefix_caching=False: this server runs with")
        print(">>> --no-enable-prefix-caching. Expect cases 1 and 2 to match, and the")
        print(">>> cache counters to stay at exactly zero (the lookup never runs).")
    elif caching == "True":
        print("\n>>> enable_prefix_caching=True (the v0.30.0 default). Expect case 2's")
        print(">>> TTFT to collapse and case 3 to be cold again.")
    else:
        print("\n>>> Could not read enable_prefix_caching from /metrics; the measured")
        print(">>> effect will tell you which configuration you are on.")
    print(f">>> block_size={block_size} tokens: reuse is whole-block, so a partial")
    print(">>> trailing block is never shareable.")

    shared_tokens = len(SHARED_START) // 4
    print(f"\nshared prefix : {len(SHARED_START):,} chars, ~{shared_tokens:,} tokens "
          f"(~{shared_tokens // block_size} blocks at block_size={block_size})")
    print(f"decode length : exactly {MAX_TOKENS} tokens per case (ignore_eos, temp 0)")
    if NONCE:
        print(f"prefix nonce  : {NONCE!r} (a fresh prefix, so case 1 is truly cold)")

    print("\n" + "=" * 74)
    print("THREE CASES (streamed, so TTFT is measured from the wire)")
    print("=" * 74)
    cases = [
        ("1 cold (first use of this prefix)", SHARED_START, "What is RAG?"),
        ("2 warm (identical prefix)", SHARED_START, "Name one downside of it."),
        ("3 control (different FIRST token)", OTHER_START, "What is RAG?"),
    ]
    results = []
    for label, prefix, question in cases:
        result = run_case(label, prefix, question, model)
        results.append(result)
        print(f"\n  {label}")
        print(f"    TTFT            : {result['ttft']:.3f}s")
        print(f"    total           : {result['total']:.3f}s "
              f"(decode after TTFT {result['decode']:.3f}s, {result['deltas']} deltas)")
        print(f"    cache counters  : +{result['queries']:,.0f} queries, "
              f"+{result['hits']:,.0f} hit tokens")
        print(f"    answer          : {result['answer']!r}")

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    cold = results[0]["ttft"]
    print()
    c.print_table(
        ["case", "TTFT s", "vs cold", "total s", "cache queries", "cache hits", "hit rate"],
        [[r["case"], f"{r['ttft']:.3f}",
          f"{cold / r['ttft']:.2f}x" if r["ttft"] > 0 else "?",
          f"{r['total']:.3f}", f"{r['queries']:,.0f}", f"{r['hits']:,.0f}",
          f"{r['hits'] / r['queries'] * 100:.1f}%" if r["queries"] > 0 else "n/a"]
         for r in results],
    )

    warm = results[1]
    control = results[2]
    if warm["ttft"] > 0:
        print(f"\n  case 2 vs case 1 TTFT : {cold / warm['ttft']:.2f}x faster")
    if control["ttft"] > 0:
        print(f"  case 3 vs case 1 TTFT : {cold / control['ttft']:.2f}x faster")
    if cold > 0 and control["ttft"] > 0 and warm["ttft"] > 0.7 * cold and caching == "True":
        print("\n  NOTE: case 1 does not look cold - its prefix was probably already")
        print("  cached by an earlier run (prefix caching is on by default). Restart the")
        print("  server, or force a fresh prefix, and run it again:")
        print("      PREFIX_NONCE=$(date +%s) python labs/05_prefix_caching.py")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print(f"""  PREFIX CACHING ON (the default) - what you should see:
    * case 1 pays full prefill for ~{shared_tokens:,} tokens; TTFT is high.
    * case 2 finds case 1's blocks already computed and prefills only its own
      question, so TTFT falls sharply and vllm:prefix_cache_hits jumps.
    * case 3 shares the same filler but diverges at the first content token, so the
      hash chain invalidates everything after it: hit rate ~0, TTFT back to case 1.
      The control is what proves case 2 was REUSE and not just warm hardware.

  PREFIX CACHING OFF (--no-enable-prefix-caching) - what you should see instead:
    * cases 1 and 2 have the same TTFT within noise.
    * vllm:prefix_cache_queries and _hits BOTH stay at exactly zero. Not "hits=0
      with queries>0": the lookup does not run at all, so nothing is counted
      (vllm/v1/core/kv_cache_manager.py returns early when caching is disabled).
    * record both tables in notes/: the case-2 TTFT difference between the two runs
      IS the value of the feature, in seconds.

  TWO DETAILS THAT EXPLAIN IMPERFECT NUMBERS
    * The hit rate can never reach 100%: vLLM caps a cache hit at
      `num_tokens - 1`, because the last prompt token must be recomputed to produce
      logits (vllm/v1/core/kv_cache_manager.py). Expect roughly
      shared_tokens / total_prompt_tokens.
    * The first request also pays one-off costs (kernel autotuning, CUDA graph
      replay warm-up). That is why case 1 looks cold even on a warm server, and why
      case 3 - not case 1 - is the honest baseline.

  A NOTE ON NAMES: in v0.30.0 the counters are `vllm:prefix_cache_queries` and
  `vllm:prefix_cache_hits` (vllm/v1/metrics/loggers.py). Older releases used
  `vllm:gpu_prefix_cache_queries` / `_hits` and, before that, a
  `vllm:gpu_prefix_cache_hit_rate` gauge. If the cache columns print "n/a", scrape
  /metrics yourself and check the names before concluding the feature is broken.

  CHECKPOINT
    - What exactly is hashed to decide a block is reusable, and why content-based
      rather than address-based?
    - Why can an unfilled trailing block never be shared?
    - Requests are served in a different order than they arrive. What does that
      imply for the hit rate in production?""")

    print(f"\nNow run this again with the server restarted as:")
    print(f"    bash scripts/serve.sh {model} --no-enable-prefix-caching")
    print("and keep both tables - the difference between them is the point.")

    c.record(
        f"lab 05 prefix caching (model={model}, prefix_caching={caching}, "
        f"max_tokens={MAX_TOKENS})",
        enable_prefix_caching=caching,
        block_size=block_size,
        shared_prefix_tokens=shared_tokens,
        cold_ttft_seconds=round(results[0]["ttft"], 3),
        warm_same_prefix_ttft_seconds=round(warm["ttft"], 3),
        control_different_first_token_ttft_seconds=round(control["ttft"], 3),
        warm_speedup_vs_cold=(round(cold / warm["ttft"], 2) if warm["ttft"] else "n/a"),
        warm_cache_hit_tokens=warm["hits"],
        warm_cache_queries=warm["queries"],
        control_cache_hits=control["hits"],
        prefix_nonce=NONCE or "(deterministic prefix)",
    )


if __name__ == "__main__":
    main()
