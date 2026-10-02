#!/usr/bin/env python3
"""Lab 03 - Stage 2: continuous batching, measured on a real engine.

WHAT IT DEMONSTRATES
  The experiment this project exists for. N identical requests are sent two ways,
  back to back, against the same running server with the same prompts:
    (a) SEQUENTIALLY, one request at a time -- what a naive server does, and what
        "static batching" degrades to when arrivals are spread out;
    (b) CONCURRENTLY, all N in flight -- the scheduler decides at every decode step
        which sequences are in the batch, so one forward pass serves all of them.
  Total wall clock should collapse while per-request latency rises only slightly.
  That gap is the value of continuous batching, and it is why vLLM exists.

  To be precise about the comparison: vLLM has no "static batching" mode to switch
  on, so arm A is the baseline such a server would give you -- one request in the
  engine at a time -- and arm B is vLLM's continuous batching. Same engine, same
  weights, same prompts, same parameters. Nothing here is simulated.

HOW TO RUN
    # terminal 1
    bash scripts/serve.sh
    # terminal 2
    source scripts/env.sh
    python labs/03_continuous_batching.py
    N=16 MAX_TOKENS=32 python labs/03_continuous_batching.py

PREREQUISITES
    The server must already be running. N=8 with MAX_TOKENS=64 finishes in well
    under a minute on the tiny model; raise N slowly and find the knee.

PREFIX CACHING CONTAMINATES THIS COMPARISON - READ THIS
    Prefix caching is ON by default (vllm/config/cache.py). Both arms start from the
    same warm cache (this lab warms up first), but the *sequential* arm gets a warm
    cache for every later request, which flatters it. The clean comparison runs
    against a server with caching off:

        bash scripts/serve.sh $MODEL --no-enable-prefix-caching   # restart the server

    Re-run this lab there and compare the speedup factor. Lab 05 isolates the
    caching effect on its own.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import statistics
import sys
import time

import _client as c

N = int(os.environ.get("N", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
TIMEOUT = float(os.environ.get("TIMEOUT", "300"))

FILLER = (
    "Continuous batching makes a scheduling decision at every decode step: "
    "finished sequences leave the batch, waiting ones join it, and a prefill can "
    "share a step with ongoing decodes. "
)
QUESTIONS = [
    "Why does batching raise aggregate throughput?",
    "What does one forward pass serve?",
    "Why does per-request latency not scale linearly?",
    "Where does a finished request free its slot?",
]

# ONE list of prompts, used by BOTH arms: identical work, so the only variable is
# how the requests are submitted.
PROMPTS = [
    FILLER * 6 + f"\n\nQuestion: {QUESTIONS[i % len(QUESTIONS)]} (request {i})"
    for i in range(N)
]

# temperature=0 and ignore_eos=True: every request decodes exactly MAX_TOKENS
# tokens, so both arms do identical work and the token counts are comparable.
FIXED = {"temperature": 0.0, "ignore_eos": True}


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation: N is small)."""
    if not values:
        return float("nan")
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def one_request(i: int, model: str) -> tuple[float, int]:
    """Send request i non-streaming; return (latency_seconds, generated_tokens)."""
    _, elapsed, usage = c.chat(
        PROMPTS[i], model=model, max_tokens=MAX_TOKENS, timeout=TIMEOUT, **FIXED
    )
    return elapsed, int(usage.get("completion_tokens", MAX_TOKENS))


def engine_tokens() -> float:
    """Engine-side generation-token counter, or NaN if /metrics is unavailable."""
    try:
        return float(c.metrics()["vllm:generation_tokens"])
    except Exception:
        return float("nan")


def summarise(label: str, results: list[tuple[float, int]], wall: float,
              engine_generated: float) -> dict:
    """Turn one arm's raw results into the numbers the comparison needs."""
    latencies = [r[0] for r in results]
    tokens = sum(r[1] for r in results)
    return {
        "arm": label,
        "wall_s": wall,
        "mean_lat_s": statistics.mean(latencies),
        "p50_s": percentile(latencies, 0.50),
        "p95_s": percentile(latencies, 0.95),
        "gen_tokens": tokens,
        "tok_per_s": tokens / wall,
        "engine_tok_per_s": engine_generated / wall,
    }


def print_arm(name: str, s: dict) -> None:
    """Print one arm's numbers, including the engine counter cross-check."""
    print(f"\n  {name}")
    print(f"    total wall clock     : {s['wall_s']:.2f}s")
    print(f"    per-request latency  : mean {s['mean_lat_s']:.2f}s  p50 {s['p50_s']:.2f}s  "
          f"p95 {s['p95_s']:.2f}s")
    print(f"    generated tokens     : {s['gen_tokens']}")
    print(f"    aggregate throughput : {s['tok_per_s']:.1f} output tok/s (from usage)")
    if s["engine_tok_per_s"] == s["engine_tok_per_s"]:  # NaN check
        print(f"    engine counter says  : {s['engine_tok_per_s']:.1f} output tok/s "
              "(vllm:generation_tokens delta / wall)")
        print("                           The two should agree: /metrics counts the same")
        print("                           tokens the responses report in usage.")


def main() -> None:
    print(f"Looking for a vLLM server at {c.BASE} ...")
    try:
        served = c.models()
    except c.ServerNotRunning as exc:
        sys.exit(f"\n{exc}")
    model = served[0]
    print(f"OK - serving {served}")
    print(f"model={model}  N={N}  max_tokens={MAX_TOKENS}  temperature=0  ignore_eos=True")

    print("\nWarming up: loads kernels, captures CUDA graphs, fills the prefix cache.")
    print("Both arms therefore start from the same warm state - and the shared")
    print("instruction prefix is already cached before either arm runs.")
    c.chat(PROMPTS[0], model=model, max_tokens=4, timeout=TIMEOUT, **FIXED)

    # --- arm A: sequential ---------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"ARM A: {N} requests SEQUENTIALLY (one at a time, as a naive server would)")
    print("=" * 72)
    before = engine_tokens()
    t0 = time.perf_counter()
    sequential = [one_request(i, model) for i in range(N)]
    seq_wall = time.perf_counter() - t0
    seq_engine_tokens = engine_tokens() - before
    print(f"done in {seq_wall:.2f}s")

    # --- arm B: concurrent ---------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"ARM B: {N} requests CONCURRENTLY (all in flight, one thread pool)")
    print("=" * 72)
    before = engine_tokens()
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        concurrent_results = list(pool.map(lambda i: one_request(i, model), range(N)))
    con_wall = time.perf_counter() - t0
    con_engine_tokens = engine_tokens() - before
    print(f"done in {con_wall:.2f}s")

    seq = summarise("sequential", sequential, seq_wall, seq_engine_tokens)
    con = summarise("concurrent", concurrent_results, con_wall, con_engine_tokens)

    # --- comparison table ----------------------------------------------------
    print(f"\n{'=' * 72}")
    print("RESULTS")
    print("=" * 72)
    print()
    c.print_table(
        ["arm", "wall s", "mean lat s", "p50 s", "p95 s", "gen tok", "tok/s"],
        [[s["arm"], f"{s['wall_s']:.2f}", f"{s['mean_lat_s']:.2f}", f"{s['p50_s']:.2f}",
          f"{s['p95_s']:.2f}", f"{s['gen_tokens']}", f"{s['tok_per_s']:.1f}"]
         for s in (seq, con)],
    )
    print_arm("ARM A detail (sequential)", seq)
    print_arm("ARM B detail (concurrent)", con)

    speedup = seq_wall / con_wall if con_wall > 0 else float("inf")
    lat_ratio = con["mean_lat_s"] / seq["mean_lat_s"] if seq["mean_lat_s"] > 0 else float("inf")
    thr_ratio = con["tok_per_s"] / seq["tok_per_s"] if seq["tok_per_s"] > 0 else float("inf")
    print(f"\n  wall-clock speedup from concurrency : {speedup:.2f}x  "
          f"({seq_wall:.2f}s -> {con_wall:.2f}s)")
    print(f"  mean per-request latency            : {seq['mean_lat_s']:.2f}s -> "
          f"{con['mean_lat_s']:.2f}s  ({lat_ratio:.2f}x)")
    print(f"  aggregate throughput                : {seq['tok_per_s']:.1f} -> "
          f"{con['tok_per_s']:.1f} output tok/s  ({thr_ratio:.2f}x)")

    print(f"\n{'=' * 72}")
    print("HOW TO READ THIS")
    print("=" * 72)
    print(f"""  Both of these are true at once, and that is the whole point:
    * aggregate throughput rose {thr_ratio:.1f}x with {N} requests in flight;
    * per-request latency rose only {lat_ratio:.2f}x - nowhere near {N}x.

  Sequentially the GPU idles through most of every decode step: one sequence means
  one token per forward pass, and each pass is dominated by reading the weights.
  Concurrently one pass serves all {N} sequences, so the weight read is amortised
  and the marginal cost of a sequence is its KV cache read, not another whole pass.

  This is static-vs-continuous batching measured on a real engine: arm A is the
  one-request-at-a-time baseline a naive server gives you, arm B is vLLM's
  continuous batching.

  PREFIX CACHING CAVEAT. The default server has prefix caching ON, and arm A's
  later requests hit the blocks its earlier requests filled - which makes arm A
  look *better* than it would on a cold engine, so the speedup you measured is a
  lower bound. For the clean number, restart with caching off and re-run:

      # Ctrl-C the server, then:
      bash scripts/serve.sh {model} --no-enable-prefix-caching
      python labs/03_continuous_batching.py

  CHECKPOINT
    - Raise N (16, 32). At what N does per-request latency start growing faster
      than throughput? That knee is your capacity on this card.
    - Which arm does prefix caching help more, and why?""")

    c.record(
        f"lab 03 continuous batching (model={model}, N={N}, max_tokens={MAX_TOKENS})",
        sequential_wall_seconds=round(seq_wall, 2),
        concurrent_wall_seconds=round(con_wall, 2),
        speedup_factor=round(speedup, 2),
        sequential_mean_latency_seconds=round(seq["mean_lat_s"], 2),
        concurrent_mean_latency_seconds=round(con["mean_lat_s"], 2),
        latency_ratio=round(lat_ratio, 2),
        sequential_tokens_per_second=round(seq["tok_per_s"], 1),
        concurrent_tokens_per_second=round(con["tok_per_s"], 1),
        generated_tokens_per_arm=seq["gen_tokens"],
        prefix_caching_setting="ON (re-run with --no-enable-prefix-caching)",
    )


if __name__ == "__main__":
    main()
