#!/usr/bin/env python3
"""Lab 03 - continuous batching, measured.

You cannot *turn off* continuous batching in vLLM, so instead we measure the
consequence: what happens to aggregate throughput and to per-request latency as
concurrency rises.

Start the server first:   bash scripts/serve.sh
Then:                     bash scripts/py.sh labs/03_continuous_batching.py

Useful env knobs:
    N=8              number of requests in each phase
    MAX_TOKENS=64    tokens generated per request
"""

from __future__ import annotations

import concurrent.futures
import os
import statistics
import time

import _client as c

BASE = c.DEFAULT_BASE
N = int(os.environ.get("N", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))

PROMPT = (
    "Write a short, self-contained paragraph explaining why batching multiple "
    "requests together improves the throughput of an LLM inference server. "
    "Be concrete and avoid filler."
)

CHAT_KWARGS = {"chat_template_kwargs": {"enable_thinking": False}}


def one_request(i: int) -> tuple[float, int]:
    _, elapsed, usage = c.chat_raw(
        f"{PROMPT} (request {i})",
        base_url=BASE,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        **CHAT_KWARGS,
    )
    return elapsed, int(usage.get("completion_tokens", 0))


def summarise(label: str, results: list[tuple[float, int]], wall: float) -> dict:
    latencies = sorted(r[0] for r in results)
    tokens = sum(r[1] for r in results)
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    return {
        "mode": label,
        "wall_s": f"{wall:.2f}",
        "mean_lat_s": f"{statistics.mean(latencies):.2f}",
        "p50_s": f"{p50:.2f}",
        "p95_s": f"{p95:.2f}",
        "tokens": str(tokens),
        "tok_per_s": f"{tokens / wall:.1f}",
    }


def main() -> None:
    print(f"Waiting for a vLLM server at {BASE} ...")
    c.wait_for_server(BASE, timeout=30)
    model = c.list_models(BASE)[0]
    print(f"OK. model={model}  N={N}  max_tokens={MAX_TOKENS}")

    # Warm up so model/compile warmup does not pollute the sequential phase.
    print("\nWarming up ...")
    c.chat_raw("Hello.", base_url=BASE, model=model, max_tokens=8, **CHAT_KWARGS)

    print(f"\nPhase 1: {N} requests sent SEQUENTIALLY (one at a time)")
    t0 = time.perf_counter()
    sequential = [one_request(i) for i in range(N)]
    seq_wall = time.perf_counter() - t0
    print(f"  done in {seq_wall:.2f}s")

    print(f"\nPhase 2: {N} requests sent CONCURRENTLY (all in flight at once)")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        concurrent_results = list(pool.map(one_request, range(N)))
    con_wall = time.perf_counter() - t0
    print(f"  done in {con_wall:.2f}s")

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    rows = [
        summarise("sequential", sequential, seq_wall),
        summarise("concurrent", concurrent_results, con_wall),
    ]
    keys = list(rows[0].keys())
    c.print_table(keys, [[r[k] for k in keys] for r in rows])

    speedup = seq_wall / con_wall if con_wall else float("inf")
    print(f"\n  wall-clock speedup from concurrency: {speedup:.2f}x")
    print(f"  mean per-request latency: "
          f"{statistics.mean([r[0] for r in sequential]):.2f}s -> "
          f"{statistics.mean([r[0] for r in concurrent_results]):.2f}s")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""  The interesting result is that BOTH can be true at once:
    - aggregate throughput goes UP a lot with concurrency
    - per-request latency goes UP, but nowhere near N times

  Sequential, the accelerator is idle during most of every decode step. Concurrent,
  one forward pass serves all N requests, so the per-step cost is shared. The cost
  is real but sublinear - that gap is the entire value of batching.

  This is exactly what batching buys you. Two things to keep in mind when reading
  the numbers:
    1. They are hardware-real, so the bottleneck can shift between compute, memory
       bandwidth and CPU overhead as concurrency changes.
    2. vLLM also has a queue and a scheduler in front. At higher N you will start
       to see queueing latency, not just compute latency.

  CHECKPOINT
    - At what N does per-request latency start growing faster than throughput?
      That knee is your capacity.
    - Which of the two phases does TTFT dominate, and which does decode dominate?
    - Run it again with N=32. Does the speedup keep scaling? Why not?""")


if __name__ == "__main__":
    main()
