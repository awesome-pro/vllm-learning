#!/usr/bin/env python3
"""Lab 06 - what the engine tells you about itself under load.

Scrapes vLLM's Prometheus endpoint while driving a load burst, so you can watch
the scheduler and KV cache behave instead of guessing.

Start the server first:   bash scripts/serve.sh
Then:                     python labs/06_metrics_and_bench.py

Env knobs:
    N=24             concurrent requests in the burst
    MAX_TOKENS=96    tokens per request
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import statistics
import threading
import time
from collections import defaultdict

import _client as c

BASE = c.DEFAULT_BASE
N = int(os.environ.get("N", "24"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))

GAUGES = {
    "vllm:num_requests_running": "requests executing right now",
    "vllm:num_requests_waiting": "requests queued, not yet admitted",
    "vllm:kv_cache_usage_perc": "fraction of KV blocks in use",
}
COUNTERS = {
    "vllm:prompt_tokens": "prefill tokens processed",
    "vllm:generation_tokens": "decode tokens generated",
    "vllm:prefix_cache_hits": "blocks served from prefix cache",
    "vllm:prefix_cache_queries": "block lookups made",
    "vllm:num_preemptions": "requests preempted (memory pressure!)",
    "vllm:request_success": "requests completed successfully",
}

PROMPT = (
    "Summarise, in a single paragraph of about eighty words, the tradeoff between "
    "time-to-first-token and inter-token latency when serving many concurrent users."
)
CHAT_KWARGS = {"chat_template_kwargs": {"enable_thinking": False}}


def scrape() -> tuple[dict[str, float], dict[str, float]]:
    """Return (gauges, counters). Gauges take the max across label sets; counters sum."""
    text = c.get_text(BASE, "/metrics", timeout=10)
    gauge_vals: dict[str, float] = defaultdict(lambda: float("-inf"))
    counter_vals: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        raw = line.split("{", 1)[0].split(" ", 1)[0]
        # Prometheus appends _total to counters; normalize so our names match.
        name = raw[: -len("_total")] if raw.endswith("_total") else raw
        m = re.search(r"\s(\S+)$", line)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        if name in GAUGES:
            gauge_vals[name] = max(gauge_vals[name], value)
        elif name in COUNTERS:
            counter_vals[name] += value
    return dict(gauge_vals), dict(counter_vals)


def one_request(i: int) -> float:
    _, elapsed, _ = c.chat_raw(
        f"{PROMPT} (variant {i})",
        base_url=BASE,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        **CHAT_KWARGS,
    )
    return elapsed


def main() -> None:
    print(f"Waiting for a vLLM server at {BASE} ...")
    c.wait_for_server(BASE, timeout=30)

    print("Warming up ...")
    c.chat_raw("Hello.", base_url=BASE, max_tokens=8, **CHAT_KWARGS)

    g0, c0 = scrape()
    print(f"\nBaseline (idle):")
    for name, desc in GAUGES.items():
        print(f"  {name:34s} {g0.get(name, 0):>10.3f}   {desc}")

    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                g, _ = scrape()
                samples.append(g)
            except Exception:
                pass
            time.sleep(0.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    print(f"\nFiring {N} concurrent requests x {MAX_TOKENS} tokens ...")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        latencies = list(pool.map(one_request, range(N)))
    wall = time.perf_counter() - t0
    stop.set()
    thread.join(timeout=2)

    g1, c1 = scrape()

    print("\n" + "=" * 74)
    print("DURING THE BURST (peak values sampled every 250 ms)")
    print("=" * 74)
    for name, desc in GAUGES.items():
        peak = max((s.get(name, 0) for s in samples), default=0.0)
        print(f"  {name:34s} peak={peak:>8.3f}   {desc}")

    print("\n" + "=" * 74)
    print("COUNTER DELTAS OVER THE BURST")
    print("=" * 74)
    for name, desc in COUNTERS.items():
        delta = c1.get(name, 0) - c0.get(name, 0)
        if delta or name in ("vllm:prefix_cache_hits", "vllm:num_preemptions"):
            print(f"  {name:34s} +{delta:>10.0f}   {desc}")

    hits = c1.get("vllm:prefix_cache_hits", 0) - c0.get("vllm:prefix_cache_hits", 0)
    queries = c1.get("vllm:prefix_cache_queries", 0) - c0.get("vllm:prefix_cache_queries", 0)
    if queries:
        print(f"\n  prefix cache hit rate this burst: {hits / queries * 100:.1f}%  "
              f"(shared instruction prefix only)")
    gen = c1.get("vllm:generation_tokens", 0) - c0.get("vllm:generation_tokens", 0)
    print(f"  decode throughput              : {gen / wall:.1f} tok/s")
    print(f"  wall clock                     : {wall:.2f}s for {N} requests")
    print(f"  per-request latency            : mean {statistics.mean(latencies):.2f}s, "
          f"max {max(latencies):.2f}s")

    waiting = max((s.get("vllm:num_requests_waiting", 0) for s in samples), default=0.0)
    kv = max((s.get("vllm:kv_cache_usage_perc", 0) for s in samples), default=0.0)
    print("\n" + "=" * 74)
    print("DIAGNOSIS")
    print("=" * 74)
    if waiting > 0:
        print(f"  Queueing happened (peak waiting = {waiting:.0f}). The scheduler had more")
        print("  work than it could admit at once, so latency includes queue time.")
    else:
        print("  No queueing (waiting stayed at 0). The engine always had room.")
    if kv > 0.9:
        print(f"  KV cache nearly full (peak {kv:.0%}) - preemption is likely next.")
    else:
        print(f"  KV cache peak usage was {kv:.0%} - memory was not the limit here.")
    print("\n  Increase N and watch which of these two moves FIRST. That tells you")
    print("  whether you are compute/scheduler-bound or memory-bound - the single most")
    print("  useful diagnostic in LLM serving.")

    print("\n" + "=" * 74)
    print("CHECKPOINT")
    print("=" * 74)
    print("""  - num_requests_waiting > 0 means the scheduler refused work. Why would it do
    that even when the KV cache has free blocks? (Hint: token budget.)
  - vllm:num_preemptions counts requests the engine had to evict and later
    recompute. What does a non-zero value cost you in wall-clock terms?
  - Which metric would you alert on in production, and what threshold?""")


if __name__ == "__main__":
    main()
