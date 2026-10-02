#!/usr/bin/env python3
"""Lab 05 - prefix caching: pay for the prompt once.

The experiment: send several requests that share a long system prompt. If prefix
caching is working, the 2nd and later requests skip prefill for the shared part,
and TTFT collapses. Then send a request with a *different* system prompt and
watch TTFT go back up.

Start the server first:   bash scripts/serve.sh
Then:                     python labs/05_prefix_caching.py

To see the effect, also run the server with caching disabled and compare:

    bash scripts/serve.sh Qwen/Qwen3-0.6B --no-enable-prefix-caching
"""

from __future__ import annotations

import re

import _client as c

BASE = c.DEFAULT_BASE

FILLER_SENTENCE = (
    "Retrieval augmented generation grounds a model in documents it was not trained on, "
    "which reduces hallucination but makes the prompt long and repetitive across requests. "
)

SHARED_SYSTEM = "You are a helpful assistant. Use the following reference material.\n\n" + (
    FILLER_SENTENCE * 40
)

OTHER_SYSTEM = "You are a terse assistant that answers only in single words.\n\n" + (
    FILLER_SENTENCE * 40
)

CHAT_KWARGS = {
    "max_tokens": 32,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


def prefix_metrics() -> tuple[int, int]:
    """Return (hits, queries) from /metrics, or (-1, -1) if unavailable."""
    try:
        text = c.get_text(BASE, "/metrics", timeout=10)
    except Exception:
        return -1, -1
    hits = queries = 0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("vllm:prefix_cache_hits"):
            hits += int(float(re.match(r".*\}\s+(\S+)$", line).group(1)))
        elif line.startswith("vllm:prefix_cache_queries"):
            queries += int(float(re.match(r".*\}\s+(\S+)$", line).group(1)))
    return hits, queries


def main() -> None:
    print(f"Waiting for a vLLM server at {BASE} ...")
    c.wait_for_server(BASE, timeout=30)

    approx_tokens = len(SHARED_SYSTEM) // 4
    print(f"Shared system prompt: {len(SHARED_SYSTEM):,} chars  (~{approx_tokens:,} tokens)")
    print("Both system prompts share the SAME filler text but differ in their first line,")
    print("so they diverge at the very first block.")

    h0, q0 = prefix_metrics()

    print("\nRunning requests (each is streamed so TTFT is accurate) ...\n")
    cases = [
        ("cold        ", "What is RAG?", SHARED_SYSTEM),
        ("warm #1     ", "Why does it reduce hallucination?", SHARED_SYSTEM),
        ("warm #2     ", "Name one downside.", SHARED_SYSTEM),
        ("different   ", "What is RAG?", OTHER_SYSTEM),
    ]

    rows = []
    baseline = None
    for label, question, system in cases:
        ttft, total, _ = c.measure_ttft(question, base_url=BASE, system=system, **CHAT_KWARGS)
        if baseline is None:
            baseline = ttft
        speedup = f"{baseline / ttft:.1f}x" if ttft > 0 else "?"
        rows.append([label, question[:28], f"{ttft:.3f}", f"{total:.3f}", speedup])

    c.print_table(["case", "question", "TTFT s", "total s", "vs cold"], rows)

    h1, q1 = prefix_metrics()
    if q1 >= 0:
        d_hits, d_queries = h1 - h0, q1 - q0
        rate = (d_hits / d_queries * 100) if d_queries else 0.0
        print(f"\nprefix_cache_queries delta : {d_queries}")
        print(f"prefix_cache_hits    delta : {d_hits}")
        print(f"hit rate this run          : {rate:.1f}%")
        print("\nThe engine counts *blocks* asked for and *blocks* found in the cache.")
        print("'different' contributes queries with almost no hits - that is the control.")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("""  - 'cold' pays full prefill for ~N tokens. Its TTFT is prefill-bound.
  - 'warm' requests find the shared prefix blocks already in the cache, so they
    only prefill their own few tokens. TTFT drops sharply.
  - 'different' shares nothing with the others, so it is cold again. That control
    is what proves the speedup came from prefix reuse and not from warm hardware.

  Note the shared prefix ends before the question: the reusable unit is a FULL
  block. A partial block is never cached, which is why 'different' diverging at
  the first line invalidates everything.

  Now restart the server with --no-enable-prefix-caching and run this again.
  The warm cases should fall back to cold timings - that is your proof.

  CHECKPOINT
    - Why is the cache key a hash chain (block content + parent hash) rather than
      just the block's own tokens?
    - Why can a partial (unfilled) block never be shared?
    - LLM requests are usually served in a different order than they arrive. What
      does that imply for prefix cache hit rate in production?""")


if __name__ == "__main__":
    main()
