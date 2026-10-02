#!/usr/bin/env python3
"""Lab 02 - the online entrypoint: a real OpenAI-compatible server.

Start the server in another shell first:

    bash scripts/serve.sh

Then:

    python labs/02_openai_client.py

This talks raw HTTP to /v1 so you can see the actual request/response shape.
"""

from __future__ import annotations

import sys

import _client as c

BASE = c.DEFAULT_BASE


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print(f"Waiting for a vLLM server at {BASE} ...")
    try:
        c.wait_for_server(BASE, timeout=30)
    except TimeoutError:
        print(f"\nNo server at {BASE}.")
        print("Start one in another shell:  bash scripts/serve.sh")
        sys.exit(1)

    models = c.list_models(BASE)
    model = models[0]
    print(f"OK - server is up. Models: {models}")

    section("1. Non-streaming chat completion")
    text, elapsed = c.chat(
        "In one sentence, what is a KV cache?",
        model=model,
        max_tokens=80,
        chat_template_kwargs={"enable_thinking": False},
    )
    print(f"response ({elapsed:.2f}s):\n{text.strip()}")
    print("\nNOTE: everything is done when you get the response. No tokens arrive")
    print("early, so 'latency' here is the whole generation, not TTFT.")

    section("2. Streaming - watch tokens arrive")
    print("(each line is a cumulative snapshot; note how fast the first one lands)")
    last = ""
    ttft = None
    for text, ttft_v, elapsed in c.chat_stream(
        "Count from 1 to 10, one number per word.",
        model=model,
        max_tokens=60,
        chat_template_kwargs={"enable_thinking": False},
    ):
        if ttft is None and ttft_v is not None:
            ttft = ttft_v
        if len(text) - len(last) >= 10:
            print(f"  [{elapsed:5.2f}s] {text!r}")
            last = text
    print(f"\n  TTFT (time to first token) : {ttft:.3f}s" if ttft else "  no content received")
    print(f"  total                      : {elapsed:.3f}s")
    if ttft:
        print(f"  decode time after TTFT     : {elapsed - ttft:.3f}s")
        print("\n  This split is the whole point: TTFT is prefill + queueing, the rest")
        print("  is decode. Every latency problem you will ever debug is one of these two.")

    section("3. Token accounting (the usage block)")
    _, _, usage = c.chat_raw(
        "Explain paged attention in two sentences.",
        model=model,
        max_tokens=100,
        chat_template_kwargs={"enable_thinking": False},
    )
    for k, v in usage.items():
        print(f"  {k:24s}: {v}")
    print("\n  prompt_tokens are prefill work; completion_tokens are decode work.")
    print("  They cost very different amounts of time per token.")

    section("4. Determinism")
    a, _ = c.chat("Say exactly: hello", model=model, max_tokens=10, temperature=0.0,
                  chat_template_kwargs={"enable_thinking": False})
    b, _ = c.chat("Say exactly: hello", model=model, max_tokens=10, temperature=0.0,
                  chat_template_kwargs={"enable_thinking": False})
    print(f"  run 1: {a.strip()!r}")
    print(f"  run 2: {b.strip()!r}")
    print(f"  identical: {a.strip() == b.strip()}")

    section("CHECKPOINT")
    print("  - Which HTTP endpoint did we hit, and which vLLM process handled the")
    print("    tokenization? (Hint: it is not the engine core.)")
    print("  - In part 2, why is TTFT roughly constant regardless of how many tokens")
    print("    we ask for, while total time is not?")
    print("  - Find these same numbers in the server log. What else does it report?")


if __name__ == "__main__":
    main()
