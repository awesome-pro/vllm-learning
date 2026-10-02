#!/usr/bin/env python3
"""Lab 01 - offline inference with the LLM class.

The offline entrypoint: no HTTP server, no separate process. This is the same
engine that `vllm serve` drives, so everything you learn here transfers.

    python labs/01_offline_inference.py
    MODEL=Qwen/Qwen3-0.6B python labs/01_offline_inference.py

Watch for the log lines printed while the engine initialises - they tell you the
KV cache size and the maximum concurrency, which lab 04 predicts from first
principles.
"""

from __future__ import annotations

import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
# vLLM sizes its paged KV cache from this fraction of the accelerator budget and
# allocates it all at startup. The Metal plugin defaults to 0.92 (~15.7 GB on a
# 24 GB Mac) - far too much for a laptop. 0.25 gives ~2.95 GB of KV cache.
GPU_MEMORY_UTILIZATION = float(os.environ.get("FRACTION", "0.25"))

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue:",
    "Write a haiku about paged memory:",
]


def main() -> None:
    print(f"Loading {MODEL} (this is the slow part - watch the log lines)...")
    print(f"KV cache budget: {GPU_MEMORY_UTILIZATION:.0%} of accelerator memory")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    load_s = time.perf_counter() - t0
    print(f"\nModel loaded in {load_s:.1f}s\n")

    print("=" * 70)
    print("GREEDY (deterministic - same input always gives the same output)")
    print("=" * 70)
    greedy = SamplingParams(temperature=0.0, max_tokens=48)
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, greedy)
    greedy_s = time.perf_counter() - t0

    total_out_tokens = 0
    for out in outputs:
        text = out.outputs[0].text.strip().replace("\n", " ")
        n = len(out.outputs[0].token_ids)
        total_out_tokens += n
        print(f"\nprompt   : {out.prompt!r}")
        print(f"output   : {text!r}")
        print(f"tokens   : {n} generated, finish_reason={out.outputs[0].finish_reason}")

    print("\n" + "=" * 70)
    print("SAMPLED (temperature=0.8 - results vary run to run)")
    print("=" * 70)
    sampled = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=48)
    outputs2 = llm.generate([PROMPTS[2]] * 3, sampled)
    for i, out in enumerate(outputs2, 1):
        print(f"  sample {i}: {out.outputs[0].text.strip()!r}")

    print("\n" + "=" * 70)
    print("MEASUREMENTS")
    print("=" * 70)
    n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"model load time        : {load_s:.2f} s")
    print(f"batched generate time  : {greedy_s:.2f} s  ({len(PROMPTS)} prompts, one call)")
    print(f"tokens generated       : {n_out}")
    print(f"effective output rate  : {n_out / greedy_s:.1f} tok/s (batched)")
    print()
    print("Note the shape of this API: you hand over ALL prompts at once and get all")
    print("results back. The engine batched them internally - you never chose a batch")
    print("size. That is continuous batching doing its job.")
    print()
    print("CHECKPOINT: why is this called 'offline' when it is clearly doing inference")
    print("online-style batching? Answer in terms of process architecture, not latency.")


if __name__ == "__main__":
    main()
