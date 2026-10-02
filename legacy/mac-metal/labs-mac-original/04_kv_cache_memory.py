#!/usr/bin/env python3
"""Lab 04 - predict the KV cache, then check your prediction against the engine.

Block size and memory pressure, done on real hardware: derive capacity from first
principles, then read what vLLM actually allocated.

Works in two modes:
  * no server running  -> prints the prediction only
  * server running     -> also scrapes /metrics and compares

    bash scripts/py.sh labs/04_kv_cache_memory.py
    MODEL=Qwen/Qwen3-0.6B bash scripts/py.sh labs/04_kv_cache_memory.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

import _client as c

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
BASE = c.DEFAULT_BASE
BYTES_PER_PARAM = 2  # bfloat16


def load_config(model: str) -> dict:
    """Fetch config.json, preferring the local Hugging Face cache."""
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model, "config.json")) as fh:
            return json.load(fh)
    except Exception:
        url = f"https://huggingface.co/{model}/raw/main/config.json"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)


def predict(cfg: dict, block_size: int, max_model_len: int) -> dict:
    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]

    per_token = 2 * kv_heads * head_dim * BYTES_PER_PARAM * layers  # K and V
    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "per_token_bytes": per_token,
        "per_block_bytes": per_token * block_size,
        "full_context_bytes": per_token * max_model_len,
        "block_size": block_size,
        "max_model_len": max_model_len,
    }


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:.1f} GiB"


def scrape_config_info(base_url: str) -> dict | None:
    """Pull the vllm:cache_config_info labels out of /metrics."""
    try:
        text = c.get_text(base_url, "/metrics", timeout=10)
    except (urllib.error.URLError, OSError):
        return None
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
            return labels
    return None


def main() -> None:
    cfg = load_config(MODEL)
    print(f"Model: {MODEL}")
    print(f"  num_hidden_layers   = {cfg['num_hidden_layers']}")
    print(f"  num_attention_heads = {cfg['num_attention_heads']}")
    print(f"  num_key_value_heads = {cfg.get('num_key_value_heads')}   <- GQA: fewer KV heads")
    print(f"  head_dim            = {cfg.get('head_dim')}")
    print(f"  max_position_embeds = {cfg.get('max_position_embeddings')}")

    print("\n" + "=" * 70)
    print("PREDICTION (write this down before you look at the server)")
    print("=" * 70)
    print("""
  KV bytes per token = 2 (K and V) x kv_heads x head_dim x dtype_bytes x layers
""")

    for block_size in (8, 16, 32, 64):
        p = predict(cfg, block_size, 4096)
        marker = "   <- vLLM default" if block_size == 16 else ""
        print(
            f"  block_size={block_size:3d}: per-token={human(p['per_token_bytes']):>10s}  "
            f"per-block={human(p['per_block_bytes']):>10s}  "
            f"4k-context={human(p['full_context_bytes']):>10s}{marker}"
        )

    p = predict(cfg, 16, 4096)
    default_ctx = cfg.get("max_position_embeddings", 4096)
    full = predict(cfg, 16, default_ctx)
    print(f"\n  One full {default_ctx:,}-token sequence costs "
          f"{human(full['full_context_bytes'])} of KV cache.")
    print("  This is why max_model_len is the most expensive knob you can turn.")

    print("\n" + "=" * 70)
    print("MEASUREMENT (/metrics from a running server)")
    print("=" * 70)
    info = scrape_config_info(BASE)
    if info is None:
        print(f"  No server at {BASE}. Start one to compare:")
        print("      bash scripts/serve.sh")
        print("  Then watch its startup log for the line:")
        print('      "<DEVICE> KV cache size: <N> tokens, Maximum concurrency for ..."')
        print("  and re-run this lab.")
        return

    print("  vllm:cache_config_info labels reported by the engine:")
    for k in sorted(info):
        print(f"      {k:24s}= {info[k]}")

    try:
        engine_block_size = int(info.get("block_size", 16))
        num_blocks = int(info.get("num_gpu_blocks") or info.get("num_blocks") or 0)
    except ValueError:
        engine_block_size, num_blocks = 16, 0

    if num_blocks:
        capacity_tokens = num_blocks * engine_block_size
        print(f"\n  blocks            : {num_blocks:,}")
        print(f"  block_size        : {engine_block_size}")
        print(f"  KV capacity       : {capacity_tokens:,} tokens")
        print(f"  capacity in bytes : {human(capacity_tokens * p['per_token_bytes'])}")
        print(f"\n  Your predicted per-token cost was {human(p['per_token_bytes'])}; "
              f"blocks x block_size x that = {human(num_blocks * engine_block_size * p['per_token_bytes'])}.")
        print("  If that matches the memory you granted the server, you now understand")
        print("  the entire memory budget of an inference engine.")
    else:
        print("\n  (No block count in the labels; read it from the server startup log.)")

    print("\n" + "=" * 70)
    print("CHECKPOINT")
    print("=" * 70)
    print("""  - Doubling block_size halves the number of blocks but not the bytes. What
    does it actually change? (Hint: fragmentation and kernel efficiency.)
  - GQA uses 8 KV heads for 16 attention heads. What would KV cost be without
    GQA, and why is that a feature rather than a detail?
  - Restart the server with --max-model-len 2048 and re-run. Did capacity in
    tokens change? Did capacity in bytes? Explain.""")


if __name__ == "__main__":
    main()
