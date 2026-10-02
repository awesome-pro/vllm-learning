#!/usr/bin/env python3
"""Lab 04 - Stage 3: predict the KV cache from the model config, then check it.

WHAT IT DEMONSTRATES
  The arithmetic that turns a model into a deployment. Three steps:
    1. read the model's config.json (from $HF_HOME first, HuggingFace second) and
       compute KV bytes per token: 2 (K and V) x layers x kv_heads x head_dim x dtype
    2. take the KV figures the engine printed at startup (paste them in, below)
    3. compare, then derive the two questions that actually size a deployment:
       how many tokens a KV budget buys, and how much concurrency that is at a
       given context length.

  If prediction and measurement disagree by more than a few percent you have
  misread a config field. The last section says which one to suspect, in order.

HOW TO RUN
    # terminal 1
    bash scripts/serve.sh            # its startup log prints the KV line once
    # terminal 2 - paste the numbers from that log (or the whole line):
    source scripts/env.sh
    KV_LOG_LINE='GPU KV cache size: 34,912 tokens, Maximum concurrency for 8,192 tokens per request: 4.26x' \\
      python labs/04_kv_cache_memory.py
    KV_TOKENS=34912 KV_AVAIL_GIB=3.73 KV_CONCURRENCY=4.26 python labs/04_kv_cache_memory.py

    MODEL=$MODEL_MID python labs/04_kv_cache_memory.py     # a different model
    KV_DTYPE_BYTES=1 python labs/04_kv_cache_memory.py     # --kv-cache-dtype fp8

PREREQUISITES
    The server should be running, for the live /metrics half. Without it the lab
    still prints the full prediction - it just cannot compare against your engine.

WHERE THE NUMBERS COME FROM (all verified in the v0.30.0 source)
    * config.json: num_hidden_layers, num_attention_heads, num_key_value_heads,
      head_dim, torch_dtype.
    * Startup log, ONE merged line (vllm/v1/core/kv_cache_utils.py):
        GPU KV cache size: 34,912 tokens, Maximum concurrency for 8,192 tokens per request: 4.26x
      plus, from vllm/v1/worker/gpu_worker.py:
        Available KV cache memory: 3.73 GiB
    * /metrics: vllm:kv_cache_usage_perc (live blocks in use) and the labelled
      gauge vllm:cache_config_info (block_size, gpu_memory_utilization,
      enable_prefix_caching, num_gpu_blocks, kv_cache_size_tokens).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

import _client as c

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
UTIL = float(os.environ.get("UTIL", "0.90"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "16"))  # CacheConfig.DEFAULT_BLOCK_SIZE
# bf16/fp16 KV = 2 bytes per element; --kv-cache-dtype fp8 halves it.
KV_DTYPE_BYTES = int(os.environ.get("KV_DTYPE_BYTES", "2"))
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# The engine's own numbers. Paste any of these; KV_LOG_LINE fills all three.
KV_LOG_LINE = os.environ.get("KV_LOG_LINE", "")
KV_TOKENS = int(os.environ.get("KV_TOKENS", "0"))
KV_AVAIL_GIB = float(os.environ.get("KV_AVAIL_GIB", "0"))
KV_CONCURRENCY = float(os.environ.get("KV_CONCURRENCY", "0"))
KV_BUDGET_GIB = float(os.environ.get("KV_BUDGET_GIB", "4.0"))


def parse_log_line(line: str) -> dict:
    """Pull the three startup numbers out of the pasted log text, if present."""
    found: dict = {}
    match = re.search(r"KV cache size:\s*([\d,]+)\s*tokens", line)
    if match:
        found["tokens"] = int(match.group(1).replace(",", ""))
    match = re.search(
        r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", line
    )
    if match:
        found["context"] = int(match.group(1).replace(",", ""))
        found["concurrency"] = float(match.group(2))
    match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line)
    if match:
        found["available_gib"] = float(match.group(1))
    return found


def load_config(model: str) -> tuple[dict, str]:
    """Return (config.json, where it came from).

    Local cache first - $HF_HOME/hub/models--<org>--<name>/snapshots/*/config.json,
    plus the pre-`hub/` layout - then the HuggingFace raw endpoint.
    """
    escaped = model.replace("/", "--")
    patterns = [
        os.path.join(HF_HOME, "hub", f"models--{escaped}", "snapshots", "*", "config.json"),
        os.path.join(HF_HOME, f"models--{escaped}", "snapshots", "*", "config.json"),
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{escaped}/snapshots/*/config.json"
        ),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle), f"cache: {path}"
            except (OSError, json.JSONDecodeError):
                continue
    url = f"https://huggingface.co/{model}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read().decode()), f"fetched: {url}"
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        sys.exit(
            f"Could not load config.json for {model}.\n"
            f"  Tried the HF cache under {HF_HOME} and {url}\n"
            f"  Error: {type(exc).__name__}: {exc}\n"
            f"  Fix: bash scripts/prefetch.sh   (downloads the model ladder)"
        )


def kib(n: float) -> str:
    """Bytes as KiB - the unit the project's KV tables use."""
    return f"{n / 1024:,.1f} KiB"


def gib(n: float) -> str:
    """Bytes as GiB, for the budget tables."""
    return f"{n / 2**30:,.2f} GiB"


def main() -> None:
    cfg, source = load_config(MODEL)
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
    per_token = 2 * layers * kv_heads * head_dim * KV_DTYPE_BYTES  # K and V
    per_token_dense = 2 * layers * heads * head_dim * KV_DTYPE_BYTES  # if no GQA

    print("=" * 74)
    print(f"1. THE MODEL: {MODEL}")
    print("=" * 74)
    print(f"  config.json          : {source}")
    print(f"  num_hidden_layers    : {layers}")
    print(f"  num_attention_heads  : {heads}")
    print(f"  num_key_value_heads  : {kv_heads}"
          f"{'   <- GQA: fewer KV heads than query heads' if kv_heads != heads else ''}")
    print(f"  head_dim             : {head_dim}")
    print(f"  torch_dtype          : {cfg.get('torch_dtype', '?')} "
          f"(counting {KV_DTYPE_BYTES} byte(s) per element in the KV cache)")
    print(f"  max_position_embeddings: {cfg.get('max_position_embeddings', '?')}")
    print()
    print("  KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes")
    print(f"                 = 2 x {layers} x {kv_heads} x {head_dim} x {KV_DTYPE_BYTES}"
          f" = {per_token:,} bytes")
    print(f"  expected (predicted): {kib(per_token)} per token, "
          f"{gib(per_token * 1_000_000)} per 1M cached tokens")
    if kv_heads != heads:
        print(f"  GQA payoff: with {heads} KV heads (no grouping) it would be "
              f"{kib(per_token_dense)}/token,")
        print(f"              i.e. {per_token_dense / per_token:.0f}x more memory - the "
              "single biggest reason these models are servable at all.")
    print(f"  one full {MAXLEN:,}-token sequence costs {gib(per_token * MAXLEN)} of KV cache.")

    # --- what the engine reported -------------------------------------------
    print("\n" + "=" * 74)
    print("2. WHAT THE ENGINE REPORTED")
    print("=" * 74)
    from_log = parse_log_line(KV_LOG_LINE) if KV_LOG_LINE else {}
    reported_tokens = int(from_log.get("tokens", 0)) or KV_TOKENS
    reported_gib = float(from_log.get("available_gib", 0.0)) or KV_AVAIL_GIB
    reported_conc = float(from_log.get("concurrency", 0.0)) or KV_CONCURRENCY
    reported_ctx = int(from_log.get("context", 0)) or MAXLEN

    info: dict = {}
    live_usage = float("nan")
    try:
        info = c.labels("vllm:cache_config_info")
        live_usage = float(c.metrics().get("vllm:kv_cache_usage_perc", float("nan")))
        print(f"  server at {c.BASE} is up. vllm:cache_config_info labels:")
        for key in ("block_size", "cache_dtype", "enable_prefix_caching",
                    "gpu_memory_utilization", "num_gpu_blocks", "kv_cache_size_tokens"):
            if key in info:
                print(f"      {key:<24}= {info[key]}")
        print("      (vLLM may emit this gauge with 'None' placeholders as well; the")
        print("       values above are the real label set. The startup log is still")
        print("       the primary source - it is the only place the KV budget appears.)")
    except c.ServerNotRunning:
        print(f"  no server at {c.BASE}, so live /metrics is unavailable.")
        print("  The prediction above and below still holds; run scripts/serve.sh to")
        print("  compare it against a real engine.")
    except Exception as exc:  # an HTTP error from the route itself
        print(f"  /metrics unavailable: {type(exc).__name__}: {exc}")

    if not reported_tokens and info.get("kv_cache_size_tokens", "None") not in ("None", ""):
        reported_tokens = int(float(info["kv_cache_size_tokens"]))
        print(f"  (nothing pasted, so using /metrics kv_cache_size_tokens="
              f"{reported_tokens:,}; the startup log is the better source)")
    if not reported_tokens and info.get("num_gpu_blocks", "None") not in ("None", ""):
        block_size = int(float(info.get("block_size", BLOCK_SIZE)))
        blocks = int(float(info["num_gpu_blocks"]))
        reported_tokens = blocks * block_size
        print(f"  (nothing pasted, so num_gpu_blocks x block_size = "
              f"{blocks:,} x {block_size} = {reported_tokens:,} tokens)")

    if not reported_tokens:
        print("\n  No reported KV figure. Paste the engine's startup line:")
        print("      KV_LOG_LINE='<the GPU KV cache size line>' \\")
        print("        python labs/04_kv_cache_memory.py")
        print("  or just the numbers:")
        print("      KV_TOKENS=34912 KV_AVAIL_GIB=3.73 KV_CONCURRENCY=4.26 \\")
        print("        python labs/04_kv_cache_memory.py")
        print("  (Printed once, by vllm/v1/core/kv_cache_utils.py, in the terminal")
        print("   where you ran scripts/serve.sh. Scroll up.)")
    else:
        print(f"  reported KV capacity : {reported_tokens:,} tokens")
        if reported_gib:
            print(f"  reported KV budget   : {reported_gib:.2f} GiB "
                  "(the 'Available KV cache memory' line)")
            print(f"  -> implied bytes/token: {reported_gib * 2**30 / reported_tokens:,.0f} B "
                  f"measured vs {per_token:,} B predicted")
        if reported_conc:
            print(f"  reported concurrency : {reported_conc:.2f}x at "
                  f"{reported_ctx:,} tokens/request")
        if not reported_gib:
            print(f"  -> those {reported_tokens:,} tokens are worth "
                  f"{gib(reported_tokens * per_token)} of KV cache by the formula")

    # --- predicted vs reported ----------------------------------------------
    print("\n" + "=" * 74)
    print("3. PREDICTED vs REPORTED")
    print("=" * 74)
    rows = []
    if reported_tokens and reported_gib:
        predicted_tokens = reported_gib * 2**30 / per_token
        diff = (reported_tokens - predicted_tokens) / predicted_tokens * 100
        rows.append(["max cached tokens", f"{predicted_tokens:,.0f}",
                     f"{reported_tokens:,}", f"{diff:+.2f}%"])
    if reported_conc and reported_tokens:
        predicted_conc = reported_tokens / reported_ctx
        diff = (reported_conc - predicted_conc) / predicted_conc * 100
        rows.append([f"max concurrency @ {reported_ctx:,}",
                     f"{predicted_conc:,.2f}x", f"{reported_conc:.2f}x", f"{diff:+.2f}%"])
    if rows:
        print()
        c.print_table(["quantity", "predicted", "reported", "difference"], rows)
        print("\n  The concurrency row is a consistency check on the engine's own")
        print("  arithmetic (num_tokens / max_model_len), so it should be ~0.00%.")
        print("  The token row is the real test of your formula against its memory")
        print("  accounting: a few percent low is expected (block padding and group")
        print("  unification); tens of percent means you misread the config.")
    else:
        print("  Nothing to compare yet: paste KV_TOKENS + KV_AVAIL_GIB (or")
        print("  KV_LOG_LINE). The prediction above is the part you derive yourself.")

    # --- derived capacity tables ---------------------------------------------
    print("\n" + "=" * 74)
    print("4. DERIVED: what a KV budget buys, and at what context")
    print("=" * 74)
    if reported_gib:
        budget_gib = reported_gib
    else:
        budget_gib = KV_BUDGET_GIB
        print(f"  (no 'Available KV cache memory' pasted; using the illustrative")
        print(f"   KV_BUDGET_GIB={budget_gib:g}. Paste yours for real numbers.)")
    print()
    budgets = sorted({1.0, 2.0, 4.0, 8.0, round(budget_gib, 2)})
    rows = []
    for budget in budgets:
        tokens = budget * 2**30 / per_token
        tag = "  <- your server" if abs(budget - budget_gib) < 0.005 else ""
        rows.append([f"{budget:,.2f} GiB", f"{tokens:,.0f}",
                     f"{tokens / MAXLEN:,.1f}x", f"{tokens / 32768:,.1f}x{tag}"])
    c.print_table(
        ["KV budget", "cached tokens", f"concurrency @{MAXLEN // 1024}k",
         "concurrency @32k"], rows
    )
    print("\n  vLLM's own concurrency figure is num_tokens / max_model_len, computed")
    print("  per cache group (vllm/v1/core/kv_cache_utils.py). Raising --max-model-len")
    print("  does not create more cache: the same bytes now carry a bigger per-request")
    print("  claim, so concurrency falls proportionally. That is the tradeoff in one line.")

    if live_usage == live_usage:  # NaN check
        print("\n" + "=" * 74)
        print("5. LIVE KV USAGE  (vllm:kv_cache_usage_perc)")
        print("=" * 74)
        print(f"  right now: {live_usage:.4f}   (1.0 = every KV block allocated)")
        print("  Send a request and scrape again: it rises, then does NOT fall back to")
        print("  zero immediately. Freed blocks stay in the cache until they are evicted")
        print("  or overwritten - that is prefix caching's mechanism, and it is why this")
        print("  number behaves like a high-water mark under steady traffic.")

    print("\n" + "=" * 74)
    print("HOW TO INTERPRET A MISMATCH (the actual lesson of this lab)")
    print("=" * 74)
    print(f"""  Agreement within a few percent means your formula and the engine's memory
  accounting describe the same thing. If not, suspect these, in order:

    1. num_key_value_heads vs num_attention_heads. Using the query-head count is
       the classic 4x error on GQA models like Qwen3. Read the config again.
    2. head_dim. Some models set it explicitly and it is NOT hidden_size/heads.
    3. Hybrid attention. Sliding-window layers cache fewer tokens per layer, so
       "every layer is full attention" over-predicts.
    4. KV dtype. --kv-cache-dtype fp8 stores 1 byte per element and roughly doubles
       capacity. Re-run with KV_DTYPE_BYTES=1 and watch the prediction move.
    5. Padding and unification. vLLM pads cache groups to a common size and rounds
       to whole blocks (block_size={BLOCK_SIZE} here), so the reported capacity is
       usually a little LOWER than the pure formula. A few percent is expected.
    6. The budget you compared against. --gpu-memory-utilization is a fraction of
       the WHOLE card; weights, activations and the CUDA context come out of it
       before a single KV block exists. Only the 'Available KV cache memory' line
       tells you what actually reached the cache.""")

    c.record(
        f"lab 04 KV cache (model={MODEL}, max_model_len={MAXLEN}, util={UTIL}, "
        f"kv_dtype_bytes={KV_DTYPE_BYTES})",
        predicted_kv_bytes_per_token=per_token,
        predicted_kv_kib_per_token=round(per_token / 1024, 1),
        layers=layers,
        kv_heads=kv_heads,
        attention_heads=heads,
        head_dim=head_dim,
        reported_kv_tokens=reported_tokens if reported_tokens else "not pasted",
        reported_available_kv_gib=round(reported_gib, 3) if reported_gib else "not pasted",
        reported_max_concurrency=reported_conc if reported_conc else "not pasted",
        predicted_tokens_from_budget=(
            round(reported_gib * 2**30 / per_token) if reported_gib else "n/a"
        ),
        live_kv_cache_usage_perc=(
            round(live_usage, 4) if live_usage == live_usage else "no server"
        ),
    )


if __name__ == "__main__":
    main()
