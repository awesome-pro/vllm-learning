#!/usr/bin/env bash
# Lab 06 - Stage 5: the scheduler's two budgets (max_num_batched_tokens, max_num_seqs).
#
# WHAT IT DEMONSTRATES
#   The scheduler is governed by two numbers, and confusing them is the most common
#   misunderstanding in vLLM:
#     --max-num-batched-tokens  bounds how many TOKENS OF WORK one step may contain
#     --max-num-seqs            bounds how many SEQUENCES may be in flight at once
#   It also shows the trap: on a card below 70 GB, `vllm serve` defaults the token
#   budget to 2048 while the offline LLM() class defaults to 8192. Same model, same
#   GPU, 4x the per-step token budget, purely because of the entrypoint.
#
#   Part 1 reads the defaults THIS machine will actually use - straight out of
#   EngineArgs.get_batch_defaults(), plus the running server's own log and /metrics.
#   Part 2 drives load at several concurrency levels and reports throughput and
#   p50/p99 latency from /metrics, so you can find the knee: the point past which
#   more concurrency buys no throughput and only adds latency.
#   Part 3 gives the exact commands to restart the server with the other budgets.
#
#   Restarting the server is deliberately NOT done for you. A sweep that silently
#   restarts things is a sweep you cannot trust: you change one flag, then re-run.
#
# HOW TO RUN
#     # terminal 1
#     bash scripts/serve.sh 2>&1 | tee notes/serve.log   # tee so Part 1 can read it
#     # terminal 2
#     cd /workspace/vlearning && source scripts/env.sh
#     bash labs/06_scheduler_budgets.sh
#     LEVELS="1 2 4 8 16 32" MAXTOK=64 bash labs/06_scheduler_budgets.sh
#
# PREREQUISITES
#     The server must already be running. Runtime ~30-60 s on the tiny model with the
#     default sweep. HTTP, JSON and the histogram arithmetic are done by two small
#     python3 programs (stdlib only, reusing labs/_client.py) written to a temp dir;
#     the sweep, the tables and the interpretation are this script.
#
# FINDING THE KNEE, CONCRETELY
#     Throughput that stops rising while p99 keeps climbing = you are past the knee.
#     vllm:num_requests_waiting > 0 means the scheduler refused work it had room to
#     queue but not to run - a TOKEN-budget or KV limit, not a GPU limit.

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

LEVELS="${LEVELS:-1 2 4 8 16}"
MAXTOK="${MAXTOK:-48}"
TIMEOUT="${TIMEOUT:-300}"
SERVER_LOG="${SERVER_LOG:-$VL_ROOT/notes/serve.log}"
export VL_ROOT MODEL PORT

# The python helpers live in files rather than in `$(python3 - <<EOF ...)` on
# purpose: heredocs inside command substitution misreport their exit status on
# bash 3.2, and a silent false failure in a measurement lab is unacceptable.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/lab06.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

hr() { printf '%s\n' "------------------------------------------------------------------------"; }

cat > "$WORK/probe.py" <<'PYEOF'
"""Read the real batch defaults out of vLLM and the live config out of /metrics.

Nothing here is guessed. `EngineArgs.get_batch_defaults` (vllm/engine/arg_utils.py,
v0.30.0) is the function that `_set_default_max_num_seqs_and_batched_tokens_args`
calls to pick these numbers from device memory and the usage context.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("VL_ROOT", "."), "labs"))
import _client as c  # noqa: E402

from vllm.config.scheduler import SchedulerConfig  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402

print("  device reported by vLLM:")
try:
    from vllm.platforms import current_platform

    print(f"    name        : {current_platform.get_device_name()}")
    print(f"    total VRAM  : {current_platform.get_device_total_memory() / 2**30:,.1f} GiB")
    print(f"    device type : {current_platform.device_type}")
except Exception as exc:  # CPU-only import, or no GPU visible
    print(f"    could not query the platform: {type(exc).__name__}: {exc}")

batched, seqs = EngineArgs.get_batch_defaults(world_size=1)
print()
print("  EngineArgs.get_batch_defaults(world_size=1):")
print(f"    {'usage context':<24}{'max_num_batched_tokens':>24}{'max_num_seqs':>14}")
for context in (UsageContext.LLM_CLASS, UsageContext.OPENAI_API_SERVER):
    print(f"    {context.value:<24}{batched.get(context, -1):>24}{seqs.get(context, -1):>14}")

print()
print("  generic fallbacks in SchedulerConfig (used when the context is unknown):")
print(f"    DEFAULT_MAX_NUM_BATCHED_TOKENS = {SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS}")
print(f"    DEFAULT_MAX_NUM_SEQS           = {SchedulerConfig.DEFAULT_MAX_NUM_SEQS}")

print()
print("  live server config from /metrics (vllm:cache_config_info):")
try:
    labels = c.labels("vllm:cache_config_info")
    if labels:
        for key in ("block_size", "gpu_memory_utilization", "enable_prefix_caching",
                    "cache_dtype", "num_gpu_blocks", "kv_cache_size_tokens"):
            if key in labels:
                print(f"    {key:<24}= {labels[key]}")
    else:
        print("    gauge not present")
    live = c.metrics()
    print(f"    vllm:num_requests_running = {live.get('vllm:num_requests_running', 0):.0f}"
          "  (an idle server should be at 0)")
    print(f"    vllm:num_requests_waiting = {live.get('vllm:num_requests_waiting', 0):.0f}")
except c.ServerNotRunning:
    print(f"    NO SERVER at {c.BASE}")
    print("    start one first:  bash scripts/serve.sh")
    sys.exit(3)
PYEOF

cat > "$WORK/load.py" <<'PYEOF'
"""Drive LEVEL concurrent requests and print ONE tab-separated row on stdout.

Progress and warnings go to stderr so the caller can capture stdout cleanly.

Latency quantiles come from the ENGINE's own Prometheus histograms, differenced
across the burst so each level is measured on its own requests rather than on the
server's whole lifetime:

    vllm:e2e_request_latency_seconds    observed when a request finishes
    vllm:time_to_first_token_seconds    observed when its first token appears

The bucket layout is vLLM's (vllm/v1/metrics/buckets.py) and the estimate is
PromQL's histogram_quantile: interpolate inside the bucket that holds the rank.
"""
import concurrent.futures
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.environ.get("VL_ROOT", "."), "labs"))
import _client as c  # noqa: E402

LEVEL = int(os.environ["LEVEL"])
MAXTOK = int(os.environ["MAXTOK"])
TIMEOUT = float(os.environ["TIMEOUT"])

FILLER = ("A scheduler decides at every step how much work fits in the token "
          "budget and how many sequences may run at once. ")
PROMPT = FILLER * 6 + "\n\nQuestion: what does the token budget bound?"


def scrape_buckets(metric: str) -> dict:
    """{le: cumulative count} for one histogram family, from a fresh scrape."""
    buckets: dict = {}
    # _client._samples is the shared Prometheus exposition parser; the flat
    # c.metrics() dict deliberately collapses buckets, which is exactly what we
    # must NOT do here.
    for name, labels, value in c._samples(c.get_text("/metrics", timeout=15)):
        if name == f"{metric}_bucket" and "le" in labels:
            buckets[labels["le"]] = value
    return buckets


def quantile(before: dict, after: dict, q: float) -> float:
    """Histogram quantile over the observations added during this burst."""
    deltas = sorted((float(le), after.get(le, 0.0) - before.get(le, 0.0)) for le in after)
    if not deltas or deltas[-1][1] <= 0:
        return float("nan")
    rank, prev_le, prev_count = q * deltas[-1][1], 0.0, 0.0
    for le, count in deltas:
        if count >= rank:
            if le == float("inf") or count == prev_count:
                return prev_le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return deltas[-1][0]


def one(_: int) -> float:
    """One non-streaming request; returns its client-side latency in seconds."""
    _, elapsed, _ = c.chat(PROMPT, max_tokens=MAXTOK, timeout=TIMEOUT,
                           temperature=0.0, ignore_eos=True)
    return elapsed


def pct(values: list, q: float) -> float:
    """Nearest-rank percentile on the client-observed latencies."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))]


before_tokens = float(c.metrics().get("vllm:generation_tokens", 0.0))
e2e_before = scrape_buckets("vllm:e2e_request_latency_seconds")
ttft_before = scrape_buckets("vllm:time_to_first_token_seconds")

peaks = {"running": 0.0, "waiting": 0.0, "kv": 0.0}
stop = threading.Event()


def sampler() -> None:
    """Poll /metrics during the burst: the peaks explain WHY latency grew."""
    while not stop.is_set():
        try:
            m = c.metrics()
            peaks["running"] = max(peaks["running"], m.get("vllm:num_requests_running", 0.0))
            peaks["waiting"] = max(peaks["waiting"], m.get("vllm:num_requests_waiting", 0.0))
            peaks["kv"] = max(peaks["kv"], m.get("vllm:kv_cache_usage_perc", 0.0))
        except Exception:
            pass  # a scrape during the burst must never kill the measurement
        time.sleep(0.1)


watcher = threading.Thread(target=sampler, daemon=True)
watcher.start()
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=LEVEL) as pool:
    client_latencies = list(pool.map(one, range(LEVEL)))
wall = time.perf_counter() - start
stop.set()
watcher.join(timeout=1.0)

generated = float(c.metrics().get("vllm:generation_tokens", 0.0)) - before_tokens
e2e_after = scrape_buckets("vllm:e2e_request_latency_seconds")
ttft_after = scrape_buckets("vllm:time_to_first_token_seconds")

print(f"     wall {wall:.2f}s, {generated:.0f} engine-counted tokens, "
      f"{generated / wall:.1f} tok/s, peak running {peaks['running']:.0f}, "
      f"peak waiting {peaks['waiting']:.0f}, peak KV {peaks['kv']:.2f}", file=sys.stderr)

print("\t".join([
    str(LEVEL),
    f"{wall:.2f}",
    str(LEVEL),
    f"{generated:.0f}",
    f"{generated / wall:.1f}",
    f"{pct(client_latencies, 0.50) * 1e3:.1f}",
    f"{pct(client_latencies, 0.99) * 1e3:.1f}",
    f"{quantile(e2e_before, e2e_after, 0.50) * 1e3:.1f}",
    f"{quantile(e2e_before, e2e_after, 0.99) * 1e3:.1f}",
    f"{quantile(ttft_before, ttft_after, 0.50) * 1e3:.1f}",
    f"{peaks['running']:.0f}",
    f"{peaks['waiting']:.0f}",
    f"{peaks['kv']:.2f}",
]))
PYEOF

# ---------------------------------------------------------------------------
# Part 1 - the defaults this machine will actually use
# ---------------------------------------------------------------------------
hr
echo "Lab 06 - Stage 5: scheduler budgets   ($(date -u '+%Y-%m-%d %H:%M UTC'))"
hr
echo "Part 1: the defaults vLLM will actually use on THIS machine"
echo

if probe_out="$(python3 "$WORK/probe.py" 2>&1)"; then
  printf '%s\n' "$probe_out"
else
  printf '%s\n' "$probe_out"
  echo
  echo "Fix the problem above, then re-run this lab (scripts/serve.sh must be running)."
  exit 1
fi

# The effective token budget is logged by the scheduler itself, once, at startup.
echo
if [ -f "$SERVER_LOG" ]; then
  echo "  the running server's own words ($SERVER_LOG):"
  grep -E 'Chunked prefill is enabled|KV cache size|Available KV cache memory' "$SERVER_LOG" \
    | tail -3 | sed 's/^/    /' || true
else
  echo "  no server log at $SERVER_LOG"
  echo "  (start the server as:  bash scripts/serve.sh 2>&1 | tee $SERVER_LOG"
  echo "   and the effective --max-num-batched-tokens plus the KV line show up here.)"
fi

cat <<'EOF'

  THE ASYMMETRY, AND WHY IT MATTERS
    On a < 70 GB card (the RTX 4090 here) vLLM picks different token budgets by
    ENTRYPOINT, on purpose:

      entrypoint                      max_num_batched_tokens   max_num_seqs
      LLM()          (offline)                 8192                256
      vllm serve     (HTTP)                    2048                256
      H100/H200 (>=70GB), serve                8192               1024
      B200/B300 (>=160GB)                     16384               1024

    Why: a larger token budget means longer individual steps, which is good for
    throughput and bad for latency, and an interactive HTTP server is expected to
    answer promptly, while an offline batch job only cares about total time. It is a
    deliberate latency/throughput choice, not a bug - and it is the single biggest
    reason "my server is slower than my benchmark script" surprises people.
EOF

# ---------------------------------------------------------------------------
# Part 2 - the sweep
# ---------------------------------------------------------------------------
echo
hr
echo "Part 2: load sweep  (levels: $LEVELS, max_tokens=$MAXTOK per request)"
echo "Each level fires K concurrent non-streaming requests (temperature=0,"
echo "ignore_eos=True, so every request decodes exactly $MAXTOK tokens)."
echo "All requests share ONE prompt on purpose: prefix caching then does its best, so"
echo "this sweep measures SCHEDULING, not prefill bandwidth. Give each request a"
echo "unique suffix if you want prefill contention to show up instead."
hr

echo "  (warming up the engine first: one 4-token request, so kernel autotuning and"
echo "   CUDA graph capture do not land on the concurrency-1 row)"
if ! MAXTOK=4 LEVEL=1 TIMEOUT="$TIMEOUT" python3 "$WORK/load.py" >/dev/null 2>"$WORK/warmup.err"; then
  echo "  warm-up failed - is the server still alive? Check its terminal:"
  sed 's/^/    /' "$WORK/warmup.err"
  exit 1
fi

# One TSV row per level: concurrency, wall, reqs, tokens, tok/s, client p50/p99,
# engine e2e p50/p99, ttft p50, peak running/waiting/KV.
ROWS=()
for level in $LEVELS; do
  case "$level" in
    ''|*[!0-9]*) echo "  skipping invalid level '$level' (must be a positive integer)"; continue ;;
  esac
  [ "$level" -ge 1 ] || { echo "  skipping level 0"; continue; }
  echo
  echo "  -> concurrency $level ..."
  if row="$(LEVEL="$level" MAXTOK="$MAXTOK" TIMEOUT="$TIMEOUT" python3 "$WORK/load.py")"; then
    ROWS+=("$row")
  else
    echo "     level $level FAILED (see the traceback above)"
    echo "     a connection error means the server died - check its terminal."
  fi
done

# ---------------------------------------------------------------------------
# Part 3 - the table
# ---------------------------------------------------------------------------
echo
hr
echo "RESULTS  (copy this block straight into notes/)"
hr
printf '%-6s %-8s %-7s %-9s %-9s %-11s %-11s %-10s %-10s %-10s %-9s %-10s %-8s\n' \
  "conc" "wall_s" "reqs" "gen_tok" "tok/s" "client_p50" "client_p99" \
  "e2e_p50" "e2e_p99" "ttft_p50" "peak_run" "peak_wait" "peak_kv"
printf '%-6s %-8s %-7s %-9s %-9s %-11s %-11s %-10s %-10s %-10s %-9s %-10s %-8s\n' \
  "----" "------" "----" "-------" "-----" "----------" "----------" \
  "-------" "-------" "--------" "--------" "---------" "-------"
if [ "${#ROWS[@]}" -eq 0 ]; then
  echo "  (no successful level - nothing to report)"
else
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r conc wall reqs gen tps cp50 cp99 ep50 ep99 tp50 prun pwait pkv <<<"$row"
    printf '%-6s %-8s %-7s %-9s %-9s %-11s %-11s %-10s %-10s %-10s %-9s %-10s %-8s\n' \
      "$conc" "$wall" "$reqs" "$gen" "$tps" "${cp50}ms" "${cp99}ms" \
      "${ep50}ms" "${ep99}ms" "${tp50}ms" "$prun" "$pwait" "$pkv"
  done
fi

echo
echo "COLUMNS"
cat <<'EOF'
  client_p50/p99  per-request latency measured by the client (exact, per level)
  e2e_p50/p99     the same thing from vllm:e2e_request_latency_seconds, differenced
                  across the level - the engine's own view, bucket-quantised
  ttft_p50        vllm:time_to_first_token_seconds: is latency growing in PREFILL
                  (TTFT) or in DECODE (e2e - ttft)? That tells you which budget bites
  peak_run        vllm:num_requests_running: how many sequences actually ran together
  peak_wait       vllm:num_requests_waiting: >0 means the scheduler REFUSED work
  peak_kv         vllm:kv_cache_usage_perc: 1.0 means every KV block was allocated
EOF

echo
echo "FINDING THE KNEE"
cat <<'EOF'
  Read the table twice: once down the tok/s column, once down p99.

  * tok/s still rising, p99 roughly flat  -> add concurrency, it is nearly free.
  * tok/s flat, p99 climbing              -> you are PAST the knee. The GPU is
    saturated and the extra requests only queue.
  * peak_wait > 0                         -> the scheduler refused to admit work.
    That is a budget limit (--max-num-batched-tokens, --max-num-seqs) or a KV limit,
    NOT a compute limit. If peak_kv is also near 1.0, the KV cache is the binding one.
  * peak_kv low while peak_wait > 0       -> the token budget is the binding one.
    This is the 2048-vs-8192 asymmetry from Part 1, showing up as latency.
EOF

# ---------------------------------------------------------------------------
# Part 4 - changing the budgets
# ---------------------------------------------------------------------------
echo
hr
echo "Part 4: change one budget at a time, then re-run this lab"
hr
cat <<EOF
  Stop the server (Ctrl-C in its terminal) and restart it with the flag under test.
  Same model, same prompt, same sweep - only the flag changes:

    # 1. the default you just measured (token budget 2048 on this card)
    bash scripts/serve.sh $MODEL

    # 2. four times the per-step token budget: fewer chunked-prefill splits
    bash scripts/serve.sh $MODEL --max-num-batched-tokens 8192

    # 3. more sequences in flight, token budget left at its default
    bash scripts/serve.sh $MODEL --max-num-seqs 32

    # 4. both, which is the configuration closest to the offline LLM() defaults
    bash scripts/serve.sh $MODEL --max-num-batched-tokens 8192 --max-num-seqs 32

  Then, in this terminal (it is the same sweep, so the tables line up):

    LEVELS="$LEVELS" MAXTOK=$MAXTOK bash labs/06_scheduler_budgets.sh

  What to expect - predicted from the scheduler's own semantics; your measurements
  are the real data:
    * Raising --max-num-batched-tokens should raise throughput on PROMPT-heavy load
      and raise TTFT for everyone else, because one long prefill now occupies more
      of each step. With this lab's short prompts the effect is small: raise the
      prompt length and MAXTOK together to see it.
    * Raising --max-num-seqs alone, past what the token budget can feed, changes
      little: the token budget rather than the sequence count is the binding
      constraint.
    * Both raised is where throughput stops being scheduler-bound and becomes
      GPU-bound - and where p99 starts to matter.
    * Watch peak_wait in every run: the configuration where it stays 0 at the
      highest concurrency is the one with headroom.

  vLLM's own benchmarking CLI does this properly at scale (Stage 7):
    vllm bench serve --model $MODEL --num-prompts 200 --max-concurrency 16 \\
      --request-rate inf --save-result
EOF

hr
cat <<EOF
RECORD: lab 06 scheduler budgets  (add the vLLM version and the date)
  defaults_offline_max_num_batched_tokens   <- Part 1, LLM_CLASS row
  defaults_serve_max_num_batched_tokens     <- Part 1, OPENAI_API_SERVER row
  defaults_max_num_seqs                     <- Part 1 (both contexts on this card)
  effective_token_budget_from_server_log    <- Part 1 (grep of notes/serve.log)
  device_name_and_vram_gib                  <- Part 1
  kv_cache_size_tokens_from_server_log      <- Part 1
  sweep_levels                              <- the conc column
  tokens_per_second_per_level               <- the tok/s column
  client_p50_p99_ms_per_level               <- client_p50 / client_p99
  e2e_p50_p99_ms_per_level                  <- e2e_p50 / e2e_p99 (engine histograms)
  ttft_p50_ms_per_level                     <- ttft_p50
  peak_requests_waiting_per_level           <- peak_wait (>0 = scheduler refused work)
  peak_kv_cache_usage_perc_per_level        <- peak_kv
  knee_concurrency                          <- the level where tok/s flattens
  then repeat the sweep for each flag combination in Part 4 and keep every table
------------------------------------------------------------------------
Write these into notes/05-scheduler-budgets.md. A sweep is only useful next to the
flag values that produced it, so record the exact serve.sh command line too.
EOF
