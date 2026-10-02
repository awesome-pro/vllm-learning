#!/usr/bin/env bash
# Lab 08 — Stage 7: /metrics under load, and vLLM's own benchmark, reconciled.
#
# WHAT IT DEMONSTRATES
#   (a) Snapshot /metrics three times — before the load, while it is running, and after —
#       and diff the series that actually explain what happened:
#         requests running, requests waiting (and waiting *by reason*),
#         KV cache usage, prefix cache hits/queries, the TTFT histogram buckets,
#         and the preemption counter.
#   (b) Then run vLLM's own benchmark client (`vllm bench serve`) against the same
#       server, with its own metrics window, and reconcile the client's numbers with
#       the engine's numbers. They should agree; where they cannot, the reason is the
#       lesson.
#   (c) End with a guided interpretation: which metric moved FIRST, and how to tell
#       "GPU saturated" from "scheduler starved" using metrics alone.
#
# RUN
#   bash scripts/serve.sh                 # in another terminal — this lab needs a server
#   bash labs/08_metrics_and_bench.sh              # burst + bench + interpretation
#   bash labs/08_metrics_and_bench.sh burst        # only the hand-driven burst
#   bash labs/08_metrics_and_bench.sh bench        # only `vllm bench serve`
#   bash labs/08_metrics_and_bench.sh status       # what has been recorded so far
#
# PREREQUISITES
#   * A running vLLM OpenAI server on $PORT (default 8000). Start it with scripts/serve.sh.
#   * `curl`, `python3`, and the `vllm` CLI on PATH (i.e. `source scripts/env.sh` first).
#   * No other traffic on the server while this runs — the whole method is "one variable
#     at a time", and a second client would silently ruin the metric deltas.
#
# ENV KNOBS
#   N=48              requests in the hand-driven burst
#   CONC=16           concurrency of the burst
#   MAX_TOKENS=96     generated tokens per burst request (exact: ignore_eos is set)
#   BENCH_N=64        --num-prompts for `vllm bench serve`
#   BENCH_CONC=16     --max-concurrency for the bench
#   BENCH_IN=512      --random-input-len
#   BENCH_OUT=128     --random-output-len
#   BENCH_SEED=0      --seed (the random dataset is reproducible for a fixed seed)
#   SAMPLE_INTERVAL=0.5   seconds between gauge samples during load
#   OUTDIR=$VL_ROOT/notes/lab08
#
# ---------------------------------------------------------------------------
# VERIFIED FACTS THIS LAB RELIES ON (vLLM v0.30.0, checked against the source tree)
#
#   Metric names — all read out of `vllm/v1/metrics/loggers.py`:
#     vllm:num_requests_running                gauge   (line 521)
#     vllm:num_requests_waiting                gauge   (line 531)
#     vllm:num_requests_waiting_by_reason      gauge   (line 541) labels: reason=capacity|deferred
#     vllm:kv_cache_usage_perc                 gauge   (line 613) "1 means 100 percent usage"
#     vllm:prefix_cache_queries                counter (line 636) — in TOKENS, not blocks
#     vllm:prefix_cache_hits                   counter (line 647) — in TOKENS, not blocks
#     vllm:num_preemptions                     counter (line 713)
#     vllm:prompt_tokens                       counter (line 722)
#     vllm:generation_tokens                   counter (line 756)
#     vllm:request_success                     counter (line 766) label: finished_reason
#     vllm:time_to_first_token_seconds         histogram (line 871)
#     vllm:inter_token_latency_seconds         histogram (line 881)
#     vllm:request_queue_time_seconds          histogram (line 911)
#     vllm:request_prefill_time_seconds        histogram (line 931)
#     vllm:request_decode_time_seconds         histogram (line 941)
#     vllm:request_num_preemptions             histogram (line 951)
#     vllm:request_prefill_kv_computed_tokens  histogram (line 961)
#   There is NO `vllm:gpu_cache_usage_perc` anywhere in the tree — that was the old name
#   and it is gone. The current name is `vllm:kv_cache_usage_perc`. (Older guides,
#   including vLLM's own Grafana notes, still say otherwise.)
#
#   Exposition format — counters are exported with a `_total` suffix; gauges are not.
#   `$VLLM_SRC/docs/design/metrics.md:336-363` shows the literal text:
#       # TYPE vllm:generation_tokens_total counter
#       vllm:generation_tokens_total{model_name="..."} 27453.0
#       # TYPE vllm:num_requests_running gauge
#       vllm:num_requests_running{model_name="..."} 8.0
#   Every series carries `model_name` and `engine` labels
#   (`vllm/v1/metrics/loggers.py:484`), so this lab sums counters over label sets and
#   takes the max over label sets for gauges. The awk below accepts both `name` and
#   `name_total` spellings so it keeps working if that convention ever changes.
#
#   Histogram buckets — the TTFT family is `TIME_TO_FIRST_TOKEN_BUCKETS` in
#   `vllm/v1/metrics/buckets.py:57-80`: 0.001 0.005 0.01 0.02 0.04 0.06 0.08 0.1 0.25
#   0.5 0.75 1.0 2.5 5.0 7.5 10.0 20.0 40.0 80.0 160.0 640.0 2560.0 seconds. Note the
#   spacing: dense around interactive latencies, coarse in the tail. p99 read off these
#   buckets is therefore only as precise as the bucket it lands in.
#
#   `vllm bench serve` — verified in `vllm/benchmarks/serve.py` and
#   `vllm/benchmarks/datasets/datasets.py`:
#     * subcommand name `serve` (`vllm/entrypoints/cli/benchmark/serve.py:13`)
#     * `--backend` choices come from ASYNC_REQUEST_FUNCS
#       (`vllm/benchmarks/lib/endpoint_request_func.py:1091-1107`): vllm, openai,
#       openai-chat, openai-responses, openai-audio, openai-embeddings, ...
#     * `--endpoint` default is `/v1/completions` (serve.py:1635); the chat backend
#       asserts the URL ends in `chat/completions`
#       (`endpoint_request_func.py:353`), so `--backend openai-chat` REQUIRES
#       `--endpoint /v1/chat/completions`
#     * `--dataset-name` default `random`, choices include random/sharegpt/sonnet/hf/
#       custom/prefix_repetition/... (datasets.py:1594-1615)
#     * `--random-input-len` default 1024, `--random-output-len` default 128,
#       `--random-range-ratio` default "0.0", `--random-prefix-len` default 0
#       (datasets.py:1911-1944)
#     * `--num-prompts` default 1000 (datasets.py:1591), `--seed` default 0 (datasets.py:1587)
#     * `--percentile-metrics` default "ttft,tpot,itl" for generative models (serve.py:1818)
#     * `--metric-percentiles` default "99" (serve.py:1829)
#     * `--ready-check-timeout-sec` default 0 = readiness check SKIPPED (serve.py:1958-1963)
#     * `--ignore-eos` (serve.py:1801) is forwarded into the chat payload
#       (`endpoint_request_func.py:135-136`), and `max_completion_tokens` is set to the
#       dataset's output length — which is what makes "generated = num_prompts x output_len"
#       exact for `--random-range-ratio 0`.
#     * printed labels (`serve.py:1192-1408`): "Successful requests:", "Benchmark duration (s):",
#       "Total input tokens:", "Total generated tokens:", "Request throughput (req/s):",
#       "Output token throughput (tok/s):", "Total token throughput (tok/s):",
#       "Peak concurrent requests:", "Mean TTFT (ms):", "Median TTFT (ms):", "P99 TTFT (ms):",
#       "Mean TPOT (ms):", "Mean ITL (ms):", "P99 ITL (ms):", "Mean E2EL (ms):"
# ---------------------------------------------------------------------------

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"

N="${N:-48}"
CONC="${CONC:-16}"
MAX_TOKENS="${MAX_TOKENS:-96}"
REQ_TIMEOUT="${REQ_TIMEOUT:-300}"

BENCH_N="${BENCH_N:-64}"
BENCH_CONC="${BENCH_CONC:-16}"
BENCH_IN="${BENCH_IN:-512}"
BENCH_OUT="${BENCH_OUT:-128}"
BENCH_SEED="${BENCH_SEED:-0}"

SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"
OUTDIR="${OUTDIR:-$VL_ROOT/notes/lab08}"

mkdir -p "$OUTDIR"

hr() { printf '%s\n' "------------------------------------------------------------------------------"; }
hr2() { printf '%s\n' ".............................................................................."; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

require_server() {
  if ! curl -fsS -m 5 "${BASE}/health" >/dev/null 2>&1; then
    die "no vLLM server answering ${BASE}/health.
  Start one in another terminal first:
      bash scripts/serve.sh
  then re-run this lab. (Or point it elsewhere: PORT=8001 bash labs/08_metrics_and_bench.sh)"
  fi
}

served_model_id() {
  curl -fsS -m 10 "${BASE}/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "")'
}

# ---------------------------------------------------------------------------
# /metrics plumbing
# ---------------------------------------------------------------------------
# Reads /metrics into <label>.prom (raw text) and <label>.values (key<TAB>value).
#
# Conventions handled here:
#   * counters may appear as `name` or `name_total` in the exposition — accept both;
#   * every series has model_name/engine labels, so strip labels before matching;
#   * gauges and histograms appear once per label set: sum counters, max gauges;
#   * `reason="capacity"` / `reason="deferred"` come out of the waiting-by-reason gauge.
read_metrics() {
  local label="$1"
  local raw="$OUTDIR/${label}.prom"
  local parsed="$OUTDIR/${label}.values"
  curl -fsS -m 10 "${BASE}/metrics" > "$raw" \
    || die "could not scrape ${BASE}/metrics"
  awk '
    function base(k) { sub(/\{.*$/, "", k); return k }
    /^#/ { next }
    NF < 2 { next }
    {
      n = base($1); v = $NF + 0
      if (n == "vllm:num_requests_running")                { if (!seen["running"] || v > g_run) g_run = v; seen["running"]=1 }
      else if (n == "vllm:num_requests_waiting")           { if (!seen["waiting"] || v > g_wait) g_wait = v; seen["waiting"]=1 }
      else if (n == "vllm:num_requests_waiting_by_reason") {
        if ($1 ~ /reason="capacity"/)                     { if (!seen["wcap"] || v > g_wcap) g_wcap = v; seen["wcap"]=1 }
        else if ($1 ~ /reason="deferred"/)                { if (!seen["wdef"] || v > g_wdef) g_wdef = v; seen["wdef"]=1 }
      }
      else if (n == "vllm:kv_cache_usage_perc")            { if (!seen["kv"] || v > g_kv) g_kv = v; seen["kv"]=1 }
      else if (n == "vllm:generation_tokens" || n == "vllm:generation_tokens_total")       c_gen += v
      else if (n == "vllm:prompt_tokens" || n == "vllm:prompt_tokens_total")               c_prompt += v
      else if (n == "vllm:prefix_cache_queries" || n == "vllm:prefix_cache_queries_total") c_pq += v
      else if (n == "vllm:prefix_cache_hits" || n == "vllm:prefix_cache_hits_total")       c_ph += v
      else if (n == "vllm:num_preemptions" || n == "vllm:num_preemptions_total")           c_pre += v
      else if (n == "vllm:request_success" || n == "vllm:request_success_total")           c_ok += v
      else if (n == "vllm:time_to_first_token_seconds_count")  c_ttft_n += v
      else if (n == "vllm:time_to_first_token_seconds_sum")    c_ttft_sum += v
      else if (n == "vllm:time_to_first_token_seconds_bucket") {
        if (match($0, /le="[^"]*"/)) {
          le = substr($0, RSTART + 4, RLENGTH - 5)
          b_ttft[le] += v
        }
      }
      else if (n == "vllm:inter_token_latency_seconds_count")  c_itl_n += v
      else if (n == "vllm:inter_token_latency_seconds_sum")    c_itl_sum += v
      else if (n == "vllm:e2e_request_latency_seconds_count")  c_e2e_n += v
      else if (n == "vllm:e2e_request_latency_seconds_sum")    c_e2e_sum += v
      else if (n == "vllm:request_queue_time_seconds_sum")     c_q_sum += v
      else if (n == "vllm:request_queue_time_seconds_count")   c_q_n += v
      else if (n == "vllm:request_prefill_time_seconds_sum")   c_pf_sum += v
      else if (n == "vllm:request_prefill_time_seconds_count") c_pf_n += v
      else if (n == "vllm:request_decode_time_seconds_sum")    c_dc_sum += v
      else if (n == "vllm:request_decode_time_seconds_count")  c_dc_n += v
    }
    END {
      if (seen["running"]) printf "running\t%.0f\n", g_run
      if (seen["waiting"]) printf "waiting\t%.0f\n", g_wait
      if (seen["wcap"])    printf "waiting_capacity\t%.0f\n", g_wcap
      if (seen["wdef"])    printf "waiting_deferred\t%.0f\n", g_wdef
      if (seen["kv"])      printf "kv_perc\t%.6f\n", g_kv
      printf "generation_tokens\t%.0f\n", c_gen
      printf "prompt_tokens\t%.0f\n", c_prompt
      printf "prefix_queries\t%.0f\n", c_pq
      printf "prefix_hits\t%.0f\n", c_ph
      printf "preemptions\t%.0f\n", c_pre
      printf "request_success\t%.0f\n", c_ok
      printf "ttft_count\t%.0f\n", c_ttft_n
      printf "ttft_sum_s\t%.6f\n", c_ttft_sum
      printf "itl_count\t%.0f\n", c_itl_n
      printf "itl_sum_s\t%.6f\n", c_itl_sum
      printf "e2e_count\t%.0f\n", c_e2e_n
      printf "e2e_sum_s\t%.6f\n", c_e2e_sum
      printf "queue_count\t%.0f\n", c_q_n
      printf "queue_sum_s\t%.6f\n", c_q_sum
      printf "prefill_count\t%.0f\n", c_pf_n
      printf "prefill_sum_s\t%.6f\n", c_pf_sum
      printf "decode_count\t%.0f\n", c_dc_n
      printf "decode_sum_s\t%.6f\n", c_dc_sum
      for (le in b_ttft) printf "ttft_bucket_%s\t%.0f\n", le, b_ttft[le]
    }
  ' "$raw" > "$parsed"
  printf '%s' "$parsed"
}

# mval <file> <key> [default]
mval() {
  local v
  v="$(awk -F'\t' -v k="$2" '$1 == k { print $2; exit }' "$1" 2>/dev/null || true)"
  if [ -z "$v" ]; then printf '%s' "${3:-0}"; else printf '%s' "$v"; fi
}

# delta <before-file> <after-file> <key>   (integer-valued counters)
delta() { awk -v a="$(mval "$1" "$3")" -v b="$(mval "$2" "$3")" 'BEGIN { printf "%.0f", b - a }'; }

# deltasum <before-file> <after-file> <key>   (counters that hold SECONDS — keep the
# fraction. Rounding a `_sum` series to an integer loses the entire measurement.)
deltasum() { awk -v a="$(mval "$1" "$3")" -v b="$(mval "$2" "$3")" 'BEGIN { printf "%.6f", b - a }'; }

# Gauge sampler: one curl per sample, appends ts/run/wait/kv to the file until the stop
# file appears. Used to get PEAK gauges, which no post-hoc diff can recover.
sample_gauges() {
  local out="$1" stop="$2"
  while [ ! -f "$stop" ]; do
    curl -fsS -m 3 "${BASE}/metrics" 2>/dev/null | awk -v ts="$(date +%s.%N)" '
      function base(k) { sub(/\{.*$/, "", k); return k }
      /^#/ { next } NF < 2 { next }
      { n = base($1); v = $NF + 0
        if (n == "vllm:num_requests_running")  { if (v > r) r = v }
        else if (n == "vllm:num_requests_waiting") { if (v > w) w = v }
        else if (n == "vllm:num_requests_waiting_by_reason" && $1 ~ /reason="capacity"/) { if (v > wc) wc = v }
        else if (n == "vllm:kv_cache_usage_perc")  { if (v > kv) kv = v }
      }
      END { printf "%s\t%.0f\t%.0f\t%.0f\t%.6f\n", ts, r, w, wc, kv }
    ' >> "$out" 2>/dev/null || true
    sleep "$SAMPLE_INTERVAL"
  done
}

# peak <samples-file> <column 2..5>
peak() {
  [ -s "$1" ] || { printf 'n/a'; return 0; }
  awk -v c="$2" '{ v = $c + 0; if (NR == 1 || v > m) m = v } END { printf "%.4g", m }' "$1"
}

# mean of a column, over samples
sample_mean() {
  [ -s "$1" ] || { printf 'n/a'; return 0; }
  awk -v c="$2" '{ s += $c + 0 } END { printf "%.4g", s / NR }' "$1"
}

# run_window <label> <cmd...>   — snapshot, sample while running, snapshot again
run_window() {
  local label="$1"; shift
  local samples="$OUTDIR/${label}.samples.tsv"
  local stop="$OUTDIR/${label}.stop"
  rm -f "$stop"
  : > "$samples"
  read_metrics "${label}.before" >/dev/null
  local t0 t1
  t0="$(date +%s.%N)"
  sample_gauges "$samples" "$stop" &
  local sampler=$!
  "$@" || printf 'WARNING: command exited non-zero\n'
  t1="$(date +%s.%N)"
  touch "$stop"
  wait "$sampler" 2>/dev/null || true
  read_metrics "${label}.after" >/dev/null
  awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.3f", a - b }' > "$OUTDIR/${label}.wall"
}

# ---------------------------------------------------------------------------
# the hand-driven burst
# ---------------------------------------------------------------------------
PROMPT="You are explaining a caching system to a colleague. Describe, in a single paragraph and without using any lists, what a least-recently-used eviction policy does, why it is a reasonable default, and one workload where it behaves badly."

drive_burst() {
  local latfile="$1" bodyfile="$2"
  : > "$latfile"
  seq 1 "$N" | xargs -P "$CONC" -I{} \
    curl -sS -m "$REQ_TIMEOUT" -o /dev/null -w '%{time_total}\n' \
      -H 'Content-Type: application/json' \
      --data-binary "@${bodyfile}" \
      "${BASE}/v1/completions" >> "$latfile" || true
  local got
  got="$(wc -l < "$latfile" | tr -d ' ')"
  [ "$got" -eq "$N" ] || printf 'WARNING: only %s of %s requests returned a timing\n' "$got" "$N"
}

pctl() {
  sort -n "$1" | awk -v p="$2" '{ v[NR] = $1 + 0 }
    END { if (NR == 0) { print "n/a"; exit }
          r = (p / 100) * (NR - 1); i = int(r) + 1; f = r - int(r)
          if (i + 1 <= NR) printf "%.4f", v[i] + f * (v[i + 1] - v[i]); else printf "%.4f", v[i] }'
}

# ttft_deltas <before> <after>   -> "le<TAB>cumulative_delta" per bucket, sorted,
# followed by "__total<TAB>n".
#
# IMPORTANT: Prometheus histograms are CUMULATIVE — `_bucket{le="0.1"}` is the number of
# observations <= 0.1 s, not the number in [prev, 0.1]. So the delta of that series is
# already "how many requests in this window finished prefill within 0.1 s". The total
# number of observations is the `+Inf` bucket (equivalently `_count`), NOT the sum of the
# rows. Getting this wrong makes every percentile wrong, so it is worth reading twice.
#
# The "+Inf" bucket is used only for the total and is not printed as a row.
ttft_deltas() {
  awk -F'\t' '
    FILENAME == ARGV[1] { before[$1] = $2 + 0; next }
    { after[$1] = $2 + 0 }
    END {
      n = 0
      for (k in after)  if (k ~ /^ttft_bucket_[0-9.]+$/) { v = substr(k, 13) + 0; if (!(v in seen)) { seen[v] = 1; les[++n] = v } }
      for (k in before) if (k ~ /^ttft_bucket_[0-9.]+$/) { v = substr(k, 13) + 0; if (!(v in seen)) { seen[v] = 1; les[++n] = v } }
      for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
        if (les[j] < les[i]) { t = les[i]; les[i] = les[j]; les[j] = t }
      last = 0
      for (i = 1; i <= n; i++) {
        k = sprintf("ttft_bucket_%g", les[i])
        d = after[k] - before[k]
        if (d < 0) d = 0
        printf "%g\t%.0f\n", les[i], d
        last = d
      }
      total = after["ttft_bucket_+Inf"] - before["ttft_bucket_+Inf"]
      if (total < 0 || total != total) total = 0
      if (total == 0) total = last   # +Inf missing: best available fallback
      printf "__total\t%.0f\n", total
    }' "$1" "$2"
}

# ttft_pct <deltas-file> <percentile>  -> the bucket boundary it lands in, milliseconds.
# This is a bucket estimate, not an exact percentile: the engine only records which bucket
# a request fell into (vllm/v1/metrics/buckets.py:57-80). Read the client for exact
# percentiles; read this to see the SHAPE.
ttft_pct() {
  awk -F'\t' -v p="$2" '
    $1 == "__total" { total = $2 + 0; next }
    { n++; le[n] = $1 + 0; cum[n] = $2 + 0 }
    END {
      if (total <= 0 || n == 0) { print "n/a"; exit }
      target = total * p / 100
      for (i = 1; i <= n; i++) if (cum[i] >= target) { printf "%.1f", le[i] * 1000; exit }
      printf ">%.0f", le[n] * 1000
    }' "$1"
}

show_ttft_buckets() {
  local deltas="$1" count="$2"
  hr2
  printf 'TTFT histogram, this burst only (cumulative bucket delta)\n'
  hr2
  awk -F'\t' -v count="$count" '
    $1 == "__total" { total = $2 + 0; next }
    { n++; le[n] = $1 + 0; cum[n] = $2 + 0 }
    END {
      if (n == 0 || total <= 0) { print "  (no TTFT samples in this burst)"; exit }
      for (i = 1; i <= n; i++) {
        pct = cum[i] / total * 100
        bar = ""; bars = int(pct / 4)
        for (j = 0; j < bars; j++) bar = bar "#"
        printf "  prefill done within %8.3f s : %6d  %6.1f%%  %s\n", le[i], cum[i], pct, bar
        if (cum[i] >= total) break
      }
      printf "  %d TTFT observations in this burst (ttft_count delta said %s)\n", total, count
    }' "$deltas"
}

do_burst() {
  require_server
  hr
  printf 'PHASE A — hand-driven burst, with /metrics before / during / after\n'
  hr
  local served
  served="$(served_model_id)"
  [ -n "$served" ] || die "could not read a model id from ${BASE}/v1/models"
  printf '  server      : %s\n' "$BASE"
  printf '  model       : %s\n' "$served"
  printf '  workload    : %s requests x %s tokens, concurrency %s (ignore_eos -> exact)\n' \
    "$N" "$MAX_TOKENS" "$CONC"
  printf '  sampling    : every %ss while the load runs\n\n' "$SAMPLE_INTERVAL"

  local body="$OUTDIR/burst.body.json"
  local lat="$OUTDIR/burst.lat"
  cat > "$body" <<JSON
{"model": "${served}", "prompt": "${PROMPT}", "max_tokens": ${MAX_TOKENS}, "temperature": 0, "ignore_eos": true}
JSON

  printf 'warmup ...\n'
  curl -sS -m "$REQ_TIMEOUT" -o /dev/null -H 'Content-Type: application/json' \
    --data-binary "@${body}" "${BASE}/v1/completions" || die "warmup request failed"

  printf 'running ...\n'
  run_window burst drive_burst "$lat" "$body"

  local before="$OUTDIR/burst.before.values" after="$OUTDIR/burst.after.values"
  local samples="$OUTDIR/burst.samples.tsv"
  local wall dgen dprompt dph dpq dpre dck hitrate
  wall="$(cat "$OUTDIR/burst.wall")"
  dgen="$(delta "$before" "$after" generation_tokens)"
  dprompt="$(delta "$before" "$after" prompt_tokens)"
  dph="$(delta "$before" "$after" prefix_hits)"
  dpq="$(delta "$before" "$after" prefix_queries)"
  dpre="$(delta "$before" "$after" preemptions)"
  dck="$(delta "$before" "$after" request_success)"
  hitrate="$(awk -v h="$dph" -v q="$dpq" 'BEGIN { if (q > 0) printf "%.1f%%", h / q * 100; else print "n/a (no queries)" }')"

  hr
  printf 'A1 — the series that matter, before / peak / after\n'
  hr
  printf '%-46s %12s %12s %12s\n' "series" "before" "PEAK" "after"
  printf '%-46s %12s %12s %12s\n' "----------------------------------------------" "------" "----" "-----"
  printf '%-46s %12s %12s %12s\n' "vllm:num_requests_running" \
    "$(mval "$before" running)" "$(peak "$samples" 2)" "$(mval "$after" running)"
  printf '%-46s %12s %12s %12s\n' "vllm:num_requests_waiting" \
    "$(mval "$before" waiting)" "$(peak "$samples" 3)" "$(mval "$after" waiting)"
  printf '%-46s %12s %12s %12s\n' "  ...by_reason{reason=capacity}" \
    "$(mval "$before" waiting_capacity)" "$(peak "$samples" 4)" "$(mval "$after" waiting_capacity)"
  printf '%-46s %12s %12s %12s\n' "  ...by_reason{reason=deferred}" \
    "$(mval "$before" waiting_deferred)" "-" "$(mval "$after" waiting_deferred)"
  printf '%-46s %12s %12s %12s\n' "vllm:kv_cache_usage_perc  (1.0 = 100%)" \
    "$(mval "$before" kv_perc)" "$(peak "$samples" 5)" "$(mval "$after" kv_perc)"
  printf '%-46s %12s %12s %12s\n' "vllm:num_preemptions_total (counter)" \
    "$(mval "$before" preemptions)" "-" "$(mval "$after" preemptions)"
  printf '\n'

  hr
  printf 'A2 — what the burst actually did (counter deltas)\n'
  hr
  printf '  wall clock                 : %s s for %s requests\n' "$wall" "$N"
  printf '  requests finished          : %s   (success counter delta)\n' "$dck"
  printf '  generation tokens          : %s   (nominal %s = %s x %s)\n' "$dgen" "$(( N * MAX_TOKENS ))" "$N" "$MAX_TOKENS"
  printf '  prompt tokens              : %s\n' "$dprompt"
  printf '  prefix cache queries/hits  : %s / %s   -> hit rate %s\n' "$dpq" "$dph" "$hitrate"
  printf '  preemptions                : %s\n' "$dpre"
  printf '  decode throughput          : %s tok/s   (generation delta / wall)\n' \
    "$(awk -v g="$dgen" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", g / w; else print "n/a" }')"
  printf '  client-side p50 / p99      : %s / %s s   (from curl, not from vLLM)\n' \
    "$(pctl "$lat" 50)" "$(pctl "$lat" 99)"
  printf '\n'
  printf '  NOTE on the prefix counters: they count TOKENS, not blocks\n'
  printf '  (vllm/v1/metrics/loggers.py:636-653). Every request here shares one prompt, so a\n'
  printf '  hit rate near 100%% after the first request is expected — that is the control\n'
  printf '  showing the burst was decode-dominated, not prefill-dominated.\n'

  ttft_deltas "$before" "$after" > "$OUTDIR/burst.ttft_deltas.tsv"
  show_ttft_buckets "$OUTDIR/burst.ttft_deltas.tsv" "$(delta "$before" "$after" ttft_count)"

  hr
  printf 'A3 — where the time went, from the engine'"'"'s own histograms\n'
  hr
  local qmean pfmean dcmean e2emean itlmean
  qmean="$(awk -v s="$(deltasum "$before" "$after" queue_sum_s)" -v n="$(delta "$before" "$after" queue_count)" 'BEGIN { if (n > 0) printf "%.1f", s / n * 1000; else print "n/a" }')"
  pfmean="$(awk -v s="$(deltasum "$before" "$after" prefill_sum_s)" -v n="$(delta "$before" "$after" prefill_count)" 'BEGIN { if (n > 0) printf "%.1f", s / n * 1000; else print "n/a" }')"
  dcmean="$(awk -v s="$(deltasum "$before" "$after" decode_sum_s)" -v n="$(delta "$before" "$after" decode_count)" 'BEGIN { if (n > 0) printf "%.1f", s / n * 1000; else print "n/a" }')"
  e2emean="$(awk -v s="$(deltasum "$before" "$after" e2e_sum_s)" -v n="$(delta "$before" "$after" e2e_count)" 'BEGIN { if (n > 0) printf "%.1f", s / n * 1000; else print "n/a" }')"
  itlmean="$(awk -v s="$(deltasum "$before" "$after" itl_sum_s)" -v n="$(delta "$before" "$after" itl_count)" 'BEGIN { if (n > 0) printf "%.1f", s / n * 1000; else print "n/a" }')"
  printf '  %-42s %10s ms\n' "vllm:request_queue_time_seconds (mean)" "$qmean"
  printf '  %-42s %10s ms\n' "vllm:request_prefill_time_seconds (mean)" "$pfmean"
  printf '  %-42s %10s ms\n' "vllm:request_decode_time_seconds (mean)" "$dcmean"
  printf '  %-42s %10s ms\n' "vllm:e2e_request_latency_seconds (mean)" "$e2emean"
  printf '  %-42s %10s ms\n' "vllm:inter_token_latency_seconds (mean)" "$itlmean"
  printf '\n  e2e = queue + prefill + decode, so whichever term dominates IS your bottleneck.\n'
  printf '  If queue_time is the big one, you are starving or over-subscribing the engine.\n'
  printf '  If decode_time is the big one, you are GPU-bound. Those are different problems.\n\n'

  # persist the table rows for notes/
  {
    printf 'metric\tbefore\tpeak\tafter\n'
    printf 'vllm:num_requests_running\t%s\t%s\t%s\n' "$(mval "$before" running)" "$(peak "$samples" 2)" "$(mval "$after" running)"
    printf 'vllm:num_requests_waiting\t%s\t%s\t%s\n' "$(mval "$before" waiting)" "$(peak "$samples" 3)" "$(mval "$after" waiting)"
    printf 'vllm:num_requests_waiting_by_reason{capacity}\t%s\t%s\t%s\n' "$(mval "$before" waiting_capacity)" "$(peak "$samples" 4)" "$(mval "$after" waiting_capacity)"
    printf 'vllm:kv_cache_usage_perc\t%s\t%s\t%s\n' "$(mval "$before" kv_perc)" "$(peak "$samples" 5)" "$(mval "$after" kv_perc)"
  } > "$OUTDIR/burst.table.tsv"
  {
    printf 'wall_s\t%s\n' "$wall"
    printf 'gen_tokens\t%s\n' "$dgen"
    printf 'prompt_tokens\t%s\n' "$dprompt"
    printf 'prefix_queries\t%s\n' "$dpq"
    printf 'prefix_hits\t%s\n' "$dph"
    printf 'prefix_hit_rate\t%s\n' "$hitrate"
    printf 'preemptions\t%s\n' "$dpre"
    printf 'running_peak\t%s\n' "$(peak "$samples" 2)"
    printf 'waiting_peak\t%s\n' "$(peak "$samples" 3)"
    printf 'kv_peak\t%s\n' "$(peak "$samples" 5)"
    printf 'client_p50_s\t%s\n' "$(pctl "$lat" 50)"
    printf 'client_p99_s\t%s\n' "$(pctl "$lat" 99)"
    printf 'ttft_p50_ms\t%s\n' "$(ttft_pct "$OUTDIR/burst.ttft_deltas.tsv" 50)"
    printf 'ttft_p99_ms\t%s\n' "$(ttft_pct "$OUTDIR/burst.ttft_deltas.tsv" 99)"
    printf 'queue_mean_ms\t%s\n' "$qmean"
    printf 'prefill_mean_ms\t%s\n' "$pfmean"
    printf 'decode_mean_ms\t%s\n' "$dcmean"
    printf 'e2e_mean_ms\t%s\n' "$e2emean"
    printf 'itl_mean_ms\t%s\n' "$itlmean"
  } > "$OUTDIR/burst.results.tsv"
  printf 'recorded %s and %s\n' "$OUTDIR/burst.table.tsv" "$OUTDIR/burst.results.tsv"
}

# ---------------------------------------------------------------------------
# vllm bench serve
# ---------------------------------------------------------------------------
do_bench() {
  require_server
  hr
  printf 'PHASE B — vLLM'"'"'s own benchmark client, same server, its own metrics window\n'
  hr
  local served
  served="$(served_model_id)"
  [ -n "$served" ] || die "could not read a model id from ${BASE}/v1/models"

  mkdir -p "$OUTDIR/bench"
  local stamp benchout
  stamp="$(date +%Y%m%d-%H%M%S)"
  benchout="$OUTDIR/bench/bench-${stamp}.txt"

  # Every flag below was read out of vllm/benchmarks/serve.py and
  # vllm/benchmarks/datasets/datasets.py in THIS checkout — see the header block.
  # `--ignore-eos` + `--random-range-ratio 0` make the generated token count exactly
  # num_prompts * random_output_len, which is the anchor for the reconciliation below.
  # shellcheck disable=SC2054  # the commas live inside --percentile-metrics' value
  local -a bench_cmd=(
    vllm bench serve
    --backend openai-chat
    --base-url "$BASE"
    --endpoint /v1/chat/completions
    --model "$served"
    --dataset-name random
    --random-input-len "$BENCH_IN"
    --random-output-len "$BENCH_OUT"
    --random-range-ratio 0
    --num-prompts "$BENCH_N"
    --max-concurrency "$BENCH_CONC"
    --ignore-eos
    --seed "$BENCH_SEED"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,99
    --ready-check-timeout-sec 60
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
    --disable-tqdm
    --save-result
    --result-dir "$OUTDIR/bench"
    --result-filename "bench-${stamp}.json"
  )

  # Print the command with one flag (plus its value) per continuation line, so the
  # result can be copied straight into a terminal. Values containing spaces or braces
  # are quoted, which matters for --extra-body.
  printf '  the exact command:\n\n'
  printf '    vllm bench serve'
  local i=3 flag value
  while [ "$i" -lt "${#bench_cmd[@]}" ]; do
    flag="${bench_cmd[$i]}"
    case "$flag" in
      --ignore-eos|--disable-tqdm|--save-result)
        printf ' \\\n      %s' "$flag"
        i=$((i + 1))
        ;;
      *)
        value="${bench_cmd[$((i + 1))]:-}"
        case "$value" in
          *" "*|*"{"*) printf ' \\\n      %s %s' "$flag" "'$value'" ;;
          *)           printf ' \\\n      %s %s' "$flag" "$value" ;;
        esac
        i=$((i + 2))
        ;;
    esac
  done
  printf '\n\n'

  if ! command -v vllm >/dev/null 2>&1; then
    die "the 'vllm' CLI is not on PATH.
  Run:  source scripts/env.sh   (it activates \$VENV)
  or:   \$VENV/bin/vllm bench serve ..."
  fi

  printf 'running (this imports vLLM client-side and can take ~30 s to start) ...\n\n'
  run_window bench "${bench_cmd[@]}" > "$benchout" 2>&1 || true
  cat "$benchout"

  # ---- parse the client's report -----------------------------------------
  local f="$benchout"
  bfield() { grep -m1 -F "$1" "$f" 2>/dev/null | awk '{ print $NF }' || true; }
  local b_dur b_ok b_fail b_gen b_in b_reqthr b_outthr b_peak b_ttft_mean b_ttft_p50 b_ttft_p99
  local b_itl_mean b_itl_p99 b_e2e_p99 b_tpot_mean
  b_dur="$(bfield 'Benchmark duration (s):')"
  b_ok="$(bfield 'Successful requests:')"
  b_fail="$(bfield 'Failed requests:')"
  b_gen="$(bfield 'Total generated tokens:')"
  b_in="$(bfield 'Total input tokens:')"
  b_reqthr="$(bfield 'Request throughput (req/s):')"
  b_outthr="$(bfield 'Output token throughput (tok/s):')"
  b_peak="$(bfield 'Peak concurrent requests:')"
  b_ttft_mean="$(bfield 'Mean TTFT (ms):')"
  b_ttft_p50="$(bfield 'P50 TTFT (ms):')"
  b_ttft_p99="$(bfield 'P99 TTFT (ms):')"
  b_itl_mean="$(bfield 'Mean ITL (ms):')"
  b_itl_p99="$(bfield 'P99 ITL (ms):')"
  b_e2e_p99="$(bfield 'P99 E2EL (ms):')"
  b_tpot_mean="$(bfield 'Mean TPOT (ms):')"

  [ -n "$b_dur" ] || die "could not parse a 'Benchmark duration (s):' line out of $f.
  The most common causes: the server was not reachable at ${BASE}, or
  --endpoint did not match --backend (openai-chat needs /v1/chat/completions)."

  # ---- the engine's view of the same window ------------------------------
  local before="$OUTDIR/bench.before.values" after="$OUTDIR/bench.after.values"
  local samples="$OUTDIR/bench.samples.tsv"
  ttft_deltas "$before" "$after" > "$OUTDIR/bench.ttft_deltas.tsv"
  local dgen dprompt dpq dph dpre dck
  dgen="$(delta "$before" "$after" generation_tokens)"
  dprompt="$(delta "$before" "$after" prompt_tokens)"
  dpq="$(delta "$before" "$after" prefix_queries)"
  dph="$(delta "$before" "$after" prefix_hits)"
  dpre="$(delta "$before" "$after" preemptions)"
  dck="$(delta "$before" "$after" request_success)"

  hr
  printf 'B1 — reconciliation: client report vs engine metrics, SAME window\n'
  hr
  printf '%-30s %18s %18s %s\n' "quantity" "vllm bench serve" "/metrics delta" "verdict"
  printf '%-30s %18s %18s %s\n' "------------------------------" "------------------" "------------------" "-------"

  reconcile() {
    # reconcile <label> <client value> <metric value> <tolerance-relative>
    awk -v l="$1" -v c="$2" -v m="$3" -v tol="$4" 'BEGIN {
      if (c == "" || m == "") { printf "%-30s %18s %18s %s\n", l, (c == "" ? "n/a" : c), (m == "" ? "n/a" : m), "not comparable"; exit }
      d = (c + 0) - (m + 0); rel = (c + 0) != 0 ? (d < 0 ? -d : d) / (c + 0) : 0
      verdict = (rel <= tol) ? "AGREE" : (rel <= 0.10 ? "close" : "DISAGREE")
      printf "%-30s %18.4g %18.4g %s (%.1f%%)\n", l, c + 0, m + 0, verdict, rel * 100
    }'
  }
  reconcile "generated tokens" "$b_gen" "$dgen" 0.02
  reconcile "requests completed" "$b_ok" "$dck" 0.02
  # The next row is EXPECTED to disagree: $OUTDIR/bench.wall is the whole `vllm bench`
  # process (client import + tokenizer + dataset build + the run), while the client's own
  # "Benchmark duration" covers only the measured window.
  reconcile "duration vs process wall" "$b_dur" "$(cat "$OUTDIR/bench.wall")" 0.30

  printf '%-30s %18s %18s %s\n' "output tok/s" "$b_outthr" \
    "$(awk -v g="$dgen" -v d="$b_dur" 'BEGIN { if (d > 0) printf "%.2f", g / d; else print "" }')" \
    "(engine tokens / client duration)"
  printf '%-30s %18s %18s %s\n' "peak concurrent requests" "$b_peak" "$(peak "$samples" 2)" "sampled every ${SAMPLE_INTERVAL}s"
  printf '%-30s %18s %18s %s\n' "P50 TTFT (ms)" "$b_ttft_p50" "$(ttft_pct "$OUTDIR/bench.ttft_deltas.tsv" 50)" "histogram is coarse"
  printf '%-30s %18s %18s %s\n' "P99 TTFT (ms)" "$b_ttft_p99" "$(ttft_pct "$OUTDIR/bench.ttft_deltas.tsv" 99)" "histogram is coarse"
  printf '\n'
  printf '  How to read a DISAGREE:\n'
  printf '   * generated tokens disagreeing by a little = traffic outside the window\n'
  printf '     (a stray client, or the previous phase still draining).\n'
  printf '   * disagreeing a lot = you are not measuring what you think; check --endpoint\n'
  printf '     and --backend match, and that nothing else is talking to the server.\n'
  printf '   * TTFT percentiles will rarely match to the millisecond, and that is not a bug:\n'
  printf '     the client measures per request, while the engine histogram has bucket\n'
  printf '     boundaries (vllm/v1/metrics/buckets.py:57-80). The engine can only tell you\n'
  printf '     which bucket a request fell into. Read the CLIENT for percentiles.\n\n'

  printf '  Other engine counters for this window:\n'
  printf '    prompt tokens            : %s   (client says %s)\n' "$dprompt" "$b_in"
  printf '    prefix queries / hits    : %s / %s\n' "$dpq" "$dph"
  printf '    preemptions              : %s\n' "$dpre"
  printf '    client mean ITL / TPOT   : %s / %s ms\n' "$b_itl_mean" "$b_tpot_mean"
  printf '    client mean TTFT         : %s ms\n' "$b_ttft_mean"
  printf '    client P99 ITL / E2EL    : %s / %s ms\n' "$b_itl_p99" "$b_e2e_p99"
  printf '    client result JSON       : %s/bench/bench-%s.json\n' "$OUTDIR" "$stamp"
  printf '\n'
  printf '  `vllm bench serve` is reported at the CLIENT, and the docs say so explicitly\n'
  printf '  ($VLLM_SRC/docs/benchmarking/cli.md, "Understanding the Latency Metrics").\n'
  printf '  ITL records gaps between streamed outputs; TPOT is (e2e - TTFT)/(tokens - 1)\n'
  printf '  per request. They agree only when each streamed output carries one token.\n\n'

  {
    printf 'bench_duration_s\t%s\n' "$b_dur"
    printf 'bench_completed\t%s\n' "$b_ok"
    printf 'bench_failed\t%s\n' "$b_fail"
    printf 'bench_total_generated\t%s\n' "$b_gen"
    printf 'bench_total_input\t%s\n' "$b_in"
    printf 'bench_req_throughput\t%s\n' "$b_reqthr"
    printf 'bench_out_throughput\t%s\n' "$b_outthr"
    printf 'bench_peak_concurrency\t%s\n' "$b_peak"
    printf 'bench_ttft_mean_ms\t%s\n' "$b_ttft_mean"
    printf 'bench_ttft_p50_ms\t%s\n' "$b_ttft_p50"
    printf 'bench_ttft_p99_ms\t%s\n' "$b_ttft_p99"
    printf 'bench_itl_mean_ms\t%s\n' "$b_itl_mean"
    printf 'bench_itl_p99_ms\t%s\n' "$b_itl_p99"
    printf 'bench_tpot_mean_ms\t%s\n' "$b_tpot_mean"
    printf 'bench_e2e_p99_ms\t%s\n' "$b_e2e_p99"
    printf 'metrics_gen_tokens\t%s\n' "$dgen"
    printf 'metrics_prompt_tokens\t%s\n' "$dprompt"
    printf 'metrics_preemptions\t%s\n' "$dpre"
    printf 'cmd\t%s\n' "${bench_cmd[*]}"
  } > "$OUTDIR/bench.results.tsv"
  printf 'recorded %s\n' "$OUTDIR/bench.results.tsv"
}

# ---------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------
do_interpret() {
  hr
  printf 'PHASE C — interpretation\n'
  hr
  local bres="$OUTDIR/burst.results.tsv"
  local brk="$OUTDIR/bench.results.tsv"
  [ -f "$bres" ] || die "no burst results yet. Run:  bash labs/08_metrics_and_bench.sh burst"

  local waiting kv running preempt qmean dcmean itlmean hitrate
  waiting="$(mval "$bres" waiting_peak)"
  kv="$(mval "$bres" kv_peak)"
  running="$(mval "$bres" running_peak)"
  preempt="$(mval "$bres" preemptions)"
  qmean="$(mval "$bres" queue_mean_ms)"
  dcmean="$(mval "$bres" decode_mean_ms)"
  itlmean="$(mval "$bres" itl_mean_ms)"
  hitrate="$(mval "$bres" prefix_hit_rate)"

  printf 'THE QUESTION: which single metric moved FIRST and predicted the latency?\n\n'
  printf 'Answer it in this order, using your own numbers:\n'
  printf '  0. How full did the batch get?  peak num_requests_running = %s\n' "$running"
  printf '     (compare it with --max-num-seqs on the server; if it never got close,\n'
  printf '      the engine was never the bottleneck)\n'
  printf '  1. Did anything WAIT?          peak num_requests_waiting = %s\n' "$waiting"
  printf '  2. If yes, was it admission or memory?  peak kv_cache_usage_perc = %s\n' "$kv"
  printf '  3. Did the engine evict?       num_preemptions_total delta   = %s\n' "$preempt"
  printf '  4. Where did the time go?      queue %s ms | decode %s ms | ITL %s ms\n\n' "$qmean" "$dcmean" "$itlmean"

  printf 'The honest general answer: `vllm:num_requests_waiting_by_reason{reason="capacity"}`\n'
  printf 'is the earliest signal, because it becomes non-zero on the very first engine step\n'
  printf 'where the scheduler has more work than it can admit — before any of that queueing\n'
  printf 'shows up in a latency histogram. It is set from `scheduler_stats.num_waiting_reqs`\n'
  printf 'in `vllm/v1/metrics/loggers.py:1094-1099`. Latency percentiles CONFIRM the damage;\n'
  printf 'the waiting gauge PREDICTS it.\n\n'
  printf 'Your run: waiting peaked at %s, KV peaked at %s.\n' "$waiting" "$kv"
  awk -v w="$waiting" -v kv="$kv" -v p="$preempt" 'BEGIN {
    if (w + 0 > 0 && kv + 0 >= 0.95) {
      print "  => KV-cache-bound. The queue existed because there were no free blocks."
      print "     Fix: smaller --max-model-len, lower concurrency, --kv-cache-dtype fp8 (lab 10),"
      print "     or more GPUs (lab 11). Raising --max-num-seqs will NOT help."
    } else if (w + 0 > 0 && kv + 0 < 0.95) {
      print "  => Budget-bound, not memory-bound. Blocks were free but the scheduler could not"
      print "     admit: --max-num-batched-tokens (2048 by default for `vllm serve` below 70 GB)"
      print "     or --max-num-seqs was the binding constraint. This is the Stage 5 lesson."
    } else {
      print "  => Arrival-limited (starved). Nothing queued, so the engine always had room and"
      print "     the latency you measured is the model + overhead, not contention. Raise N or"
      print "     CONC to find the knee."
    }
    if (p + 0 > 0) print "  !! num_preemptions_total moved: requests were evicted and will be recomputed."
  }'
  printf '\n  Your prefix cache hit rate this burst: %s\n' "$hitrate"
  printf '  A rate near zero with a shared prompt means reuse is NOT happening — check that the\n'
  printf '  shared part is at least one full block (16 tokens by default, CacheConfig.DEFAULT_BLOCK_SIZE)\n'
  printf '  and that the server was not started with --no-enable-prefix-caching.\n\n'

  hr
  printf 'METRICS -> DIAGNOSIS (the table to keep)\n'
  hr
  cat <<'TABLE'
| observed pattern                                            | likely cause                              | confirm with                                        |
| ---                                                         | ---                                       | ---                                                 |
| kv_cache_usage_perc ~1.0 AND waiting{capacity} rising        | KV cache exhausted; cannot admit          | num_preemptions_total rising; queue_time rising     |
| waiting{capacity} > 0 BUT kv_cache_usage_perc low            | token budget (max-num-batched-tokens)     | iteration_tokens_total histogram top bucket         |
| running pinned at max-num-seqs AND waiting > 0               | sequence budget saturated                 | startup log's max_num_seqs; num_requests_running     |
| num_preemptions_total climbing                               | memory pressure, evict + recompute        | request_prefill_kv_computed_tokens (recomputed work) |
| prefix_cache_queries rising, hits flat                       | no reuse: unique prompts / <1 block shared| cache_config_info{enable_prefix_caching}             |
| TTFT rising, inter_token_latency_seconds flat                | queue/prefill grew; decode is healthy     | request_queue_time vs request_decode_time           |
| inter_token_latency_seconds rising, TTFT flat                | decode batch grew; GPU saturated          | num_requests_running, kv_cache_usage_perc            |
| running < max-num-seqs AND waiting == 0                      | arrival-limited (starved)                 | request rate; prompt_tokens rate                     |
| generated tokens/s flat while prompt tokens/s spikes         | prefill stealing engine steps             | chunked prefill on; iteration_tokens_total           |
| all counters flat during "load"                              | you are not talking to that server        | /v1/models id; scrape /metrics by hand               |
TABLE
  printf '\n'

  hr
  printf 'RECONCILIATION CHECKLIST (fill in from B1)\n'
  hr
  if [ -f "$brk" ]; then
    printf '  client total generated  : %s\n' "$(mval "$brk" bench_total_generated)"
    printf '  engine generated delta  : %s\n' "$(mval "$brk" metrics_gen_tokens)"
    printf '  client output tok/s     : %s\n' "$(mval "$brk" bench_out_throughput)"
    printf '  client TTFT p50/p99 ms  : %s / %s\n' "$(mval "$brk" bench_ttft_p50_ms)" "$(mval "$brk" bench_ttft_p99_ms)"
  else
    printf '  (run `bash labs/08_metrics_and_bench.sh bench` to fill this in)\n'
  fi
  printf '\n'
}

show_status() {
  hr
  printf 'Lab 08 — Stage 7 status   (results in %s)\n' "$OUTDIR"
  hr
  if curl -fsS -m 3 "${BASE}/health" >/dev/null 2>&1; then
    printf 'server: UP at %s  (model: %s)\n\n' "$BASE" "$(served_model_id || echo '?')"
  else
    printf 'server: DOWN at %s\n\n' "$BASE"
  fi
  local f
  for f in burst.results.tsv bench.results.tsv; do
    if [ -f "$OUTDIR/$f" ]; then
      printf '%s:\n' "$f"
      awk -F'\t' '{ printf "  %-26s %s\n", $1, $2 }' "$OUTDIR/$f"
      printf '\n'
    else
      printf '%s: not recorded yet\n\n' "$f"
    fi
  done
  printf 'Commands: burst | bench | interpret | all | status\n'
}

case "${1:-all}" in
  all)       do_burst; do_bench; do_interpret ;;
  burst)     do_burst ;;
  bench)     do_bench ;;
  interpret) do_interpret ;;
  status)    show_status ;;
  *)         die "unknown argument '$1'. Use: burst | bench | interpret | all | status" ;;
esac

# ---------------------------------------------------------------------------
# RECORD: write these into notes/stage-07-metrics.md
#
#   date / vLLM version / torch version / GPU + driver
#   the exact `vllm serve` command the server is running, and the exact `vllm bench serve`
#   command (both are stored in $OUTDIR/*.results.tsv)
#
#   From PHASE A (the hand-driven burst):
#     - peak vllm:num_requests_running
#     - peak vllm:num_requests_waiting, and peak by_reason{capacity}
#     - peak vllm:kv_cache_usage_perc
#     - delta vllm:num_preemptions_total
#     - delta vllm:prefix_cache_queries_total / _hits_total and the hit rate
#     - delta vllm:generation_tokens_total and decode tok/s
#     - client p50 and p99 end-to-end latency, seconds
#     - the TTFT histogram bucket table (paste it), and the bucket-derived p50/p99
#     - mean queue / prefill / decode / e2e / inter-token time, ms
#
#   From PHASE B (`vllm bench serve`):
#     - Successful requests, Benchmark duration (s)
#     - Total generated tokens, Total input tokens
#     - Request throughput (req/s), Output token throughput (tok/s)
#     - Mean / Median / P50 / P99 TTFT (ms)
#     - Mean / P99 ITL (ms), Mean TPOT (ms), P99 E2EL (ms)
#     - Peak concurrent requests
#     - AND the engine-side deltas over the same window, so the two columns can be compared
#
#   Then answer, in one sentence each:
#     - Which single metric moved first, and why is it a leading rather than lagging signal?
#     - Where did the time go: queue, prefill, or decode? Quote the three numbers.
#     - Was this run GPU-saturated, budget-bound, or starved? Quote the evidence.
#     - Which two numbers from the client and the engine did NOT reconcile, and why?
#
# Nothing in this lab has been run on a GPU. Every number it prints is yours to produce.
# ---------------------------------------------------------------------------
