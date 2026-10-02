#!/usr/bin/env bash
# Lab 07 — Stage 6: compilation levels, CUDA graphs, and what they cost at startup.
#
# WHAT IT DEMONSTRATES
#   The same fixed workload, served four ways that differ ONLY in how vLLM compiles
#   the model and captures CUDA graphs:
#
#     o0       -O0   no torch.compile, no CUDA graphs
#     o1       -O1   torch.compile + PIECEWISE CUDA graphs
#     o2       -O2   the default: torch.compile + FULL_AND_PIECEWISE CUDA graphs
#     o2-fdo   -O2 with cudagraph_mode=FULL_DECODE_ONLY (full graphs for decode only)
#
#   For each configuration this lab makes you capture:
#     (a) the exact `vllm serve` command that was used,
#     (b) startup wall-clock time and the KV-cache startup line,
#     (c) decode throughput and p50 latency on a fixed number of identical requests,
#     (d) the process list, so you can see the API server / engine-core / worker split.
#
#   Then it prints a summary table row and tells you to Ctrl-C and start the next one.
#   The loop is under YOUR control: this script never restarts a server by itself.
#
# RUN
#   bash labs/07_compilation_levels.sh              # status: what is done, and the next command
#   bash labs/07_compilation_levels.sh o0           # guided run for one configuration
#   bash labs/07_compilation_levels.sh o1
#   bash labs/07_compilation_levels.sh o2
#   bash labs/07_compilation_levels.sh o2-fdo
#   bash labs/07_compilation_levels.sh table        # the summary table (also printed after each run)
#   bash labs/07_compilation_levels.sh procs        # process list of the running server
#   bash labs/07_compilation_levels.sh reset        # drop the recorded runs and start over
#
#   Resumable: each run writes $OUTDIR/<cfg>.tsv, and `status` recomputes from those
#   files. Ctrl-C between configurations costs you nothing.
#
# PREREQUISITES
#   * A Linux GPU pod with vLLM installed in $VENV (see docs/03-runpod-setup.md).
#   * Run from the project root:  source scripts/env.sh
#   * `curl` and `python3` on PATH (both are in the pod image).
#   * This lab needs a real GPU: `-O0` vs `-O2` is meaningless on CPU.
#
# ENV KNOBS (all have defaults)
#   N=32            requests per configuration
#   CONC=8          concurrent requests
#   MAX_TOKENS=128  generated tokens per request (exact: ignore_eos is set)
#   WAIT_SECS=900   how long to wait for /health before giving up
#   OUTDIR=$VL_ROOT/notes/lab07
#
# ---------------------------------------------------------------------------
# VERIFIED FACTS THIS LAB RELIES ON (vLLM v0.30.0, checked against the source tree)
#
#   * `-O0/-O1/-O2/-O3` are real CLI spellings. `vllm/utils/argparse_utils.py`
#     rewrites any `-O<n>` into `--optimization-level <n>` (lines 353-364).
#   * Default is `O2`: `optimization_level: OptimizationLevel = OptimizationLevel.O2`
#     (`vllm/config/vllm.py:442`). O2 = torch.compile + FULL_AND_PIECEWISE CUDA graphs
#     (`OPTIMIZATION_LEVEL_02`, `vllm/config/vllm.py:300-322`).
#   * O0 sets `cudagraph_mode=NONE` and `mode=NONE`; O1 sets `cudagraph_mode=PIECEWISE`
#     (`vllm/config/vllm.py:254-299`).
#   * CUDAGraphMode members are NONE, PIECEWISE, FULL, FULL_DECODE_ONLY,
#     FULL_AND_PIECEWISE (`vllm/config/compilation.py:53-63`).
#   * THERE IS NO `--cuda-graph-mode` FLAG. `git log -S "cuda-graph-mode"` over the
#     vLLM tree returns nothing — it has never existed. The CUDA-graph mode is a field
#     of CompilationConfig and is set either as JSON
#     (`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`, the form shown in
#     `$VLLM_SRC/docs/design/cuda_graphs.md:197-200`) or with the dotted shorthand
#     (`-cc.cudagraph_mode=FULL_DECODE_ONLY`, handled by
#     `vllm/utils/argparse_utils.py:420-449`). This lab uses the documented JSON form.
#   * User-set fields win over optimization-level defaults:
#     `vllm/config/vllm.py:1034-1062` ("User specified fields will not be overridden").
#     So `-O2` + an explicit cudagraph_mode is a legal combination.
#   * `--enforce-eager` is the older, narrower equivalent of "turn the graphs off": it
#     sets BOTH `compilation_config.mode = NONE` and `cudagraph_mode = NONE`
#     (`vllm/config/vllm.py:1689-1691`), and vLLM logs
#     "Cudagraph is disabled under eager mode" (`vllm/config/vllm.py:1963`).
#     `-O0` does the same two things *and* disables the fusion pass config and the
#     FlashInfer autotune (`OPTIMIZATION_LEVEL_00`, `vllm/config/vllm.py:254-276`).
#     So: `--enforce-eager` ≈ `-O0`, but not bit-identical.
#   * The KV-cache startup line is a SINGLE merged line, `logger.info_once` in
#     `vllm/v1/core/kv_cache_utils.py:2481-2488`:
#         "GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 8,192
#          tokens per request: 150.70x"
#     "Available KV cache memory: X GiB" comes from `vllm/v1/worker/gpu_worker.py:722`.
#   * `/health` returns 200 when the engine answers and 503 when the engine is dead
#     (`vllm/entrypoints/serve/instrumentator/health.py:22-33`). The API socket binds
#     AFTER the engine is built (`vllm/entrypoints/launchers/api_server/entry.py:120-140`),
#     so "first 200 on /health" is a fair "the server is usable" timestamp.
#   * Process titles are `"$VLLM_PROCESS_NAME_PREFIX::<name>"` with the prefix defaulting
#     to `VLLM` (`vllm/utils/system_utils.py:179-193`, `vllm/envs.py:1900`); the names are
#     `APIServer_<i>` (`vllm/v1/utils.py:551`), `EngineCore` (`vllm/v1/engine/core.py:1370`)
#     and `Worker_<rank>` (`vllm/v1/executor/multiproc_executor.py:912`).
#   * `ignore_eos` is a real request field (`vllm/entrypoints/openai/completion/protocol.py:82`),
#     which is what makes "exactly MAX_TOKENS tokens per request" true.
# ---------------------------------------------------------------------------

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
N="${N:-32}"
CONC="${CONC:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WAIT_SECS="${WAIT_SECS:-900}"
REQ_TIMEOUT="${REQ_TIMEOUT:-300}"
OUTDIR="${OUTDIR:-$VL_ROOT/notes/lab07}"

CONFIGS=(o0 o1 o2 o2-fdo)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
hr() { printf '%s\n' "----------------------------------------------------------------------"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cfg_label() {
  case "$1" in
    o0)     printf '%s' "-O0      no torch.compile, no CUDA graphs" ;;
    o1)     printf '%s' "-O1      torch.compile + PIECEWISE CUDA graphs" ;;
    o2)     printf '%s' "-O2      default: torch.compile + FULL_AND_PIECEWISE CUDA graphs" ;;
    o2-fdo) printf '%s' "-O2      + cudagraph_mode=FULL_DECODE_ONLY (full graphs, decode only)" ;;
    *)      printf '%s' "unknown" ;;
  esac
}

# Sets the global array CFG_ARGS for a configuration key. These are the RAW argv
# elements handed to `vllm serve` — no shell quoting, because there is no shell
# in between once we exec.
CFG_ARGS=()
cfg_args() {
  case "$1" in
    o0)     CFG_ARGS=(-O0) ;;
    o1)     CFG_ARGS=(-O1) ;;
    o2)     CFG_ARGS=(-O2) ;;
    o2-fdo) CFG_ARGS=(-O2 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}') ;;
    *)      die "unknown configuration '$1' (expected one of: ${CONFIGS[*]})" ;;
  esac
}

# The same flags as a string a human can copy-paste. Note the single quotes around
# the JSON: without them your shell eats the double quotes and vLLM gets garbage.
cfg_args_display() {
  case "$1" in
    o2-fdo) printf '%s' "-O2 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'" ;;
    *)      printf '%s' "${CFG_ARGS[*]}" ;;
  esac
}

server_up() { curl -fsS -m 3 "${BASE}/health" >/dev/null 2>&1; }

served_model_id() {
  curl -fsS -m 10 "${BASE}/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "")' 2>/dev/null
}

# metric_sum <name>   sum a counter over all label sets (Prometheus appends _total)
metric_sum() {
  curl -fsS -m 5 "${BASE}/metrics" | awk -v m="$1" '
    /^#/ { next }
    NF < 2 { next }
    { key = $1; sub(/\{.*$/, "", key)
      if (key == m || key == m "_total") s += ($NF + 0) }
    END { printf "%.0f", s }'
}

# metric_max <name>   max a gauge over all label sets
metric_max() {
  curl -fsS -m 5 "${BASE}/metrics" | awk -v m="$1" '
    BEGIN { mx = "" }
    /^#/ { next }
    NF < 2 { next }
    { key = $1; sub(/\{.*$/, "", key)
      if (key == m || key == m "_total") {
        v = $NF + 0
        if (mx == "" || v > mx) mx = v
      } }
    END { if (mx == "") print "n/a"; else printf "%.4f", mx }'
}

# val <file> <key>   read a key=value line written by this lab
val() {
  [ -f "$1" ] || return 0
  awk -F'\t' -v k="$2" '$1 == k { print $2; exit }' "$1"
}

# pctl <file> <percentile>
pctl() {
  sort -n "$1" | awk -v p="$2" '
    { v[NR] = $1 + 0 }
    END {
      if (NR == 0) { print "n/a"; exit }
      r = (p / 100) * (NR - 1); i = int(r) + 1; f = r - int(r)
      if (i + 1 <= NR) printf "%.4f", v[i] + f * (v[i + 1] - v[i])
      else printf "%.4f", v[i]
    }'
}

mean_of() { awk '{ s += $1 + 0 } END { if (NR == 0) print "n/a"; else printf "%.4f", s / NR }' "$1"; }

wait_for_health() {
  local cfg="$1" t0 t elapsed
  t0="$(date +%s)"
  printf 'waiting for %s ' "${BASE}/health"
  while :; do
    if server_up; then
      t="$(date +%s)"
      printf ' up after %ss\n' "$((t - t0))"
      return 0
    fi
    elapsed=$(( $(date +%s) - t0 ))
    if [ "$elapsed" -ge "$WAIT_SECS" ]; then
      printf '\n'
      die "no server on ${BASE} after ${WAIT_SECS}s.
  - Is the server still loading?      tail -f ${OUTDIR}/${cfg}.log
  - Did it crash with CUDA OOM?       UTIL=0.85 bash labs/07_compilation_levels.sh ${cfg}, or a smaller MODEL
  - Is it on a different port?        PORT=8001 bash labs/07_compilation_levels.sh ${cfg}"
    fi
    printf '.'
    sleep 3
  done
}

show_procs() {
  printf '\n-- processes (setproctitle -> "%s::<name>") --\n' "${VLLM_PROCESS_NAME_PREFIX:-VLLM}"
  ps -eo pid,ppid,pcpu,rss,etime,comm,args 2>/dev/null \
    | grep -E 'VLLM::|vllm serve|EngineCore|APIServer' \
    | grep -v grep || printf '  (none found)\n'
  printf '\n-- GPU processes --\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
      || printf '  (nvidia-smi query failed)\n'
  else
    printf '  (nvidia-smi not available)\n'
  fi
  printf '\n  Expected on one GPU at -O2: one `vllm serve` parent, a %s::APIServer process,\n' "${VLLM_PROCESS_NAME_PREFIX:-VLLM}"
  printf '  one %s::EngineCore process, and the worker living inside it\n' "${VLLM_PROCESS_NAME_PREFIX:-VLLM}"
  printf '  (UniProcExecutor). Count them, then compare -O2 against -O0: the COUNT does\n'
  printf '  not change, only the time spent before the socket opens.\n'
}

read_startup_log() {
  local cfg="$1" log="$2"
  printf '\n-- startup log, configuration %s --\n' "$cfg"
  if [ ! -f "$log" ]; then
    printf '  (no log at %s — start the server with the 2>&1 | tee form below)\n' "$log"
    return 0
  fi
  local line
  for pat in 'Available KV cache memory' 'KV cache size:' 'Chunked prefill is enabled' \
             'Cudagraph is disabled under eager mode' 'Enabled custom fusions' \
             'Starting vLLM server on' 'Resolved architecture'; do
    line="$(grep -m1 -F "$pat" "$log" 2>/dev/null || true)"
    if [ -n "$line" ]; then
      printf '  %s\n' "$line"
    fi
  done
}

# ---------------------------------------------------------------------------
# the workload
# ---------------------------------------------------------------------------
# One request per line of latency output. Requests are identical on purpose:
#   * the prompt is fixed, so with prefix caching ON (the default) every request
#     after the first reuses the prompt's blocks and prefill cost collapses;
#   * ignore_eos + temperature 0 make every request generate EXACTLY MAX_TOKENS
#     tokens, so "generation tokens = N * MAX_TOKENS" is arithmetic, not a guess.
PROMPT="Explain, in plain language and without lists, why an inference engine batches decode steps across many independent requests instead of running each request to completion before starting the next one."

drive_load() {
  local latfile="$1" bodyfile="$2"
  : > "$latfile"
  seq 1 "$N" | xargs -P "$CONC" -I{} \
    curl -sS -m "$REQ_TIMEOUT" -o /dev/null -w '%{time_total}\n' \
      -H 'Content-Type: application/json' \
      --data-binary "@${bodyfile}" \
      "${BASE}/v1/completions" >> "$latfile" || true
  # xargs exits non-zero if any invocation fails; an incomplete latency file is
  # worth knowing about, so say so instead of silently reporting a good p50.
  local got
  got="$(wc -l < "$latfile" | tr -d ' ')"
  if [ "$got" -lt "$N" ]; then
    printf '  WARNING: only %s of %s requests produced a timing — check the server log.\n' "$got" "$N"
  fi
}

do_run() {
  local cfg="$1"
  cfg_args "$cfg"

  local result="${OUTDIR}/${cfg}.tsv"
  local log="${OUTDIR}/${cfg}.log"
  local startfile="${OUTDIR}/${cfg}.start"
  local donefile="${OUTDIR}/${cfg}.done"

  mkdir -p "$OUTDIR"

  hr
  printf 'Lab 07 — configuration %s\n' "$cfg"
  hr
  printf '  %s\n\n' "$(cfg_label "$cfg")"

  # (a) the exact command ---------------------------------------------------
  printf 'STEP 1 — the exact server command for this configuration\n'
  hr
  printf 'Ctrl-C any server that is already running, then paste these four lines into\n'
  printf 'THEIR OWN terminal (the first two only set up the run; the third launches):\n\n'
  printf '  cd %s\n' "$VL_ROOT"
  printf '  rm -f %s\n' "$donefile"
  printf '  date +%%s.%%N > %s\n' "$startfile"
  printf '  bash scripts/serve.sh "$MODEL" %s 2>&1 | tee %s\n\n' "$(cfg_args_display "$cfg")" "$log"
  printf '  The `date` writes a start marker immediately before the server launches, so the\n'
  printf '  startup time measured below is wall-clock from the real launch, not from your\n'
  printf '  keypress. The `tee` is how this script can read the KV-cache line back.\n\n'
  printf '  MODEL=%s  PORT=%s  UTIL=%s  MAXLEN=%s\n' "$MODEL" "$PORT" "$UTIL" "$MAXLEN"
  printf '  equivalent long form:  vllm serve %s --host 0.0.0.0 --port %s --max-model-len %s \\\n' \
    "$MODEL" "$PORT" "$MAXLEN"
  printf '      --gpu-memory-utilization %s %s\n\n' "$UTIL" "$(cfg_args_display "$cfg")"

  # already running? --------------------------------------------------------
  if server_up; then
    if [ -f "$startfile" ] && { [ ! -f "$donefile" ] || [ "$startfile" -nt "$donefile" ]; }; then
      printf 'A server is already answering %s/health and its start marker is fresh.\n' "$BASE"
      printf 'Assuming it is the %s server you just started; continuing in 5s (Ctrl-C to abort).\n' "$cfg"
      sleep 5
    else
      die "a server is already running on ${BASE}, but it was NOT started for '$cfg'.
  Restarting the server needs a human: Ctrl-C it, paste the command above, then re-run
      bash labs/07_compilation_levels.sh $cfg
  (Measuring the wrong flags is worse than not measuring at all.)"
    fi
  else
    printf 'STEP 2 — start the server with the command above. This script will wait.\n'
    printf '  log file: %s\n' "$log"
    printf '  timeout : %ss\n' "$WAIT_SECS"
    wait_for_health "$cfg"
  fi

  local up_epoch startup startup_src
  up_epoch="$(date +%s.%N)"
  if [ -f "$startfile" ]; then
    startup="$(awk -v a="$up_epoch" -v b="$(cat "$startfile")" 'BEGIN { printf "%.1f", a - b }')"
    startup_src="marker→/health"
  else
    startup="n/a"
    startup_src="missing start marker"
  fi

  # (b) KV-cache line ------------------------------------------------------
  printf '\nSTEP 3 — the numbers vLLM printed before any traffic\n'
  hr
  read_startup_log "$cfg" "$log"
  local kv_tokens avail_kv max_conc
  kv_tokens="$(grep -m1 -oE 'KV cache size: [0-9,]+ tokens' "$log" 2>/dev/null | grep -oE '[0-9,]+' | tr -d ',' || true)"
  avail_kv="$(grep -m1 -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" 2>/dev/null | grep -oE '[0-9.]+' || true)"
  max_conc="$(grep -m1 -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" 2>/dev/null | grep -oE '[0-9.]+x$' || true)"
  printf '\n  startup wall-clock   : %s s  (%s)\n' "$startup" "$startup_src"
  printf '  Available KV memory  : %s GiB\n' "${avail_kv:-n/a}"
  printf '  GPU KV cache size    : %s tokens\n' "${kv_tokens:-n/a}"
  printf '  Maximum concurrency  : %s\n' "${max_conc:-n/a}"

  # (c) the fixed workload -------------------------------------------------
  printf '\nSTEP 4 — fixed workload: %s requests x %s tokens, concurrency %s\n' "$N" "$MAX_TOKENS" "$CONC"
  hr
  local served body latfile
  served="$(served_model_id)"
  [ -n "$served" ] || die "could not read a model id from ${BASE}/v1/models"
  printf '  served model id: %s\n' "$served"
  body="${OUTDIR}/${cfg}.body.json"
  cat > "$body" <<JSON
{"model": "${served}", "prompt": "${PROMPT}", "max_tokens": ${MAX_TOKENS}, "temperature": 0, "ignore_eos": true}
JSON
  latfile="${OUTDIR}/${cfg}.lat"

  printf '  warmup (1 request, not counted) ...\n'
  curl -sS -m "$REQ_TIMEOUT" -o /dev/null -H 'Content-Type: application/json' \
    --data-binary "@${body}" "${BASE}/v1/completions" || die "warmup request failed"

  local gen0 gen1 p0 p1 wall
  gen0="$(metric_sum vllm:generation_tokens)"
  p0="$(metric_sum vllm:prompt_tokens)"
  local t0 t1
  t0="$(date +%s.%N)"
  drive_load "$latfile" "$body"
  t1="$(date +%s.%N)"
  wall="$(awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.3f", a - b }')"
  gen1="$(metric_sum vllm:generation_tokens)"
  p1="$(metric_sum vllm:prompt_tokens)"

  local d_gen d_prompt thr_nominal thr_measured p50 p99 mean
  d_gen=$(( gen1 - gen0 ))
  d_prompt=$(( p1 - p0 ))
  thr_nominal="$(awk -v n="$N" -v mt="$MAX_TOKENS" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", (n * mt) / w; else print "n/a" }')"
  thr_measured="$(awk -v g="$d_gen" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", g / w; else print "n/a" }')"
  p50="$(pctl "$latfile" 50)"
  p99="$(pctl "$latfile" 99)"
  mean="$(mean_of "$latfile")"

  printf '\n  wall clock            : %s s for %s requests\n' "$wall" "$N"
  printf '  generation tokens     : %s   (nominal %s = %s x %s)\n' "$d_gen" "$(( N * MAX_TOKENS ))" "$N" "$MAX_TOKENS"
  printf '  prefill tokens        : %s   (small: prefix caching absorbed the shared prompt)\n' "$d_prompt"
  printf '  decode throughput     : %s tok/s   (from the metric delta)\n' "$thr_measured"
  printf '  decode throughput     : %s tok/s   (nominal N*max_tokens/wall)\n' "$thr_nominal"
  printf '  latency mean/p50/p99  : %s / %s / %s s\n' "$mean" "$p50" "$p99"

  # (d) processes ----------------------------------------------------------
  printf '\nSTEP 5 — what the process table shows\n'
  show_procs

  # record + table row -----------------------------------------------------
  {
    printf 'cfg\t%s\n'          "$cfg"
    printf 'label\t%s\n'        "$(cfg_label "$cfg")"
    printf 'model\t%s\n'        "$served"
    printf 'startup_s\t%s\n'    "$startup"
    printf 'avail_kv_gib\t%s\n' "${avail_kv:-n/a}"
    printf 'kv_tokens\t%s\n'    "${kv_tokens:-n/a}"
    printf 'max_concurrency\t%s\n' "${max_conc:-n/a}"
    printf 'n\t%s\n'            "$N"
    printf 'conc\t%s\n'         "$CONC"
    printf 'max_tokens\t%s\n'   "$MAX_TOKENS"
    printf 'wall_s\t%s\n'       "$wall"
    printf 'gen_tokens\t%s\n'   "$d_gen"
    printf 'prefill_tokens\t%s\n' "$d_prompt"
    printf 'decode_tok_s\t%s\n' "$thr_measured"
    printf 'decode_tok_s_nominal\t%s\n' "$thr_nominal"
    printf 'lat_mean_s\t%s\n'   "$mean"
    printf 'lat_p50_s\t%s\n'    "$p50"
    printf 'lat_p99_s\t%s\n'    "$p99"
    printf 'cmdline\t%s\n'      "vllm serve ${MODEL} --host 0.0.0.0 --port ${PORT} --max-model-len ${MAXLEN} --gpu-memory-utilization ${UTIL} $(cfg_args_display "$cfg")"
  } > "$result"
  date +%s.%N > "$donefile"

  printf '\nSTEP 6 — recorded %s\n' "$result"
  hr
  printf 'NOW: Ctrl-C the server in its own terminal, then start the next configuration:\n\n'
  local next
  next="$(next_cfg || true)"
  if [ -n "$next" ]; then
    printf '  bash labs/07_compilation_levels.sh %s\n\n' "$next"
  else
    printf '  all four configurations are recorded. Print the table:\n'
    printf '  bash labs/07_compilation_levels.sh table\n\n'
  fi
  print_table
}

# ---------------------------------------------------------------------------
# status / table
# ---------------------------------------------------------------------------
next_cfg() {
  local c
  for c in "${CONFIGS[@]}"; do
    [ -f "${OUTDIR}/${c}.tsv" ] || { printf '%s' "$c"; return 0; }
  done
  return 1
}

print_table() {
  hr
  printf 'SUMMARY — Stage 6, compilation levels (all runs must share model/N/CONC/MAX_TOKENS)\n'
  hr
  printf '%-8s %9s %12s %13s %11s %10s %9s\n' \
    "cfg" "startup s" "KV tokens" "max conc" "decode t/s" "p50 s" "p99 s"
  local c f
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '%-8s %9s %12s %13s %11s %10s %9s\n' \
        "$c" "$(val "$f" startup_s)" "$(val "$f" kv_tokens)" "$(val "$f" max_concurrency)" \
        "$(val "$f" decode_tok_s)" "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
    else
      printf '%-8s %9s %12s %13s %11s %10s %9s\n' "$c" "-" "-" "-" "-" "-" "-"
    fi
  done
  printf '\nMarkdown, ready to paste into notes/:\n\n'
  printf '| config | startup s | KV tokens | max conc | decode tok/s | p50 s | p99 s |\n'
  printf '| --- | --- | --- | --- | --- | --- | --- |\n'
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '| `%s` | %s | %s | %s | %s | %s | %s |\n' \
        "$(val "$f" label)" "$(val "$f" startup_s)" "$(val "$f" kv_tokens)" \
        "$(val "$f" max_concurrency)" "$(val "$f" decode_tok_s)" \
        "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
    fi
  done
  printf '\n'
}

show_status() {
  hr
  printf 'Lab 07 — Stage 6 status   (results in %s)\n' "$OUTDIR"
  hr
  print_table
  printf 'The four configurations, and what each one changes:\n\n'
  local c
  for c in "${CONFIGS[@]}"; do
    printf '  %-8s %s\n' "$c" "$(cfg_label "$c")"
  done
  printf '\n  Reference: %s\n' "--enforce-eager"
  printf '    sets compilation_config.mode=NONE *and* cudagraph_mode=NONE\n'
  printf '    (vllm/config/vllm.py:1689-1691). That is the same two switches -O0 flips,\n'
  printf '    so if you see -O0 numbers you have effectively seen --enforce-eager too.\n'
  printf '    There is NO `--cuda-graph-mode` flag — the graph mode lives inside\n'
  printf '    CompilationConfig. See the header comment for the verified spellings.\n'
  printf '\nRun this in a SECOND terminal, with the server running:\n'
  printf '  nvidia-smi\n'
  printf '  bash labs/07_compilation_levels.sh procs\n\n'
  hr
  if server_up; then
    printf 'A server IS answering %s/health.\n' "$BASE"
  else
    printf 'No server on %s.\n' "$BASE"
  fi
  local n
  if n="$(next_cfg)"; then
    printf 'Next unrecorded configuration: %s\n\n' "$n"
    printf '  bash labs/07_compilation_levels.sh %s\n\n' "$n"
    printf '(it will print the exact `vllm serve` command and then WAIT for you to start it)\n'
  else
    printf 'All four configurations are recorded. Interpret them, then write notes/.\n'
  fi
  printf '\n'
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required (it ships with the venv)"

case "${1:-status}" in
  status|"") show_status ;;
  table)    print_table ;;
  procs)    show_procs ;;
  reset)
    hr
    printf 'This deletes every recorded run under %s\n' "$OUTDIR"
    printf '(logs, latency files, .tsv results, start/done markers). Type "yes" to continue: '
    read -r answer
    [ "$answer" = "yes" ] || die "aborted"
    rm -f "${OUTDIR}"/o0.* "${OUTDIR}"/o1.* "${OUTDIR}"/o2.* "${OUTDIR}"/o2-fdo.*
    printf 'cleared.\n'
    ;;
  o0|o1|o2|o2-fdo) do_run "$1" ;;
  *) die "unknown argument '$1'. Use: ${CONFIGS[*]} | status | table | procs | reset" ;;
esac

# ---------------------------------------------------------------------------
# RECORD: write these into notes/stage-06-compilation.md
#
#   date / vLLM version / torch version / GPU + driver     (always)
#   model, and the four exact command lines                ($OUTDIR/<cfg>.tsv -> cmdline)
#   for EACH configuration:
#     - startup wall-clock, seconds                        -> startup_s
#     - Available KV cache memory, GiB                     -> avail_kv_gib
#     - GPU KV cache size, tokens                          -> kv_tokens
#     - Maximum concurrency for <maxlen> tokens/request    -> max_concurrency
#     - decode throughput, tok/s (metric delta)            -> decode_tok_s
#     - decode throughput, tok/s (nominal)                 -> decode_tok_s_nominal
#     - p50 and p99 end-to-end latency, s                  -> lat_p50_s / lat_p99_s
#     - the process count you saw (`procs`), and whether it changed between configs
#   then:
#     - o2 startup_s  minus  o0 startup_s   = what compilation + graph capture costs you
#     - o2 decode_tok_s  minus  o0 decode_tok_s = what they buy you
#     - the same two differences for o1 and o2-fdo
#     - which configuration had the best throughput per second of startup
#
# All numbers above are PREDICTIONS until you run this on the pod. Nothing in this
# project was executed on a GPU while it was written.
# ---------------------------------------------------------------------------
