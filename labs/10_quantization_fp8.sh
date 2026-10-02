#!/usr/bin/env bash
# Lab 10 — Stage 9: quantization. FP8 weights, FP8 KV cache, and what each one costs.
#
# WHAT IT DEMONSTRATES
#   Four configurations of the SAME model and the SAME workload, differing only in
#   precision:
#
#     bf16    the baseline: bf16 weights, bf16 KV cache
#     wfp8    --quantization fp8              (W8A8: weights AND activations in fp8)
#     kvfp8   --kv-cache-dtype fp8            (KV cache in fp8, weights untouched)
#     both    both flags together
#
#   For each configuration it makes you capture, from the startup log:
#     weight memory ("Model loading took ... GiB memory"), "Available KV cache memory",
#     "GPU KV cache size: N tokens", "Maximum concurrency for M tokens per request";
#   then measures decode throughput and p50 latency on a fixed workload, and saves the
#   generated text to a file so quality can be compared by eye.
#
#   It then prints PREDICTED vs OBSERVED for the KV numbers, using the bytes/token
#   formula from Stage 3. The fp8 KV runs should roughly DOUBLE the cached tokens.
#
# RUN
#   bash labs/10_quantization_fp8.sh                # status + the next command
#   bash labs/10_quantization_fp8.sh bf16           # guided run: prints the command, waits, measures
#   bash labs/10_quantization_fp8.sh wfp8
#   bash labs/10_quantization_fp8.sh kvfp8
#   bash labs/10_quantization_fp8.sh both
#   bash labs/10_quantization_fp8.sh table          # the summary table + predicted vs observed
#   bash labs/10_quantization_fp8.sh quality        # the generated texts, side by side
#   bash labs/10_quantization_fp8.sh reset
#
#   Resumable: each run writes $OUTDIR/<cfg>.tsv. Ctrl-C between configurations is free.
#
# PREREQUISITES
#   * A GPU pod with vLLM installed and a working server setup (docs/03-runpod-setup.md).
#   * Run from the project root:  source scripts/env.sh
#   * ---------------------------------------------------------------
#   * FP8 W8A8 REQUIRES ADA (SM 8.9) OR HOPPER (SM 9.0).
#     It works on the RTX 4090 (Ada). It does NOT work on an A100 or an RTX A6000
#     (both Ampere, SM 8.0 / 8.6). This is a deliberate hardware gate, not a config
#     problem: vLLM's CUTLASS FP8 kernels check
#         cuda_device_capability >= 89  (with CUDA >= 12.4)
#     in `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-158`.
#     On Ampere that check returns false, `CutlassFP8ScaledMMLinearKernel.is_supported()`
#     returns "CUTLASS FP8 kernels not available"
#     (`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:165-172`), and vLLM falls
#     back to an FP8-Marlin weight-only path (`.../scaled_mm/marlin.py:29-45`, whose own
#     docstring says "for GPUs that lack FP8 hardware support"): the model runs, but you
#     get bf16 GEMM throughput with fp8-sized weights, so the W8A8 rows of this lab are
#     MEANINGLESS on Ampere. `--kv-cache-dtype fp8` is a separate knob and is not gated
#     the same way.
#   * Qwen3-30B-A3B (Stage 10's MoE) is the opposite case: it needs `-tp=2` or a 48 GB
#     card, and its fp8 build is ~30 GB.
#
# ENV KNOBS
#   LAB_MODEL=$MODEL_MID    model to quantize (MID by default: big enough for the KV
#                           numbers to be interesting, small enough to fit 24 GB)
#   N=32 / CONC=8 / MAX_TOKENS=128      the fixed workload (ignore_eos -> exact token count)
#   WAIT_SECS=900           how long to wait for /health
#   KV_KIB_BF16=            override the predicted bf16 KV bytes/token (see below)
#   OUTDIR=$VL_ROOT/notes/lab10
#
# ---------------------------------------------------------------------------
# VERIFIED FACTS THIS LAB RELIES ON (vLLM v0.30.0, checked against the source tree)
#
#   * `--quantization` (short `-q`) is a real serve flag (`vllm/engine/arg_utils.py:955`).
#     `--quantization fp8` on a bf16 checkpoint quantizes during loading.
#   * `--kv-cache-dtype` is real (`vllm/engine/arg_utils.py:1337`) and takes the
#     `CacheDType` literal from `vllm/config/cache.py:39-58`, which includes "fp8"
#     (= fp8_e4m3), "fp8_e4m3", "fp8_e5m2", "nvfp4", ... The default is "auto"
#     (`cache_dtype: CacheDType = "auto"`, cache.py:121).
#   * `gpu_memory_utilization` defaults to 0.92 (`vllm/config/cache.py:101`).
#     scripts/serve.sh defaults to 0.90; this lab does the same, so the four runs are
#     comparable to each other (which is the only thing that matters here).
#   * The startup lines this lab reads, and where they come from:
#       "Model loading took %s GiB memory and %.6f seconds"
#            -> vllm/v1/worker/gpu/model_runner.py:407-411 (MRV2, the default)
#       "Available KV cache memory: %s GiB"
#            -> vllm/v1/worker/gpu_worker.py:721-724
#       "GPU KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx"
#            -> vllm/v1/core/kv_cache_utils.py:2481-2488 (ONE merged line, logger.info_once)
#   * The KV bytes/token formula (Stage 3):
#       KV bytes/token = 2 (K and V) x num_layers x num_kv_heads x head_dim x dtype_bytes
#     Shapes from each model's config.json:
#       Qwen3-0.6B      28 layers x 8 kv heads x 128 = 112 KiB bf16 /  56 KiB fp8
#       Qwen3-4B        36 layers x 8 kv heads x 128 = 144 KiB bf16 /  72 KiB fp8
#       Qwen3-8B        36 layers x 8 kv heads x 128 = 144 KiB bf16 /  72 KiB fp8
#       Qwen3-30B-A3B   48 layers x 4 kv heads x 128 =  96 KiB bf16 /  48 KiB fp8
#     (0.6B and 8B read out of the local HF config.json; 4B shares the 8B's shape per the
#     project README; 30B-A3B read from its config.json on the Hub. Note that README.md's
#     ladder table lists 72 KiB for the MoE — the arithmetic above gives 96 KiB. Derive it
#     yourself at Stage 3 and trust the startup log over any table, including that one.)
#   * `--gpu-memory-utilization U` does NOT mean "U of the card is weights". vLLM
#     pre-allocates a KV cache sized to (U x total) minus the profiled activation
#     footprint and the weights; see the Note in $VLLM_SRC/docs/configuration/optimization.md
#     and `vllm/v1/worker/gpu_worker.py:615-700`. The number you actually care about is
#     the one vLLM prints: "Available KV cache memory".
#   * Careful: `$VLLM_SRC/docs/configuration/optimization.md` calls the memory knob
#     `--kv-cache-memory`. The real flag is `--kv-cache-memory-bytes`
#     (`vllm/engine/arg_utils.py:1335`). Docs lag the code; the code wins.
# ---------------------------------------------------------------------------

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-8192}"

LAB_MODEL="${LAB_MODEL:-$MODEL_MID}"
N="${N:-32}"
CONC="${CONC:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
REQ_TIMEOUT="${REQ_TIMEOUT:-300}"
WAIT_SECS="${WAIT_SECS:-900}"
OUTDIR="${OUTDIR:-$VL_ROOT/notes/lab10}"

CONFIGS=(bf16 wfp8 kvfp8 both)

hr() { printf '%s\n' "----------------------------------------------------------------------"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cfg_label() {
  case "$1" in
    bf16)  printf '%s' "baseline: bf16 weights, bf16 KV" ;;
    wfp8)  printf '%s' "--quantization fp8   (W8A8 weights+activations)" ;;
    kvfp8) printf '%s' "--kv-cache-dtype fp8 (KV cache only)" ;;
    both)  printf '%s' "--quantization fp8 + --kv-cache-dtype fp8" ;;
    *)     printf '%s' "unknown" ;;
  esac
}

CFG_ARGS=()
cfg_args() {
  case "$1" in
    bf16)  CFG_ARGS=() ;;
    wfp8)  CFG_ARGS=(--quantization fp8) ;;
    kvfp8) CFG_ARGS=(--kv-cache-dtype fp8) ;;
    both)  CFG_ARGS=(--quantization fp8 --kv-cache-dtype fp8) ;;
    *)     die "unknown configuration '$1' (expected one of: ${CONFIGS[*]})" ;;
  esac
}

cfg_args_display() {
  case "$1" in
    bf16) printf '%s' "" ;;
    *)    printf '%s' "${CFG_ARGS[*]}" ;;
  esac
}

# bf16 KV bytes per token for the model currently selected, from its config.json shape.
kv_bytes_per_token_bf16() {
  if [ -n "${KV_KIB_BF16:-}" ]; then printf '%s' "$KV_KIB_BF16"; return 0; fi
  case "$LAB_MODEL" in
    *Qwen3-0.6B*)  printf '%s' 112 ;;
    *Qwen3-4B*)    printf '%s' 144 ;;
    *Qwen3-8B*)    printf '%s' 144 ;;
    *Qwen3-30B-A3B*) printf '%s' 96 ;;
    *)             printf '%s' "" ;;
  esac
}

# ---------------------------------------------------------------------------
# helpers (same shape as lab 07 — see that file for the reasoning)
# ---------------------------------------------------------------------------
server_up() { curl -fsS -m 3 "${BASE}/health" >/dev/null 2>&1; }

served_model_id() {
  curl -fsS -m 10 "${BASE}/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "")'
}

metric_sum() {
  curl -fsS -m 5 "${BASE}/metrics" | awk -v m="$1" '
    /^#/ { next } NF < 2 { next }
    { key = $1; sub(/\{.*$/, "", key)
      if (key == m || key == m "_total") s += ($NF + 0) }
    END { printf "%.0f", s }'
}

val() {
  [ -f "$1" ] || return 0
  awk -F'\t' -v k="$2" '$1 == k { print $2; exit }' "$1"
}

pctl() {
  sort -n "$1" | awk -v p="$2" '
    { v[NR] = $1 + 0 }
    END { if (NR == 0) { print "n/a"; exit }
          r = (p / 100) * (NR - 1); i = int(r) + 1; f = r - int(r)
          if (i + 1 <= NR) printf "%.4f", v[i] + f * (v[i + 1] - v[i]); else printf "%.4f", v[i] }'
}

mean_of() { awk '{ s += $1 + 0 } END { if (NR == 0) print "n/a"; else printf "%.4f", s / NR }' "$1"; }

wait_for_health() {
  local cfg="$1" t0 t elapsed
  t0="$(date +%s)"
  printf 'waiting for %s/health ' "${BASE}/health"
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
  - Still loading?                 tail -f ${OUTDIR}/${cfg}.log
  - CUDA OOM during weight load?   this is the failure mode of the un-quantized bf16 run
                                   on a 24 GB card: LAB_MODEL=$MODEL_TINY, or lower UTIL.
  - Ampere card + --quantization fp8?  that path has no FP8 hardware (see the header)."
    fi
    printf '.'
    sleep 3
  done
}

read_startup_log() {
  local log="$1"
  printf '\n-- startup log lines that matter --\n'
  if [ ! -f "$log" ]; then
    printf '  (no log at %s — start the server with the 2>&1 | tee form below)\n' "$log"
    return 0
  fi
  local pat line
  for pat in 'Model loading took' 'Available KV cache memory' 'KV cache size:' \
             'Chunked prefill is enabled' 'quantization' 'Quantization' 'Starting vLLM server on'; do
    line="$(grep -m1 -F "$pat" "$log" 2>/dev/null || true)"
    [ -n "$line" ] && printf '  %s\n' "$line"
  done
}

# ---------------------------------------------------------------------------
# the fixed workload (identical in all four configurations)
# ---------------------------------------------------------------------------
PROMPT="Explain, in plain language, why an inference engine stores attention keys and values in a cache, and what changes when that cache is stored at lower precision. Be concrete about the tradeoff."

QUALITY_PROMPT="Answer both parts, numbered. (1) A batch has 17 requests and 3 of them finish; how many remain? (2) In two sentences, explain why storing a KV cache in fp8 instead of bf16 can change a model's answers."

drive_load() {
  local latfile="$1" bodyfile="$2"
  : > "$latfile"
  seq 1 "$N" | xargs -P "$CONC" -I{} \
    curl -sS -m "$REQ_TIMEOUT" -o /dev/null -w '%{time_total}\n' \
      -H 'Content-Type: application/json' \
      --data-binary "@${bodyfile}" "${BASE}/v1/completions" >> "$latfile" || true
  local got
  got="$(wc -l < "$latfile" | tr -d ' ')"
  [ "$got" -eq "$N" ] || printf '  WARNING: only %s of %s requests returned a timing\n' "$got" "$N"
}

do_run() {
  local cfg="$1"
  cfg_args "$cfg"

  local result="${OUTDIR}/${cfg}.tsv" log="${OUTDIR}/${cfg}.log"
  local startfile="${OUTDIR}/${cfg}.start" donefile="${OUTDIR}/${cfg}.done"
  mkdir -p "$OUTDIR"

  hr
  printf 'Lab 10 — configuration %s\n' "$cfg"
  hr
  printf '  %s\n' "$(cfg_label "$cfg")"
  if [ "$cfg" != "bf16" ]; then
    printf '  (Ampere/A100/A6000 check: --kv-cache-dtype fp8 is fine; --quantization fp8 needs\n'
    printf '   Ada or Hopper. See the header of this file.)\n'
  fi
  printf '\n'

  printf 'STEP 1 — the exact server command for this configuration\n'
  hr
  printf 'Ctrl-C any server that is already running, then paste these four lines into\n'
  printf 'THEIR OWN terminal:\n\n'
  printf '  cd %s\n' "$VL_ROOT"
  printf '  rm -f %s\n' "$donefile"
  printf '  date +%%s.%%N > %s\n' "$startfile"
  local extra_disp
  extra_disp="$(cfg_args_display "$cfg")"
  if [ -n "$extra_disp" ]; then
    printf '  MODEL=%s UTIL=%s MAXLEN=%s bash scripts/serve.sh %s %s 2>&1 | tee %s\n\n' \
      "$LAB_MODEL" "$UTIL" "$MAXLEN" "$LAB_MODEL" "$extra_disp" "$log"
  else
    printf '  MODEL=%s UTIL=%s MAXLEN=%s bash scripts/serve.sh %s 2>&1 | tee %s\n\n' \
      "$LAB_MODEL" "$UTIL" "$MAXLEN" "$LAB_MODEL" "$log"
  fi
  printf '  equivalent long form:\n'
  if [ -n "$extra_disp" ]; then
    printf '    vllm serve %s --host 0.0.0.0 --port %s --max-model-len %s --gpu-memory-utilization %s %s\n\n' \
      "$LAB_MODEL" "$PORT" "$MAXLEN" "$UTIL" "$extra_disp"
  else
    printf '    vllm serve %s --host 0.0.0.0 --port %s --max-model-len %s --gpu-memory-utilization %s\n\n' \
      "$LAB_MODEL" "$PORT" "$MAXLEN" "$UTIL"
  fi

  if server_up; then
    if [ -f "$startfile" ] && { [ ! -f "$donefile" ] || [ "$startfile" -nt "$donefile" ]; }; then
      printf 'A server is already answering %s/health and its start marker is fresh.\n' "$BASE"
      printf 'Assuming it is the %s server you just started; continuing in 5s (Ctrl-C to abort).\n' "$cfg"
      sleep 5
    else
      die "a server is already running on ${BASE}, but it was NOT started for '$cfg'.
  Ctrl-C it, paste the command above, then re-run:
      bash labs/10_quantization_fp8.sh $cfg"
    fi
  else
    printf 'STEP 2 — start the server with the command above. This script will wait.\n'
    printf '  log file: %s\n  timeout : %ss\n' "$log" "$WAIT_SECS"
    wait_for_health "$cfg"
  fi

  # ---- whatever the startup log will tell us -----------------------------
  printf '\nSTEP 3 — capacity, read off the startup log BEFORE any traffic\n'
  hr
  read_startup_log "$log"
  local weights_kv avail_kv kv_tokens max_conc
  weights_kv="$(grep -m1 -oE 'Model loading took [0-9.]+ GiB' "$log" 2>/dev/null | grep -oE '[0-9.]+' || true)"
  avail_kv="$(grep -m1 -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" 2>/dev/null | grep -oE '[0-9.]+' || true)"
  kv_tokens="$(grep -m1 -oE 'KV cache size: [0-9,]+ tokens' "$log" 2>/dev/null | grep -oE '[0-9,]+' | tr -d ',' || true)"
  max_conc="$(grep -m1 -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" 2>/dev/null | grep -oE '[0-9.]+x$' || true)"
  printf '\n  weights ("Model loading took")   : %s GiB\n' "${weights_kv:-n/a}"
  printf '  Available KV cache memory        : %s GiB\n' "${avail_kv:-n/a}"
  printf '  GPU KV cache size                : %s tokens\n' "${kv_tokens:-n/a}"
  printf '  Maximum concurrency              : %s\n' "${max_conc:-n/a}"

  # derive bytes/token from what vLLM just told us, and compare with the formula
  local model_kib derived_kib
  model_kib="$(kv_bytes_per_token_bf16)"
  derived_kib=""
  case "${kv_tokens:-}" in
    ''|*[!0-9]*) : ;;                    # missing or not a plain integer -> no derivation
    *) if [ "$kv_tokens" -gt 0 ]; then
         derived_kib="$(awk -v g="$avail_kv" -v t="$kv_tokens" 'BEGIN { printf "%.1f", (g * 1073741824 / t) / 1024 }')"
       fi ;;
  esac
  printf '\n  bytes/token from the formula     : %s\n' "${model_kib:+$model_kib KiB}"
  printf '  bytes/token derived from the log : %s\n' "${derived_kib:+$derived_kib KiB}"
  if [ -n "${derived_kib:-}" ] && [ -n "${model_kib:-}" ]; then
    awk -v a="$derived_kib" -v b="$model_kib" 'BEGIN {
      d = (a - b) / b * 100
      printf "  formula vs engine               : %+.1f%%  %s\n", d,
        (d < 5 && d > -5) ? "(agree — the formula is the model)" : "(check the config shape and max-model-len)"
    }'
  fi

  # ---- the fixed workload ------------------------------------------------
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

  local gen0 gen1 wall t0 t1 dgen thr_measured thr_nominal p50 p99 mean
  gen0="$(metric_sum vllm:generation_tokens)"
  t0="$(date +%s.%N)"
  drive_load "$latfile" "$body"
  t1="$(date +%s.%N)"
  wall="$(awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.3f", a - b }')"
  gen1="$(metric_sum vllm:generation_tokens)"
  dgen=$(( gen1 - gen0 ))
  thr_measured="$(awk -v g="$dgen" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", g / w; else print "n/a" }')"
  thr_nominal="$(awk -v n="$N" -v mt="$MAX_TOKENS" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", (n * mt) / w; else print "n/a" }')"
  p50="$(pctl "$latfile" 50)"
  p99="$(pctl "$latfile" 99)"
  mean="$(mean_of "$latfile")"

  printf '\n  wall clock            : %s s for %s requests\n' "$wall" "$N"
  printf '  generation tokens     : %s   (nominal %s)\n' "$dgen" "$(( N * MAX_TOKENS ))"
  printf '  decode throughput     : %s tok/s   (metric delta)\n' "$thr_measured"
  printf '  decode throughput     : %s tok/s   (nominal)\n' "$thr_nominal"
  printf '  latency mean/p50/p99  : %s / %s / %s s\n' "$mean" "$p50" "$p99"

  # ---- quality probe ------------------------------------------------------
  printf '\nSTEP 5 — quality probe (saved verbatim for eyeball comparison)\n'
  hr
  local qbody qfile
  qbody="${OUTDIR}/${cfg}.quality.body.json"
  qfile="${OUTDIR}/${cfg}.text"
  cat > "$qbody" <<JSON
{"model": "${served}", "prompt": "${QUALITY_PROMPT}", "max_tokens": ${MAX_TOKENS}, "temperature": 0}
JSON
  curl -sS -m "$REQ_TIMEOUT" -H 'Content-Type: application/json' \
    --data-binary "@${qbody}" "${BASE}/v1/completions" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' > "$qfile" \
    || die "quality probe failed"
  printf '  saved to %s (%s chars)\n' "$qfile" "$(wc -c < "$qfile" | tr -d ' ')"
  printf '  ---- first 320 chars ----\n'
  head -c 320 "$qfile" | sed 's/^/  | /'
  printf '\n  -------------------------\n'
  if grep -q '14' "$qfile"; then
    printf '  crude check: the answer "14" appears -> the arithmetic survived.\n'
  else
    printf '  crude check: "14" NOT found. Read the whole file before blaming quantization:\n'
    printf '  with max_tokens=%s a long-winded answer may simply have been truncated.\n' "$MAX_TOKENS"
  fi

  # ---- record -------------------------------------------------------------
  {
    printf 'cfg\t%s\n'              "$cfg"
    printf 'label\t%s\n'            "$(cfg_label "$cfg")"
    printf 'model\t%s\n'            "$served"
    printf 'weights_gib\t%s\n'      "${weights_kv:-n/a}"
    printf 'avail_kv_gib\t%s\n'     "${avail_kv:-n/a}"
    printf 'kv_tokens\t%s\n'        "${kv_tokens:-n/a}"
    printf 'max_concurrency\t%s\n'  "${max_conc:-n/a}"
    printf 'derived_kib_per_token\t%s\n' "${derived_kib:-n/a}"
    printf 'formula_kib_per_token\t%s\n' "${model_kib:-n/a}"
    printf 'n\t%s\n'                "$N"
    printf 'conc\t%s\n'             "$CONC"
    printf 'max_tokens\t%s\n'       "$MAX_TOKENS"
    printf 'wall_s\t%s\n'           "$wall"
    printf 'gen_tokens\t%s\n'       "$dgen"
    printf 'decode_tok_s\t%s\n'     "$thr_measured"
    printf 'decode_tok_s_nominal\t%s\n' "$thr_nominal"
    printf 'lat_mean_s\t%s\n'       "$mean"
    printf 'lat_p50_s\t%s\n'        "$p50"
    printf 'lat_p99_s\t%s\n'        "$p99"
    printf 'text_file\t%s\n'        "$qfile"
    printf 'cmdline\t%s\n'          "vllm serve ${LAB_MODEL} --host 0.0.0.0 --port ${PORT} --max-model-len ${MAXLEN} --gpu-memory-utilization ${UTIL} $(cfg_args_display "$cfg")"
  } > "$result"
  date +%s.%N > "$donefile"

  printf '\nSTEP 6 — recorded %s\n' "$result"
  hr
  print_predictions "$cfg"
  printf 'NOW: Ctrl-C the server, then start the next configuration:\n\n'
  local next
  if next="$(next_cfg)"; then
    printf '  bash labs/10_quantization_fp8.sh %s\n\n' "$next"
  else
    printf '  all four are recorded:\n  bash labs/10_quantization_fp8.sh table\n\n'
  fi
  print_table
}

# ---------------------------------------------------------------------------
# predicted vs observed
# ---------------------------------------------------------------------------
print_predictions() {
  local cfg="$1"
  local base="${OUTDIR}/bf16.tsv"
  hr
  printf 'PREDICTED vs OBSERVED (KV tokens)\n'
  hr
  if [ ! -f "$base" ]; then
    printf '  Run the bf16 baseline first: the predictions are anchored to it.\n'
    printf '      bash labs/10_quantization_fp8.sh bf16\n\n'
    return 0
  fi
  local b_tokens b_avail b_weights b_kib
  b_tokens="$(val "$base" kv_tokens)"
  b_avail="$(val "$base" avail_kv_gib)"
  b_weights="$(val "$base" weights_gib)"
  b_kib="$(val "$base" formula_kib_per_token)"

  if [ "$cfg" = "bf16" ]; then
    printf '  baseline observed: %s tokens in %s GiB of KV cache (%s KiB/token)\n\n' \
      "$b_tokens" "$b_avail" "${b_kib:-?}"
    printf '  Predictions for the other three configurations:\n'
    printf '    wfp8  : the weights halve, so the freed bytes become KV cache.\n'
    printf '            predicted extra KV memory = %s GiB (half of %s GiB of weights)\n' \
      "$(awk -v w="$b_weights" 'BEGIN { printf "%.2f", w / 2 }')" "$b_weights"
    printf '    kvfp8 : same KV memory, HALF the bytes per token -> ~2.00x the tokens.\n'
    printf '    both  : both effects together -> ~2.00x the tokens of wfp8.\n\n'
    return 0
  fi

  local o_tokens o_avail o_weights predicted ratio
  o_tokens="$(val "${OUTDIR}/${cfg}.tsv" kv_tokens)"
  o_avail="$(val "${OUTDIR}/${cfg}.tsv" avail_kv_gib)"
  o_weights="$(val "${OUTDIR}/${cfg}.tsv" weights_gib)"
  predicted=""
  case "$cfg" in
    wfp8)
      predicted="$(awk -v bt="$b_tokens" -v ba="$b_avail" -v bw="$b_weights" -v k="$b_kib" 'BEGIN {
        if (ba <= 0 || k <= 0 || bw <= 0) { print ""; exit }
        freed = bw / 2                                   # fp8 halves the weight bytes
        printf "%.0f", bt * (ba + freed) / ba
      }')"
      ;;
    kvfp8)
      predicted="$(awk -v bt="$b_tokens" 'BEGIN { printf "%.0f", bt * 2 }')"
      ;;
    both)
      local w_pred
      w_pred="$(awk -v bt="$b_tokens" -v ba="$b_avail" -v bw="$b_weights" 'BEGIN {
        if (ba <= 0 || bw <= 0) { print ""; exit }
        printf "%.0f", bt * (ba + bw / 2) / ba
      }')"
      [ -n "$w_pred" ] && predicted="$(awk -v w="$w_pred" 'BEGIN { printf "%.0f", w * 2 }')"
      ;;
  esac

  printf '  baseline bf16        : %s tokens, %s GiB KV, %s GiB weights\n' "$b_tokens" "$b_avail" "$b_weights"
  printf '  observed %-11s : %s tokens, %s GiB KV, %s GiB weights\n' "$cfg" "${o_tokens:-n/a}" "${o_avail:-n/a}" "${o_weights:-n/a}"
  if [ -n "$predicted" ]; then
    ratio="$(awk -v o="${o_tokens:-0}" -v b="$b_tokens" 'BEGIN { if (b > 0) printf "%.2fx", o / b; else print "n/a" }')"
    printf '  predicted            : %s tokens   (observed/baseline = %s)\n' "$predicted" "$ratio"
    awk -v p="$predicted" -v o="${o_tokens:-0}" 'BEGIN {
      if (p <= 0) exit
      d = (o - p) / p * 100
      printf "  predicted vs observed: %+.1f%%  %s\n", d, (d < 10 && d > -10) ? "(within 10% — the formula held)" : "(out by more than 10% — explain why before recording)"
    }'
  else
    printf '  prediction           : unavailable (need the bf16 baseline numbers)\n'
  fi
  printf '\n'
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
  printf 'SUMMARY — Stage 9 quantization (all runs must share LAB_MODEL/N/CONC/MAX_TOKENS/UTIL)\n'
  hr
  printf '%-6s %9s %10s %12s %12s %12s %10s %9s\n' \
    "cfg" "weights" "KV GiB" "KV tokens" "max conc" "decode t/s" "p50 s" "p99 s"
  local c f
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '%-6s %9s %10s %12s %12s %12s %10s %9s\n' \
        "$c" "$(val "$f" weights_gib)" "$(val "$f" avail_kv_gib)" "$(val "$f" kv_tokens)" \
        "$(val "$f" max_concurrency)" "$(val "$f" decode_tok_s)" \
        "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
    else
      printf '%-6s %9s %10s %12s %12s %12s %10s %9s\n' "$c" "-" "-" "-" "-" "-" "-" "-"
    fi
  done
  printf '\nMarkdown, ready to paste into notes/:\n\n'
  printf '| config | weights GiB | KV GiB | KV tokens | max conc | decode tok/s | p50 s | p99 s |\n'
  printf '| --- | --- | --- | --- | --- | --- | --- | --- |\n'
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '| `%s` | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$(val "$f" label)" "$(val "$f" weights_gib)" "$(val "$f" avail_kv_gib)" \
        "$(val "$f" kv_tokens)" "$(val "$f" max_concurrency)" \
        "$(val "$f" decode_tok_s)" "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
    fi
  done
  printf '\n'
  printf 'The KV cache is sized in BYTES, not tokens. --gpu-memory-utilization and the\n'
  printf 'weights decide the bytes; --kv-cache-dtype decides how many tokens fit in them.\n'
  printf 'So compare the "KV tokens" column between bf16 and kvfp8, and expect ~2x.\n\n'
}

show_quality() {
  hr
  printf 'GENERATED TEXT, by configuration (temperature 0, same prompt, same max_tokens)\n'
  hr
  printf 'Prompt: %s\n\n' "$QUALITY_PROMPT"
  local c f
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.text"
    printf '=== %s — %s ===\n' "$c" "$(cfg_label "$c")"
    if [ -f "$f" ]; then
      sed 's/^/  /' "$f"
    else
      printf '  (not recorded yet)\n'
    fi
    printf '\n'
  done
  printf 'Read for: the arithmetic answer, whether the explanation is coherent, and whether\n'
  printf 'either fp8 run degenerates into repetition or loses the thread. One prompt is not a\n'
  printf 'benchmark — but a regression that shows up here would show up in eval too.\n\n'
}

show_status() {
  hr
  printf 'Lab 10 — Stage 9 status   (results in %s)\n' "$OUTDIR"
  hr
  printf 'model: %s   (LAB_MODEL; override with LAB_MODEL=...)\n' "$LAB_MODEL"
  printf 'kv bytes/token (bf16, from the config shape): %s KiB\n\n' "$(kv_bytes_per_token_bf16)"
  print_table
  local c
  printf 'The four configurations:\n'
  for c in "${CONFIGS[@]}"; do printf '  %-6s %s\n' "$c" "$(cfg_label "$c")"; done
  printf '\nFP8 W8A8 needs Ada (SM 8.9) or Hopper (SM 9.0).\n'
  printf '  RTX 4090  -> Ada    -> supported\n'
  printf '  A100/A6000 -> Ampere -> NOT supported (falls back to weight-only FP8 Marlin)\n'
  printf '  verified at csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-158\n\n'
  if server_up; then
    printf 'A server IS answering %s/health (model %s).\n' "$BASE" "$(served_model_id || echo '?')"
  else
    printf 'No server on %s.\n' "$BASE"
  fi
  local n
  if n="$(next_cfg)"; then
    printf 'Next unrecorded configuration: %s\n\n' "$n"
    printf '  bash labs/10_quantization_fp8.sh %s\n\n' "$n"
  else
    printf 'All four recorded. Then:\n'
    printf '  bash labs/10_quantization_fp8.sh quality\n'
    printf '  bash labs/10_quantization_fp8.sh table\n\n'
  fi
}

case "${1:-status}" in
  status|"") show_status ;;
  table)     print_table ;;
  quality)   show_quality ;;
  reset)
    hr
    printf 'This deletes every recorded run under %s. Type "yes" to continue: ' "$OUTDIR"
    read -r answer
    [ "$answer" = "yes" ] || die "aborted"
    rm -f "${OUTDIR}"/bf16.* "${OUTDIR}"/wfp8.* "${OUTDIR}"/kvfp8.* "${OUTDIR}"/both.*
    printf 'cleared.\n'
    ;;
  bf16|wfp8|kvfp8|both) do_run "$1" ;;
  *) die "unknown argument '$1'. Use: ${CONFIGS[*]} | status | table | quality | reset" ;;
esac

# ---------------------------------------------------------------------------
# RECORD: write these into notes/stage-09-quantization.md
#
#   date / vLLM version / torch version / GPU + driver / CUDA version
#   LAB_MODEL, --max-model-len, --gpu-memory-utilization, and the four exact commands
#
#   For EACH of the four configurations:
#     - "Model loading took X GiB memory"      -> weights_gib
#     - "Available KV cache memory: X GiB"     -> avail_kv_gib
#     - "GPU KV cache size: N tokens"          -> kv_tokens
#     - "Maximum concurrency for M tokens per request: Kx"  -> max_concurrency
#     - bytes/token derived from the log (KV GiB / tokens)  -> derived_kib_per_token
#     - decode throughput, tok/s                -> decode_tok_s
#     - p50 and p99 latency, s                  -> lat_p50_s / lat_p99_s
#     - the generated text for the quality probe (paste it, or say "see notes/lab10/<cfg>.text")
#   Then:
#     - kvfp8 observed KV tokens / bf16 observed KV tokens  (expect ~2.00x)
#     - both  observed KV tokens / bf16 observed KV tokens  (expect > 2x: the weights shrank too)
#     - wfp8  observed weights / bf16 observed weights      (expect ~0.5x)
#     - the delta between derived_kib_per_token and the formula value, per configuration
#     - whether the two fp8 runs changed the arithmetic answer or the coherence of the text
#     - which configuration you would deploy for a latency SLO, and which for throughput
#
# Every number above is a prediction until you run it. Nothing in this project was executed
# on a GPU while it was written.
# ---------------------------------------------------------------------------
