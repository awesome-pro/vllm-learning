#!/usr/bin/env bash
# Lab 11 — Stage 10: tensor parallelism, and the model that does not fit.
#
# WHAT IT DEMONSTRATES
#   Part A — sharding on 2 x 24 GB
#     Serve $MODEL_BIG twice: once with -tp=1 (the 24 GB baseline) and once with -tp=2.
#     Read per-GPU memory from nvidia-smi before and during load, confirm the WEIGHTS
#     roughly halve per GPU, and compare throughput. Then work out why the TOTAL memory
#     is NOT exactly halved: each rank keeps its own activations, its own CUDA-graph and
#     communication buffers, and its own KV cache allocation.
#
#   Part B — a model that genuinely needs more than one card
#     Serve the FP8 MoE $MODEL_MOE with -tp=2 and watch it come up; then (optionally) try
#     -tp=1 and watch it fail, because ~30 GB of FP8 weights do not fit in a 24 GB card's
#     budget. The exact commands for the 1 x 48 GB alternative are printed by `moe-48g`.
#
# RUN
#   bash labs/11_tensor_parallel.sh              # status: GPU inventory, next command
#   bash labs/11_tensor_parallel.sh tp1          # guided run: 1 GPU, $MODEL_BIG
#   bash labs/11_tensor_parallel.sh tp2          # guided run: 2 GPUs, -tp=2
#   bash labs/11_tensor_parallel.sh moe-tp2      # Part B: FP8 MoE on 2 GPUs
#   bash labs/11_tensor_parallel.sh moe-tp1      # Part B: the deliberate OOM (non-fatal)
#   bash labs/11_tensor_parallel.sh moe-48g      # the 1 x 48 GB single-card commands
#   bash labs/11_tensor_parallel.sh table|gpu|reset
#
# PREREQUISITES
#   * A pod with TWO visible GPUs for `tp2` / `moe-tp2` / `moe-tp1` (2 x RTX 4090 24 GB is
#     the cheap option). `tp1` needs one GPU. The script checks this and stops with an
#     actionable message rather than starting something that cannot work.
#   * Run from the project root:  source scripts/env.sh
#   * `nvidia-smi`, `curl`, `python3` on PATH.
#
# COST — read this before you deploy
#   A 2-GPU pod costs TWICE the hourly rate of a 1-GPU pod, and RunPod gives you both
#   cards in one container (docs/03-runpod-setup.md §9). At the Community rates recorded
#   in that doc: 2 x RTX 4090 = $0.68/hr vs 1 x RTX 4090 = $0.34/hr; 1 x RTX 6000 Ada
#   48 GB = $0.74/hr; 1 x RTX A6000 48 GB = $0.33/hr (Ampere — no FP8 W8A8).
#   Two hours of this stage is ~$1.40 on 2 x 4090 and ~$0.66 on a single A6000. Prices
#   move; re-check before you deploy. Stop the pod when you are done.
#
# ENV KNOBS
#   N=32 / CONC=8 / MAX_TOKENS=128   the fixed workload (ignore_eos -> exact token count)
#   UTIL=0.90 / MAXLEN=8192          passed straight to scripts/serve.sh
#   WAIT_SECS=1800                   model download + load can be slow the first time
#   OUTDIR=$VL_ROOT/notes/lab11
#
# ---------------------------------------------------------------------------
# VERIFIED FACTS THIS LAB RELIES ON (vLLM v0.30.0, checked against the source tree)
#
#   * `--tensor-parallel-size` / `-tp` is real (`vllm/engine/arg_utils.py:1154`);
#     `--pipeline-parallel-size` is the neighbouring flag (line 1116).
#   * `--quantization` (short `-q`) is real (`vllm/engine/arg_utils.py:955`).
#     In v0.30.0 `--quantization fp8` on a bf16 checkpoint is a DEPRECATED ALIAS for
#     online per-tensor FP8: the loader logs
#         "--quantization fp8 is deprecated for online quantization;
#          use --quantization fp8_per_tensor instead."
#     and resolves to `_ONLINE_SHORTHANDS["fp8_per_tensor"]`
#     (`vllm/model_executor/model_loader/weight_utils.py:341-349`,
#      `vllm/config/quantization.py:187-227`). Both spellings work; this lab uses `fp8`
#     because the curriculum does, and greps the warning out of the log so you see it.
#   * FP8 W8A8 needs Ada (SM 8.9) or Hopper (SM 9.0): the per-tensor CUTLASS FP8 kernels
#     require `cuda_device_capability >= 89` with CUDA >= 12.4
#     (`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-158`), and
#     the per-tensor online path uses exactly those (`kFp8StaticTensorSym` /
#     `kFp8DynamicTokenSym`, `vllm/model_executor/layers/quantization/online/fp8.py:163-180`).
#     A 4090 is Ada, so this works. An A100 (SM 8.0) or A6000 (SM 8.6) is Ampere: the
#     check fails, `CutlassFP8ScaledMMLinearKernel.is_supported()` returns
#     "CUTLASS FP8 kernels not available"
#     (`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:165-172`), and vLLM falls
#     back to FP8-Marlin, which is weight-only (`.../scaled_mm/marlin.py:29-45`, docstring:
#     "for GPUs that lack FP8 hardware support"). You get the memory win, not the compute win.
#   * BLOCK-wise FP8 is a different gate again: `cutlass_scaled_mm_supports_block_fp8`
#     requires SM >= 90 (`.../cutlass/scaled_mm_entry.cu:161-173`). `Qwen/Qwen3-30B-A3B-FP8`
#     is block-wise (`weight_block_size: [128, 128]` in its config.json), so on Ada it runs
#     through the Triton block-scaled kernel instead
#     (`vllm/model_executor/kernels/linear/scaled_mm/triton.py:159-164` — supported on any
#     CUDA-alike device). It works; it is just not the fastest path. Use the bf16 checkpoint
#     plus `--quantization fp8` when you are on Ada and care about speed.
#   * `gpu_memory_utilization` is PER RANK: "This is a per-instance limit, and only applies
#     to the current vLLM instance" (`vllm/config/cache.py:101-110`). With `-tp=2` each of
#     the two processes gets its own `UTIL x total_memory` budget.
#   * The startup lines this lab reads:
#       "Model loading took %s GiB memory and %.6f seconds"
#            -> vllm/v1/worker/gpu/model_runner.py:407-411 — printed ONCE PER RANK
#       "Available KV cache memory: %s GiB"                -> vllm/v1/worker/gpu_worker.py:721-724
#       "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Kx"
#            -> vllm/v1/core/kv_cache_utils.py:2481-2488 — N is the ENGINE-WIDE token
#               capacity, because TP shards the KV heads across ranks.
#   * When it does not fit, you get one of exactly two messages:
#       "No available memory for the cache blocks."  -> vllm/v1/core/kv_cache_utils.py:922-928
#       a "Free memory on device (X/Y GiB) on startup. Desired GPU memory utilization is ..."
#       INFO line -> vllm/v1/worker/gpu_worker.py:967-990, which helpfully suggests
#       `--kv-cache-memory=<N>`. THAT FLAG DOES NOT EXIST: the real one is
#       `--kv-cache-memory-bytes` (`vllm/engine/arg_utils.py:1335`). Even vLLM's own
#       log lines can name a knob that is not real; check the source.
#   * Qwen3-8B bf16 is ~15.3 GB of weights and 144 KiB/token of KV; Qwen3-30B-A3B is
#     ~61.1 GB bf16 / ~30.6 GB at fp8 and 96 KiB/token of KV (bf16) / 48 KiB (fp8).
#     Both repos are non-gated apache-2.0 (checked against the HuggingFace API).
#     `Qwen/Qwen3-30B-A3B-FP8` is 32.4 GB on disk and also non-gated.
#
# PREDICTED (arithmetic, not measurement — 2 x 24 GB, UTIL=0.90 -> 21.6 GB budget/rank):
#   tp1, Qwen3-8B :  weights 15.3 GB, activations ~1.5 GB -> ~4.8 GB KV
#                    -> 4.8 GiB / 144 KiB ~= 34,900 tokens
#                    (scripts/serve.sh quotes 34,912 tokens / 4.26x as its example line)
#   tp2, Qwen3-8B :  weights ~7.65 GB/rank, activations ~1.5 GB/rank -> ~12.4 GB KV/rank
#                    AND the bytes/token per rank halve to 72 KiB (4 of 8 KV heads)
#                    -> 12.4 GiB / 72 KiB ~= 181,000 tokens
#                    So the prediction is roughly 5x, NOT 2x. If you expected 2x, that is
#                    exactly the misconception this stage exists to remove.
#   tp2, MoE fp8  :  weights ~15.3 GB/rank -> ~4.5 GB KV/rank, 48 KiB/token/rank
#                    -> ~98,000 tokens. Tight but real.
#   tp1, MoE fp8  :  ~30.6 GB of weights against a 21.6 GB budget -> must fail.
#   Throughput    : on PCIe with no NVLink, expect TP=2 to be NEAR-or-BELOW TP=1 for an
#                    8B model, because every layer now ends in an all-reduce. TP buys
#                    capacity and lower latency per token under load, not free tok/s.
# ---------------------------------------------------------------------------

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-8192}"
N="${N:-32}"
CONC="${CONC:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
REQ_TIMEOUT="${REQ_TIMEOUT:-600}"
WAIT_SECS="${WAIT_SECS:-1800}"
OUTDIR="${OUTDIR:-$VL_ROOT/notes/lab11}"

DENSE_MODEL="${DENSE_MODEL:-$MODEL_BIG}"
MOE_MODEL="${MOE_MODEL:-$MODEL_MOE}"

CONFIGS=(tp1 tp2 moe-tp2)

hr() { printf '%s\n' "----------------------------------------------------------------------"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cfg_label() {
  case "$1" in
    tp1)     printf '%s' "$DENSE_MODEL with -tp=1 (1 GPU, the 24 GB baseline)" ;;
    tp2)     printf '%s' "$DENSE_MODEL with -tp=2 (weights sharded across 2 GPUs)" ;;
    moe-tp2) printf '%s' "$MOE_MODEL with --quantization fp8 -tp=2" ;;
    *)       printf '%s' "unknown" ;;
  esac
}

CFG_ARGS=()
CFG_MODEL=""
cfg_args() {
  case "$1" in
    tp1)     CFG_MODEL="$DENSE_MODEL"; CFG_ARGS=(-tp 1) ;;
    tp2)     CFG_MODEL="$DENSE_MODEL"; CFG_ARGS=(-tp 2) ;;
    moe-tp2) CFG_MODEL="$MOE_MODEL";   CFG_ARGS=(--quantization fp8 -tp 2) ;;
    moe-tp1) CFG_MODEL="$MOE_MODEL";   CFG_ARGS=(--quantization fp8 -tp 1) ;;
    *)       die "unknown configuration '$1'" ;;
  esac
}

gpus_needed() {
  case "$1" in
    tp1)     printf '1' ;;
    *)       printf '2' ;;
  esac
}

# ---------------------------------------------------------------------------
# GPU plumbing
# ---------------------------------------------------------------------------
have_nvidia_smi() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

gpu_count() {
  have_nvidia_smi || { printf '0'; return 0; }
  local n
  n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c . || true)"
  printf '%s' "${n:-0}"
}

require_gpus() {
  local need="$1" cfg="$2" have why
  have="$(gpu_count)"
  have="${have:-0}"
  if [ "$have" -lt "$need" ]; then
    if [ "$have" -eq 0 ]; then
      why="There is no GPU visible at all. Are you actually on the pod, or still on your laptop?"
    else
      why="RunPod gives you both cards in ONE container, but only if you DEPLOY a 2-GPU pod.
  You cannot add a second GPU to a running pod: stop it and redeploy with GPU count = 2
  (docs/03-runpod-setup.md section 9). The 2-GPU rate is 2x the 1-GPU rate."
    fi
    die "configuration '$cfg' needs $need visible GPU(s); this machine reports $have.
  $why

  The single-GPU parts of this lab still work:  bash labs/11_tensor_parallel.sh tp1
  For the 48 GB one-card alternative, see:      bash labs/11_tensor_parallel.sh moe-48g"
  fi
}

# One line per GPU: "index used_mib total_mib"
gpu_snapshot() {
  have_nvidia_smi || return 0
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{ gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); printf "%s %s %s\n", $1, $2, $3 }'
}

show_gpu() {
  printf '\n-- GPU inventory --\n'
  if ! have_nvidia_smi; then
    printf '  nvidia-smi is not available. This lab needs a real GPU.\n'
    return 0
  fi
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
             --format=csv 2>/dev/null | sed 's/^/  /'
  printf '  GPUs visible: %s\n' "$(gpu_count)"
  printf '\n-- per-process GPU memory --\n'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
    | sed 's/^/  /' || printf '  (query failed)\n'
}

gpu_mem_total_used() {
  # $1 = snapshot file -> total MiB used across all rows
  awk '{ s += $2 } END { printf "%.4g", s }' "$1"
}

# Samples per-GPU used memory into a file until the stop file appears.
gpu_sampler() {
  local out="$1" stop="$2"
  while [ ! -f "$stop" ]; do
    gpu_snapshot | awk -v ts="$(date +%s.%N)" '{ printf "%s %s %s\n", ts, $1, $2 }' >> "$out"
    sleep 0.5
  done
}

gpu_peak() {
  # $1 = samples file, $2 = gpu index
  [ -s "$1" ] || { printf 'n/a'; return 0; }
  awk -v g="$2" '{ if ($2 == g) { if (!seen || $3 > m) { m = $3; seen = 1 } } }
       END { if (seen) printf "%.0f", m; else print "n/a" }' "$1"
}

# ---------------------------------------------------------------------------
# generic helpers (same shape as labs 07 and 10)
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

val() { [ -f "$1" ] || return 0; awk -F'\t' -v k="$2" '$1 == k { print $2; exit }' "$1"; }

pctl() {
  sort -n "$1" | awk -v p="$2" '{ v[NR] = $1 + 0 }
    END { if (NR == 0) { print "n/a"; exit }
          r = (p / 100) * (NR - 1); i = int(r) + 1; f = r - int(r)
          if (i + 1 <= NR) printf "%.4f", v[i] + f * (v[i + 1] - v[i]); else printf "%.4f", v[i] }'
}

wait_for_health() {
  local cfg="$1" log="$2" t0 t elapsed
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
      printf '\n-- last 25 log lines --\n'
      tail -n 25 "$log" 2>/dev/null | sed 's/^/  /' || true
      die "no server on ${BASE} after ${WAIT_SECS}s.
  The whole point of Part B is that some of these runs are SUPPOSED to fail.
  Read the log tail above and match it against:
    'No available memory for the cache blocks.'   -> vllm/v1/core/kv_cache_utils.py:922
    'Free memory on device ... on startup.'       -> vllm/v1/worker/gpu_worker.py:967
    'CUDA out of memory'                          -> the weights did not fit at all
  Full log: $log"
    fi
    printf '.'
    sleep 5
  done
}

read_startup_log() {
  local log="$1"
  printf '\n-- startup log lines that matter --\n'
  if [ ! -f "$log" ]; then
    printf '  (no log at %s — start the server with the 2>&1 | tee form below)\n' "$log"
    return 0
  fi
  local pat line n
  # "Model loading took" is printed once PER RANK — count them, that is the proof of sharding.
  n="$(grep -c -F 'Model loading took' "$log" 2>/dev/null || true)"
  printf '  "Model loading took" lines: %s  (one per rank)\n' "${n:-0}"
  while IFS= read -r line; do printf '  %s\n' "$line"; done < <(
    grep -E -m2 -F 'Model loading took' "$log" 2>/dev/null || true
  )
  for pat in 'Available KV cache memory' 'KV cache size:' 'deprecated for online quantization' \
             'Starting vLLM server on'; do
    line="$(grep -m1 -F "$pat" "$log" 2>/dev/null || true)"
    [ -n "$line" ] && printf '  %s\n' "$line"
  done
}

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

PROMPT="Explain what tensor parallelism does to a transformer layer, and why the memory saving per GPU is not exactly the reciprocal of the tensor-parallel size. Be concrete."

# ---------------------------------------------------------------------------
# the guided run
# ---------------------------------------------------------------------------
do_run() {
  local cfg="$1"
  cfg_args "$cfg"
  local need
  need="$(gpus_needed "$cfg")"
  require_gpus "$need" "$cfg"

  local result="${OUTDIR}/${cfg}.tsv" log="${OUTDIR}/${cfg}.log"
  local startfile="${OUTDIR}/${cfg}.start" donefile="${OUTDIR}/${cfg}.done"
  mkdir -p "$OUTDIR"

  hr
  printf 'Lab 11 — configuration %s\n' "$cfg"
  hr
  printf '  %s\n' "$(cfg_label "$cfg")"
  printf '  GPUs required: %s   visible: %s\n\n' "$need" "$(gpu_count)"

  printf 'STEP 1 — the exact server command\n'
  hr
  printf 'Ctrl-C any server that is already running, then paste these four lines into\n'
  printf 'THEIR OWN terminal:\n\n'
  printf '  cd %s\n' "$VL_ROOT"
  printf '  rm -f %s\n' "$donefile"
  printf '  date +%%s.%%N > %s\n' "$startfile"
  printf '  MODEL=%s UTIL=%s MAXLEN=%s bash scripts/serve.sh %s %s 2>&1 | tee %s\n\n' \
    "$CFG_MODEL" "$UTIL" "$MAXLEN" "$CFG_MODEL" "${CFG_ARGS[*]}" "$log"
  printf '  equivalent long form:\n'
  printf '    vllm serve %s --host 0.0.0.0 --port %s --max-model-len %s --gpu-memory-utilization %s %s\n\n' \
    "$CFG_MODEL" "$PORT" "$MAXLEN" "$UTIL" "${CFG_ARGS[*]}"
  if [ "$cfg" = "moe-tp1" ]; then
    printf '  NOTE: this run is EXPECTED TO FAIL. That is the experiment. Let it fail, then\n'
    printf '  re-run this subcommand — it will read the failure out of the log for you.\n\n'
  fi

  # ---- baseline GPU memory BEFORE the server is up ----------------------
  local prefile="${OUTDIR}/${cfg}.gpu.before"
  gpu_snapshot > "$prefile"
  printf '%s\n' '-- GPU memory before the server loads --'
  sed 's/^/  index used_MiB total_MiB: /' "$prefile" 2>/dev/null | head -8 || true
  printf '\n'

  if server_up; then
    if [ -f "$startfile" ] && { [ ! -f "$donefile" ] || [ "$startfile" -nt "$donefile" ]; }; then
      printf 'A server is already answering %s/health and its start marker is fresh.\n' "$BASE"
      printf 'Assuming it is the %s server you just started; continuing in 5s (Ctrl-C to abort).\n' "$cfg"
      sleep 5
    else
      die "a server is already running on ${BASE}, but it was NOT started for '$cfg'.
  Ctrl-C it, paste the command above, then re-run:
      bash labs/11_tensor_parallel.sh $cfg"
    fi
  else
    printf 'STEP 2 — start the server. This script waits up to %ss (first load downloads weights).\n' "$WAIT_SECS"
    if ! wait_for_health "$cfg" "$log"; then
      # wait_for_health already died; this is unreachable, kept for clarity
      :
    fi
  fi

  # ---- post-load memory: this is where the weights are visible ----------
  local idlefile="${OUTDIR}/${cfg}.gpu.idle"
  sleep 3   # let the profiler settle
  gpu_snapshot > "$idlefile"
  local weights_lines weights_val avail_kv kv_tokens max_conc
  weights_lines="$(grep -c -F 'Model loading took' "$log" 2>/dev/null || true)"
  weights_val="$(grep -m1 -oE 'Model loading took [0-9.]+ GiB' "$log" 2>/dev/null | grep -oE '[0-9.]+' || true)"
  avail_kv="$(grep -m1 -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" 2>/dev/null | grep -oE '[0-9.]+' || true)"
  kv_tokens="$(grep -m1 -oE 'KV cache size: [0-9,]+ tokens' "$log" 2>/dev/null | grep -oE '[0-9,]+' | tr -d ',' || true)"
  max_conc="$(grep -m1 -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" 2>/dev/null | grep -oE '[0-9.]+x$' || true)"

  printf '\nSTEP 3 — what the engine says about itself\n'
  hr
  read_startup_log "$log"
  printf '\n  per-GPU memory AFTER load (nvidia-smi, MiB used):\n'
  sed 's/^/    gpu /' "$idlefile" 2>/dev/null || true
  printf '  total across GPUs: %s MiB\n' "$(gpu_mem_total_used "$idlefile")"
  printf '\n  weights per rank ("Model loading took") : %s GiB   (%s line(s))\n' "${weights_val:-n/a}" "${weights_lines:-0}"
  printf '  Available KV cache memory (per rank)    : %s GiB\n' "${avail_kv:-n/a}"
  printf '  GPU KV cache size (ENGINE-WIDE tokens)  : %s\n' "${kv_tokens:-n/a}"
  printf '  Maximum concurrency                     : %s\n' "${max_conc:-n/a}"

  # sharding check against the tp1 baseline
  local base="${OUTDIR}/tp1.tsv"
  if [ "$cfg" != "tp1" ] && [ -f "$base" ]; then
    local b_w b_tok b_gpu0
    b_w="$(val "$base" weights_gib)"; b_tok="$(val "$base" kv_tokens)"
    b_gpu0="$(val "$base" gpu0_idle_mib)"
    printf '\n-- Part A sharding check, against the tp1 baseline --\n'
    awk -v bw="$b_w" -v w="${weights_val:-0}" 'BEGIN {
      if (bw <= 0 || w <= 0) { print "  (no usable weight numbers on one side)"; exit }
      printf "  weights per rank   : tp1 %s GiB -> tp2 %s GiB   (ratio %.2fx)\n", bw, w, w / bw
      printf "  %s\n", (w / bw > 0.40 && w / bw < 0.60) ? "  => ~0.5x: the weights ARE sharded." : "  => NOT ~0.5x: read the note below before concluding anything."
    }'
    printf '  gpu0 idle memory   : tp1 %s MiB -> tp2 %s MiB\n' "${b_gpu0:-n/a}" "$(awk '{ if ($1 == 0) print $2 }' "$idlefile")"
    awk -v bt="$b_tok" -v t="${kv_tokens:-0}" 'BEGIN {
      if (bt <= 0 || t <= 0) exit
      printf "  engine KV tokens   : tp1 %s -> tp2 %s   (%.2fx)\n", bt, t, t / bt
      printf "  Note: the prediction in this file is ~5x, not 2x — TP halves the per-rank\n"
      printf "  WEIGHTS (freeing memory for KV) AND halves the bytes/token per rank (KV heads\n"
      printf "  are sharded too). Two effects multiply; that is why the KV capacity grows faster\n"
      printf "  than the GPU count.\n"
    }'
    printf '\n  Why the TOTAL is still not exactly halved:\n'
    printf '    - embeddings, norms, biases and the sampler are replicated, not sharded, so\n'
    printf '      weights/rank is a little MORE than W/2 (measure it: the ratio above);\n'
    printf '    - activations, CUDA-graph capture buffers and NCCL all-reduce buffers are\n'
    printf '      allocated PER RANK and do not shrink when TP grows;\n'
    printf '    - every rank runs its own CUDA context, cuBLAS/cuDNN workspaces and KV\n'
    printf '      cache allocation, so there is a fixed per-process overhead you pay twice;\n'
    printf '    - gpu_memory_utilization is a PER-RANK limit (vllm/config/cache.py:101-110),\n'
    printf '      so 2 GPUs at 0.90 give you 2 x 21.6 GB of BUDGET, not 43.2 GB of usable KV.\n'
  fi

  # ---- the workload, with per-GPU peak sampling -------------------------
  printf '\nSTEP 4 — fixed workload: %s requests x %s tokens, concurrency %s\n' "$N" "$MAX_TOKENS" "$CONC"
  hr
  local served body latfile samples stopfile sampler wall t0 t1 dgen g0 g1
  served="$(served_model_id)"
  [ -n "$served" ] || die "could not read a model id from ${BASE}/v1/models"
  body="${OUTDIR}/${cfg}.body.json"
  latfile="${OUTDIR}/${cfg}.lat"
  samples="${OUTDIR}/${cfg}.gpu.samples"
  stopfile="${OUTDIR}/${cfg}.gpu.stop"
  cat > "$body" <<JSON
{"model": "${served}", "prompt": "${PROMPT}", "max_tokens": ${MAX_TOKENS}, "temperature": 0, "ignore_eos": true}
JSON

  printf '  warmup ...\n'
  curl -sS -m "$REQ_TIMEOUT" -o /dev/null -H 'Content-Type: application/json' \
    --data-binary "@${body}" "${BASE}/v1/completions" || die "warmup failed"

  rm -f "$stopfile"; : > "$samples"
  gpu_sampler "$samples" "$stopfile" &
  sampler=$!
  g0="$(metric_sum vllm:generation_tokens)"
  t0="$(date +%s.%N)"
  drive_load "$latfile" "$body"
  t1="$(date +%s.%N)"
  g1="$(metric_sum vllm:generation_tokens)"
  touch "$stopfile"; wait "$sampler" 2>/dev/null || true

  wall="$(awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.3f", a - b }')"
  dgen=$(( g1 - g0 ))
  local thr p50 p99 mean
  thr="$(awk -v g="$dgen" -v w="$wall" 'BEGIN { if (w > 0) printf "%.1f", g / w; else print "n/a" }')"
  p50="$(pctl "$latfile" 50)"; p99="$(pctl "$latfile" 99)"; mean="$(awk '{s+=$1+0} END { if (NR) printf "%.4f", s/NR; else print "n/a" }' "$latfile")"

  printf '\n  wall clock            : %s s for %s requests\n' "$wall" "$N"
  printf '  generation tokens     : %s   (nominal %s)\n' "$dgen" "$(( N * MAX_TOKENS ))"
  printf '  decode throughput     : %s tok/s\n' "$thr"
  printf '  latency mean/p50/p99  : %s / %s / %s s\n' "$mean" "$p50" "$p99"
  printf '\n  per-GPU PEAK memory during the load (MiB):\n'
  local idx
  for idx in $(seq 0 $(( need - 1 ))); do
    printf '    gpu %s peak: %s MiB\n' "$idx" "$(gpu_peak "$samples" "$idx")"
  done
  printf '  sum of per-GPU sampled peaks: %s MiB (samples are not simultaneous; treat as\n' "$(gpu_peak_sum "$samples")"
  printf '  an upper bound, not an exact total)\n'

  # ---- record -----------------------------------------------------------
  local TP_VALUE
  TP_VALUE="$(printf '%s' "${CFG_ARGS[*]}" | grep -oE -- '-tp [0-9]+' | awk '{print $2}' || true)"
  {
    printf 'cfg\t%s\n'             "$cfg"
    printf 'label\t%s\n'           "$(cfg_label "$cfg")"
    printf 'model\t%s\n'           "$CFG_MODEL"
    printf 'tp\t%s\n'              "$TP_VALUE"
    printf 'weights_gib\t%s\n'     "${weights_val:-n/a}"
    printf 'weights_lines\t%s\n'   "${weights_lines:-0}"
    printf 'avail_kv_gib\t%s\n'    "${avail_kv:-n/a}"
    printf 'kv_tokens\t%s\n'       "${kv_tokens:-n/a}"
    printf 'max_concurrency\t%s\n' "${max_conc:-n/a}"
    printf 'gpu0_idle_mib\t%s\n'   "$(awk '{ if ($1 == 0) print $2 }' "$idlefile")"
    printf 'gpu1_idle_mib\t%s\n'   "$(awk '{ if ($1 == 1) print $2 }' "$idlefile")"
    printf 'gpu0_peak_mib\t%s\n'   "$(gpu_peak "$samples" 0)"
    printf 'gpu1_peak_mib\t%s\n'   "$(gpu_peak "$samples" 1)"
    printf 'gpus_visible\t%s\n'    "$(gpu_count)"
    printf 'n\t%s\n'               "$N"
    printf 'conc\t%s\n'            "$CONC"
    printf 'max_tokens\t%s\n'      "$MAX_TOKENS"
    printf 'wall_s\t%s\n'          "$wall"
    printf 'gen_tokens\t%s\n'      "$dgen"
    printf 'decode_tok_s\t%s\n'    "$thr"
    printf 'lat_mean_s\t%s\n'      "$mean"
    printf 'lat_p50_s\t%s\n'      "$p50"
    printf 'lat_p99_s\t%s\n'      "$p99"
    printf 'cmdline\t%s\n'         "vllm serve ${CFG_MODEL} --host 0.0.0.0 --port ${PORT} --max-model-len ${MAXLEN} --gpu-memory-utilization ${UTIL} ${CFG_ARGS[*]}"
  } > "$result"
  date +%s.%N > "$donefile"

  printf '\nSTEP 5 — recorded %s\n' "$result"
  hr
  printf 'NOW: Ctrl-C the server, then:\n\n'
  local next
  if next="$(next_cfg)"; then
    printf '  bash labs/11_tensor_parallel.sh %s\n\n' "$next"
  else
    printf '  bash labs/11_tensor_parallel.sh table\n'
    printf '  bash labs/11_tensor_parallel.sh moe-48g\n\n'
  fi
  print_table
}

gpu_peak_sum() {
  [ -s "$1" ] || { printf 'n/a'; return 0; }
  awk '{ if ($3 > m[$2]) m[$2] = $3 } END { s = 0; for (g in m) s += m[g]; printf "%.0f", s }' "$1"
}

# ---------------------------------------------------------------------------
# Part B, the deliberate failure
# ---------------------------------------------------------------------------
do_moe_tp1_fail() {
  local cfg="moe-tp1"
  cfg_args "$cfg"
  require_gpus 1 "$cfg"
  mkdir -p "$OUTDIR"
  local log="${OUTDIR}/${cfg}.log" startfile="${OUTDIR}/${cfg}.start"

  hr
  printf 'Lab 11 — Part B: the run that MUST NOT fit\n'
  hr
  printf '  %s with --quantization fp8 -tp=1 on a 24 GB card\n' "$MOE_MODEL"
  printf '  predicted weights: ~30.6 GB against a %s x 24 = %s GB budget\n\n' \
    "$UTIL" "$(awk -v u="$UTIL" 'BEGIN { printf "%.1f", u * 24 }')"
  printf '  1. Ctrl-C any running server.\n'
  printf '  2. Paste:\n\n'
  printf '     cd %s\n' "$VL_ROOT"
  printf '     date +%%s.%%N > %s\n' "$startfile"
  printf '     MODEL=%s UTIL=%s MAXLEN=%s bash scripts/serve.sh %s --quantization fp8 -tp 1 2>&1 | tee %s\n\n' \
    "$MOE_MODEL" "$UTIL" "$MAXLEN" "$MOE_MODEL" "$log"
  printf '  3. Watch it fail, then press Enter here.\n\n'
  printf '  What you are looking for (exact strings, verified in the source):\n'
  printf '    "No available memory for the cache blocks."   vllm/v1/core/kv_cache_utils.py:922-928\n'
  printf '    "Free memory on device (..) on startup."      vllm/v1/worker/gpu_worker.py:967-990\n'
  printf '    "CUDA out of memory"                          the weights themselves did not fit\n\n'
  read -r _ || true

  if [ ! -f "$log" ]; then
    printf 'No log at %s — did you use the 2>&1 | tee form above?\n' "$log"
    return 0
  fi
  printf '\n-- verdict --\n'
  local saw=0
  if grep -q 'No available memory for the cache blocks' "$log"; then
    printf '  FOUND: "No available memory for the cache blocks."\n'
    printf '  Meaning: the weights and activations consumed the whole budget, so vLLM could\n'
    printf '  not allocate a single KV block. The engine refuses rather than OOM-ing later —\n'
    printf '  a deliberate design choice, so you find out at startup, not at 3am.\n'
    saw=1
  fi
  if grep -q 'Free memory on device' "$log"; then
    printf '  FOUND: the "Free memory on device ..." INFO line:\n'
    grep -m1 -A2 'Free memory on device' "$log" | sed 's/^/    /'
    printf '  That line suggests a `--kv-cache-memory=<N>` value. THERE IS NO SUCH FLAG:\n'
    printf '  the real one is `--kv-cache-memory-bytes` (vllm/engine/arg_utils.py:1335).\n'
    printf '  The suggestion is also useless here — the problem is the weights, not the KV.\n'
    saw=1
  fi
  if grep -qi 'out of memory' "$log"; then
    printf '  FOUND: a CUDA out-of-memory during weight load.\n'
    saw=1
  fi
  if [ "$saw" -eq 0 ]; then
    if server_up; then
      printf '  The server came UP. That contradicts the prediction. Possibilities:\n'
      printf '    - the fp8 weights are smaller than predicted (read "Model loading took");\n'
      printf '    - the log is from another run;\n'
      printf '    - you are on a card bigger than 24 GB.\n'
      printf '  Record what actually happened — a wrong prediction is the useful outcome.\n'
    else
      printf '  The server is not up and none of the expected strings are in the log.\n'
      printf '  Read the log yourself and record the real error:\n    %s\n' "$log"
    fi
  fi
  printf '\n  Now do it properly:\n\n    bash labs/11_tensor_parallel.sh moe-tp2\n\n'
}

# ---------------------------------------------------------------------------
# the 48 GB single-card alternative (printed, not run)
# ---------------------------------------------------------------------------
do_moe_48g() {
  hr
  printf 'Lab 11 — Part B alternative: ONE 48 GB card\n'
  hr
  cat <<'TXT'
  Why this exists: 2 x 24 GB with -tp=2 works, but every layer pays an all-reduce over
  PCIe. One 48 GB card has no inter-GPU traffic at all. On RunPod (docs/03-runpod-setup.md
  §9) the options are:

    1 x RTX 6000 Ada 48 GB   $0.74/hr   Ada  -> FP8 W8A8 works. The best single card here.
    1 x RTX A6000    48 GB   $0.33/hr   Ampere -> NO FP8 W8A8. Cheap, but see the caveat.
    2 x RTX 4090     24 GB   $0.68/hr   (that is the pod this lab's Part B assumes)

  Route 1 — the bf16 checkpoint, quantized online by vLLM (~30.6 GB resident, 61 GB download):

    MODEL=Qwen/Qwen3-30B-A3B UTIL=0.90 MAXLEN=8192 \
      bash scripts/serve.sh Qwen/Qwen3-30B-A3B --quantization fp8

    (long form)
    vllm serve Qwen/Qwen3-30B-A3B \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --quantization fp8

    `--quantization fp8` is a deprecated alias for `--quantization fp8_per_tensor` in
    v0.30.0 (the loader logs the deprecation and resolves it — see the header). Use the
    per-tensor route on Ada: it lands on the CUTLASS FP8 kernels that SM 8.9 supports.

  Route 2 — the pre-quantized checkpoint (32.4 GB download, non-gated, apache-2.0):

    vllm serve Qwen/Qwen3-30B-A3B-FP8 \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90

    Do NOT pass --quantization here: the checkpoint declares
    quantization_config.quant_method = "fp8" with weight_block_size [128, 128], and vLLM
    reads it from config.json.

    Caveat, verified in the source: that checkpoint is BLOCK-wise FP8, and vLLM's
    CUTLASS block-FP8 kernels require SM >= 90
    (csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:161-173). On Ada
    (SM 8.9) that check fails and vLLM uses the Triton block-scaled kernel
    (vllm/model_executor/kernels/linear/scaled_mm/triton.py:159-164), which is supported
    on any CUDA device but is not the fastest path. On Hopper it would use CUTLASS.
    So: Route 1 for speed on a 4090 or 6000 Ada, Route 2 for the smallest download.

  Route 3 — the A6000 (48 GB, Ampere) at $0.33/hr:

    vllm serve Qwen/Qwen3-30B-A3B-FP8 --max-model-len 8192 --gpu-memory-utilization 0.90

    It will load and serve, and you get the full memory win (32 GB of weights plus room
    for KV). You do not get FP8 tensor cores: the per-tensor CUTLASS gate is SM >= 89
    (scaled_mm_entry.cu:145-158), so vLLM falls back to weight-only FP8 Marlin
    (vllm/model_executor/kernels/linear/scaled_mm/marlin.py:29-45). Expect bf16-class
    throughput with fp8-sized weights. That is the whole lesson of Stage 9: VRAM is not
    the only spec that matters.

  Predicted, 48 GB, UTIL=0.90 -> 43.2 GB budget:
    Route 1/2: weights ~30.6 GB, activations ~2 GB -> ~10.6 GB KV
               at 96 KiB/token (bf16 KV) ~= 115,000 tokens
               at 48 KiB/token (add --kv-cache-dtype fp8) ~= 230,000 tokens
    Route 3  : the same arithmetic, minus the FP8 compute win.

  Do not take those numbers on faith. Run the server, read "Available KV cache memory"
  and "GPU KV cache size", and compare — that is lab 10's method applied to a bigger model.
TXT
  printf '\n'
}

# ---------------------------------------------------------------------------
# table / status
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
  printf 'SUMMARY — Stage 10 tensor parallelism\n'
  hr
  printf '%-9s %6s %8s %10s %12s %12s %12s %10s %9s\n' \
    "cfg" "tp" "w/rank" "KV GiB" "KV tokens" "max conc" "decode t/s" "p50 s" "p99 s"
  local c f
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '%-9s %6s %8s %10s %12s %12s %12s %10s %9s\n' \
        "$c" "$(val "$f" tp)" "$(val "$f" weights_gib)" "$(val "$f" avail_kv_gib)" \
        "$(val "$f" kv_tokens)" "$(val "$f" max_concurrency)" \
        "$(val "$f" decode_tok_s)" "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
    else
      printf '%-9s %6s %8s %10s %12s %12s %12s %10s %9s\n' "$c" "-" "-" "-" "-" "-" "-" "-" "-"
    fi
  done
  printf '\nPer-GPU memory, idle after load and peak during the workload (MiB):\n'
  printf '%-9s %12s %12s %12s %12s\n' "cfg" "gpu0 idle" "gpu1 idle" "gpu0 peak" "gpu1 peak"
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    if [ -f "$f" ]; then
      printf '%-9s %12s %12s %12s %12s\n' "$c" \
        "$(val "$f" gpu0_idle_mib)" "$(val "$f" gpu1_idle_mib)" \
        "$(val "$f" gpu0_peak_mib)" "$(val "$f" gpu1_peak_mib)"
    fi
  done
  printf '\nMarkdown, ready to paste into notes/:\n\n'
  printf '| config | tp | weights/rank GiB | KV GiB | KV tokens | max conc | decode tok/s | p50 s | p99 s |\n'
  printf '| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
  for c in "${CONFIGS[@]}"; do
    f="${OUTDIR}/${c}.tsv"
    [ -f "$f" ] && printf '| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
      "$(val "$f" label)" "$(val "$f" tp)" "$(val "$f" weights_gib)" "$(val "$f" avail_kv_gib)" \
      "$(val "$f" kv_tokens)" "$(val "$f" max_concurrency)" "$(val "$f" decode_tok_s)" \
      "$(val "$f" lat_p50_s)" "$(val "$f" lat_p99_s)"
  done
  printf '\n'
}

show_status() {
  hr
  printf 'Lab 11 — Stage 10 status   (results in %s)\n' "$OUTDIR"
  hr
  printf 'dense model : %s\n' "$DENSE_MODEL"
  printf 'MoE model   : %s\n' "$MOE_MODEL"
  show_gpu
  print_table
  printf 'Configurations:\n'
  local c
  for c in "${CONFIGS[@]}"; do printf '  %-9s %-3s GPUs  %s\n' "$c" "$(gpus_needed "$c")" "$(cfg_label "$c")"; done
  printf '  %-9s %-3s GPUs  the deliberate failure (Part B)\n' "moe-tp1" "1"
  printf '\nCOST: a 2-GPU pod is 2x the hourly rate (Community 2 x RTX 4090 = $0.68/hr vs\n'
  printf '$0.34/hr). Re-check the price before deploying, and stop the pod afterwards.\n\n'
  if server_up; then
    printf 'A server IS answering %s/health (model %s).\n' "$BASE" "$(served_model_id || echo '?')"
  else
    printf 'No server on %s.\n' "$BASE"
  fi
  local n
  if n="$(next_cfg)"; then
    printf 'Next unrecorded configuration: %s\n\n  bash labs/11_tensor_parallel.sh %s\n\n' "$n" "$n"
  else
    printf 'All three recorded. Then:\n'
    printf '  bash labs/11_tensor_parallel.sh moe-tp1    # watch it not fit\n'
    printf '  bash labs/11_tensor_parallel.sh moe-48g   # the 48 GB alternative\n\n'
  fi
}

case "${1:-status}" in
  status|"") show_status ;;
  gpu)       show_gpu ;;
  table)     print_table ;;
  reset)
    hr
    printf 'This deletes every recorded run under %s. Type "yes" to continue: ' "$OUTDIR"
    read -r answer
    [ "$answer" = "yes" ] || die "aborted"
    rm -f "${OUTDIR}"/tp1.* "${OUTDIR}"/tp2.* "${OUTDIR}"/moe-tp2.* "${OUTDIR}"/moe-tp1.*
    printf 'cleared.\n'
    ;;
  tp1|tp2|moe-tp2) do_run "$1" ;;
  moe-tp1)   do_moe_tp1_fail ;;
  moe-48g)   do_moe_48g ;;
  *) die "unknown argument '$1'. Use: tp1 | tp2 | moe-tp2 | moe-tp1 | moe-48g | status | table | gpu | reset" ;;
esac

# ---------------------------------------------------------------------------
# RECORD: write these into notes/stage-10-scaling.md
#
#   date / vLLM version / torch version / CUDA version / GPU model(s) + driver
#   the pod shape (how many GPUs, $/hr) and how long the session ran
#
#   Part A (-tp=1 vs -tp=2 on the same model):
#     - weights per rank ("Model loading took") for each, and the ratio
#       (expect ~0.5x — if it is not, say why in your own words)
#     - per-GPU memory from nvidia-smi: idle after load, and peak during the workload
#     - "Available KV cache memory" and "GPU KV cache size" for each
#     - decode throughput and p50/p99 latency for each
#     - the KV-token ratio (tp2/tp1). Record your PREDICTION first; the arithmetic in this
#       file says ~5x, not 2x. Were you right?
#     - whether throughput went UP, FLAT or DOWN with TP=2, and your explanation
#       (PCIe all-reduce per layer is the thing to look for)
#
#   Part B:
#     - the exact -tp=2 command for the MoE, its weights/rank, KV GiB, KV tokens
#     - the error string from the -tp=1 attempt, verbatim
#     - whether the failure was "No available memory for the cache blocks" or a CUDA OOM,
#       and which of the two you would rather see in production
#     - which single 48 GB card you would rent, and whether FP8 W8A8 is available on it
#
#   Every number above is a prediction until you run it. Nothing in this project was
#   executed on a GPU while it was written.
# ---------------------------------------------------------------------------
