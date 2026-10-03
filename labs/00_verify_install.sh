#!/usr/bin/env bash
# Lab 00 - Stage 0: prove the machine works and record exactly what we are running.
#
# WHAT IT DEMONSTRATES
#   That this is a real GPU pod, that vLLM is importable *from the venv*, that the
#   source checkout is the version this guide was written against, that model
#   weights and the KV cache live on the persistent volume, and that one token can
#   actually come out of the GPU. Nothing later in the guide is worth doing if any
#   of these is false, and every failure prints the command that fixes it.
#
# HOW TO RUN
#   cd /workspace/vlearning
#   source scripts/env.sh
#   bash labs/00_verify_install.sh
#
# PREREQUISITES
#   None - no server needed. It is a GPU lab, so on a laptop the pod/GPU/vLLM
#   checks report FAIL, which is the correct answer. The first run may download
#   $MODEL_TINY (~1.4 GB) into $HF_HOME.
#
# SAFE TO RE-RUN: read-only checks plus one tiny in-process generation. ~10 s warm,
#   ~2 min if the tiny model still has to come down.
#
# Path convention: $VLLM_SRC is the checkout (scripts/env.sh -> /workspace/src/vllm).
# There is no third-party mirror of vLLM inside this project; the checkout is it.

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

PASS=0
FAIL=0
FIXES=()

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
pass() { PASS=$((PASS + 1)); printf '  [PASS] %s\n' "$1"; }
fail() {
  FAIL=$((FAIL + 1))
  printf '  [FAIL] %s\n' "$1"
  printf '         fix : %s\n' "$2"
  FIXES+=("$1  ->  $2")
}
note() { printf '         %s\n' "$1"; }

# Use the venv interpreter explicitly so this works whether or not the venv is
# active in the current shell (env.sh activates it only if it exists).
VL_PY="$VENV/bin/python"
[ -x "$VL_PY" ] || VL_PY="$(command -v python3 || true)"

hr
echo "Lab 00 - environment verification   ($(date -u '+%Y-%m-%d %H:%M UTC'))"
hr
printf 'guide root : %s\n' "$VL_ROOT"
printf 'venv       : %s\n' "$VENV"
printf 'VLLM_SRC   : %s\n' "$VLLM_SRC"
printf 'HF_HOME    : %s\n' "$HF_HOME"
printf 'MODEL_TINY : %s\n' "$MODEL_TINY"
echo

# --- 1. Are we on a pod? -----------------------------------------------------
hr
echo "1. Host / pod"
if [ -d /workspace ]; then
  pass "/workspace exists (a pod: everything durable is on the volume)"
else
  fail "/workspace does not exist" \
    "run this on the pod - see docs/03-runpod-setup.md; locally the labs cannot run"
fi
printf '         host   : %s %s\n' "$(uname -s)" "$(uname -m)"
printf '         kernel : %s\n' "$(uname -r)"
printf '         python : %s\n' "$(python3 --version 2>&1)"

# --- 2. GPU, driver, CUDA ----------------------------------------------------
hr
echo "2. GPU, driver, CUDA"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  pass "nvidia-smi works"
  while IFS= read -r line; do printf '         GPU    : %s\n' "$line"; done < <(
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
      --format=csv,noheader 2>/dev/null || true
  )
  cuda_drv="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1 || true)"
  printf '         driver reports max CUDA %s\n' "${cuda_drv:-unknown}"
  note "That is the driver's ceiling. The CUDA the wheels were built against is"
  note "torch.version.cuda from check 4 (vLLM 0.30.0 publishes CUDA 12.9 builds)."
else
  fail "nvidia-smi missing or failing" \
    "wrong pod/image - redeploy with runpod/pytorch:1.4.0-cu1290-torch291-ubuntu2404 (docs/03-runpod-setup.md section 2)"
fi

# --- 3. vLLM importable from the venv ---------------------------------------
hr
echo "3. vLLM in the venv"
if [ -x "$VL_PY" ]; then
  pass "interpreter: $VL_PY"
  if vllm_version="$("$VL_PY" -c 'import vllm; print(vllm.__version__)' 2>&1)"; then
    case "$vllm_version" in
      0.30.0*) pass "vllm.__version__ = $vllm_version (what this guide targets)" ;;
      *) fail "vllm.__version__ = $vllm_version, expected 0.30.0" \
           "source \$VENV/bin/activate && uv pip install 'vllm==0.30.0' (scripts/bootstrap.sh does this)" ;;
    esac
  else
    fail "import vllm failed" "bash scripts/bootstrap.sh - it creates \$VENV and installs vLLM"
    note "$(printf '%s' "$vllm_version" | tail -3)"
  fi
else
  fail "no interpreter at $VENV/bin/python" "bash scripts/bootstrap.sh"
fi

# --- 4. torch sees the GPU, and how much VRAM is free -----------------------
hr
echo "4. torch.cuda"
if [ -x "$VL_PY" ]; then
  if torch_out="$("$VL_PY" - <<'PY' 2>&1
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
# device_count() reads NVML, exactly like nvidia-smi does. It will happily report
# a device on a pod where the CUDA runtime cannot create a context at all, so ask
# the runtime itself before trusting the count.
try:
    torch.cuda.init()
    runtime = "ok"
except Exception as exc:      # cudaErrorUnknown, cudaErrorInsufficientDriver, ...
    runtime = f"{type(exc).__name__}: {exc}"
print("runtime", runtime)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if runtime == "ok":
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"gpu{i} {torch.cuda.get_device_name(i)} "
              f"free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB")
PY
)"; then
    while IFS= read -r line; do printf '         %s\n' "$line"; done <<<"$torch_out"
    if grep -q '^runtime ok' <<<"$torch_out"; then
      pass "CUDA runtime initialized - torch.cuda.init() created a context"
      if grep -q '^device_count 0' <<<"$torch_out"; then
        fail "torch sees 0 CUDA devices" \
          "confirm the pod has a GPU and that /dev/nvidia* exists inside it"
      else
        pass "torch sees $(grep -c '^gpu[0-9]' <<<"$torch_out") CUDA device(s), with free VRAM above"
      fi
    elif grep -q '^runtime ' <<<"$torch_out"; then
      fail "CUDA runtime cannot initialize: $(grep -m1 '^runtime ' <<<"$torch_out" | cut -c9-)" \
        "the install is fine: nvidia-smi and device_count() use NVML and still see the card, but CUDA cannot make a context. Work through docs/03-runpod-setup.md section 6"
      note "the device_count above comes from NVML, not from CUDA, so a non-zero"
      note "count does NOT mean CUDA works - every stage after this one needs it."
      if grep -qi 'forward compat' <<<"$torch_out"; then
        note "error 804 is specific and fixable: RunPod's forward-compat shim is being"
        note "loaded ahead of the host driver, which GeForce cards cannot use. Repair:"
        note "  mkdir -p /root/disabled-ld-so-conf"
        note "  mv \"\$(grep -rl /compat /etc/ld.so.conf.d/)\" /root/disabled-ld-so-conf/ && ldconfig"
        note "bootstrap.sh step 1 now does this automatically on a GeForce GPU."
      fi
    else
      fail "torch.cuda.is_available() is False" \
        "CPU-only install - reinstall vLLM inside the venv with: uv pip install vllm --torch-backend=auto"
    fi
  else
    fail "could not query torch.cuda" "reinstall torch in the venv: bash scripts/bootstrap.sh"
    note "$(printf '%s' "$torch_out" | tail -3)"
  fi
fi

# --- 5. The vLLM source checkout --------------------------------------------
hr
echo "5. vLLM source checkout"
if [ -d "$VLLM_SRC/vllm" ]; then
  pass "checkout present: $VLLM_SRC"
  tag="$(git -C "$VLLM_SRC" describe --tags 2>/dev/null || echo 'not-a-git-repo')"
  case "$tag" in
    v0.30.0*) pass "git -C \$VLLM_SRC describe --tags -> $tag" ;;
    *) fail "git describe --tags -> $tag, expected v0.30.0" \
         "git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git $VLLM_SRC" ;;
  esac
  echo "         design docs readable locally, no network needed:"
  ls "$VLLM_SRC/docs/design" 2>/dev/null | head -6 | sed 's/^/           docs\/design\//' || true
else
  fail "no vLLM checkout at $VLLM_SRC" \
    "git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git $VLLM_SRC"
fi

# --- 6. Weights on the persistent volume ------------------------------------
hr
echo "6. Model cache and disk"
case "$HF_HOME" in
  /workspace/*) pass "HF_HOME is on the volume: $HF_HOME" ;;
  *) fail "HF_HOME=$HF_HOME is NOT under /workspace" \
       "export HF_HOME=/workspace/hf - the container disk is wiped on stop, so weights would re-download every session" ;;
esac
if [ -d "$HF_HOME" ]; then
  printf '         HF_HOME size : %s\n' "$(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || true)"
  printf '         models cached: %s\n' \
    "$(find "$HF_HOME" -maxdepth 2 -name 'models--*' 2>/dev/null | wc -l | tr -d ' ' || true)"
else
  note "HF_HOME does not exist yet; the generation below creates it"
fi
# `|| true` matters: with `set -e` + `pipefail`, a failing df/du in a command
# substitution would abort this deliberately non-fatal check.
disk_avail_kb="$(df -Pk /workspace 2>/dev/null | awk 'NR==2 {print $4}' || true)"
case "${disk_avail_kb:-}" in
  ''|*[!0-9]*) disk_avail_kb="" ;;
esac
if [ -n "$disk_avail_kb" ]; then
  printf '         /workspace free: %s\n' "$(df -Ph /workspace 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  if [ "$disk_avail_kb" -gt 5000000 ]; then
    pass "more than 5 GB free on /workspace"
  else
    fail "under 5 GB free on /workspace" \
      "du -sh /workspace/* and delete stale models; the whole ladder needs ~25 GB"
  fi
else
  fail "could not read free space for /workspace" "is /workspace mounted?"
fi

# --- 7. One tiny generation, offline, no server ------------------------------
hr
echo "7. Offline generation with $MODEL_TINY"
note "The LLM class is Stage 0's entrypoint. enforce_eager=True skips torch.compile"
note "so a verification run stays short; UTIL=$UTIL sets the KV cache budget."
if [ -x "$VL_PY" ]; then
  if gen_out="$(MODEL_TINY="$MODEL_TINY" UTIL="$UTIL" "$VL_PY" - <<'PY' 2>&1
import os
import time

from vllm import LLM, SamplingParams

t0 = time.perf_counter()
llm = LLM(
    model=os.environ["MODEL_TINY"],
    max_model_len=512,
    gpu_memory_utilization=float(os.environ["UTIL"]),
    enforce_eager=True,  # no torch.compile / CUDA graphs: this is a smoke test
)
print("LOAD_SECONDS", round(time.perf_counter() - t0, 1))
out = llm.generate(
    ["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8)
)[0]
print("OUTPUT", repr(out.outputs[0].text.strip()))
PY
)"; then
    printf '%s\n' "$gen_out" \
      | grep -E 'KV cache size|Available KV cache|LOAD_SECONDS|OUTPUT' \
      | sed 's/^/         /' || true
    if grep -q '^OUTPUT' <<<"$gen_out"; then
      pass "a token came out of the GPU: $(grep '^OUTPUT' <<<"$gen_out" | cut -c1-64)"
    else
      fail "generation produced no OUTPUT line" \
        "run it by hand to see the traceback; a CUDA OOM means lower UTIL"
      note "$(printf '%s' "$gen_out" | tail -4)"
    fi
    kv_line="$(grep -m1 'KV cache size' <<<"$gen_out" || true)"
    if [ -n "$kv_line" ]; then
      note "the line every later stage reads: ${kv_line#*] }"
    fi
  else
    fail "the tiny offline generation crashed" \
      "python -c 'from vllm import LLM; LLM(\"$MODEL_TINY\")' and read the traceback"
    note "$(printf '%s' "$gen_out" | tail -4)"
  fi
fi

# --- summary -----------------------------------------------------------------
hr
if [ "$FAIL" -eq 0 ]; then
  printf 'SUMMARY: %d passed, 0 failed - the Stage 0 environment is good.\n' "$PASS"
else
  printf 'SUMMARY: %d passed, %d FAILED\n' "$PASS" "$FAIL"
  if [ "${#FIXES[@]}" -gt 0 ]; then
    for f in "${FIXES[@]}"; do printf '  - %s\n' "$f"; done
  fi
fi
hr

cat <<'EOF'
RECORD: lab 00 environment  (add today's date and the vLLM version from check 3)
  gpu_name_and_memory        <- check 2 (nvidia-smi --query-gpu)
  driver_version             <- check 2
  torch_cuda_version         <- check 4 (torch.version.cuda)
  vllm_version               <- check 3
  vllm_src_tag               <- check 5 (git describe --tags)
  hf_home_path               <- check 6
  workspace_free_gb          <- check 6
  model_load_seconds         <- check 7 (LOAD_SECONDS)
  kv_cache_size_tokens       <- check 7 (the "KV cache size: N tokens" line)
------------------------------------------------------------------------
Write these into notes/00-environment.md. Facts about vLLM go stale fast: always
record the version next to the number.
EOF

[ "$FAIL" -eq 0 ]
