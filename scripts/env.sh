# shellcheck shell=bash
# scripts/env.sh — shared environment for every lab. Source it, do not execute it.
#
#   source scripts/env.sh
#
# Sets up paths, the model ladder, and activates the pod's virtualenv.
# Safe to source repeatedly.

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export VL_ROOT

# --- Where things live -------------------------------------------------------
# Two supported machines, and they differ in ways that matter:
#
#   the pod   durable state on the volume at /workspace; a 24 GB card with its own
#             VRAM, so vLLM may take 0.90 of it.
#   a laptop  no volume. vLLM comes from vllm-metal's venv (see docs/mac-onramp.md)
#             and weights come from the normal HuggingFace cache you already have.
#             Crucially the memory is UNIFIED: `--gpu-memory-utilization` is a ceiling
#             on the same pool macOS and your apps are using, so the pod's 0.90 would
#             try to reserve ~23 GB of a 26 GB Mac and bring the machine to its knees.
if [ -d /workspace ]; then
  export WS="${WS:-/workspace}"
  export HF_HOME="${HF_HOME:-$WS/hf}"         # model weights live on the volume
  export VENV="${VENV:-$WS/venv}"             # vLLM's virtualenv, also on the volume
  export UTIL="${UTIL:-0.90}"                 # 24 GB of VRAM: leave ~2.4 GB headroom
  export MAXLEN="${MAXLEN:-8192}"
  export HOST="${HOST:-0.0.0.0}"              # reachable from your laptop via the proxy
else
  export WS="${WS:-$VL_ROOT}"
  # Reuse the machine's existing HF cache; do not start an empty one beside the guide.
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  # vllm-metal's installer puts the whole toolchain here (docs/mac-onramp.md).
  export VENV="${VENV:-$HOME/.venv-vllm-metal}"
  export UTIL="${UTIL:-0.30}"                 # unified memory: keep the Mac usable
  export MAXLEN="${MAXLEN:-4096}"             # KV per request; halved for the laptop
  export HOST="${HOST:-127.0.0.1}"            # loopback: no LAN exposure, no firewall prompt
fi

# --- The vLLM source you will be reading -------------------------------------
# Every doc here cites source as `$VLLM_SRC/vllm/...`. On the pod that is the pinned
# v0.30.0 clone under /workspace. A laptop has no clone there -- but you probably
# already have a vLLM checkout somewhere, and locating it is what makes those paths
# clickable while you read. That reading is most of the work you can do without a GPU,
# so it is worth getting right. An explicit VLLM_SRC always wins.
if [ -z "${VLLM_SRC:-}" ] && [ ! -d "$WS/src/vllm/vllm" ]; then
  for candidate in "$HOME/Desktop/vllm" "$HOME/vllm" "$VL_ROOT/../vllm"; do
    if [ -d "$candidate/vllm" ]; then
      VLLM_SRC="$candidate"
      printf 'env: source checkout = %s (local, not the pod clone -- check the version)\n' "$candidate"
      break
    fi
  done
fi
export VLLM_SRC="${VLLM_SRC:-$WS/src/vllm}"   # shallow clone of upstream at v0.30.0

# --- Keep the toolchain on the volume too ------------------------------------
# RunPod CLEARS the container disk when a pod stops, but /workspace survives. If
# uv's managed CPython or its wheel cache live in the container, the venv on
# /workspace is left pointing at a deleted interpreter after every restart, and
# every wheel gets re-downloaded. Both belong on the volume.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WS/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WS/uv-cache}"
export PATH="$HOME/.local/bin:$PATH"          # uv installs itself here

# --- Downloads ---------------------------------------------------------------
# HF_XET_HIGH_PERFORMANCE is the current switch for fast multi-connection downloads.
# It supersedes HF_HUB_ENABLE_HF_TRANSFER, which huggingface_hub no longer uses and
# now warns about ("hf_transfer is not used anymore" — seen on hub 1.33.0, the
# version vLLM 0.30.0 resolves to). Set on a pod, so it is a no-op elsewhere.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM=false

# --- The model ladder --------------------------------------------------------
# Stage 0-2 use TINY. Stages 3-9 use MID, then repeat on BIG for the real numbers.
# MOE only fits on a 48 GB+ card (or 2x24 GB with -tp=2).
export MODEL_TINY="${MODEL_TINY:-Qwen/Qwen3-0.6B}"    # ~1.5 GB bf16
export MODEL_MID="${MODEL_MID:-Qwen/Qwen3-4B}"        # ~8.0 GB bf16
export MODEL_BIG="${MODEL_BIG:-Qwen/Qwen3-8B}"        # 15.3 GB on disk, ~16.4 GB in VRAM — tight on 24 GB
export MODEL_MOE="${MODEL_MOE:-Qwen/Qwen3-30B-A3B}"   # ~61 GB bf16 — needs 80 GB / 2x48 GB

export MODEL="${MODEL:-$MODEL_TINY}"                  # default model for labs

# --- Server defaults ---------------------------------------------------------
# UTIL and MAXLEN were already defaulted above, per platform: 0.90/8192 on the pod's
# dedicated 24 GB of VRAM, 0.30/4096 on a laptop's unified memory. Override either on
# the command line for a single run, e.g.  UTIL=0.45 bash scripts/serve.sh
export PORT="${PORT:-8000}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export HOST="${HOST:-0.0.0.0}"

# --- Activate vLLM's virtualenv ----------------------------------------------
# The pod's venv, or vllm-metal's on a laptop. Both are ordinary venvs, so the same
# labs run on either machine with no changes.
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
fi

# --- Tiny helpers the labs use ----------------------------------------------
vl_have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

vl_banner() {
  printf '\n=== %s ===\n' "$1"
  printf 'vLLM src : %s\n' "$VLLM_SRC"
  printf 'HF cache : %s\n' "$HF_HOME"
  printf 'Model    : %s\n' "${2:-$MODEL}"
  printf 'Endpoint : http://0.0.0.0:%s/v1\n\n' "$PORT"
}
