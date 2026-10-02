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
# On the pod everything durable is on the volume, mounted at /workspace.
if [ -d /workspace ]; then
  export WS="${WS:-/workspace}"
else
  # Local machine (reading docs / editing). Labs will not run here.
  export WS="${WS:-$VL_ROOT}"
fi

export VLLM_SRC="${VLLM_SRC:-$WS/src/vllm}"   # shallow clone of upstream at v0.30.0
export HF_HOME="${HF_HOME:-$WS/hf}"           # model weights live on the volume
export VENV="${VENV:-$WS/venv}"               # vLLM's virtualenv, also on the volume

# --- Keep the toolchain on the volume too ------------------------------------
# RunPod CLEARS the container disk when a pod stops, but /workspace survives. If
# uv's managed CPython or its wheel cache live in the container, the venv on
# /workspace is left pointing at a deleted interpreter after every restart, and
# every wheel gets re-downloaded. Both belong on the volume.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WS/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WS/uv-cache}"
export PATH="$HOME/.local/bin:$PATH"          # uv installs itself here

# --- Downloads ---------------------------------------------------------------
# hf_transfer is installed by bootstrap.sh and makes weight downloads much faster.
if python3 -c "import hf_transfer" 2>/dev/null || [ -d "$VENV/lib" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
fi
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
export PORT="${PORT:-8000}"
export UTIL="${UTIL:-0.90}"        # gpu-memory-utilization: 0.90 leaves ~2.4 GB headroom on 24 GB
export MAXLEN="${MAXLEN:-8192}"

# --- Activate the pod venv ---------------------------------------------------
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
