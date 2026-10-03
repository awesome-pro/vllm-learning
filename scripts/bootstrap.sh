#!/usr/bin/env bash
# bootstrap.sh — one-shot setup for a fresh RunPod GPU pod (idempotent).
#
#   cd /workspace && bash vlearning/scripts/bootstrap.sh
#
# Installs uv, creates /workspace/venv, installs vLLM, clones the vLLM source at
# the pinned tag, persists the environment in ~/.bashrc, and optionally warms the
# model cache. Re-run it any time; it notices what already exists.
#
# Env knobs:
#   VLLM_VERSION=0.30.0     vLLM release to install
#   TORCH_BACKEND=cu130     force a PyTorch CUDA backend (cu129/cu130/auto); default detected
#   LADDER=tiny|tiny+mid|all   how much of the model ladder to pre-download
#   SKIP_PREFETCH=1         do not download any models now
#
# WHY THE CUDA VERSION MATTERS (verified 2026-10-02):
#   The PyPI wheel for vllm 0.30.0 is built against CUDA 13.0 — its metadata requires
#   nvidia-cutlass-dsl[cu13], and installing with --torch-backend=cu130 pulls
#   torch 2.13.0+cu130 and nvidia-cuda-runtime 13.0.96. Installing that same wheel with
#   --torch-backend=cu129 satisfies the resolver but pairs a CUDA-12 torch with a
#   CUDA-13 vLLM, which fails at runtime.
#   CUDA 13 needs driver R580+. A consumer card (RTX 4090) cannot use CUDA
#   forward-compatibility mode, so on an older driver we fall back to the explicitly
#   CUDA 12.9 wheel published in the GitHub release assets.
set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
VENV="${VENV:-$WS/venv}"
HF_HOME="${HF_HOME:-$WS/hf}"
VLLM_SRC="${VLLM_SRC:-$WS/src/vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
LADDER="${LADDER:-tiny+mid}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

# RunPod clears the container disk on stop; /workspace survives. Anything the venv
# depends on must therefore live on the volume, including uv's managed interpreter
# and its wheel cache — otherwise a restarted pod finds a venv whose Python is gone.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WS/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WS/uv-cache}"
export PATH="$HOME/.local/bin:$PATH"

# Export these so the child prefetch.sh (and the sanity check) inherit them.
export WS VENV HF_HOME VLLM_SRC
export HF_HUB_ENABLE_HF_TRANSFER=1     # requires hf_transfer, installed below
export TOKENIZERS_PARALLELISM=false

# Decide the CUDA stack unless the caller forced one.
USE_CU129_WHEEL=0
if [ -n "${TORCH_BACKEND:-}" ]; then
  [ "$TORCH_BACKEND" = "cu129" ] && USE_CU129_WHEEL=1
else
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
                  | head -1 | cut -d. -f1 | tr -dc '0-9')"
  if [ -n "$driver_major" ] && [ "$driver_major" -ge 580 ] 2>/dev/null; then
    TORCH_BACKEND=cu130
  else
    TORCH_BACKEND=cu129
    USE_CU129_WHEEL=1
    warn "driver ${driver_major:-unknown} < 580: using the CUDA 12.9 vLLM wheel instead of CUDA 13"
  fi
fi

# ---------------------------------------------------------------------------
say "1/7  Checking the machine"
if [ ! -d "$WS" ]; then
  echo "error: $WS does not exist. Are you on the pod?" >&2
  echo "       Override with WS=/some/path if you know what you are doing." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  warn "nvidia-smi not found — this looks like a CPU-only machine. vLLM will not run."
fi
echo "workspace : $WS"
echo "venv      : $VENV"
echo "hf home   : $HF_HOME"
echo "vllm src  : $VLLM_SRC"

# ---------------------------------------------------------------------------
say "2/7  Installing uv"
if command -v uv >/dev/null 2>&1; then
  uv --version
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "error: uv install failed" >&2; exit 1; }
  uv --version
fi

# ---------------------------------------------------------------------------
say "3/7  Creating the virtualenv"
mkdir -p "$HF_HOME"
if [ -x "$VENV/bin/python" ]; then
  echo "already exists: $VENV"
else
  uv venv --python 3.12 --seed --managed-python "$VENV"
fi

# ---------------------------------------------------------------------------
say "4/7  Installing vLLM $VLLM_VERSION  (torch backend: $TORCH_BACKEND)"
have_version="$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
# Accept "0.30.0", "0.30.0+cu129", "0.30.0.dev0+g1234" as the same release.
if [ "${have_version%%+*}" = "$VLLM_VERSION" ]; then
  echo "already installed: vllm $have_version"
else
  [ -n "$have_version" ] && echo "found vllm $have_version, installing $VLLM_VERSION"
  if [ "$USE_CU129_WHEEL" = "1" ]; then
    # The CUDA 12.9 build is not on PyPI; it is a GitHub release asset.
    arch="$(uname -m)"   # x86_64 on RunPod
    wheel="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_${arch}.whl"
    echo "installing the CUDA 12.9 wheel: $wheel"
    uv pip install --python "$VENV/bin/python" "$wheel" --torch-backend=cu129
  else
    uv pip install --python "$VENV/bin/python" \
      "vllm==$VLLM_VERSION" \
      --torch-backend="$TORCH_BACKEND"
  fi
fi
# hf_transfer makes weight downloads several times faster.
uv pip install --python "$VENV/bin/python" hf_transfer >/dev/null

# ---------------------------------------------------------------------------
say "5/7  Cloning the vLLM source at v$VLLM_VERSION"
if [ -d "$VLLM_SRC/.git" ]; then
  echo "already cloned: $VLLM_SRC"
  git -C "$VLLM_SRC" describe --tags 2>/dev/null || true
else
  mkdir -p "$(dirname "$VLLM_SRC")"
  git clone --depth 1 --branch "v$VLLM_VERSION" \
    https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi
echo "read the source here: \$VLLM_SRC  ($VLLM_SRC)"

# ---------------------------------------------------------------------------
say "6/7  Persisting the environment in ~/.bashrc"
MARK="# >>> vlearning >>>"
if grep -qF "$MARK" "$HOME/.bashrc" 2>/dev/null; then
  echo "already present in ~/.bashrc"
else
  cat >> "$HOME/.bashrc" <<EOF

$MARK
export WS="$WS"
export VLLM_SRC="$VLLM_SRC"
export HF_HOME="$HF_HOME"
export VENV="$VENV"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PATH="\$HOME/.local/bin:\$PATH"
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
# <<< vlearning <<<
EOF
  echo "appended to ~/.bashrc (re-login, or: source ~/.bashrc)"
fi

# ---------------------------------------------------------------------------
say "7/7  Sanity check"
# The interpreter the venv points at must survive a pod restart. If it is not on
# the volume, a stopped/started pod comes back to a broken venv.
if ! "$VENV/bin/python" -c 'import sys' 2>/dev/null; then
  warn "the venv's interpreter is missing — it was on the container disk, which RunPod clears on stop."
  warn "re-run this script to rebuild it on the volume."
fi

# These are two different questions, and asking only the first is how a broken
# pod looks healthy right up to the first model load:
#   torch.cuda.device_count() -> "can NVML see a card?"  Same library nvidia-smi
#       uses. It happily answers "1" on a pod where CUDA is completely unusable.
#   torch.cuda.init()         -> "can the CUDA runtime create a context?"  This
#       is the question every vLLM stage actually depends on.
GPU_OK=1
if ! "$VENV/bin/python" - <<'PY'
import torch, vllm
print(f"vllm    : {vllm.__version__}")
print(f"torch   : {torch.__version__}")
print(f"cuda    : {torch.version.cuda}")
print(f"nvml    : {torch.cuda.device_count()} device(s) visible to nvidia-smi/NVML")
try:
    torch.cuda.init()
except Exception as exc:
    print(f"runtime : FAILED - {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print("runtime : ok - CUDA context created")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  [{i}] {torch.cuda.get_device_name(i)}  {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
PY
then
  GPU_OK=0
fi

# Weights live on /workspace and survive a stop, so download them even when the
# GPU is unusable: fixing the GPU later should not cost another 9 GB.
if [ "${SKIP_PREFETCH:-0}" != "1" ]; then
  if [ "$GPU_OK" = "0" ]; then
    warn "downloading the model ladder anyway — the weights are on /workspace and persist,"
    warn "so fixing the GPU later will not mean re-downloading them."
  fi
  bash "$VL_ROOT/scripts/prefetch.sh" "$LADDER"
fi

if [ "$GPU_OK" = "1" ]; then
  STATUS="Bootstrap complete."
else
  STATUS="Bootstrap finished, but the GPU check FAILED — read below."
fi

cat <<EOF

------------------------------------------------------------------
 $STATUS

   source ~/.bashrc            # or log out and back in
   cd $VL_ROOT
   source scripts/env.sh
   bash labs/00_verify_install.sh

 Then: CURRICULUM.md Stage 0 → 1 → 2 …
 Note: vLLM $VLLM_VERSION was pinned deliberately. Check for a newer
       release before a long session:  docs/03-runpod-setup.md §9
------------------------------------------------------------------
EOF

if [ "$GPU_OK" = "0" ]; then
  cat <<EOF

------------------------------------------------------------------
 The INSTALL is fine. The CUDA RUNTIME is not.

  nvidia-smi works because it uses NVML. CUDA needs a *context* on the
  device, and creating that is what just failed — so the install above is
  untouched and you do not need to reinstall anything. On RunPod this is
  one of three things, cheapest to check first:

  1. A stale forward-compat libcuda is shadowing the real driver. That is
     what the image's /usr/local/cuda/compat directory is for, and a CUDA 13
     runtime will refuse it on a 580 driver.

       LD_DEBUG=libs "$VENV/bin/python" -c "import torch" 2>&1 | grep -i libcuda | head -3
       unset CUDA_VISIBLE_DEVICES
       export LD_LIBRARY_PATH="\$(printf '%s' "\$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vE '/usr/local/cuda/(compat|lib64/stubs)' | paste -sd: -)"
       "$VENV/bin/python" -c "import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))"

     If the last line prints the card, make the change permanent by adding
     the same LD_LIBRARY_PATH line to scripts/env.sh.

  2. A glitched GPU attach. Stop the pod and start it again — restarting
     the container is not always enough — then re-run this script.

  3. Still broken after a restart? Do not fight CUDA 13, skip it:

       TORCH_BACKEND=cu129 bash "$VL_ROOT/scripts/bootstrap.sh"

     That installs the published CUDA 12.9 wheel, which pairs with a CUDA 12
     torch and runs happily on the same 580 driver. Every vLLM concept in
     this guide is identical; only the toolkit version changes.

  Full writeup: docs/03-runpod-setup.md section 6.
------------------------------------------------------------------
EOF
  exit 1
fi
