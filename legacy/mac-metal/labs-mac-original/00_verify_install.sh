#!/usr/bin/env bash
# Lab 00 - verify the vLLM install and record exactly what we are running.
#
#   bash labs/00_verify_install.sh

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
hr() { printf '%s\n' "------------------------------------------------------------"; }

# The Homebrew formula installs vLLM into a private virtualenv.
if PREFIX="$(brew --prefix vllm-metal 2>/dev/null)" && [ -x "${PREFIX}/libexec/bin/python" ]; then
  VLLM_PY="${PREFIX}/libexec/bin/python"
  VLLM_PREFIX="${PREFIX}"
elif [ -x /opt/homebrew/opt/vllm-metal/libexec/bin/python ]; then
  VLLM_PY=/opt/homebrew/opt/vllm-metal/libexec/bin/python
  VLLM_PREFIX=/opt/homebrew/opt/vllm-metal
else
  VLLM_PY=""
  VLLM_PREFIX=""
fi

hr
echo "Lab 00 - environment verification"
hr

echo "## Platform"
printf 'machine    : %s\n' "$(uname -m)"
printf 'macOS      : %s\n' "$(sw_vers -productVersion 2>/dev/null || echo unknown)"
printf 'chip       : %s\n' "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
printf 'memory     : %s bytes\n' "$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
printf 'disk free  : %s\n' "$(df -h / | awk 'NR==2 {print $4}')"

hr
echo "## vLLM installation"
if [ -n "${VLLM_PY}" ]; then
  printf 'prefix     : %s\n' "${VLLM_PREFIX}"
  printf 'interpreter: %s\n' "${VLLM_PY}"
  printf 'vllm CLI   : %s\n' "$(command -v vllm || echo 'not on PATH')"
  printf 'vllm ver   : %s\n' "$(vllm --version 2>/dev/null | tail -1)"
  echo
  echo "NOTE: vLLM lives in Homebrew's private venv. Your shell's 'python3' does NOT"
  echo "      see it. Use 'bash scripts/py.sh <script.py>' for Python labs, or the"
  echo "      'vllm' CLI directly."
else
  echo "NOT INSTALLED. Install with:"
  echo "  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal"
  echo "  brew install vllm-project/vllm-metal/vllm-metal"
fi

hr
echo "## Backend and plugin detection"
if [ -n "${VLLM_PY}" ]; then
  "${VLLM_PY}" - <<'PY' 2>&1 | grep -vE '^(INFO|WARNING)' | head -30
import importlib
import platform

print("python :", platform.python_version())
print("machine:", platform.machine())

for mod in ("vllm", "vllm_metal", "mlx", "mlx_lm", "torch", "transformers"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s}: {getattr(m, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{mod:14s}: NOT AVAILABLE ({type(exc).__name__})")

try:
    from vllm.platforms import current_platform
    print("vllm platform  :", type(current_platform).__name__)
    print("device type    :", getattr(current_platform, "device_type", "?"))
except Exception as exc:
    print("vllm platform  : could not resolve:", exc)
PY
else
  echo "skipped (not installed)"
fi

hr
echo "## vllm serve - available config groups"
if command -v vllm >/dev/null 2>&1; then
  vllm serve --help 2>/dev/null | sed -n '/Config Groups:/,/^$/p' | head -40
else
  echo "skipped (vllm not on PATH)"
fi

hr
echo "## Source checkout (read-only reference)"
SRC="${HERE}/../vendor/vllm"
if [ -d "${SRC}" ]; then
  printf 'path    : %s\n' "${SRC}"
  printf 'commit  : %s\n' "$(git -C "${SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  printf 'docs    : %s design docs in vendor/vllm/docs/design\n' \
    "$(ls "${SRC}/docs/design" 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "missing - clone with:"
  echo "  git clone --depth 1 https://github.com/vllm-project/vllm.git vendor/vllm"
fi

hr
echo "Done. Write the version + commit above into notes/ - vLLM facts go stale fast."
