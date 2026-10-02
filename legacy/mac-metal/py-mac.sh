#!/usr/bin/env bash
# Run a Python script with the interpreter that actually has vLLM installed.
#
# The Homebrew vllm-metal formula installs vLLM into a *private* virtualenv
# (no activation needed for the `vllm` CLI, but `python` does not see it).
# Use this wrapper for any script that does `from vllm import ...`.
#
#   bash scripts/py.sh labs/01_offline_inference.py
#   bash scripts/py.sh -c "import vllm; print(vllm.__version__)"
#
# Override by exporting VLLM_PY if you have your own environment.

set -euo pipefail

if [ -n "${VLLM_PY:-}" ]; then
  PY="${VLLM_PY}"
elif PREFIX="$(brew --prefix vllm-metal 2>/dev/null)" && [ -x "${PREFIX}/libexec/bin/python" ]; then
  PY="${PREFIX}/libexec/bin/python"
elif [ -x /opt/homebrew/opt/vllm-metal/libexec/bin/python ]; then
  PY=/opt/homebrew/opt/vllm-metal/libexec/bin/python
else
  echo "error: could not find the vLLM interpreter." >&2
  echo "       install with: brew install vllm-project/vllm-metal/vllm-metal" >&2
  echo "       or export VLLM_PY=/path/to/python" >&2
  exit 1
fi

exec "${PY}" "$@"
