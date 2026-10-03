#!/usr/bin/env bash
# serve.sh — start the vLLM OpenAI-compatible server on the pod.
#
#   bash scripts/serve.sh                          # $MODEL_TINY, port 8000
#   bash scripts/serve.sh Qwen/Qwen3-4B            # explicit model
#   UTIL=0.85 bash scripts/serve.sh Qwen/Qwen3-8B  # tighter KV budget
#   bash scripts/serve.sh Qwen/Qwen3-4B --no-enable-prefix-caching   # extra flags pass through
#
# ---------------------------------------------------------------------------
# TWO DEFAULTS WORTH KNOWING (both verified in v0.30.0 source):
#
#   --gpu-memory-utilization 0.92 is vLLM's default, which on a 24 GB card
#   reserves ~22 GB and leaves almost nothing for the rest of the system. We
#   default to 0.90. If the pod feels unstable, go to 0.85.
#
#   --max-num-batched-tokens defaults to 2048 for `vllm serve` on a card
#   below 70 GB (it is 8192 for the offline LLM() class). That is a real
#   throughput ceiling on long prompts, and it is WHY Stage 5 exists. We do
#   not override it here on purpose: you should see the default first, then
#   change it deliberately in Lab 06.
# ---------------------------------------------------------------------------
#
# Watch the startup log for the one line that tells you your real capacity:
#
#   GPU KV cache size: 176,192 tokens, Maximum concurrency for 8,192 tokens
#   per request: 21.51x
#
# That is the number of tokens you can cache, and the concurrency it buys at
# the context length you asked for. It is printed before any traffic arrives.
#
# Those figures are MEASURED, not predicted: a RunPod RTX 4090 (23.5 GiB) running this
# script with its own defaults -- Qwen3-0.6B, UTIL=0.90, max-model-len 8192 -- where it
# also printed "Available KV cache memory: 18.82 GiB". Check the arithmetic yourself:
# 18.82 GiB / 176,192 tokens = 112.0 KiB per token, the 0.6B figure in README.md's
# ladder. Change the model or the context length and both numbers move; that is the
# labs' job, and this line is the one to predict before you run them.
# ---------------------------------------------------------------------------

set -euo pipefail

VL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$VL_ROOT/scripts/env.sh"

MODEL="${1:-$MODEL_TINY}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-8192}"
shift || true

cat <<EOF
------------------------------------------------------------------
 Serving   : ${MODEL}
 Endpoint  : http://127.0.0.1:${PORT}/v1          (from the pod)
 Public    : https://<pod-id>-${PORT}.proxy.runpod.net   (via RunPod proxy)
 Docs/UI   : http://127.0.0.1:${PORT}/docs
 max-model-len          : ${MAXLEN}
 gpu-memory-utilization : ${UTIL}    (sets the KV cache budget)
 extra args             : $*
 Stop with : Ctrl-C
------------------------------------------------------------------
EOF

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${UTIL}" \
  "$@"
