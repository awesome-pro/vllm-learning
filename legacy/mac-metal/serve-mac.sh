#!/usr/bin/env bash
# Start a real vLLM OpenAI-compatible server on this Mac (Metal / MLX backend).
#
#   bash scripts/serve.sh                      # default model, port 8000
#   bash scripts/serve.sh Qwen/Qwen3-0.6B      # explicit model
#   PORT=8001 bash scripts/serve.sh            # different port
#   FRACTION=0.3 bash scripts/serve.sh         # even smaller KV cache
#   bash scripts/serve.sh Qwen/Qwen3-0.6B --max-model-len 16384
#
# MEMORY NOTE -----------------------------------------------------------------
# vLLM allocates the entire paged KV cache up front, sized from
# --gpu-memory-utilization. The Metal plugin's own default is 0.92, which on a
# 24 GB Mac reserves ~15.7 GB of KV cache - enough to make the machine crawl.
# We default to FRACTION=0.25: ~2.95 GB of KV cache, 25,760 cached tokens,
# ~6.3x concurrency at a 4096-token context. Raise it if you want more
# concurrent requests; lower it if the Mac feels pressured.
# -----------------------------------------------------------------------------

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
FRACTION="${FRACTION:-0.25}"
shift || true

cat <<EOF
------------------------------------------------------------------
 Serving   : ${MODEL}
 Endpoint  : http://127.0.0.1:${PORT}/v1
 Docs/UI   : http://127.0.0.1:${PORT}/docs
 max-model-len         : ${MAX_MODEL_LEN}
 gpu-memory-utilization: ${FRACTION}   (size of the KV cache budget)
 Stop with : Ctrl-C
------------------------------------------------------------------
 Watch the startup log for:
   "Paged attention memory breakdown: ... kv_budget=..."
   "GPU KV cache size: N tokens, Maximum concurrency ..."
 If kv_budget looks too big, restart with a smaller FRACTION.
------------------------------------------------------------------
EOF

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${FRACTION}" \
  "$@"
