#!/usr/bin/env bash
# prefetch.sh — warm the model cache before lab time, so labs are not download time.
#
#   bash scripts/prefetch.sh                # tiny + mid (default, 8.9 GB)
#   bash scripts/prefetch.sh all            # + the 8B model (24.2 GB total)
#   bash scripts/prefetch.sh tiny           # just the 0.6B (1.4 GB)
#
# Weights land in $HF_HOME (on the volume), so they survive a pod restart.
set -euo pipefail

WS="${WS:-/workspace}"
HF_HOME="${HF_HOME:-$WS/hf}"
VENV="${VENV:-$WS/venv}"
export HF_HOME

MODEL_TINY="${MODEL_TINY:-Qwen/Qwen3-0.6B}"
MODEL_MID="${MODEL_MID:-Qwen/Qwen3-4B}"
MODEL_BIG="${MODEL_BIG:-Qwen/Qwen3-8B}"
MODEL_MOE="${MODEL_MOE:-Qwen/Qwen3-30B-A3B}"

LADDER="${1:-tiny+mid}"
case "$LADDER" in
  tiny)     MODELS=("$MODEL_TINY") ;;
  tiny+mid) MODELS=("$MODEL_TINY" "$MODEL_MID") ;;
  all)      MODELS=("$MODEL_TINY" "$MODEL_MID" "$MODEL_BIG") ;;
  *) echo "usage: $0 [tiny|tiny+mid|all]" >&2; exit 2 ;;
esac

# Prefer the pod venv's CLI; fall back to whatever is on PATH.
if   [ -x "$VENV/bin/hf" ];              then DL=("$VENV/bin/hf" download)
elif [ -x "$VENV/bin/huggingface-cli" ]; then DL=("$VENV/bin/huggingface-cli" download)
elif command -v hf >/dev/null 2>&1;      then DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then DL=(huggingface-cli download)
else
  echo "error: no huggingface CLI found. Run scripts/bootstrap.sh first, or:" >&2
  echo "       uv pip install --python $VENV/bin/python 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "$HF_HOME"
echo "cache: $HF_HOME"
echo "plan : $LADDER  →  ${MODELS[*]}"
echo

for m in "${MODELS[@]}"; do
  printf '\n=== downloading %s ===\n' "$m"
  "${DL[@]}" "$m" || {
    echo "warning: failed to download $m — labs that need it will fetch it on first use" >&2
    continue
  }
done

cat <<EOF

------------------------------------------------------------------
 Cached models:
$(du -sh "$HF_HOME"/hub/models--* 2>/dev/null | sed 's/^/   /' || echo "   (none found)")

 Not prefetched on purpose:
   $MODEL_BIG  (15.3 GB) — bash scripts/prefetch.sh all
   $MODEL_MOE  (61 GB bf16) — Stage 10 only, on a 48 GB+ card
------------------------------------------------------------------
EOF
