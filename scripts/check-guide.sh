#!/usr/bin/env bash
# check-guide.sh — verify this guide against a real vLLM checkout.
#
#   bash scripts/check-guide.sh                     # uses $VLLM_SRC (or ~/Desktop/vllm)
#   VLLM_SRC=/workspace/src/vllm bash scripts/check-guide.sh
#   CHECK_URLS=1 bash scripts/check-guide.sh        # also HTTP-check every docs.vllm.ai link
#
# Run this after every vLLM upgrade. It checks:
#   1. every `vllm/...` / `tests/...` source path cited in the guide exists in the checkout
#   2. every relative markdown link resolves to a file that exists
#   3. every lab referenced by CURRICULUM.md exists, and every lab file is referenced
#   4. (optional) every docs.vllm.ai URL returns HTTP 200
#
# Exit code 0 = the guide matches the checkout. Non-zero = something moved upstream.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SRC="${VLLM_SRC:-/workspace/src/vllm}"
[ -d "$VLLM_SRC/vllm" ] || VLLM_SRC="$HOME/Desktop/vllm"
CHECK_URLS="${CHECK_URLS:-0}"

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=$((fail + 1)); }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }

printf '\n\033[1mcheck-guide\033[0m  guide=%s  vllm=%s\n' "$ROOT" "$VLLM_SRC"
if [ ! -d "$VLLM_SRC/vllm" ]; then
  echo "error: no vLLM checkout at $VLLM_SRC — set VLLM_SRC=/path/to/vllm" >&2
  exit 2
fi
if [ -d "$VLLM_SRC/.git" ]; then
  printf 'vllm checkout: %s\n' "$(git -C "$VLLM_SRC" describe --tags --always 2>/dev/null || echo '?')"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m1. source paths cited in the guide\033[0m\n'
# Only unambiguous vLLM-internal prefixes; skip globs, placeholders, GitHub release URLs,
# and the legacy/ archive (which legitimately cites the retired vendor/vllm/ layout).
paths="$(
  grep -rhoE '(vllm|tests|csrc)/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*' \
      --include='*.md' --include='*.sh' --include='*.py' \
      --exclude-dir=legacy "$ROOT" 2>/dev/null \
    | sed 's/[.,;:)]*$//' \
    | grep -v '[*<>]' \
    | grep -v '^vllm/releases/' \
    | sort -u
)"
n_paths=0; n_missing=0
while IFS= read -r p; do
  [ -n "$p" ] || continue
  # only treat as a source citation if it looks like a real path (has an extension or 2+ slashes)
  case "$p" in
    *.*|*/*/*) ;;
    *) continue ;;
  esac
  n_paths=$((n_paths + 1))
  if [ ! -e "$VLLM_SRC/$p" ]; then
    bad "missing in checkout: $p"
    n_missing=$((n_missing + 1))
  fi
done <<< "$paths"
if [ "$n_missing" -eq 0 ]; then
  ok "$n_paths distinct source paths all exist"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m2. relative markdown links\033[0m\n'
n_links=0; n_bad=0
while IFS= read -r line; do
  file="${line%%:*}"
  rest="${line#*:}"
  # extract ](target) targets, one per match
  while IFS= read -r target; do
    [ -n "$target" ] || continue
    case "$target" in
      http*|mailto:*|\#*|'') continue ;;
    esac
    target="${target%%#*}"                     # drop anchors
    [ -n "$target" ] || continue
    n_links=$((n_links + 1))
    if [ ! -e "$(dirname "$file")/$target" ]; then
      bad "$(realpath --relative-to="$ROOT" "$file" 2>/dev/null || echo "$file") → $target"
      n_bad=$((n_bad + 1))
    fi
  done < <(printf '%s' "$rest" | grep -oE '\]\([^)]+\)' | sed 's/^](//; s/)$//')
done < <(grep -rHoE '\]\([^)]+\)' --include='*.md' --exclude-dir=legacy "$ROOT" 2>/dev/null)
if [ "$n_bad" -eq 0 ]; then
  ok "$n_links relative links all resolve"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m3. labs referenced vs labs present\033[0m\n'
referenced="$(grep -ohE 'labs/[A-Za-z0-9_]+\.(py|sh)' "$ROOT/CURRICULUM.md" 2>/dev/null | sort -u)"
n_ref=0; n_miss=0
while IFS= read -r l; do
  [ -n "$l" ] || continue
  n_ref=$((n_ref + 1))
  [ -f "$ROOT/$l" ] || { bad "CURRICULUM.md references a missing lab: $l"; n_miss=$((n_miss + 1)); }
done <<< "$referenced"
while IFS= read -r f; do
  rel="labs/$(basename "$f")"
  if ! grep -qF "$rel" "$ROOT/CURRICULUM.md" 2>/dev/null; then
    bad "lab exists but is not referenced by CURRICULUM.md: $rel"
    n_miss=$((n_miss + 1))
  fi
done < <(find "$ROOT/labs" -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) ! -name '_*' | sort)
if [ "$n_miss" -eq 0 ]; then
  ok "$n_ref lab references match the files on disk"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m4. links that are stale by default\033[0m\n'
# legacy/ is excluded: it is a frozen archive of the Mac edition, and its paths
# (vendor/vllm/, docs/03-apple-silicon-setup.md) were correct when it was written.
# This script is excluded too, since it necessarily contains the patterns it looks for.
n_stale=0
if grep -rqE '\]\(https://docs\.vllm\.ai/en/latest/' "$ROOT" --include='*.md' --exclude-dir=legacy 2>/dev/null; then
  bad "some linked docs URLs use /en/latest/ (developer preview) — link to /en/stable/ instead"
  n_stale=1
fi
if grep -rq 'vendor/vllm' "$ROOT" --include='*.md' --include='*.sh' --include='*.py' \
     --exclude-dir=legacy --exclude=check-guide.sh 2>/dev/null; then
  bad "the guide still cites the retired vendor/vllm/ path — use \$VLLM_SRC"
  grep -rln 'vendor/vllm' "$ROOT" --include='*.md' --include='*.sh' --include='*.py' \
     --exclude-dir=legacy --exclude=check-guide.sh 2>/dev/null | sed 's|^|      |'
  n_stale=1
fi
if grep -rq 'custom_logits_processors' "$ROOT" --include='*.md' \
     --exclude-dir=legacy --exclude=check-guide.sh 2>/dev/null; then
  bad "docs URL /features/custom_logits_processors/ does not exist — it is /features/custom_logitsprocs/"
  n_stale=1
fi
[ "$n_stale" -eq 0 ] && ok "no known-stale references"

# ---------------------------------------------------------------------------
if [ "$CHECK_URLS" = "1" ]; then
  printf '\n\033[1m5. HTTP check of every docs.vllm.ai link\033[0m\n'
  urls="$(grep -rhoE 'https://docs\.vllm\.ai/[A-Za-z0-9_./-]+' "$ROOT" --include='*.md' 2>/dev/null | sed 's/[.,)]*$//' | sort -u)"
  n_urls=0; n_dead=0
  while IFS= read -r u; do
    [ -n "$u" ] || continue
    n_urls=$((n_urls + 1))
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$u" 2>/dev/null)"
    case "$code" in
      200|301|302) ;;
      *) bad "$code  $u"; n_dead=$((n_dead + 1)) ;;
    esac
  done <<< "$urls"
  [ "$n_dead" -eq 0 ] && ok "$n_urls doc URLs return 200"
fi

# ---------------------------------------------------------------------------
if [ "$fail" -eq 0 ]; then
  printf '\n\033[32mguide matches the checkout — nothing to fix\033[0m\n\n'
  exit 0
fi
printf '\n\033[31m%s problem(s) found.\033[0m Upstream moved, or the guide has a typo.\n' "$fail"
printf 'Fix the paths, or update the guide to the new release. See CURRICULUM.md § "Staying current".\n\n'
exit 1
