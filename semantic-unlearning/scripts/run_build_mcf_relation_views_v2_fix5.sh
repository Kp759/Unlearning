#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${FORGET_DIRECT:?Set FORGET_DIRECT to the sanitized training_visible_forget_direct.json}"
RELATION_V2_CORPUS="${RELATION_V2_CORPUS:-$ROOT/outputs/mcf_relation_views_v2_seed1/relation_views_v2_fix5.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -f "$FORGET_DIRECT" ]]; then
  printf 'Sanitized forget input does not exist: %s\n' "$FORGET_DIRECT" >&2
  exit 2
fi
exec "$PYTHON_BIN" scripts/build_mcf_relation_views_v2_fix5.py \
  --forget-direct "$FORGET_DIRECT" \
  --out "$RELATION_V2_CORPUS" \
  "$@"
