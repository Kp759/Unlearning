#!/usr/bin/env bash
# MQuAKE answer aliases, one seed, layer 19.
#   1. alias-leak eval of the existing regular run (no retraining): is a
#      forgotten answer still produced through a Wikidata alias?
#   2. (ALIAS_TRAIN=1) rows retrained with --alias-targets behind the SAME
#      router, official MQuAKE eval, alias-leak eval, comparison.
#
#   bash scripts/run_mquake_alias_seed1.sh
#
# Env: SEED (1), REF_TAG (multiseed_regular_v1), ALIAS_TAG (alias_targets_v1),
# ALIAS_TRAIN (1), TRAINING_ROUTE (router), NORM_SCALE (auto), MAX_TRAIN_SECONDS (7200).
set -euo pipefail

SEED="${SEED:-1}"
REF_TAG="${REF_TAG:-multiseed_regular_v1}"
ALIAS_TAG="${ALIAS_TAG:-alias_targets_v1}"
ALIAS_TRAIN="${ALIAS_TRAIN:-1}"
TRAINING_ROUTE="${TRAINING_ROUTE:-router}"
NORM_SCALE="${NORM_SCALE:-auto}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MQUAKE_PATH="$ROOT/data/MQuAKE-CF-3k-v2.json"
test -f "$MQUAKE_PATH" || { echo "Missing $MQUAKE_PATH" >&2; exit 2; }
REF="$ROOT/outputs/mquake_${REF_TAG}/seed${SEED}/L19"
test -f "$REF/linear_global/fact_association_embeddings.pt" \
  || { echo "Missing reference run $REF/linear_global" >&2; exit 2; }
test -f "$REF/router/fact_association_embeddings.pt" \
  || { echo "Missing reference router $REF/router" >&2; exit 2; }
OUT="$ROOT/outputs/mquake_${ALIAS_TAG}/seed${SEED}/L19"
mkdir -p "$OUT"

if [[ ! -f "$REF/linear_global/alias_leak_eval.json" ]]; then
  echo "===== [MQuAKE s$SEED] 1/4 alias-leak eval of the reference run | $(date) ====="
  python -u scripts/evaluate_mquake_alias_leak.py --run-dir "$REF/linear_global" \
    --mquake-path "$MQUAKE_PATH" --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only
fi
[[ "$ALIAS_TRAIN" == "1" ]] || { echo "ALIAS_TRAIN=0: done -> $REF/linear_global/alias_leak_eval.md"; exit 0; }

FINAL="$OUT/linear_global"
if [[ ! -f "$FINAL/fact_association_embeddings.pt" ]]; then
  if [[ -e "$FINAL" ]]; then mv "$FINAL" "$FINAL.incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  echo "===== [MQuAKE s$SEED] 2/4 rows with alias targets (route=$TRAINING_ROUTE) | $(date) ====="
  python -u scripts/train_mquake_linear_router_rows.py \
    --router-dir "$REF/router" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --max-training-seconds "$MAX_TRAIN_SECONDS" \
    --alias-targets --mquake-path "$MQUAKE_PATH" --device cuda --local-files-only
fi

if [[ ! -f "$FINAL/official_mquake_eval.json" ]]; then
  echo "===== [MQuAKE s$SEED] 3/4 official MQuAKE eval | $(date) ====="
  python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mquake-path "$MQUAKE_PATH" --wikidata-dir "$ROOT/data/wikidata" \
    --seed "$SEED" --device cuda --dtype bfloat16 --batch-size 8 --local-files-only \
    --allow-imperfect-direct-routing
fi
if [[ ! -f "$FINAL/alias_leak_eval.json" ]]; then
  python -u scripts/evaluate_mquake_alias_leak.py --run-dir "$FINAL" \
    --mquake-path "$MQUAKE_PATH" --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only
fi

echo "===== [MQuAKE s$SEED] 4/4 comparison | $(date) ====="
python -u scripts/compare_mquake_alias_targets.py --reference "$REF/linear_global" \
  --candidate "$FINAL" --out-prefix "$OUT/comparison"
echo "===== COMPLETE -> $OUT/comparison.md ====="
