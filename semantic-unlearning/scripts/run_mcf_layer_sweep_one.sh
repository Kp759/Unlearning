#!/usr/bin/env bash
# One layer of the MCF layer-wise study, linear classifier only (no Router V2).
#
#   bash scripts/run_mcf_layer_sweep_one.sh LAYER
#
# Read layer == write layer == LAYER. Stages:
#   1. prep          data + untrained rows at LAYER            -> L??/prep
#   2. router        fit the linear classifier at LAYER         -> L??/router
#                    (global threshold, recall-first 0.98, placement 0.1:
#                    the frozen seed-1 MCF policy)
#   3. rows          train the residual rows                    -> L??/linear_global
#                    TRAINING_ROUTE=router : routed by the linear classifier (regular)
#                    TRAINING_ROUTE=oracle : genie routing (write-side ceiling)
#   4. official MCF eval of L??/linear_global (linear classifier routing)
#   5. decomposition: linear classifier vs genie on the same rows (optional)
#
# Env overrides: SWEEP_TAG (default layer_sweep_linear_v1), TRAINING_ROUTE
# (router|oracle, default oracle), NORM_SCALE (default auto),
# WITH_DECOMPOSITION (default 1).
set -euo pipefail

LAYER="${1:?Usage: bash scripts/run_mcf_layer_sweep_one.sh LAYER}"
SWEEP_TAG="${SWEEP_TAG:-layer_sweep_linear_v1}"
TRAINING_ROUTE="${TRAINING_ROUTE:-oracle}"
NORM_SCALE="${NORM_SCALE:-auto}"
WITH_DECOMPOSITION="${WITH_DECOMPOSITION:-1}"
case "$TRAINING_ROUTE" in router|oracle) ;; *)
  echo "TRAINING_ROUTE must be router or oracle, got '$TRAINING_ROUTE'" >&2; exit 2;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
test -f "$REF/association_manifest.json" || {
  echo "Missing seed-1 manifest (model/data paths only): $REF/association_manifest.json" >&2
  exit 2
}
MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
MCF_PATH="$(jq -r '.mcf_path' "$REF/association_manifest.json")"

BASE="$ROOT/outputs/mcf_${SWEEP_TAG}/L$(printf '%02d' "$LAYER")"
PREP="$BASE/prep"
ROUTER="$BASE/router"
FINAL="$BASE/linear_global"
mkdir -p "$BASE"

# Each stage is skipped once its output exists, so a stopped run can be
# restarted. Nothing is ever overwritten; a half-written stage must be moved.
stage_ready() {  # dir, marker
  if [[ -f "$1/$2" ]]; then return 0; fi
  test ! -e "$1" || { echo "Incomplete stage dir (move it aside first): $1" >&2; exit 2; }
  return 1
}

if ! stage_ready "$PREP" fact_association_embeddings.pt; then
  echo "===== [L$LAYER] 1/5 PREP data + untrained rows ====="
  python -u scripts/prepare_mcf_association_source.py \
    --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
    --output-dir "$PREP" --layer "$LAYER" --device cuda --local-files-only
fi

if ! stage_ready "$ROUTER" fact_association_embeddings.pt; then
  echo "===== [L$LAYER] 2/5 FIT linear classifier router ====="
  python -u scripts/fit_linear_router.py \
    --run-dir "$PREP" --output-dir "$ROUTER" \
    --device cuda --local-files-only \
    --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [L$LAYER] 3/5 TRAIN rows (route=$TRAINING_ROUTE, norm-scale=$NORM_SCALE) ====="
  python -u scripts/train_mcf_linear_router_rows.py \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --device cuda --local-files-only
fi

if [[ ! -f "$FINAL/official_mcf_eval.json" ]]; then
  echo "===== [L$LAYER] 4/5 OFFICIAL MCF eval (linear classifier routing) ====="
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mcf-path "$MCF_PATH" \
    --wikidata-dir "$ROOT/data/wikidata" \
    --device cuda --dtype bfloat16 --local-files-only
fi

if [[ "$WITH_DECOMPOSITION" == "1" && ! -f "$FINAL/decomposition/router_decomposition.json" ]]; then
  # The script's "v2" arm is whatever router the artifact holds: here the
  # linear classifier. "oracle" is genie routing on the same rows.
  echo "===== [L$LAYER] 5/5 DECOMPOSITION (linear classifier vs genie) ====="
  python -u scripts/evaluate_router_decomposition.py \
    --run-dir "$FINAL" --mcf-path "$MCF_PATH" \
    --output-dir "$FINAL/decomposition" \
    --arms base,v2,oracle --device cuda --local-files-only
fi

echo "===== [L$LAYER] COMPLETE -> $BASE ====="
