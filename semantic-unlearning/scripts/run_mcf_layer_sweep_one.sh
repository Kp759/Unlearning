#!/usr/bin/env bash
# One layer of the MCF layer-wise study: train -> linear router -> official eval.
#
#   bash scripts/run_mcf_layer_sweep_one.sh LAYER
#
# Read layer == write layer == LAYER (the SURE method moved as a whole).
# Rows are trained with oracle routing and a norm-matched step size, so the
# comparison across layers is not decided by the V2 gate's quality at that
# layer or by the residual-stream norm growing with depth. The learned linear
# router is then refit at LAYER with the frozen seed-1 MCF policy
# (global threshold, recall-first 0.98, placement 0.1), and the official MCF
# evaluator scores the result exactly as for layer 19.
#
# Env overrides: SWEEP_TAG (default layer_sweep_v1), NORM_SCALE (default auto),
# TRAINING_ROUTE (default oracle), WITH_DECOMPOSITION (default 1).
set -euo pipefail

LAYER="${1:?Usage: bash scripts/run_mcf_layer_sweep_one.sh LAYER}"
SWEEP_TAG="${SWEEP_TAG:-layer_sweep_v1}"
NORM_SCALE="${NORM_SCALE:-auto}"
TRAINING_ROUTE="${TRAINING_ROUTE:-oracle}"
WITH_DECOMPOSITION="${WITH_DECOMPOSITION:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
test -f "$REF/association_manifest.json" || {
  echo "Missing seed-1 manifest (model/data paths): $REF/association_manifest.json" >&2
  exit 2
}
MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
MCF_PATH="$(jq -r '.mcf_path' "$REF/association_manifest.json")"

BASE="$ROOT/outputs/mcf_${SWEEP_TAG}/L$(printf '%02d' "$LAYER")"
TRAIN="$BASE/rows"
ROUTER="$BASE/linear_global"
mkdir -p "$BASE"

# Each stage is skipped if its output already exists, so a timed-out job can
# be resubmitted without redoing finished work. Nothing is ever overwritten.
if [[ ! -f "$TRAIN/fact_association_embeddings.pt" ]]; then
  test ! -e "$TRAIN" || {
    echo "Incomplete training dir exists (move it aside first): $TRAIN" >&2
    exit 2
  }
  echo "===== [L$LAYER] TRAIN rows (route=$TRAINING_ROUTE, norm-scale=$NORM_SCALE) ====="
  python -u scripts/run_mcf_fact_association_router_v2_seed1.py \
    --model-path "$MODEL_PATH" \
    --mcf-path "$MCF_PATH" \
    --output-dir "$TRAIN" \
    --device cuda \
    --local-files-only \
    --layer "$LAYER" \
    --training-route "$TRAINING_ROUTE" \
    --norm-scale "$NORM_SCALE"
fi

if [[ ! -f "$ROUTER/fact_association_embeddings.pt" ]]; then
  test ! -e "$ROUTER" || {
    echo "Incomplete router dir exists (move it aside first): $ROUTER" >&2
    exit 2
  }
  echo "===== [L$LAYER] FIT linear router (global, recall-first) ====="
  python -u scripts/fit_linear_router.py \
    --run-dir "$TRAIN" \
    --output-dir "$ROUTER" \
    --device cuda \
    --local-files-only \
    --threshold-policy global \
    --min-recall 0.98 \
    --threshold-placement-fraction 0.1
fi

if [[ ! -f "$ROUTER/official_mcf_eval.json" ]]; then
  echo "===== [L$LAYER] OFFICIAL MCF eval ====="
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$ROUTER" \
    --mcf-path "$MCF_PATH" \
    --wikidata-dir "$ROOT/data/wikidata" \
    --device cuda \
    --dtype bfloat16 \
    --local-files-only
fi

if [[ "$WITH_DECOMPOSITION" == "1" && ! -f "$ROUTER/decomposition/router_decomposition.json" ]]; then
  # Read-vs-write attribution: the oracle arm is the write layer's ceiling,
  # the learned-router arm ("v2" slot) is the full method.
  echo "===== [L$LAYER] ROUTER DECOMPOSITION (router vs oracle) ====="
  python -u scripts/evaluate_router_decomposition.py \
    --run-dir "$ROUTER" \
    --mcf-path "$MCF_PATH" \
    --output-dir "$ROUTER/decomposition" \
    --arms base,v2,oracle \
    --device cuda \
    --local-files-only
fi

echo "===== [L$LAYER] COMPLETE -> $BASE ====="
