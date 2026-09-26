#!/usr/bin/env bash
# One layer of the MQuAKE layer-wise study, linear classifier only (no Router V2).
#
#   bash scripts/run_mquake_layer_sweep_one.sh LAYER
#
# Stages (each skipped once done; half-written stages are moved aside):
#   1. prep    locked seed-1 forget associations + untrained rows at LAYER
#   2. router  linear classifier at LAYER, same calibration settings as the
#              shipped MQuAKE linear router (read from its report when present)
#   3. rows    TRAINING_ROUTE=router (regular) | oracle (genie)
#   4. official MQuAKE eval (Eff, AtomicGen, retain, PPL) under the linear
#      classifier; direct rewrites it misroutes are recorded, not fatal
#
# Env: SWEEP_TAG (default layer_sweep_linear_v1), TRAINING_ROUTE (default
# oracle), NORM_SCALE (default auto), MAX_TRAIN_SECONDS (default 7200, as the
# shipped MQuAKE run), MOVE_INCOMPLETE (default 1).
set -euo pipefail

LAYER="${1:?Usage: bash scripts/run_mquake_layer_sweep_one.sh LAYER}"
SWEEP_TAG="${SWEEP_TAG:-layer_sweep_linear_v1}"
TRAINING_ROUTE="${TRAINING_ROUTE:-oracle}"
NORM_SCALE="${NORM_SCALE:-auto}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
case "$TRAINING_ROUTE" in router|oracle) ;; *)
  echo "TRAINING_ROUTE must be router or oracle, got '$TRAINING_ROUTE'" >&2; exit 2;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Paths only (model, locked split) from an existing MQuAKE run.
REF=""
for cand in "$ROOT/outputs/mquake_fact_assoc_router_v2_seed1" \
            "$ROOT/outputs/mquake_fact_assoc_seed1_uniqueassoc_train"; do
  if [[ -f "$cand/association_manifest.json" ]]; then REF="$cand"; break; fi
done
test -n "$REF" || { echo "No MQuAKE reference manifest found under outputs/" >&2; exit 2; }
MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
VISIBLE="$(jq -r '.training_visible_path' "$REF/association_manifest.json")"
SPLIT="$(jq -r '.split_manifest_path' "$REF/association_manifest.json")"
MQUAKE_PATH="$ROOT/data/MQuAKE-CF-3k-v2.json"
test -f "$MQUAKE_PATH" || { echo "Missing $MQUAKE_PATH" >&2; exit 2; }

# Router calibration: copy the shipped MQuAKE linear router's settings.
ROUTER_REPORT="$ROOT/outputs/mquake_linear_2x2_seed1_v24/linear_router_report.json"
ROUTER_ARGS=(--threshold-policy global --threshold-placement-fraction 0.1)
if [[ -f "$ROUTER_REPORT" ]]; then
  MIN_RECALL="$(jq -r '.router_fit.min_recall // empty' "$ROUTER_REPORT")"
  TARGET_FPR="$(jq -r '.router_fit.target_fpr // empty' "$ROUTER_REPORT")"
  PLACEMENT="$(jq -r '.router_fit.threshold_placement_fraction // empty' "$ROUTER_REPORT")"
  MARGIN="$(jq -r '.router_fit.ambiguity_margin // empty' "$ROUTER_REPORT")"
  ROUTER_ARGS=(--threshold-policy global)
  [[ -n "$MIN_RECALL" ]] && ROUTER_ARGS+=(--min-recall "$MIN_RECALL")
  [[ -n "$TARGET_FPR" && -z "$MIN_RECALL" ]] && ROUTER_ARGS+=(--target-fpr "$TARGET_FPR")
  [[ -n "$PLACEMENT" ]] && ROUTER_ARGS+=(--threshold-placement-fraction "$PLACEMENT")
  [[ -n "$MARGIN" ]] && ROUTER_ARGS+=(--ambiguity-margin "$MARGIN")
  echo "Router settings from $ROUTER_REPORT: ${ROUTER_ARGS[*]}"
else
  ROUTER_ARGS+=(--min-recall 0.98)
  echo "No shipped MQuAKE router report; using ${ROUTER_ARGS[*]}"
fi

BASE="$ROOT/outputs/mquake_${SWEEP_TAG}/L$(printf '%02d' "$LAYER")"
PREP="$BASE/prep"
ROUTER="$BASE/router"
FINAL="$BASE/linear_global"
mkdir -p "$BASE"

stage_ready() {  # dir, marker
  if [[ -f "$1/$2" ]]; then return 0; fi
  if [[ -e "$1" ]]; then
    if [[ "$MOVE_INCOMPLETE" == "1" ]]; then
      local aside="$1.incomplete_$(date +%Y%m%d_%H%M%S)"
      echo "Moving incomplete stage dir aside: $1 -> $aside" >&2
      mv "$1" "$aside"
    else
      echo "Incomplete stage dir (move it aside first): $1" >&2; exit 2
    fi
  fi
  return 1
}

if ! stage_ready "$PREP" fact_association_embeddings.pt; then
  echo "===== [MQuAKE L$LAYER] 1/4 PREP ====="
  python -u scripts/prepare_mquake_association_source.py \
    --model-path "$MODEL_PATH" --training-visible "$VISIBLE" --split-manifest "$SPLIT" \
    --output-dir "$PREP" --layer "$LAYER" --device cuda --local-files-only
fi

if ! stage_ready "$ROUTER" fact_association_embeddings.pt; then
  echo "===== [MQuAKE L$LAYER] 2/4 FIT linear classifier router ====="
  python -u scripts/fit_linear_router.py \
    --run-dir "$PREP" --output-dir "$ROUTER" \
    --device cuda --local-files-only "${ROUTER_ARGS[@]}"
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [MQuAKE L$LAYER] 3/4 TRAIN rows (route=$TRAINING_ROUTE, norm-scale=$NORM_SCALE) ====="
  python -u scripts/train_mquake_linear_router_rows.py \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --max-training-seconds "$MAX_TRAIN_SECONDS" \
    --device cuda --local-files-only
fi

if [[ ! -f "$FINAL/official_mquake_eval.json" ]]; then
  echo "===== [MQuAKE L$LAYER] 4/4 OFFICIAL MQuAKE eval (linear classifier routing) ====="
  python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mquake-path "$MQUAKE_PATH" \
    --wikidata-dir "$ROOT/data/wikidata" \
    --device cuda --dtype bfloat16 --batch-size 8 --local-files-only \
    --allow-imperfect-direct-routing
fi

echo "===== [MQuAKE L$LAYER] COMPLETE -> $BASE ====="
