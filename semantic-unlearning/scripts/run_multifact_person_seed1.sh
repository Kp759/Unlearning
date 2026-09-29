#!/usr/bin/env bash
# Multi-fact person benchmark, one seed, one layer: SURE with the linear
# classifier (L-BFGS, calibrated cutoff folded into the bias), regular mode.
#
#   bash scripts/run_multifact_person_seed1.sh
#
# Stages (each skipped once done; half-written stage dirs are moved aside):
#   1. data    build the locked split (base-model knowledge filter, seed sampling)
#   2. prep    forget facts + untrained rows at LAYER
#   3. router  fit_linear_router.py (L-BFGS; --decision-rule calibrated_bias; calibration
#              settings copied from the frozen MQuAKE linear router when present)
#   4. rows    train_direct_linear_router_rows.py --dataset multifact (30 updates/fact)
#   5. eval    evaluate_multifact_person.py: base vs SURE on direct / single / multi-fact probes
#
# Env: SEED (1), LAYER (19), TAG (multifact_person_v1), TRAINING_ROUTE (router),
# NORM_SCALE (1), MAX_TRAIN_SECONDS (7200), MOVE_INCOMPLETE (1), BUILD_ARGS (extra
# flags for the builder, e.g. "--min-facts 3").
set -euo pipefail

SEED="${SEED:-1}"
LAYER="${LAYER:-19}"
TAG="${TAG:-multifact_person_v1}"
TRAINING_ROUTE="${TRAINING_ROUTE:-router}"
NORM_SCALE="${NORM_SCALE:-1}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
BUILD_ARGS="${BUILD_ARGS:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"

# Model path from an existing MQuAKE run (paths only).
MODEL_PATH=""
for cand in "$ROOT/outputs/mquake_fact_assoc_router_v2_seed1" \
            "$ROOT/outputs/mquake_multiseed_regular_v1/seed1/L19/prep" \
            "$ROOT/outputs/mquake_fact_assoc_seed1_uniqueassoc_train"; do
  if [[ -f "$cand/association_manifest.json" ]]; then
    MODEL_PATH="$(jq -r '.model_path' "$cand/association_manifest.json")"; break
  fi
done
test -n "$MODEL_PATH" || { echo "No MQuAKE manifest to take the model path from" >&2; exit 2; }

MCF_PATH="$ROOT/data/multi_counterfact.json"
MQ3K="$ROOT/data/MQuAKE-CF-3k-v2.json"
MQCF="$ROOT/data/MQuAKE-CF.json"   # 9k cases: many more people with 3+ facts
test -f "$MCF_PATH" || { echo "Missing $MCF_PATH" >&2; exit 2; }

OUT="$ROOT/outputs/${TAG}/seed${SEED}"
DATA="$OUT/data"
BASE="$OUT/L${LL}"
PREP="$BASE/prep"
ROUTER="$BASE/router"
FINAL="$BASE/linear_global"
mkdir -p "$OUT" "$BASE"

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

if ! stage_ready "$DATA" split_manifest.json; then
  echo "===== [multifact s$SEED] 1/5 DATA: build locked split | $(date) ====="
  if [[ ! -f "$MQCF" ]]; then
    curl -fsSL -o "$MQCF.tmp" \
      "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF.json" \
      && mv "$MQCF.tmp" "$MQCF" \
      || { rm -f "$MQCF.tmp"; echo "WARNING: could not download MQuAKE-CF.json; building from MCF + MQuAKE-CF-3k only (fewer 3-fact people)" >&2; }
  fi
  MQ_ARGS=()
  for f in "$MQ3K" "$MQCF"; do
    if [[ -f "$f" ]]; then MQ_ARGS+=(--mquake-path "$f"); fi
  done
  # shellcheck disable=SC2086
  python -u scripts/build_multifact_person_dataset.py \
    --mcf-path "$MCF_PATH" ${MQ_ARGS[@]+"${MQ_ARGS[@]}"} \
    --model-path "$MODEL_PATH" --output-dir "$DATA" --seed "$SEED" \
    --device cuda --local-files-only $BUILD_ARGS
fi

if ! stage_ready "$PREP" fact_association_embeddings.pt; then
  echo "===== [multifact s$SEED L$LL] 2/5 PREP | $(date) ====="
  python -u scripts/prepare_multifact_association_source.py \
    --model-path "$MODEL_PATH" --training-visible "$DATA/training_visible_forget.json" \
    --split-manifest "$DATA/split_manifest.json" \
    --output-dir "$PREP" --layer "$LAYER" --device cuda --local-files-only
fi

# Calibration settings of the frozen MQuAKE linear router (same record format and
# same-subject retain structure); defaults otherwise.
ROUTER_REPORT="$ROOT/outputs/mquake_linear_2x2_seed1_v24/linear_router_report.json"
ROUTER_ARGS=(--threshold-policy global --decision-rule calibrated_bias)
if [[ -f "$ROUTER_REPORT" ]]; then
  MIN_RECALL="$(jq -r '.router_fit.min_recall // empty' "$ROUTER_REPORT")"
  TARGET_FPR="$(jq -r '.router_fit.target_fpr // empty' "$ROUTER_REPORT")"
  PLACEMENT="$(jq -r '.router_fit.threshold_placement_fraction // empty' "$ROUTER_REPORT")"
  MARGIN="$(jq -r '.router_fit.ambiguity_margin // empty' "$ROUTER_REPORT")"
  if [[ -n "$MIN_RECALL" ]]; then ROUTER_ARGS+=(--min-recall "$MIN_RECALL"); fi
  if [[ -n "$TARGET_FPR" && -z "$MIN_RECALL" ]]; then ROUTER_ARGS+=(--target-fpr "$TARGET_FPR"); fi
  if [[ -n "$PLACEMENT" ]]; then ROUTER_ARGS+=(--threshold-placement-fraction "$PLACEMENT"); fi
  if [[ -n "$MARGIN" ]]; then ROUTER_ARGS+=(--ambiguity-margin "$MARGIN"); fi
  echo "Router settings from $ROUTER_REPORT: ${ROUTER_ARGS[*]}"
else
  ROUTER_ARGS+=(--min-recall 0.98 --threshold-placement-fraction 0.1)
  echo "No frozen MQuAKE router report; using ${ROUTER_ARGS[*]}"
fi

if ! stage_ready "$ROUTER" fact_association_embeddings.pt; then
  echo "===== [multifact s$SEED L$LL] 3/5 ROUTER: linear classifier (L-BFGS, folded bias) | $(date) ====="
  python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$ROUTER" \
    --device cuda --local-files-only "${ROUTER_ARGS[@]}"
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [multifact s$SEED L$LL] 4/5 ROWS (route=$TRAINING_ROUTE) | $(date) ====="
  python -u scripts/train_direct_linear_router_rows.py --dataset multifact \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --max-training-seconds "$MAX_TRAIN_SECONDS" --device cuda --local-files-only
fi

if [[ ! -f "$FINAL/official_multifact_eval.json" ]]; then
  echo "===== [multifact s$SEED L$LL] 5/5 EVAL: base vs SURE | $(date) ====="
  python -u scripts/evaluate_multifact_person.py --run-dir "$FINAL" \
    --wikidata-dir "$ROOT/data/wikidata" --device cuda --dtype bfloat16 --local-files-only
fi

echo "===== [multifact s$SEED L$LL] COMPLETE -> $FINAL/official_multifact_eval.md ====="
