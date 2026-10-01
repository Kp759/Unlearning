#!/usr/bin/env bash
# Eval-DU+ (FT-Mul-Chunk) with SURE, seed 1, one layer:
# linear classifier (L-BFGS, calibrated cutoff folded into the bias), regular mode.
#
#   bash scripts/run_evaldu_plus_seed1.sh
#
# Stages (each skipped once done; half-written stage dirs are moved aside):
#   0. upstream  clone of github.com/wrh14/learning_time_shapes_unlearning (data only)
#   1. finetune  base model on FT-Mul-Chunk (paper recipe)       -> $OUT/ft_mul_chunk
#   2. data      split: forget facts, UL-Mul training prompts, probes
#   3. prep      forget facts + untrained rows at LAYER (fine-tuned model)
#   4. router    fit_linear_router.py (L-BFGS; --decision-rule calibrated_bias)
#   5. rows      train_direct_linear_router_rows.py --dataset evaldu (30 updates/fact)
#   6. eval      evaluate_evaldu_plus.py: fine-tuned model vs SURE, paper's knowledge score
#
# Env: SEED (1), LAYER (19), TAG (evaldu_plus_v1), SPLIT (facts100 | people12),
# UNLEARN_DATA (mul | single), FT_ARGS (extra flags for the fine-tune, e.g. "--epochs 8"),
# TRAINING_ROUTE (router), NORM_SCALE (1), MAX_TRAIN_SECONDS (7200), MOVE_INCOMPLETE (1),
# REWORD (0 | 1): router trained and calibrated on generated rewordings of the
#   forget facts' UL prefixes (scripts/evaldu_router_rewordings.py); own dirs
#   L<LL>_reworded; rewordings shared per split/seed; rows unchanged.
# REWORD_GEN_ARGS ("--consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4").
set -euo pipefail

SEED="${SEED:-1}"
LAYER="${LAYER:-19}"
TAG="${TAG:-evaldu_plus_v1}"
SPLIT="${SPLIT:-facts100}"
UNLEARN_DATA="${UNLEARN_DATA:-mul}"
FT_ARGS="${FT_ARGS:-}"
TRAINING_ROUTE="${TRAINING_ROUTE:-router}"
NORM_SCALE="${NORM_SCALE:-1}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
REWORD="${REWORD:-0}"
REWORD_GEN_ARGS="${REWORD_GEN_ARGS:---consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"

BASE_MODEL=""
for cand in "$ROOT/outputs/mquake_fact_assoc_router_v2_seed1" \
            "$ROOT/outputs/mquake_multiseed_regular_v1/seed1/L19/prep" \
            "$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"; do
  if [[ -f "$cand/association_manifest.json" ]]; then
    BASE_MODEL="$(jq -r '.model_path' "$cand/association_manifest.json")"; break
  fi
done
test -n "$BASE_MODEL" || { echo "No reference manifest to take the base model path from" >&2; exit 2; }

UPSTREAM="$ROOT/data/evaldu_plus_upstream"
OUT="$ROOT/outputs/${TAG}"
FT="$OUT/ft_mul_chunk"
SPLIT_TAG="seed${SEED}"
if [[ "$SPLIT" != "facts100" || "$UNLEARN_DATA" != "mul" ]]; then
  SPLIT_TAG="seed${SEED}_${SPLIT}_ul${UNLEARN_DATA}"
fi
DATA="$OUT/$SPLIT_TAG/data"
BASE="$OUT/$SPLIT_TAG/L${LL}"
REWORDINGS="$OUT/$SPLIT_TAG/rewordings_v1.json"
if [[ "$REWORD" == "1" ]]; then BASE="${BASE}_reworded"; fi
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

if [[ ! -f "$UPSTREAM/synthetic_data/ft_mul_chunk.json" ]]; then
  echo "===== [evaldu] 0/6 clone upstream data | $(date) ====="
  rm -rf "$UPSTREAM.tmp"
  git clone --depth 1 https://github.com/wrh14/learning_time_shapes_unlearning "$UPSTREAM.tmp" \
    && mv "$UPSTREAM.tmp" "$UPSTREAM" \
    || { echo "Could not clone the upstream data. On the login node run:
  git clone --depth 1 https://github.com/wrh14/learning_time_shapes_unlearning $UPSTREAM" >&2; exit 2; }
fi
echo "upstream commit: $(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null || echo unknown) | base model: $BASE_MODEL"

if ! stage_ready "$FT" finetune_report.json; then
  echo "===== [evaldu] 1/6 FINE-TUNE on FT-Mul-Chunk | $(date) ====="
  # shellcheck disable=SC2086
  python -u scripts/finetune_evaldu_plus.py --model-path "$BASE_MODEL" --upstream "$UPSTREAM" \
    --output-dir "$FT" --device cuda --local-files-only $FT_ARGS
fi
jq '{recipe: .recipe, knowledge_before: (.knowledge_before // {} | with_entries(.value |= .mean_knowledge_score)),
     knowledge_after: (.knowledge_after // {} | with_entries(.value |= .mean_knowledge_score))}' \
  "$FT/finetune_report.json" || true

if ! stage_ready "$DATA" split_manifest.json; then
  echo "===== [evaldu $SPLIT_TAG] 2/6 DATA | $(date) ====="
  python -u scripts/build_evaldu_plus_split.py --upstream "$UPSTREAM" --output-dir "$DATA" \
    --split "$SPLIT" --unlearn-data "$UNLEARN_DATA" --seed "$SEED"
fi

if ! stage_ready "$PREP" fact_association_embeddings.pt; then
  echo "===== [evaldu $SPLIT_TAG L$LL] 3/6 PREP | $(date) ====="
  python -u scripts/prepare_evaldu_association_source.py --model-path "$FT" \
    --split-manifest "$DATA/split_manifest.json" --output-dir "$PREP" \
    --layer "$LAYER" --device cuda --local-files-only
fi

if [[ "$REWORD" == "1" && ! -f "$PREP/association_examples.json" ]]; then
  test ! -e "$BASE/router" || { echo "Router already fit without rewordings: $BASE/router (move it aside)" >&2; exit 2; }
  if [[ ! -s "$REWORDINGS" ]]; then
    echo "===== [evaldu $SPLIT_TAG] 3b/6 REWORDINGS of the forget prefixes | $(date) ====="
    # shellcheck disable=SC2086
    python -u scripts/evaldu_router_rewordings.py generate --prep-dir "$PREP" --out "$REWORDINGS" \
      --device cuda --local-files-only $REWORD_GEN_ARGS
  fi
  python -u scripts/evaldu_router_rewordings.py examples --prep-dir "$PREP" --rewordings "$REWORDINGS"
fi

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
  echo "===== [evaldu $SPLIT_TAG L$LL] 4/6 ROUTER: linear classifier (L-BFGS, folded bias, reword=$REWORD) | $(date) ====="
  python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$ROUTER" \
    --device cuda --local-files-only "${ROUTER_ARGS[@]}"
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [evaldu $SPLIT_TAG L$LL] 5/6 ROWS (route=$TRAINING_ROUTE) | $(date) ====="
  python -u scripts/train_direct_linear_router_rows.py --dataset evaldu \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --max-training-seconds "$MAX_TRAIN_SECONDS" --device cuda --local-files-only
fi

if [[ ! -f "$FINAL/official_evaldu_eval.json" ]]; then
  echo "===== [evaldu $SPLIT_TAG L$LL] 6/6 EVAL: fine-tuned model vs SURE | $(date) ====="
  python -u scripts/evaluate_evaldu_plus.py --run-dir "$FINAL" \
    --wikidata-dir "$ROOT/data/wikidata" --device cuda --dtype bfloat16 --local-files-only
fi

echo "===== [evaldu $SPLIT_TAG L$LL] COMPLETE -> $FINAL/official_evaldu_eval.md ====="
