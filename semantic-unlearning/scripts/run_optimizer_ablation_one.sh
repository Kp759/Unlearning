#!/usr/bin/env bash
# Router optimizer ablation: SGD vs the shipped L-BFGS, one benchmark.
#
#   bash scripts/run_optimizer_ablation_one.sh {mcf|zsre|mquake}
#
# Reuses the reference run's prep (same facts, rows init, layer) and copies
# every flag of its router fit; L2 and PCA are pinned to the reference CV
# selection, so the optimizer is the only change. Stages (each skipped once
# done; half-written stage dirs are moved aside):
#   1. router_sgd          SGD router                     (fit_linear_router_sgd.py)
#   2. router_lbfgs_rerun  L-BFGS again, same code path   (noise floor)      [NOISE_FLOOR=1]
#   3. swap_sgd            reference rows behind the SGD router + official eval
#   4. full_sgd            rows retrained under the SGD router + official eval
#   5. full_lbfgs_rerun    rows retrained under the rerun router + eval      [NOISE_FLOOR=1]
#   6. comparison.{json,md}
#
# Env: SEED (1), LAYER (19), REF_TAG (multiseed_regular_v1; e.g. multiseed_reworded_v2
# for the ZsRE reworded router), ABL_TAG (optimizer_ablation_v1), NOISE_FLOOR (1),
# SGD_ARGS (extra flags for fit_linear_router_sgd.py, e.g. "--sgd-epochs 600"),
# MOVE_INCOMPLETE (1). Row-training cap, norm scale and training route are read
# from the reference run's manifest.
set -euo pipefail

DATASET="${1:?Usage: bash scripts/run_optimizer_ablation_one.sh mcf|zsre|mquake}"
case "$DATASET" in mcf|zsre|mquake) ;; *) echo "unknown dataset '$DATASET'" >&2; exit 2;; esac
SEED="${SEED:-1}"
LAYER="${LAYER:-19}"
REF_TAG="${REF_TAG:-multiseed_regular_v1}"
ABL_TAG="${ABL_TAG:-optimizer_ablation_v1}"
NOISE_FLOOR="${NOISE_FLOOR:-1}"
SGD_ARGS="${SGD_ARGS:-}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"
REF="$ROOT/outputs/${DATASET}_${REF_TAG}/seed${SEED}/L${LL}"
OUT="$ROOT/outputs/${ABL_TAG}/${DATASET}/seed${SEED}/L${LL}"
EVAL_JSON="official_${DATASET}_eval.json"

for f in "$REF/prep/fact_association_embeddings.pt" \
         "$REF/router/linear_router_report.json" \
         "$REF/router/fact_association_embeddings.pt" \
         "$REF/linear_global/association_manifest.json" \
         "$REF/linear_global/$EVAL_JSON"; do
  test -f "$f" || { echo "Reference run incomplete, missing: $f" >&2; exit 2; }
done
mkdir -p "$OUT"

# Row training exactly as the reference run did it.
REF_MANIFEST="$REF/linear_global/association_manifest.json"
TRAINING_ROUTE="$(jq -r '.training_route // "router"' "$REF_MANIFEST")"
NORM_SCALE="$(jq -r '.layer_representation.norm_scale_argument // "1"' "$REF_MANIFEST")"
MAX_TRAIN_SECONDS="$(jq -r '.plan.max_training_seconds // empty' "$REF_MANIFEST")"
echo "reference: $REF"
echo "row training copied from reference: route=$TRAINING_ROUTE norm-scale=$NORM_SCALE max-seconds=${MAX_TRAIN_SECONDS:-trainer default}"

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

train_rows() {  # router_dir, output_dir
  local cap=()
  [[ -n "$MAX_TRAIN_SECONDS" ]] && cap=(--max-training-seconds "$MAX_TRAIN_SECONDS")
  case "$DATASET" in
    mcf)    python -u scripts/train_mcf_linear_router_rows.py \
              --router-dir "$1" --output-dir "$2" --training-route "$TRAINING_ROUTE" \
              --norm-scale "$NORM_SCALE" ${cap[@]+"${cap[@]}"} --device cuda --local-files-only ;;
    zsre)   python -u scripts/train_direct_linear_router_rows.py --dataset zsre \
              --router-dir "$1" --output-dir "$2" --training-route "$TRAINING_ROUTE" \
              --norm-scale "$NORM_SCALE" ${cap[@]+"${cap[@]}"} --device cuda --local-files-only ;;
    mquake) python -u scripts/train_mquake_linear_router_rows.py \
              --router-dir "$1" --output-dir "$2" --training-route "$TRAINING_ROUTE" \
              --norm-scale "$NORM_SCALE" ${cap[@]+"${cap[@]}"} --device cuda --local-files-only ;;
  esac
}

evaluate() {  # run_dir
  [[ -f "$1/$EVAL_JSON" ]] && return 0
  case "$DATASET" in
    mcf)
      local mcf_path; mcf_path="$(jq -r '.mcf_path' "$REF/router/association_manifest.json")"
      python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
        --run-dir "$1" --mcf-path "$mcf_path" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only ;;
    zsre)
      python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
        --run-dir "$1" --zsre-path "$ROOT/data/zsre_mend_eval.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only ;;
    mquake)
      python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
        --run-dir "$1" --mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only --allow-imperfect-direct-routing ;;
  esac
}

FAILED=()

if ! stage_ready "$OUT/router_sgd" optimizer_ablation.json; then
  echo "===== [$DATASET s$SEED L$LL] 1/6 SGD router | $(date) ====="
  # shellcheck disable=SC2086
  python -u scripts/fit_linear_router_sgd.py --like "$REF/router" \
    --output-dir "$OUT/router_sgd" $SGD_ARGS || FAILED+=(router_sgd)
fi

if [[ "$NOISE_FLOOR" == "1" ]] && ! stage_ready "$OUT/router_lbfgs_rerun" optimizer_ablation.json; then
  echo "===== [$DATASET s$SEED L$LL] 2/6 L-BFGS rerun router (noise floor) | $(date) ====="
  python -u scripts/fit_linear_router_sgd.py --optimizer lbfgs --like "$REF/router" \
    --output-dir "$OUT/router_lbfgs_rerun" || FAILED+=(router_lbfgs_rerun)
fi

if [[ -f "$OUT/router_sgd/optimizer_ablation.json" ]]; then
  if ! stage_ready "$OUT/swap_sgd" fact_association_embeddings.pt; then
    echo "===== [$DATASET s$SEED L$LL] 3/6 swap: reference rows behind SGD router | $(date) ====="
    python -u scripts/swap_router_rows.py --router-dir "$OUT/router_sgd" \
      --rows-from "$REF/linear_global" --output-dir "$OUT/swap_sgd" || FAILED+=(swap_sgd)
  fi
  if [[ -f "$OUT/swap_sgd/fact_association_embeddings.pt" ]]; then
    evaluate "$OUT/swap_sgd" || FAILED+=(swap_sgd_eval)
  fi

  if ! stage_ready "$OUT/full_sgd" fact_association_embeddings.pt; then
    echo "===== [$DATASET s$SEED L$LL] 4/6 rows retrained under SGD router | $(date) ====="
    train_rows "$OUT/router_sgd" "$OUT/full_sgd" || FAILED+=(full_sgd)
  fi
  if [[ -f "$OUT/full_sgd/fact_association_embeddings.pt" ]]; then
    evaluate "$OUT/full_sgd" || FAILED+=(full_sgd_eval)
  fi
fi

if [[ "$NOISE_FLOOR" == "1" && -f "$OUT/router_lbfgs_rerun/optimizer_ablation.json" ]]; then
  if ! stage_ready "$OUT/full_lbfgs_rerun" fact_association_embeddings.pt; then
    echo "===== [$DATASET s$SEED L$LL] 5/6 rows retrained under L-BFGS rerun | $(date) ====="
    train_rows "$OUT/router_lbfgs_rerun" "$OUT/full_lbfgs_rerun" || FAILED+=(full_lbfgs_rerun)
  fi
  if [[ -f "$OUT/full_lbfgs_rerun/fact_association_embeddings.pt" ]]; then
    evaluate "$OUT/full_lbfgs_rerun" || FAILED+=(full_lbfgs_rerun_eval)
  fi
fi

echo "===== [$DATASET s$SEED L$LL] 6/6 comparison | $(date) ====="
CONTROL_ARGS=()
if [[ "$NOISE_FLOOR" == "1" ]]; then
  CONTROL_ARGS=(--control-router "$OUT/router_lbfgs_rerun" --control-run "$OUT/full_lbfgs_rerun")
fi
python -u scripts/compare_router_optimizers.py --dataset "$DATASET" --seed "$SEED" --layer "$LAYER" \
  --reference-router "$REF/router" --reference-run "$REF/linear_global" \
  ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"} \
  --candidate-router "$OUT/router_sgd" --candidate-swap-run "$OUT/swap_sgd" \
  --candidate-run "$OUT/full_sgd" --out-prefix "$OUT/comparison" || FAILED+=(comparison)

echo "===== [$DATASET s$SEED L$LL] done -> $OUT | failed: ${FAILED[*]:-none} ====="
[[ ${#FAILED[@]} -eq 0 ]]
