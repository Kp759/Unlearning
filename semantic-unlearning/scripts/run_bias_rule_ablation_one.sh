#!/usr/bin/env bash
# Bias-rule ablation: plain logistic regression (stage-1 bias, fire at p >= 0.5)
# vs the shipped calibrated cutoff folded into the bias, for one benchmark and
# one router optimizer.
#
#   OPTIMIZER=lbfgs bash scripts/run_bias_rule_ablation_one.sh mcf
#   OPTIMIZER=sgd   bash scripts/run_bias_rule_ablation_one.sh zsre
#
# Both arms share the router's weights; only the bias rule differs.
#   folded      calibrated router + rows trained under it + official eval
#               lbfgs: the reference run itself (outputs/<ds>_<REF_TAG>/seed1/L19)
#               sgd:   reused from the optimizer ablation (router_sgd / full_sgd) when
#                      finished there and SGD_ARGS is empty; otherwise fit + trained here
#   router_raw  same heads, stage-1 bias, cutoff 0 (make_raw_logistic_router.py; both
#               rules scored on every router prompt)
#   raw_swap    folded rows behind the raw router + official eval (eval-time effect)
#   raw         rows retrained under the raw router + official eval (full method)
#   comparison.{json,md}
#
# Env: OPTIMIZER (lbfgs|sgd, required), SEED (1), LAYER (19), REF_TAG
# (multiseed_regular_v1), ABL_TAG (bias_rule_ablation_v1), SGD_ARGS (extra flags for
# fit_linear_router_sgd.py), REUSE_SGD (1), SGD_REUSE_TAG (optimizer_ablation_v1),
# MOVE_INCOMPLETE (1). Row training (route, norm scale, time cap) is copied from the
# reference run's manifest. Resumable: finished stages are skipped.
set -euo pipefail

DATASET="${1:?Usage: OPTIMIZER=lbfgs|sgd bash scripts/run_bias_rule_ablation_one.sh mcf|zsre|mquake}"
case "$DATASET" in mcf|zsre|mquake) ;; *) echo "unknown dataset '$DATASET'" >&2; exit 2;; esac
OPTIMIZER="${OPTIMIZER:-}"
case "$OPTIMIZER" in lbfgs|sgd) ;; *) echo "OPTIMIZER must be lbfgs or sgd, got '$OPTIMIZER'" >&2; exit 2;; esac
SEED="${SEED:-1}"
LAYER="${LAYER:-19}"
REF_TAG="${REF_TAG:-multiseed_regular_v1}"
ABL_TAG="${ABL_TAG:-bias_rule_ablation_v1}"
SGD_ARGS="${SGD_ARGS:-}"
REUSE_SGD="${REUSE_SGD:-1}"
SGD_REUSE_TAG="${SGD_REUSE_TAG:-optimizer_ablation_v1}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
# RETRAIN_RAW=0: no row retraining. Only the raw router + the existing folded rows
# behind it (raw_swap) + official eval, compared with the folded run (~1 h per task).
RETRAIN_RAW="${RETRAIN_RAW:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"
REF="$ROOT/outputs/${DATASET}_${REF_TAG}/seed${SEED}/L${LL}"
OUT="$ROOT/outputs/${ABL_TAG}/${OPTIMIZER}/${DATASET}/seed${SEED}/L${LL}"
EVAL_JSON="official_${DATASET}_eval.json"

for f in "$REF/prep/fact_association_embeddings.pt" \
         "$REF/router/linear_router_report.json" \
         "$REF/router/fact_association_embeddings.pt" \
         "$REF/linear_global/association_manifest.json" \
         "$REF/linear_global/$EVAL_JSON"; do
  test -f "$f" || { echo "Reference run incomplete, missing: $f" >&2; exit 2; }
done
mkdir -p "$OUT"

REF_MANIFEST="$REF/linear_global/association_manifest.json"
TRAINING_ROUTE="$(jq -r '.training_route // "router"' "$REF_MANIFEST")"
NORM_SCALE="$(jq -r '.layer_representation.norm_scale_argument // "1"' "$REF_MANIFEST")"
MAX_TRAIN_SECONDS="$(jq -r '.plan.max_training_seconds // empty' "$REF_MANIFEST")"
echo "reference: $REF | optimizer: $OPTIMIZER"
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
  if [[ -n "$MAX_TRAIN_SECONDS" ]]; then cap=(--max-training-seconds "$MAX_TRAIN_SECONDS"); fi
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
  if [[ -f "$1/$EVAL_JSON" ]]; then return 0; fi
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

# ---- folded arm (the shipped rule) -----------------------------------------
if [[ "$OPTIMIZER" == "lbfgs" ]]; then
  FOLDED_ROUTER="$REF/router"
  FOLDED_RUN="$REF/linear_global"
  echo "folded arm = reference L-BFGS run: $FOLDED_RUN"
else
  SGD_FROM="$ROOT/outputs/${SGD_REUSE_TAG}/${DATASET}/seed${SEED}/L${LL}"
  REUSABLE=0
  if [[ "$REUSE_SGD" == "1" && -z "$SGD_ARGS" && -f "$SGD_FROM/router_sgd/optimizer_ablation.json" ]]; then
    # Only if that SGD router was fit from this same reference router.
    LIKE="$(jq -r '.like // empty' "$SGD_FROM/router_sgd/optimizer_ablation.json")"
    if [[ -n "$LIKE" && "$(realpath "$LIKE")" == "$(realpath "$REF/router")" ]]; then REUSABLE=1
    else echo "not reusing $SGD_FROM/router_sgd: fit from '$LIKE', not $REF/router" >&2; fi
  fi
  if [[ "$REUSABLE" == "1" ]]; then
    FOLDED_ROUTER="$SGD_FROM/router_sgd"
    echo "folded SGD router reused from the optimizer ablation: $FOLDED_ROUTER"
  else
    if ! stage_ready "$OUT/router_folded" optimizer_ablation.json; then
      echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] SGD router (folded) | $(date) ====="
      # shellcheck disable=SC2086
      python -u scripts/fit_linear_router_sgd.py --like "$REF/router" \
        --output-dir "$OUT/router_folded" $SGD_ARGS || FAILED+=(router_folded)
    fi
    FOLDED_ROUTER="$OUT/router_folded"
  fi
  if [[ "$FOLDED_ROUTER" == "${SGD_FROM}/router_sgd" && -f "$SGD_FROM/full_sgd/$EVAL_JSON" ]]; then
    FOLDED_RUN="$SGD_FROM/full_sgd"
    echo "folded SGD rows reused from the optimizer ablation: $FOLDED_RUN"
  else
    FOLDED_RUN="$OUT/folded"
    if [[ -f "$FOLDED_ROUTER/fact_association_embeddings.pt" ]]; then
      if ! stage_ready "$OUT/folded" fact_association_embeddings.pt; then
        echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] rows under folded SGD router | $(date) ====="
        train_rows "$FOLDED_ROUTER" "$OUT/folded" || FAILED+=(folded)
      fi
      if [[ -f "$OUT/folded/fact_association_embeddings.pt" ]]; then
        evaluate "$OUT/folded" || FAILED+=(folded_eval)
      fi
    fi
  fi
fi
printf '{"optimizer": "%s", "dataset": "%s", "folded_router": "%s", "folded_run": "%s"}\n' \
  "$OPTIMIZER" "$DATASET" "$FOLDED_ROUTER" "$FOLDED_RUN" > "$OUT/arms.json"

# ---- raw arm (plain logistic regression, p >= 0.5 on the stage-1 bias) -------
if [[ -f "$FOLDED_ROUTER/fact_association_embeddings.pt" ]]; then
  if ! stage_ready "$OUT/router_raw" fact_association_embeddings.pt; then
    echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] raw logistic router + both rules on router prompts | $(date) ====="
    python -u scripts/make_raw_logistic_router.py --router-dir "$FOLDED_ROUTER" \
      --output-dir "$OUT/router_raw" --device cuda --local-files-only || FAILED+=(router_raw)
  fi
fi

if [[ -f "$OUT/router_raw/fact_association_embeddings.pt" ]]; then
  if [[ -f "$FOLDED_RUN/fact_association_embeddings.pt" ]]; then
    if ! stage_ready "$OUT/raw_swap" fact_association_embeddings.pt; then
      echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] swap: folded rows behind raw router | $(date) ====="
      python -u scripts/swap_router_rows.py --router-dir "$OUT/router_raw" \
        --rows-from "$FOLDED_RUN" --output-dir "$OUT/raw_swap" || FAILED+=(raw_swap)
    fi
    if [[ -f "$OUT/raw_swap/fact_association_embeddings.pt" ]]; then
      evaluate "$OUT/raw_swap" || FAILED+=(raw_swap_eval)
    fi
  fi
  if [[ "$RETRAIN_RAW" == "1" ]]; then
    if ! stage_ready "$OUT/raw" fact_association_embeddings.pt; then
      echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] rows retrained under raw router | $(date) ====="
      train_rows "$OUT/router_raw" "$OUT/raw" || FAILED+=(raw)
    fi
    if [[ -f "$OUT/raw/fact_association_embeddings.pt" ]]; then
      evaluate "$OUT/raw" || FAILED+=(raw_eval)
    fi
  fi
fi
RAW_RUN_ARGS=()
if [[ "$RETRAIN_RAW" == "1" ]]; then RAW_RUN_ARGS=(--raw-run "$OUT/raw"); fi

echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] comparison | $(date) ====="
python -u scripts/compare_bias_rules.py --dataset "$DATASET" --optimizer "$OPTIMIZER" \
  --seed "$SEED" --layer "$LAYER" \
  --folded-router "$FOLDED_ROUTER" --folded-run "$FOLDED_RUN" \
  --raw-router "$OUT/router_raw" --raw-swap-run "$OUT/raw_swap" ${RAW_RUN_ARGS[@]+"${RAW_RUN_ARGS[@]}"} \
  --out-prefix "$OUT/comparison" || FAILED+=(comparison)

echo "===== [$OPTIMIZER $DATASET s$SEED L$LL] done -> $OUT | failed: ${FAILED[*]:-none} ====="
[[ ${#FAILED[@]} -eq 0 ]]
