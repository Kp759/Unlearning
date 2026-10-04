#!/usr/bin/env bash
# Shared-vector (compressed) banks on the multiseed sweep's own routers, one task:
#
#   bash scripts/run_compressed_multiseed_one.sh DATASET SEED VALUE_MODE
#     DATASET     mcf | zsre | mquake | rwku
#     VALUE_MODE  full | tied_answer | answer_fixed | lowrank:K | answer_map:r ...
#
# The router is reused unchanged from outputs/<dataset>_<REF_TAG>/seed<S>/L<LL>/router
# (the router behind the shipped multiseed rows), so only the values differ:
#   full          one vector per fact, trained by the same joint trainer (reference)
#   tied_answer   facts with the same answer share ONE vector (+ a scalar each)
#   answer_fixed  no stored vector: one scalar on the answer token's direction
# Shipped row-wise rows for comparison: .../seed<S>/L<LL>/linear_global.
#
# Output: outputs/<OUT_TAG>/<dataset>/seed<S>/L<LL>/<mode>/{fact_association_embeddings.pt,
#   training_report.json, official_<dataset>_eval.json}. Resumable.
# Env: LAYER (19), BATCH_FACTS (8: facts whose gradients form one optimizer step),
# ABSTAIN (unset = each dataset's shipped objective: MCF trains toward " I don't know."
#   (weight 1), ZsRE/MQuAKE only suppress the answer; on = add the abstention term,
#   off = drop it), ABSTAIN_TEXT (" I don't know."), ABSTAIN_WEIGHT (1.0),
# ABSTAIN_EOS (0; 1 = end the abstention with the tokenizer's end token so generation stops
#   right after "I don't know." instead of continuing and naming the answer; needs ABSTAIN=on),
# REF_TAG (per dataset: mcf/mquake/rwku multiseed_regular_v1, zsre multiseed_reworded_v2;
#   RWKU subject gate: REF_TAG=multiseed_subject_v1),
# OUT_TAG (compressed_multiseed[_bf<N>][_idk|_noidk|_idk_eos][_<router>]_v1: _idk/_noidk only
#   when ABSTAIN differs from the dataset's default, _idk_eos whenever ABSTAIN_EOS=1, _<router>
#   (e.g. _subject) when REF_TAG is not the dataset's default; default runs reuse
#   compressed_multiseed_v1),
# MOVE_INCOMPLETE (1).
set -euo pipefail
DATASET="${1:?Usage: run_compressed_multiseed_one.sh DATASET SEED VALUE_MODE}"
SEED="${2:?seed}"
MODE="${3:?value mode}"
case "$DATASET" in mcf|zsre|mquake|rwku) ;; *) echo "DATASET must be mcf, zsre, mquake or rwku" >&2; exit 2;; esac
# RWKU prompts use Llama-3's chat template, which embeds a date: pin it as the RWKU sweep does,
# so training contexts and the evaluator see identical requests.
[[ "$DATASET" == rwku ]] && export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"
LAYER="${LAYER:-19}"
BATCH_FACTS="${BATCH_FACTS:-8}"
DEFAULT_ABSTAIN_TEXT=" I don't know."
ABSTAIN_TEXT="${ABSTAIN_TEXT:-$DEFAULT_ABSTAIN_TEXT}"
ABSTAIN_WEIGHT="${ABSTAIN_WEIGHT:-1.0}"
DEFAULT_ABSTAIN=off; [[ "$DATASET" == mcf ]] && DEFAULT_ABSTAIN=on
ABSTAIN="${ABSTAIN:-$DEFAULT_ABSTAIN}"
case "$ABSTAIN" in on|off) ;; *) echo "ABSTAIN must be on or off" >&2; exit 2;; esac
ABSTAIN_EOS="${ABSTAIN_EOS:-0}"
case "$ABSTAIN_EOS" in 0|1) ;; *) echo "ABSTAIN_EOS must be 0 or 1" >&2; exit 2;; esac
[[ "$ABSTAIN_EOS" == 1 && "$ABSTAIN" != on ]] && { echo "ABSTAIN_EOS=1 needs ABSTAIN=on" >&2; exit 2; }
DEFAULT_REF_TAG=multiseed_regular_v1; [[ "$DATASET" == zsre ]] && DEFAULT_REF_TAG=multiseed_reworded_v2
REF_TAG="${REF_TAG:-$DEFAULT_REF_TAG}"
TAG="compressed_multiseed"
[[ "$BATCH_FACTS" != 8 ]] && TAG="${TAG}_bf${BATCH_FACTS}"
if [[ "$ABSTAIN_EOS" == 1 ]]; then
  TAG="${TAG}_idk_eos"
elif [[ "$ABSTAIN" != "$DEFAULT_ABSTAIN" ]]; then
  [[ "$ABSTAIN" == on ]] && TAG="${TAG}_idk" || TAG="${TAG}_noidk"
fi
if [[ "$REF_TAG" != "$DEFAULT_REF_TAG" ]]; then
  ROUTER_NAME="${REF_TAG#multiseed_}"; TAG="${TAG}_${ROUTER_NAME%_v[0-9]*}"
fi
OUT_TAG="${OUT_TAG:-${TAG}_v1}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"
REF="$ROOT/outputs/${DATASET}_${REF_TAG}/seed${SEED}/L${LL}"
ROUTER="$REF/router"
OUT="$ROOT/outputs/${OUT_TAG}/${DATASET}/seed${SEED}/L${LL}/${MODE//:/_}"
EVAL_JSON="official_${DATASET}_eval.json"
for f in "$ROUTER/fact_association_embeddings.pt" "$ROUTER/association_manifest.json"; do
  test -f "$f" || { echo "Missing router from the multiseed sweep: $f" >&2; exit 2; }
done
echo "[$DATASET s$SEED L$LL $MODE batch=$BATCH_FACTS abstain=$ABSTAIN eos=$ABSTAIN_EOS] router $ROUTER -> $OUT"

if [[ ! -f "$OUT/training_report.json" ]]; then
  if [[ -e "$OUT" ]]; then
    if [[ "$MOVE_INCOMPLETE" == "1" ]]; then mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
    else echo "Incomplete output dir: $OUT" >&2; exit 2; fi
  fi
  mkdir -p "$(dirname "$OUT")"
  echo "===== [$DATASET s$SEED L$LL $MODE] 1/2 TRAIN compressed values in the loop ====="
  if [[ "$DATASET" == mcf ]]; then
    # MCF's objective already has the " I don't know." term (PLAN unknown_weight 1).
    UNKNOWN_WEIGHT="$ABSTAIN_WEIGHT"; [[ "$ABSTAIN" == off ]] && UNKNOWN_WEIGHT=0
    EOS_ARGS=(); [[ "$ABSTAIN_EOS" == 1 ]] && EOS_ARGS=(--unknown-eos)
    python -u scripts/train_mcf_compressed_bank.py --router-dir "$ROUTER" --output-dir "$OUT" \
      --value-mode "$MODE" --training-route router --batch-facts "$BATCH_FACTS" \
      --unknown-weight "$UNKNOWN_WEIGHT" --unknown-completion "$ABSTAIN_TEXT" \
      ${EOS_ARGS[@]+"${EOS_ARGS[@]}"} --device cuda --local-files-only
  else
    ABSTAIN_ARGS=()
    [[ "$ABSTAIN" == on ]] && ABSTAIN_ARGS=(--abstain-text "$ABSTAIN_TEXT" --abstain-weight "$ABSTAIN_WEIGHT")
    [[ "$ABSTAIN_EOS" == 1 ]] && ABSTAIN_ARGS+=(--abstain-eos)
    python -u scripts/train_direct_compressed_bank.py --dataset "$DATASET" --router-dir "$ROUTER" \
      --output-dir "$OUT" --value-mode "$MODE" --training-route router --batch-facts "$BATCH_FACTS" \
      ${ABSTAIN_ARGS[@]+"${ABSTAIN_ARGS[@]}"} --device cuda --local-files-only
  fi
fi

if [[ ! -f "$OUT/$EVAL_JSON" ]]; then
  echo "===== [$DATASET s$SEED L$LL $MODE] 2/2 OFFICIAL $DATASET eval ====="
  case "$DATASET" in
    mcf)
      MCF_PATH="$(jq -r '.mcf_path' "$ROUTER/association_manifest.json")"
      python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
        --run-dir "$OUT" --mcf-path "$MCF_PATH" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only ;;
    zsre)
      python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
        --run-dir "$OUT" --zsre-path "$ROOT/data/zsre_mend_eval.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only ;;
    mquake)
      python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
        --run-dir "$OUT" --mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only --allow-imperfect-direct-routing ;;
    rwku)
      # Generation-based RWKU eval; per-probe outputs land in details.*.prediction.
      python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
        --run-dir "$OUT" --data-root "$ROOT/data/rwku" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only --no-download \
        --allow-imperfect-direct-routing --out "$OUT/$EVAL_JSON" ;;
  esac
fi
echo "===== [$DATASET s$SEED L$LL $MODE] COMPLETE -> $OUT ====="
