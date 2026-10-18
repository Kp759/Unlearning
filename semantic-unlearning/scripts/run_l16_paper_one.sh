#!/usr/bin/env bash
# Paper table, one (dataset, seed): base | ours (L16 question-side Δe) | genie (same rows,
# perfect routing). Official evaluator of each benchmark; Eff / Gen / Spe / PPL.
#
#   bash scripts/run_l16_paper_one.sh DATASET SEED       DATASET in mcf zsre mquake rwku
#
# Stages (each skipped once its output exists; a half-written stage dir is moved aside):
#   1. router   linear classifier at L19 for this seed (the multiseed_regular_v1 router,
#               reused when present, else fitted via run_<ds>_layer_sweep_one.sh, STOP_AFTER=router)
#   2. v22      MCF only: factorized subject x relation router v2.2 (subject layer 6,
#               relation 19+23+27 picked on validation), reference = the L19 router above.
#               If no layer is deployable the seed falls back to the L19 router (logged in
#               router_choice.json).
#   3. base     official eval of the router with all-zero rows = the unedited model
#   4. ours     rows trained by train_mcf_jlens_feedback_rows.py: write layer 16, query mode
#               (last prompt token + every generated token), output loss + plain logit-lens
#               feedback at L24 + KL keep; ZsRE/MQuAKE/RWKU add teacher-forced gold views
#               (their metrics score every answer token under the gold prefix)
#   5. genie    official eval of the SAME rows with --genie (forget prompts -> own row,
#               everything else -> no row; router not run)
#
# Outputs: outputs/l16_paper_v1/<ds>/seed<S>/{base,ours,genie}/official_<ds>_eval.json
# Env: WRITE_LAYER (16), V22_SUBJECT_LAYERS ("6"), RWKU_GATE (threshold), TRAIN_ARGS (extra),
#      TAG (l16_paper_v1), MOVE_INCOMPLETE (1)
set -euo pipefail

DS="${1:?Usage: run_l16_paper_one.sh DATASET SEED}"
SEED="${2:?Usage: run_l16_paper_one.sh DATASET SEED}"
case "$DS" in mcf|zsre|mquake|rwku) ;; *) echo "DATASET must be mcf|zsre|mquake|rwku" >&2; exit 2;; esac
WRITE_LAYER="${WRITE_LAYER:-16}"
TAG="${TAG:-l16_paper_v1}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/$TAG/$DS/seed$SEED"
mkdir -p "$OUT"
log() { echo "===== [$DS s$SEED] $* | $(date)"; }

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

# ---------------------------------------------------------------- 1. L19 linear router
RTAG=multiseed_regular_v1
GATE="${RWKU_GATE:-threshold}"
[[ "$DS" == rwku && "$GATE" == subject ]] && RTAG=multiseed_subject_v1
LIN="$ROOT/outputs/${DS}_${RTAG}/seed$SEED/L19/router"
if [[ ! -f "$LIN/fact_association_embeddings.pt" ]]; then
  log "1. fit L19 linear router (STOP_AFTER=router)"
  SEED="$SEED" SWEEP_TAG="$RTAG" TRAINING_ROUTE=router NORM_SCALE=1 WITH_DECOMPOSITION=0 \
    STOP_AFTER=router GATE="$GATE" bash "scripts/run_${DS}_layer_sweep_one.sh" 19
fi
test -f "$LIN/fact_association_embeddings.pt" || { echo "no L19 router at $LIN" >&2; exit 2; }
log "1. L19 router: $LIN"

# ---------------------------------------------------------------- 2. MCF: v2.2 router
ROUTER="$LIN"
CHOICE="$OUT/router_choice.json"
if [[ "$DS" == mcf ]]; then
  V22="$OUT/router_v22"
  if [[ ! -f "$V22/summary.json" ]]; then
    log "2. fit factorized router v2.2 (subject layers ${V22_SUBJECT_LAYERS:-6})"
    mkdir -p "$V22"
    read -r -a SL <<< "${V22_SUBJECT_LAYERS:-6}"
    python -u scripts/fit_factorized_router_v2.py --reference-dir "$LIN" --output-dir "$V22" \
      --subject-layers "${SL[@]}" --relation-layers 19 23 24 27 --relation-concat 19,23,27 \
      --hard-rounds 1 --hard-fraction 0.1 --hard-weight 5 \
      --min-recall 0.95 --max-false-fire-same 0.10 --max-false-fire-other 0.01 \
      --near-miss-mode any_word --near-miss-cap 30 \
      --device cuda --local-files-only || echo "v2.2 fit exited non-zero (no deployable layer?)"
  fi
  SEL="$(jq -r '.selected_dir // empty' "$V22/summary.json" 2>/dev/null || true)"
  if [[ -n "$SEL" && -f "$V22/$SEL/fact_association_embeddings.pt" ]]; then
    ROUTER="$V22/$SEL"
    echo "{\"router\": \"factorized_v2.2\", \"dir\": \"$ROUTER\"}" > "$CHOICE"
  else
    echo "{\"router\": \"linear_L19_fallback\", \"dir\": \"$LIN\", \"reason\": \"v2.2 not deployable\"}" > "$CHOICE"
    echo "!!! v2.2 not deployable for seed $SEED: falling back to the L19 linear router" >&2
  fi
else
  echo "{\"router\": \"linear_L19\", \"dir\": \"$LIN\"}" > "$CHOICE"
fi
log "2. router for ours/genie: $ROUTER"

# ---------------------------------------------------------------- evaluator per benchmark
REF_MCF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
evaluate() {  # run_dir [--genie]
  local dir="$1"; shift
  case "$DS" in
    mcf)
      python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
        --run-dir "$dir" --mcf-path "$(jq -r '.mcf_path' "$REF_MCF")" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --local-files-only "$@" ;;
    zsre)
      python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
        --run-dir "$dir" --zsre-path "$ROOT/data/zsre_mend_eval.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --batch-size 8 --local-files-only "$@" ;;
    mquake)
      python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
        --run-dir "$dir" --mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --batch-size 8 --local-files-only \
        --allow-imperfect-direct-routing "$@" ;;
    rwku)
      python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
        --run-dir "$dir" --data-root "$ROOT/data/rwku" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only --no-download \
        --allow-imperfect-direct-routing --out "$dir/official_rwku_eval.json" "$@" ;;
  esac
}
EVAL_JSON="official_${DS}_eval.json"

link_run() {  # src dst: a run dir that points at src's artifact (so evals never touch src)
  mkdir -p "$2"
  for f in fact_association_embeddings.pt association_manifest.json linear_router_report.json \
           training_report.json association_examples.json; do
    [[ -f "$1/$f" && ! -e "$2/$f" ]] && ln -s "$1/$f" "$2/$f"
  done
  return 0
}

# ---------------------------------------------------------------- 3. base
BASE="$OUT/base"
if [[ ! -f "$BASE/$EVAL_JSON" ]]; then
  log "3. base eval (L19 router, all-zero rows = unedited model)"
  link_run "$LIN" "$BASE"
  evaluate "$BASE"
fi

# ---------------------------------------------------------------- 4. ours
OURS="$OUT/ours"
TF=()
[[ "$DS" != mcf ]] && TF=(--teacher-forced-views)
if ! stage_ready "$OURS" fact_association_embeddings.pt; then
  log "4. train ours (write L$WRITE_LAYER, query, logit lens L24, ${TF[*]:-no TF views})"
  python -u scripts/train_mcf_jlens_feedback_rows.py --router-dir "$ROUTER" --output-dir "$OURS" \
    --write-layer "$WRITE_LAYER" --write-mode query --lens-type logit --lens-layer 24 \
    "${TF[@]}" ${TRAIN_ARGS:-} --device cuda --local-files-only
fi
if [[ ! -f "$OURS/$EVAL_JSON" ]]; then
  log "4. ours eval"
  evaluate "$OURS"
fi

# ---------------------------------------------------------------- 5. genie
GENIE="$OUT/genie"
if [[ ! -f "$GENIE/$EVAL_JSON" ]]; then
  log "5. genie eval (same rows, perfect routing)"
  link_run "$OURS" "$GENIE"
  evaluate "$GENIE" --genie
fi
log "COMPLETE -> $OUT"
