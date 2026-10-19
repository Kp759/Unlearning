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
#      TAG (l16_paper_v1), MOVE_INCOMPLETE (1), ORACLE_UNROUTED (1: facts the router never
#      routes are trained with their own row forced, training only),
#      ZSRE_REWORD_TAG (reworded_v2 | reworded_v1 | none: ZsRE router trained on rewordings),
#      BASELINES ("ga gd npo rmu zerounlearn" -> <seed>/<baseline>/), SKIP_OURS (1 = baselines only),
#      BUILD_ROUTERS (1; 0 = fail instead of fitting a missing router, for baseline jobs)
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
  [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "L19 router missing: $LIN (run the main job first)" >&2; exit 2; }
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
    [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "v2.2 router missing: $V22 (run the main job first)" >&2; exit 2; }
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
elif [[ "$DS" == zsre && "${ZSRE_REWORD_TAG:-reworded_v2}" != none ]]; then
  # ZsRE: the L19 router misses paraphrases (genie Gen ~2.5 vs ours ~12.5). Router trained on
  # base-model rewordings of each direct question (official rephrases never read):
  # zsre_router_rewordings.py; v2 keeps a rewording only if the base model's answer stays as
  # likely after it as after the direct question, and drops near-duplicates.
  RWTAG="${ZSRE_REWORD_TAG:-reworded_v2}"
  RW="$ROOT/outputs/zsre_multiseed_$RWTAG/seed$SEED"
  ROUTER="$RW/L19/router"
  if [[ ! -f "$ROUTER/fact_association_embeddings.pt" ]]; then
    [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "reworded ZsRE router missing: $ROUTER (run the main job first)" >&2; exit 2; }
    mkdir -p "$RW"
    if [[ ! -f "$RW/rewordings.json" ]]; then
      GEN=()
      [[ "$RWTAG" == reworded_v2 ]] && GEN=(--consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4)
      log "2. ZsRE rewordings ($RWTAG)"
      python -u scripts/zsre_router_rewordings.py generate \
        --prep-dir "$ROOT/outputs/zsre_multiseed_regular_v1/seed$SEED/L19/prep" \
        --out "$RW/rewordings.json.tmp" --device cuda --local-files-only "${GEN[@]}"
      mv "$RW/rewordings.json.tmp" "$RW/rewordings.json"
    fi
    log "2. ZsRE reworded L19 router ($RWTAG)"
    SEED="$SEED" SWEEP_TAG="multiseed_$RWTAG" REWORDINGS="$RW/rewordings.json" TRAINING_ROUTE=router \
      NORM_SCALE=1 STOP_AFTER=router bash scripts/run_zsre_layer_sweep_one.sh 19
  fi
  test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no reworded router at $ROUTER" >&2; exit 2; }
  echo "{\"router\": \"linear_L19_$RWTAG\", \"dir\": \"$ROUTER\"}" > "$CHOICE"
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

# ---------------------------------------------------------------- 3b. baselines (BASELINES="ga npo ...")
# Same training-visible forget views as ours (the router ours trains on), retain = base
# self-generated text, scored by the same official evaluator (scripts/paper_baselines.py).
for B in ${BASELINES:-}; do
  BD="$OUT/$B"
  if [[ ! -f "$BD/$EVAL_JSON" ]]; then
    log "3b. baseline $B"
    [[ -e "$BD" ]] && mv "$BD" "$BD.incomplete_$(date +%Y%m%d_%H%M%S)"
    link_run "$LIN" "$BD"
    evaluate "$BD" --baseline "$B" --baseline-views-from "$ROUTER" ${BASELINE_ARGS:-}
  fi
done
if [[ "${SKIP_OURS:-0}" == 1 ]]; then log "COMPLETE (baselines only) -> $OUT"; exit 0; fi

# ---------------------------------------------------------------- 4. ours
OURS="$OUT/ours"
GENIE="$OUT/genie"
TF=()
[[ "$DS" != mcf ]] && TF=(--teacher-forced-views)
ORACLE=()
[[ "${ORACLE_UNROUTED:-1}" == 1 ]] && ORACLE=(--oracle-unrouted)
# A finished `ours` trained under another config (router, flags) is moved aside together
# with its genie, so a resubmit with new settings retrains instead of silently reusing it.
if [[ -f "$OURS/fact_association_embeddings.pt" ]]; then
  STALE="$(python - "$OURS" "$ROUTER" "${#TF[@]}" "${#ORACLE[@]}" "$WRITE_LAYER" <<'EOF'
import json, sys, torch
from pathlib import Path
ours, router, tf, oracle, wl = Path(sys.argv[1]), str(Path(sys.argv[2]).resolve()), sys.argv[3] != "0", sys.argv[4] != "0", int(sys.argv[5])
m = json.loads((ours / "association_manifest.json").read_text())
fb = m.get("jlens_feedback") or {}
why = []
if str(Path(m.get("router_copied_from", "")).resolve()) != router: why.append("router")
if bool(fb.get("teacher_forced_views")) != tf: why.append("teacher_forced_views")
if int(m.get("write_layer", wl)) != wl: why.append("write_layer")
if oracle and not fb.get("oracle_unrouted"):
    rows = torch.load(ours / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)["rows"]
    if int((rows.abs().sum(-1) == 0).sum()):   # untrained (zero) rows exist -> the flag matters
        why.append("zero_rows")
print(",".join(why))
EOF
)"
  if [[ -n "$STALE" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    echo "ours is stale ($STALE): moving ours/genie aside (*.superseded_$TS)" >&2
    mv "$OURS" "$OURS.superseded_$TS"
    [[ -e "$GENIE" ]] && mv "$GENIE" "$GENIE.superseded_$TS"
  fi
fi
if ! stage_ready "$OURS" fact_association_embeddings.pt; then
  log "4. train ours (write L$WRITE_LAYER, query, logit lens L24, ${TF[*]:-no TF views} ${ORACLE[*]})"
  python -u scripts/train_mcf_jlens_feedback_rows.py --router-dir "$ROUTER" --output-dir "$OURS" \
    --write-layer "$WRITE_LAYER" --write-mode query --lens-type logit --lens-layer 24 \
    "${TF[@]}" "${ORACLE[@]}" ${TRAIN_ARGS:-} --device cuda --local-files-only
fi
if [[ ! -f "$OURS/$EVAL_JSON" ]]; then
  log "4. ours eval"
  evaluate "$OURS"
fi

# ---------------------------------------------------------------- 5. genie
if [[ ! -f "$GENIE/$EVAL_JSON" ]]; then
  log "5. genie eval (same rows, perfect routing)"
  link_run "$OURS" "$GENIE"
  evaluate "$GENIE" --genie
fi
log "COMPLETE -> $OUT"
