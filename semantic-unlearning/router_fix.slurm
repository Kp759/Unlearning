#!/bin/bash
# Router fixes, measured: seed 1, Llama-3.2-3B, block-19 router, MCF / zsRE / MQuAKE (tasks 0 / 1 / 2).
# VARIANT picks the fix; every variant writes its own router / diagnostics, nothing existing is
# touched, and the baseline is outputs/router_diag_ours_v1 (the paper's router).
#
#   control         the router now at outputs/<ds>_multiseed_ours_v1, crop off, under today's
#                   code: the baseline the crop variants are compared with
#   crop_sentence   the paper's router as is, routed with route_crop=sentence: when the full
#                   prompt fires nothing although a subject is eligible, the sentence naming the
#                   subject is routed (linear_router.subject_crops). Inference only, ~15 min.
#   crop_subject    as crop_sentence, plus the text from the subject on when two or more words
#                   follow it (route_crop=sentence_subject; catches prompts with no sentence break)
#   floor_crop      refit with the rewrite coverage floor WITHOUT the judge fallback (8 kept
#                   rewrites per fact; a third round of plain questions that must pass the strict
#                   answer-transfer check) and route_crop=sentence stored in the artifact. Two
#                   diagnostics: router_diag_ours_floor_v1 (crop off: the floor alone) and
#                   router_diag_ours_floor_crop_v1 (both). ~1 h.
#   rejected        the measured and rejected run (distractor copies + floor with the judge
#                   fallback) -> router_diag_ours_fix_v1, kept so it can be reproduced
#
#   for v in crop_sentence crop_subject floor_crop; do sbatch --export=ALL,VARIANT=$v router_fix.slurm; done
#   # when all are done:
#   python scripts/router_fix_compare.py
#SBATCH --job-name=router_fix
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/router_fix_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/router_fix_%A_%a.err

set -uo pipefail
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
  set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$ENV_DIR"
fi
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available()" || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

DATASETS=(mcf zsre mquake)
DS="${DATASETS[${SLURM_ARRAY_TASK_ID:?run as an array job}]:-}"
[[ -n "$DS" ]] || { echo "array index out of range (0-2)" >&2; exit 2; }
SEED="${SEED:-1}"
VARIANT="${VARIANT:?set VARIANT=control|crop_sentence|crop_subject|floor_crop|rejected}"
FLOOR="${FLOOR:-8}"
PAPER_ROUTER="outputs/${DS}_multiseed_ours_v1/seed$SEED/L19/router"
# the paper's router recipe (ours: 20 / 20, fuzzy subject, no PCA)
OURS_FIT="--pca-dims 0 --subject-match fuzzy --positives-per-fact 20 --negatives-per-fact 20 \
--max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5"
echo "router fix | $VARIANT | $DS seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

fit() {   # fit <tag> <extra fit args> <rewrite args>: the block-19 router, as the paper's run
  local tag="$1" router="outputs/${DS}_$1/seed$SEED/L19/router"
  if [[ ! -f "$router/fact_association_embeddings.pt" ]]; then
    echo "===== fit router ($tag) | $(date)"
    FIT_EXTRA_ARGS="$OURS_FIT $2" ROUTER_REWRITES=1 REWRITE_ARGS="$3" \
      SEED="$SEED" SWEEP_TAG="$tag" TRAINING_ROUTE=router NORM_SCALE=1 WITH_DECOMPOSITION=0 \
      STOP_AFTER=router GATE=threshold bash "scripts/run_${DS}_layer_sweep_one.sh" 19 || exit $?
  fi
  test -f "$router/fact_association_embeddings.pt" || { echo "no router at $router" >&2; exit 2; }
  jq -c '{rewrites: (.router_rewrites // null)}' "$router/linear_router_report.json" 2>/dev/null || true
}

diagnose() {   # diagnose <router dir> <out root> [--route-crop MODE]
  local out="$2/$DS/seed$SEED"
  echo "===== diagnostics -> $out | $(date)"
  python -u scripts/cache_router_states.py --dataset "$DS" --seed "$SEED" --out-dir "$out" \
    --router-recipe ours --router-dir "$1" --layers 16,19 --time-layers "" \
    --device cuda --local-files-only "${@:3}" || exit $?
  python -u scripts/router_diag_report.py --cache-dir "$out" || exit $?
  jq -c '{route_crop, parity: (.hook_parity | map_values(.route_mismatches)),
          crop_fired: (.sections | map_values(.crop_fired))}' "$out/manifest.json"
}

case "$VARIANT" in
  control)
    test -f "$PAPER_ROUTER/fact_association_embeddings.pt" || { echo "no paper router at $PAPER_ROUTER" >&2; exit 2; }
    diagnose "$PAPER_ROUTER" outputs/router_diag_ours_control_v1 --route-crop none ;;
  crop_sentence)
    test -f "$PAPER_ROUTER/fact_association_embeddings.pt" || { echo "no paper router at $PAPER_ROUTER" >&2; exit 2; }
    diagnose "$PAPER_ROUTER" outputs/router_diag_ours_crop_sentence_v1 --route-crop sentence ;;
  crop_subject)
    test -f "$PAPER_ROUTER/fact_association_embeddings.pt" || { echo "no paper router at $PAPER_ROUTER" >&2; exit 2; }
    diagnose "$PAPER_ROUTER" outputs/router_diag_ours_crop_subject_v1 --route-crop sentence_subject ;;
  floor_crop)
    fit multiseed_ours_floor_v1 "--route-crop sentence" "--floor $FLOOR --floor-fallback none"
    R="outputs/${DS}_multiseed_ours_floor_v1/seed$SEED/L19/router"
    diagnose "$R" outputs/router_diag_ours_floor_v1 --route-crop none
    diagnose "$R" outputs/router_diag_ours_floor_crop_v1 ;;
  rejected)
    DISTRACTORS="${DISTRACTORS:-data/distractor_sentences.json}"
    WIKI="${WIKI:-/project/yl258/kp759/datasets/wikipedia_20231101_100k}"
    ( flock 9
      [[ -s "$DISTRACTORS" ]] || { python -u scripts/build_distractor_sentences.py --source "$WIKI" \
        --out "$DISTRACTORS.tmp" && mv "$DISTRACTORS.tmp" "$DISTRACTORS"; }
    ) 9>"$DISTRACTORS.lock"
    fit multiseed_ours_fix_v1 "--distractors $DISTRACTORS --distractors-per-fact 4" "--floor $FLOOR --floor-fallback judge"
    diagnose "outputs/${DS}_multiseed_ours_fix_v1/seed$SEED/L19/router" outputs/router_diag_ours_fix_v1 ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
echo "Finish $VARIANT $DS seed $SEED: $(date)"
