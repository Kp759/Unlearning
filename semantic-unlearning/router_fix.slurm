#!/bin/bash
# Router fixes, measured: seed 1, Llama-3.2-3B, MCF / zsRE / MQuAKE (tasks 0 / 1 / 2).
# Refits the paper's block-19 router (ours recipe: 20 / 20, fuzzy subject, no PCA) with the two
# fixes, then runs the router diagnostics on it, next to the existing ones:
#   fix 1  distractor sentences: DISTRACTORS_PER_FACT (4) positives and as many negatives per
#          fact copied with an unrelated Wikipedia sentence in front (build_distractor_sentences.py,
#          fit_linear_router.py --distractors); targets MCF's paraphrases, which prepend one
#   fix 2  rewrite coverage floor: facts with fewer than FLOOR (8) kept rewrites after two rounds
#          get a third round of plain questions, and facts the answer-transfer check cannot serve
#          use the yes/no judge (build_router_rewrites.py --floor)
# Nothing existing is touched: routers -> outputs/<ds>_multiseed_ours_fix_v1, diagnostics ->
# outputs/router_diag_ours_fix_v1/<ds>/seed1. Compare with outputs/router_diag_ours_v1:
#   sbatch router_fix.slurm
#   python scripts/router_diag_explain.py --root outputs/router_diag_ours_fix_v1
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
TAG="${TAG:-multiseed_ours_fix_v1}"
FLOOR="${FLOOR:-8}"
DISTRACTORS_PER_FACT="${DISTRACTORS_PER_FACT:-4}"
DISTRACTORS="${DISTRACTORS:-data/distractor_sentences.json}"
WIKI="${WIKI:-/project/yl258/kp759/datasets/wikipedia_20231101_100k}"
OUT="outputs/router_diag_ours_fix_v1/$DS/seed$SEED"
ROUTER="outputs/${DS}_${TAG}/seed$SEED/L19/router"
echo "router fix | $DS seed $SEED | floor $FLOOR | distractors $DISTRACTORS_PER_FACT/fact | job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

# 0. distractor sentences, built once (the three tasks share the file; flock serialises them)
(
  flock 9
  if [[ ! -s "$DISTRACTORS" ]]; then
    echo "===== building $DISTRACTORS from $WIKI"
    python -u scripts/build_distractor_sentences.py --source "$WIKI" --out "$DISTRACTORS.tmp" && mv "$DISTRACTORS.tmp" "$DISTRACTORS"
  fi
) 9>"$DISTRACTORS.lock"
[[ -s "$DISTRACTORS" ]] || { echo "no distractor sentences at $DISTRACTORS" >&2; exit 2; }

# 1. router with both fixes (the paper's ours recipe otherwise)
if [[ ! -f "$ROUTER/fact_association_embeddings.pt" ]]; then
  echo "===== 1. fit router ($TAG) | $(date)"
  export FIT_EXTRA_ARGS="--pca-dims 0 --subject-match fuzzy --positives-per-fact 20 --negatives-per-fact 20 \
--max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5 \
--distractors $DISTRACTORS --distractors-per-fact $DISTRACTORS_PER_FACT"
  export ROUTER_REWRITES=1 REWRITE_ARGS="--floor $FLOOR"
  SEED="$SEED" SWEEP_TAG="$TAG" TRAINING_ROUTE=router NORM_SCALE=1 WITH_DECOMPOSITION=0 STOP_AFTER=router \
    GATE=threshold bash "scripts/run_${DS}_layer_sweep_one.sh" 19 || exit $?
fi
test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no router at $ROUTER" >&2; exit 2; }
jq -c '{rewrites: (.router_rewrites // null)}' "$ROUTER/linear_router_report.json" 2>/dev/null || true

# 2. diagnostics on the new router (blocks 16 and 19 only: the UMAP needs both)
echo "===== 2. cache + report | $(date)"
python -u scripts/cache_router_states.py --dataset "$DS" --seed "$SEED" --out-dir "$OUT" \
  --router-recipe ours --router-dir "$ROUTER" --layers 16,19 --time-layers "" \
  --device cuda --local-files-only || exit $?
python -u scripts/router_diag_report.py --cache-dir "$OUT" || exit $?
STATUS=$?
echo "Finish $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
