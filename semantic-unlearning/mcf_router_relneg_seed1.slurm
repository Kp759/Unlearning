#!/bin/bash
# Relation-aware router, MCF seed 1. The shipped L19 router fires on ~52% of prompts that
# ask a forgotten SUBJECT about another relation: its negatives were mostly other
# subjects, so "is this subject?" was enough. Refit it exactly as before (same prep,
# features, CV, calibration: --min-recall 0.98) plus --relation-negatives: each fact's
# subject asked with the 'preserve' half of its other relations (3 MCF wordings each,
# split fit / calibration / audit). The 'eval' half of the relations is never a router
# negative, so the collateral eval stays held out.
#
# Then a routing-only comparison (old vs new; rows play no part in routing):
# paraphrase recall, neighbourhood fire, same-subject other-relation fire.
#
#   sbatch mcf_router_relneg_seed1.slurm
#   then rows with the new router (question-side write, logit lens), and the evals:
#   J=$(sbatch --parsable --dependency=afterok:<this job> --array=0 --export=ALL,LAYERS=16,WRITE_MODE=query,NAME_SUFFIX=_query_logit_rn,ROUTER_SRC=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/mcf_router_relneg_seed1/router,EXTRA_TRAIN_ARGS="--lens-type logit" mcf_early_write_seed1.slurm)
#   sbatch --dependency=afterok:$J mcf_subject_collateral_seed1.slurm
#   sbatch --dependency=afterok:$J mcf_lens_eval_seed1.slurm
#SBATCH --job-name=mcf_router_relneg
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_relneg_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_relneg_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u  # conda's activate scripts reference unset variables
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
  set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$ENV_DIR"
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
OLD="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19"
OUT="$ROOT/outputs/mcf_router_relneg_seed1"
mkdir -p "$OUT"
test -f "$OLD/prep/fact_association_embeddings.pt" || { echo "No seed-1 L19 prep at $OLD/prep" >&2; exit 2; }
echo "relation-aware router | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

if [[ ! -f "$OUT/router/fact_association_embeddings.pt" ]]; then
  rm -rf "$OUT/router"
  python -u scripts/fit_linear_router.py \
    --run-dir "$OLD/prep" --output-dir "$OUT/router" \
    --device cuda --local-files-only \
    --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1 \
    --relation-negatives --relation-negative-templates 3
fi

python -u scripts/mcf_router_specificity.py \
  --runs old="$OLD/router" relneg="$OUT/router" \
  --out "$OUT/specificity.md" --device cuda --local-files-only
echo "Finish: $(date)"
