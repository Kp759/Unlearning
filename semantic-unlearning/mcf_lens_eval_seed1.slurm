#!/bin/bash
# Internal-knowledge check, MCF seed 1: is the forgotten answer still readable inside the
# edited forward pass? Logit lens at layers 16-27 and J-lens z at layers 20/22/24/26 (24 is
# the layer J_A/J_C were trained against; 20/22/26 are held-out lenses), read at every
# position of a 16-token greedy continuation, on our training views and the held-out MCF
# paraphrases. Runs: base, Δe (old), Δe→EOS, J_A, J_C, early-write E_L* (whichever exist).
#
#   sbatch --dependency=afterok:<jlens feedback job id> mcf_lens_eval_seed1.slurm
#   cat outputs/mcf_lens_eval_seed1/summary.md
#SBATCH --job-name=mcf_lens_eval
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_lens_eval_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_lens_eval_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
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
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
RUNS=("base=$ROOT/outputs/mcf_projection_alpha_sweep_seed1/base/linear_global"
      "delta_e=$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global")
for SPEC in "R_eos=outputs/mcf_refusal_rows_seed1/R_eos" \
            "J_A=outputs/mcf_jlens_feedback_seed1/J_A" "J_C=outputs/mcf_jlens_feedback_seed1/J_C"; do
  DIR="$ROOT/${SPEC#*=}/linear_global"
  [[ -f "$DIR/fact_association_embeddings.pt" ]] && RUNS+=("${SPEC%%=*}=$DIR")
done
for ARM in "$ROOT"/outputs/mcf_early_write_seed1/E_L*; do
  [[ -f "$ARM/linear_global/fact_association_embeddings.pt" ]] && RUNS+=("$(basename "$ARM")=$ARM/linear_global")
done
echo "lens eval | runs: ${RUNS[*]} | job ${SLURM_JOB_ID:-} | $(hostname) | $(date)"
python -u scripts/mcf_lens_eval.py --runs "${RUNS[@]}" \
  --out-dir "$ROOT/outputs/mcf_lens_eval_seed1" --device cuda --local-files-only
echo "Finish: $(date)"
