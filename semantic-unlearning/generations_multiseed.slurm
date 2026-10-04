#!/bin/bash
# What does the model say after unlearning? Base vs every unlearned arm that exists, per dataset x seed.
#   sbatch generations_multiseed.slurm                      # MCF, ZsRE, MQuAKE x seeds 1-5 = 15 tasks
#   python scripts/summarize_generations.py                  # one table over all of them
# Arms (included when the run dir exists; rerun this job after new arms finish):
#   shipped      outputs/<ds>_<REF>/seed<S>/L19/linear_global         (shipped row-wise rows)
#   joint        outputs/compressed_multiseed_v1/<ds>/seed<S>/L19/full (joint trainer, dataset default:
#                MCF trains toward " I don't know.", ZsRE/MQuAKE only suppress the answer)
#   joint_idk    outputs/compressed_multiseed_idk_v1/...   (ZsRE/MQuAKE + " I don't know." term)
#   joint_noidk  outputs/compressed_multiseed_noidk_v1/... (MCF without its " I don't know." term)
#   joint_idk_eos outputs/compressed_multiseed_idk_eos_v1/... (" I don't know." + end token: stops after it)
# Output: outputs/generations/<ds>_seed<S>.{md,jsonl} (overwritten on rerun).
#SBATCH --job-name=generations
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-14%8
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/generations_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/generations_%A_%a.err

set -uo pipefail
# Environment. On a compute node `module load Miniforge3` can fail after a
# purge (Lmod needs its parent module), so try the usual routes and fall back
# to putting the env's own bin/ first on PATH. Then stop at once if the env is
# not usable, instead of failing every layer in seconds.
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
echo "python: $(command -v python)"
python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting before any layer runs" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs outputs/generations
read -r -a DS <<< "${DATASETS:-mcf zsre mquake}"
read -r -a SEEDS_A <<< "${SEEDS:-1 2 3 4 5}"
N=$(( ${#DS[@]} * ${#SEEDS_A[@]} ))
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
(( I < N )) || { echo "task $I >= $N; nothing to do"; exit 0; }
DATASET="${DS[$(( I / ${#SEEDS_A[@]} ))]}"
SEED="${SEEDS_A[$(( I % ${#SEEDS_A[@]} ))]}"
LAYER="${LAYER:-19}"; LL="$(printf '%02d' "$LAYER")"
case "$DATASET" in zsre) REF=multiseed_reworded_v2 ;; *) REF=multiseed_regular_v1 ;; esac

DIRS=(); LABELS=()
add() { [[ -f "$2/fact_association_embeddings.pt" ]] && { DIRS+=("$2"); LABELS+=("$1"); } || true; }
add shipped     "outputs/${DATASET}_${REF}/seed${SEED}/L${LL}/linear_global"
add joint       "outputs/compressed_multiseed_v1/${DATASET}/seed${SEED}/L${LL}/full"
add joint_idk   "outputs/compressed_multiseed_idk_v1/${DATASET}/seed${SEED}/L${LL}/full"
add joint_noidk "outputs/compressed_multiseed_noidk_v1/${DATASET}/seed${SEED}/L${LL}/full"
add joint_idk_eos "outputs/compressed_multiseed_idk_eos_v1/${DATASET}/seed${SEED}/L${LL}/full"
(( ${#DIRS[@]} )) || { echo "no run dirs for $DATASET seed $SEED" >&2; exit 2; }

echo "generations | job ${SLURM_ARRAY_JOB_ID}_${I} | $DATASET seed $SEED | arms: ${LABELS[*]} | node $(hostname) | $(date) | commit $(git rev-parse HEAD)"
# shellcheck disable=SC2086
python -u scripts/generate_after_unlearning.py --local-files-only --run-dirs "${DIRS[@]}" \
  --labels "${LABELS[@]}" --out "outputs/generations/${DATASET}_seed${SEED}" ${EXTRA:-}
STATUS=$?
echo "Finish: $(date) | status $STATUS"
exit $STATUS
