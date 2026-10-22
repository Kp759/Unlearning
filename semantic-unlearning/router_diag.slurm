#!/bin/bash
# Router diagnostics, one benchmark per task (0 mcf, 1 zsre, 2 mquake, 3 rwku), seed SEED (1):
#   1. cache    last-token states at every layer + L19 router outputs (scripts/cache_router_states.py)
#   2. report   L19 error analysis (scripts/router_diag_report.py)
#   3. sweep    same router refit at every layer + robustness variants (scripts/router_layer_sweep_cached.py)
# Needs the paper's L19 routers (outputs/<ds>_multiseed_*/seed<S>/L19/router). If the paper's
# ours run exists, the cache also generates base vs ours on the null prompts the router fires on.
#
#   bash scripts/submit_router_diag.sh                     # 4 tasks + the combined write-up
#   sbatch --array=3 router_diag.slurm                     # RWKU only
#   sbatch --export=ALL,SEED=2 --array=0-3 router_diag.slurm
#SBATCH --job-name=rdiag
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rdiag_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rdiag_%A_%a.err

set -uo pipefail

if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi  # /project env once built (scripts/build_env_project.sh)
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
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
python -c "import umap" 2>/dev/null \
  || echo "umap-learn not installed: the L16/L19 projection falls back to PCA (pip install umap-learn)"

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
DATASETS=(mcf zsre mquake rwku)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$TASK]:-}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-3)" >&2; exit 2; }
SEED="${SEED:-1}"
ROOT_OUT="${ROOT_OUT:-outputs/router_diag_v1}"
OUT="$ROOT_OUT/$DS/seed$SEED"
declare -A OURS=(
  [mcf]="outputs/l16_paper_mcf_linear19/mcf/seed$SEED/ours"
  [zsre]="outputs/l16_paper_v1/zsre/seed$SEED/ours"
  [mquake]="outputs/l16_paper_v1/mquake/seed$SEED/ours"
  [rwku]="outputs/l16_paper_rwku_subject_v1/rwku/seed$SEED/ours"
)
OURS_ARGS=()
[[ -f "${OURS[$DS]}/fact_association_embeddings.pt" ]] && OURS_ARGS=(--ours-dir "${OURS[$DS]}")

echo "router diagnostics | $DS seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
STATUS=0
python -u scripts/cache_router_states.py --dataset "$DS" --seed "$SEED" --out-dir "$OUT" \
  --device cuda --local-files-only "${OURS_ARGS[@]}" || STATUS=1
if [[ $STATUS == 0 ]]; then
  python -u scripts/router_diag_report.py --cache-dir "$OUT" || STATUS=1
  python -u scripts/router_layer_sweep_cached.py --cache-dirs "$OUT" --out-root "$OUT/layer_sweep" \
    --device cuda || STATUS=1
fi
# Cross-benchmark write-up from every finished sweep (nothing is refit); the last task to
# finish writes the complete $ROOT_OUT/ROUTER_LAYER_CHOICE.md.
DONE=()
for D in mcf zsre mquake rwku; do
  [[ -f "$ROOT_OUT/$D/seed$SEED/layer_sweep/sweep.json" ]] && DONE+=("$ROOT_OUT/$D/seed$SEED")
done
if (( ${#DONE[@]} )); then
  python -u scripts/router_layer_sweep_cached.py --cache-dirs "${DONE[@]}" --out-root "$ROOT_OUT" \
    --combine-only || STATUS=1
fi
echo "Finish $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
