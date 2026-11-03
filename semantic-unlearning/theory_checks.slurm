#!/bin/bash
# Measurements behind the paper's Analysis section (scripts/theory_read_layers.py,
# scripts/theory_geometry.py). Seed 1, the paper's runs; nothing is retrained.
#   0  read layers, Llama-3.2-3B MCF   gate-free heads at every block: same- vs different-answer AUC
#   1  read layers, Qwen3-4B MCF
#   2  geometry, Llama-3.2-3B MCF      selectivity of delta_i at W=16, write-block sweep, collateral
#   3  geometry, Llama-3.2-3B zsRE
#   4  geometry, Llama-3.2-3B MQuAKE
#   5  geometry, Qwen3-4B MCF          write-block sweep only (the smoke runs measured W=16..26)
#
#   cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
#   sbatch theory_checks.slurm                      # all
#   sbatch --array=2 --export=ALL,DEBUG=1 theory_checks.slurm   # 2 associations, quick
#SBATCH --job-name=theory
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/theory_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/theory_%A_%a.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
export PATH="$ENV_DIR/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_DIR" || exit 1
export PYTHONPATH="$REPO_DIR/scripts:$REPO_DIR:${PYTHONPATH:-}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
DBG=()
[[ "${DEBUG:-0}" == 1 ]] && DBG=(--max-facts 2 --sweep-facts 2 --max-collateral 16)
L3B_SWEEP=4,8,12,14,16,19,22,25
QWEN_SWEEP=12,16,18,21,23,26,30,33
echo "task $TASK on $(hostname) $(date)"
case "$TASK" in
  0) python -u scripts/theory_read_layers.py --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1 ;;
  1) python -u scripts/theory_read_layers.py --cache-dir outputs/layer_smoke_qwen3_4b/mcf/seed1 ;;
  2|3|4)
     DS=$(echo mcf zsre mquake | cut -d' ' -f$((TASK - 1)))
     SWEEP=$L3B_SWEEP; [[ "${DEBUG:-0}" == 1 ]] && SWEEP=16
     python -u scripts/theory_geometry.py --dataset "$DS" --seed 1 --local-files-only \
       --run-dir "outputs/l16_ours_v1/$DS/seed1/ours" \
       --cache-dir "outputs/router_diag_ours_control_v1/$DS/seed1" --write-layers "$SWEEP" "${DBG[@]}" ;;
  5) SWEEP=$QWEN_SWEEP; [[ "${DEBUG:-0}" == 1 ]] && SWEEP=23
     python -u scripts/theory_geometry.py --dataset mcf --seed 1 --local-files-only --parts write_sweep \
       --run-dir outputs/layer_smoke_qwen3_4b_r26_w23/mcf/seed1/ours \
       --cache-dir outputs/layer_smoke_qwen3_4b/mcf/seed1 --write-layers "$SWEEP" "${DBG[@]}" ;;
esac
echo "exit $? $(date)"
