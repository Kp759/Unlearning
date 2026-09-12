#!/usr/bin/env bash
#SBATCH --job-name=mcf-zul-vs-fact-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_zul_vs_fact_s1_%j.out
#SBATCH --error=slurm_logs/mcf_zul_vs_fact_s1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
OURS_RUN="${OURS_RUN:-${PROJECT_DIR}/outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-${PROJECT_DIR}/data/wikidata}"
ZERO_ROOT="${ZERO_ROOT:-/scratch/yl258/kp759/Unlearning/ZeroUnlearn}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/mcf_seed1_zerounlearn_vs_fact_assoc_matched_slurm_${SLURM_JOB_ID}}"

cd "${PROJECT_DIR}"

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "Matched MCF seed-1 comparison"
echo "Started:       $(date)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "CUDA devices:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project:       ${PROJECT_DIR}"
echo "Git commit:    $(git rev-parse HEAD)"
echo "Model:         ${MODEL_PATH}"
echo "Our run:       ${OURS_RUN}"
echo "MCF:           ${MCF_PATH}"
echo "Wikidata:      ${WIKIDATA_DIR}"
echo "ZeroUnlearn:   ${ZERO_ROOT}"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================================"

nvidia-smi || true
python --version

if [[ ! -f "${OURS_RUN}/fact_association_embeddings.pt" ]]; then
  echo "ERROR: missing frozen fact-association artifact:"
  echo "  ${OURS_RUN}/fact_association_embeddings.pt"
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output directory already exists; refusing overwrite:"
  echo "  ${OUTPUT_DIR}"
  exit 3
fi

python -u scripts/compare_mcf_seed1_zerounlearn_vs_fact_association.py \
  --model-path "${MODEL_PATH}" \
  --ours-run-dir "${OURS_RUN}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --zero-unlearn-root "${ZERO_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Completed: $(date)"
echo "Results:   ${OUTPUT_DIR}"
echo "Comparison:"
cat "${OUTPUT_DIR}/comparison.md" || true
echo "============================================================"
