#!/usr/bin/env bash
#SBATCH --job-name=mcf-scope-bias-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_scoped_bias_seed1_%j.out
#SBATCH --error=slurm_logs/mcf_scoped_bias_seed1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_scoped_bias_seed1_full50}"
export OUTPUT_DIR

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUTPUT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Commit=$(git rev-parse HEAD)"
echo "Model=${MODEL_PATH}"
echo "Output=${OUTPUT_DIR}"
nvidia-smi

python -u scripts/build_mcf_scoped_bias_sidecar.py \
  --model-dir "${MODEL_PATH}" \
  --mcf-path data/multi_counterfact.json \
  --out "${OUTPUT_DIR}/scoped_bias_full50.pt" \
  --seed 1 \
  --unlearn-num 50 \
  --sample-mode official \
  --writer-layer 8 \
  --penalty 512

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${MODEL_PATH}" \
  --scoped-sidecar "${OUTPUT_DIR}/scoped_bias_full50.pt" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUTPUT_DIR}/official_eval.json" \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --max-forget-eff 0 \
  --max-forget-gen 0 \
  --min-forget-margin 0.1 \
  --fail-if-gate-missed \
  --quiet

python -c 'import json,os; p=json.load(open(os.path.join(os.environ.get("OUTPUT_DIR","outputs/mcf_scoped_bias_seed1_full50"),"official_eval.json"))); print(json.dumps({"Eff":p["Eff"],"Gen":p["Gen"],"Spe":p["Spe"],"PPL":p["PPL"],"post_reload_acceptance":p.get("post_reload_acceptance"),"routing":p.get("scoped_span_edit",{}).get("per_split_prompt_fire_audit",{})},indent=2))'
