#!/usr/bin/env bash
#SBATCH --job-name=rwku-s5e600-r2-sk-s0
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/rwku_s5e600_rank2_active_seed0/%j.out
#SBATCH --error=outputs/logs/rwku_s5e600_rank2_active_seed0/%j.err

set -euo pipefail

REPOSITORY=/scratch/yl258/kp759/Unlearning/semantic-unlearning
MODEL=/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
EXPERIMENT_ID=rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1
OUTPUT_ROOT=$REPOSITORY/outputs/rwku_s5e600_rank2_active
RECEIPT=$OUTPUT_ROOT/$EXPERIMENT_ID/checkpoint_receipt.json

mkdir -p "$REPOSITORY/outputs/logs/rwku_s5e600_rank2_active_seed0"
cd "$REPOSITORY"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

./scripts/run_rwku_generated_s5e600_rank2_seed0.sh "$MODEL" preflight
./scripts/run_rwku_generated_s5e600_rank2_seed0.sh "$MODEL" prepare
./scripts/run_rwku_generated_s5e600_rank2_seed0.sh "$MODEL" train

python scripts/rwku_checkpoint_receipt.py verify --receipt "$RECEIPT"
python - "$RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("state") != "CHECKPOINT_FROZEN":
    raise SystemExit("receipt validation did not end at CHECKPOINT_FROZEN")
if receipt.get("official_evaluation_opened") is not False:
    raise SystemExit("official evaluation opened before the evaluate stage")
PY

./scripts/run_rwku_generated_s5e600_rank2_seed0.sh "$MODEL" evaluate
