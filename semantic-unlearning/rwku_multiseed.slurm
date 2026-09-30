#!/bin/bash
# RWKU layer sweep, seeds 1-5, linear classifier only (no V2), one array task per
# (gate, seed, layer). Default 70 tasks = 2 gates x 5 seeds x 7 layers:
#   threshold  the current rule of the MCF/ZsRE/MQuAKE sweeps: recall >= 0.98 on the
#              calibration split (-> outputs/rwku_multiseed_regular_v1), then the same
#              98% rule on the merged, per-fact-balanced validation set
#              (-> outputs/calibration_rules_v1/rwku/.../recall0.98_fact)
#   subject    RWKU's native entity-level gate (-> outputs/rwku_multiseed_subject_v1)
# Rows trained under the router (regular), NORM_SCALE=1, shipped RWKU optimizer.
# Batch seed s = people s..s+4 of RWKU's first ten targets.
#
# BEFORE submitting, fetch the pinned RWKU files once on the LOGIN node:
#   for s in 1 2 3 4 5; do python scripts/rwku_batch50.py --seed $s > /dev/null; done
#
#   sbatch rwku_multiseed.slurm                                              # 70 tasks
#   sbatch --array=0-34 rwku_multiseed.slurm                                 # threshold gate only
#   sbatch --array=0-34 --export=ALL,GATE_LIST=subject rwku_multiseed.slurm  # subject gate only
#   sbatch --array=0-9 --export=ALL,GATE_LIST=threshold,LAYER_LIST="19 23" rwku_multiseed.slurm
# Task i -> GATE=GATES[i/(NS*NL)], SEED=SEEDS[(i%(NS*NL))/NL], LAYER=LAYERS[i%NL].
# Resumable: finished stages are skipped. Tables:
#   python scripts/summarize_layer_sweep_seeds.py --dataset rwku --modes regular subject
#   python scripts/summarize_arm_layers.py --arm recall0.98_fact --datasets rwku
#SBATCH --job-name=rwku_ms
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-69
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rwku_ms_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rwku_ms_%A_%a.err

set -uo pipefail

ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
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
echo "python: $(command -v python)"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

GATES=(${GATE_LIST:-threshold subject})
SEEDS=(${SEED_LIST:-1 2 3 4 5})
LAYERS=(${LAYER_LIST:-1 3 7 13 19 23 27})
NS=${#SEEDS[@]}; NL=${#LAYERS[@]}
i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
if (( i >= ${#GATES[@]} * NS * NL )); then
  echo "task $i outside ${#GATES[@]} gates x $NS seeds x $NL layers; nothing to do"; exit 0
fi
export GATE="${GATES[i / (NS * NL)]}"
export SEED="${SEEDS[(i % (NS * NL)) / NL]}"
LAYER="${LAYERS[i % NL]}"
export TRAINING_ROUTE=router NORM_SCALE=1 MIN_RECALL="${MIN_RECALL:-0.98}"
export RWKU_CHAT_DATE_STRING="26 Jul 2024"
# One base-model (all-zero rows) evaluation per seed, on the first gate's layer-19 task.
export BASE_EVAL=0
[[ "$LAYER" == 19 && "$GATE" == "${GATES[0]}" ]] && export BASE_EVAL=1

# The RWKU files must already be on disk (see the header); fail fast otherwise.
python - "$SEED" <<'EOF' || { echo "RWKU data for batch seed $SEED missing; run the login-node fetch in the header" >&2; exit 4; }
import sys
sys.path.insert(0, "scripts")
from pathlib import Path
from rwku_batch50 import build_batch_split
split = build_batch_split(data_root=Path("data/rwku"), batch_seed=int(sys.argv[1]), allow_download=False)
print("RWKU batch", sys.argv[1], [t["subject"] for t in split["manifest"]["targets"]],
      "| train", len(split["forget_train"]), "| heldout L1/L2/para",
      len(split["heldout_level1"]), len(split["heldout_level2"]), len(split["heldout_paraphrase"]))
EOF

echo "RWKU multiseed | gate $GATE | seed $SEED | layer $LAYER | base_eval $BASE_EVAL | job ${SLURM_ARRAY_JOB_ID:-?}_$i | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi
bash scripts/run_rwku_layer_sweep_one.sh "$LAYER"
status=$?
echo "Finish gate $GATE seed $SEED L$LAYER: $(date) | exit $status"
exit $status
