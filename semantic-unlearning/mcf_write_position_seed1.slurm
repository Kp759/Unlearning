#!/bin/bash
# MCF seed 1, 50 forget facts, layer 19: WHERE the routed row Δe is added.
# One array task per arm (all four run in parallel on 4 GPUs):
#   task 0 = A  last prompt token (shipped)
#   task 1 = B  last subject token + last token
#   task 2 = C  whole subject span + last token
#   task 3 = D  every prompt token except BOS
# The router is the same in every arm: the seed-1 L19 linear classifier (it
# always reads the last prompt token). Rows are retrained per arm with the
# seed-1 multiseed settings (regular routing, NORM_SCALE=1, 10800 s cap so the
# 1500-step budget binds), then scored with the official MCF evaluator.
#
#   sbatch mcf_write_position_seed1.slurm              # all four arms
#   sbatch --array=1,2 mcf_write_position_seed1.slurm  # only B and C
# Resumable: finished stages are skipped. Each task prints the summary table at
# the end; the last task to finish has all four rows. Or run it any time:
#   python scripts/mcf_write_position.py summarize --root outputs/mcf_write_position_seed1
#SBATCH --job-name=mcf_writepos
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_writepos_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_writepos_%A_%a.err

set -euo pipefail

# Environment (same block as the other sweep jobs): `module load Miniforge3`
# fails on compute nodes after a purge unless `wulver` is loaded first.
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
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs

ARMS=(A B C D)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
LAYER=19
SEED=1
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-10800}"
OUT="$ROOT/outputs/mcf_write_position_seed1"
ARM_DIR="$OUT/arm_$ARM"
mkdir -p "$ARM_DIR"

REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
test -f "$REF" || { echo "Missing $REF (model/data paths)" >&2; exit 2; }
MODEL_PATH="$(jq -r '.model_path' "$REF")"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"

echo "MCF write-position | arm $ARM | job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi || true

# 1. The seed-1 L19 linear classifier (fitted, rows still zero). Reuse the
#    multiseed / exploratory sweep router; fit a private copy only if neither exists.
ROUTER_SRC="${ROUTER_SRC:-}"
if [[ -z "$ROUTER_SRC" ]]; then
  for CANDIDATE in \
      "$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router" \
      "$ROOT/outputs/mcf_layer_sweep_linear_regular_v1/L19/router"; do
    if [[ -f "$CANDIDATE/fact_association_embeddings.pt" ]]; then ROUTER_SRC="$CANDIDATE"; break; fi
  done
fi
if [[ -z "$ROUTER_SRC" ]]; then
  echo "===== [$ARM] no seed-1 L19 router found: fitting one (prep + router) ====="
  if [[ ! -f "$ARM_DIR/prep/fact_association_embeddings.pt" ]]; then
    rm -rf "$ARM_DIR/prep"
    python -u scripts/prepare_mcf_association_source.py \
      --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
      --output-dir "$ARM_DIR/prep" --layer "$LAYER" --seed "$SEED" --device cuda --local-files-only
  fi
  if [[ ! -f "$ARM_DIR/router_fit/fact_association_embeddings.pt" ]]; then
    rm -rf "$ARM_DIR/router_fit"
    python -u scripts/fit_linear_router.py \
      --run-dir "$ARM_DIR/prep" --output-dir "$ARM_DIR/router_fit" \
      --device cuda --local-files-only \
      --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1
  fi
  ROUTER_SRC="$ARM_DIR/router_fit"
fi
echo "router: $ROUTER_SRC"

# 2. Arm router = same classifier, write_mode set.
if [[ ! -f "$ARM_DIR/router/fact_association_embeddings.pt" ]]; then
  rm -rf "$ARM_DIR/router"
  python -u scripts/mcf_write_position.py make-arm \
    --router-dir "$ROUTER_SRC" --output-dir "$ARM_DIR/router" --arm "$ARM"
fi

# 3. Train the rows under this write mode (regular routing, seed-1 multiseed settings).
FINAL="$ARM_DIR/linear_global"
if [[ ! -f "$FINAL/training_report.json" ]]; then
  if [[ -e "$FINAL" ]]; then mv "$FINAL" "$FINAL.incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  echo "===== [$ARM] TRAIN rows (route=router, norm-scale=1, cap ${MAX_TRAIN_SECONDS}s) | $(date) ====="
  python -u scripts/train_mcf_linear_router_rows.py \
    --router-dir "$ARM_DIR/router" --output-dir "$FINAL" \
    --training-route router --norm-scale 1 \
    --max-training-seconds "$MAX_TRAIN_SECONDS" \
    --device cuda --local-files-only
fi

# 4. Official MCF eval (same evaluator and settings as the layer sweep).
if [[ ! -f "$FINAL/official_mcf_eval.json" ]]; then
  echo "===== [$ARM] OFFICIAL MCF eval | $(date) ====="
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mcf-path "$MCF_PATH" \
    --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
    --device cuda --dtype bfloat16 --local-files-only
fi

echo "===== [$ARM] COMPLETE | $(date) ====="
python scripts/mcf_write_position.py summarize --root "$OUT" || true
