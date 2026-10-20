#!/bin/bash
# Router-layer sweep, MCF seed 1 (routing only, no rows trained).
# Pass 1 is unedited and cached, so the router may read ANY layer, even one after the
# write layer (16). Does a later layer, where the relation is mixed into the last token,
# separate "this subject + this question" from "this subject + another question"?
#
# Per array task = layer L (default 19 21 23 25 27):
#   prep    data + zero rows at layer L (reused from the multiseed sweep if present)
#   plain   the shipped router recipe at layer L (reused if present)
#   relneg  the same + --relation-negatives (preserve half of the subject's other relations)
#   specificity of both (paraphrase recall, neighbourhood fire, held-out same-subject fire)
# The last task to finish merges every layer into outputs/mcf_router_layer_sweep_seed1/summary.md
# (or run: bash scripts/merge_router_layer_sweep.sh).
#
#   sbatch mcf_router_layer_sweep_seed1.slurm
#   sbatch --export=ALL,LAYERS="20 22 24" mcf_router_layer_sweep_seed1.slurm
#   then rows with the best router (question-side write stays at layer 16):
#   sbatch --array=0 --export=ALL,LAYERS=16,WRITE_MODE=query,NAME_SUFFIX=_query_logit_rnL<L>,ROUTER_SRC=$PWD/outputs/mcf_router_layer_sweep_seed1/L<L>/relneg,EXTRA_TRAIN_ARGS="--lens-type logit" mcf_early_write_seed1.slurm
#SBATCH --job-name=mcf_router_layers
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_layers_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_layers_%A_%a.err

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
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
read -r -a LAYER_LIST <<< "${LAYERS:-19 21 23 25 27}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX >= ${#LAYER_LIST[@]} )); then echo "no layer for array index $INDEX; nothing to do"; exit 0; fi
L="${LAYER_LIST[$INDEX]}"
LL=$(printf '%02d' "$L")
SWEEP="$ROOT/outputs/mcf_router_layer_sweep_seed1"
OUT="$SWEEP/L$LL"
OLD="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L$LL"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
MODEL_PATH="$(jq -r '.model_path' "$REF")"
mkdir -p "$OUT"
echo "router layer $L | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
ROUTER_ARGS=(--device cuda --local-files-only --threshold-policy global --min-recall 0.98
             --threshold-placement-fraction 0.1)

PREP="$OLD/prep"
if [[ ! -f "$PREP/fact_association_embeddings.pt" ]]; then
  PREP="$OUT/prep"
  if [[ ! -f "$PREP/fact_association_embeddings.pt" ]]; then
    rm -rf "$PREP"
    python -u scripts/prepare_mcf_association_source.py --model-path "$MODEL_PATH" \
      --mcf-path "$MCF_PATH" --output-dir "$PREP" --layer "$L" --seed "$SEED" \
      --device cuda --local-files-only
  fi
fi

PLAIN="$OLD/router"
if [[ ! -f "$PLAIN/fact_association_embeddings.pt" ]]; then
  PLAIN="$OUT/plain"
  if [[ ! -f "$PLAIN/fact_association_embeddings.pt" ]]; then
    rm -rf "$PLAIN"
    python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$PLAIN" "${ROUTER_ARGS[@]}"
  fi
fi

if [[ ! -f "$OUT/relneg/fact_association_embeddings.pt" ]]; then
  rm -rf "$OUT/relneg"
  python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$OUT/relneg" "${ROUTER_ARGS[@]}" \
    --relation-negatives --relation-negative-templates 3
fi

if [[ ! -f "$OUT/specificity.json" ]]; then
  python -u scripts/mcf_router_specificity.py \
    --runs "L$L plain=$PLAIN" "L$L relneg=$OUT/relneg" \
    --out "$OUT/specificity.md" --device cuda --local-files-only
fi
bash scripts/merge_router_layer_sweep.sh
echo "Finish: $(date)"
