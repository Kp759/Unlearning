#!/bin/bash
# S_idk, MCF seed 1, layer 19, setting A: ONE shared vector for all 50 forget facts,
# trained toward " I don't know." + EOS (same loss/data/router as R_idk, rows tied).
# Steps: train -> official MCF eval -> generation on our training views -> held-out
# generation (MCF paraphrase / same-subject / neighbourhood prompts) -> rebuild tables.
#
#   sbatch mcf_shared_idk_seed1.slurm
#   sbatch --export=ALL,EXTRA_TRAIN_ARGS="--max-row-norm-ratio 2" mcf_shared_idk_seed1.slurm
#   cat outputs/mcf_refusal_rows_seed1/comparison.md outputs/mcf_heldout_generation_seed1/summary.md
#SBATCH --job-name=mcf_shared_idk
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_shared_idk_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_shared_idk_%j.err

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
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
ARM="$ROOT/outputs/mcf_refusal_rows_seed1/S_idk"
HELD="$ROOT/outputs/mcf_heldout_generation_seed1"
mkdir -p "$ARM" "$HELD"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
ROUTER_SRC="${ROUTER_SRC:-$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router}"
test -f "$ROUTER_SRC/fact_association_embeddings.pt" || { echo "No seed-1 L19 router at $ROUTER_SRC" >&2; exit 2; }
echo "S_idk (shared vector) | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

if [[ ! -f "$ARM/linear_global/training_report.json" ]]; then
  rm -rf "$ARM/linear_global"
  python -u scripts/train_mcf_refusal_rows.py --router-dir "$ROUTER_SRC" \
    --output-dir "$ARM/linear_global" --target idk --shared --device cuda --local-files-only \
    $EXTRA_TRAIN_ARGS
fi
if [[ ! -f "$ARM/linear_global/official_mcf_eval.json" ]]; then
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$ARM/linear_global" --mcf-path "$MCF_PATH" --wikidata-dir "$ROOT/data/wikidata" \
    --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only
fi
if [[ ! -f "$ARM/routed_generation.json" ]]; then
  python -u scripts/mcf_projection_edit.py routed-generate --run-dir "$ARM/linear_global" \
    --out "$ARM/routed_generation.json" --device cuda --local-files-only
fi
if [[ ! -f "$HELD/S_idk.json" ]]; then
  python -u scripts/mcf_heldout_generation.py run --method router --run-dir "$ARM/linear_global" \
    --label "shared IDK (1 vector)" --out "$HELD/S_idk.json" --seed "$SEED" \
    --mcf-path "$MCF_PATH" --device cuda --local-files-only
fi
bash scripts/run_refusal_summaries.sh
echo "Finish: $(date)"
