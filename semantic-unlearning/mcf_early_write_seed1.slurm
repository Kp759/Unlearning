#!/bin/bash
# Early-write rows, MCF seed 1: route at layer 19 (same router; unrouted prompts stay exact
# base), but add each fact's own Δe row at its SUBJECT tokens after an earlier block, where
# the fact is recalled, so later layers and positions cannot re-derive the answer. Rows are
# trained with the J-lens feedback objective (no refusal target; on-policy continuations;
# answer suppressed in the output and under a layer-24 J-lens; KL to answer-masked base).
# Array task = write layer (default 8, 12, 16; set LAYERS to change, e.g. after
# mcf_recall_site_seed1.slurm):
#
#   sbatch mcf_early_write_seed1.slurm
#   sbatch --export=ALL,LAYERS="10 14 16" mcf_early_write_seed1.slurm
#   v2 (fluency): one layer, stronger KL + repetition unlikelihood, separate output name:
#   sbatch --array=0 --export=ALL,LAYERS=16,NAME_SUFFIX=_v2,EXTRA_TRAIN_ARGS="--keep-weight 1.0 --rep-weight 0.5" mcf_early_write_seed1.slurm
#   then: sbatch --dependency=afterok:<id> mcf_lens_eval_seed1.slurm
#         bash scripts/run_refusal_summaries.sh
#SBATCH --job-name=mcf_early_write
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_early_write_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_early_write_%A_%a.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
read -r -a LAYER_LIST <<< "${LAYERS:-8 12 16}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX >= ${#LAYER_LIST[@]} )); then echo "no layer for array index $INDEX; nothing to do"; exit 0; fi
L="${LAYER_LIST[$INDEX]}"
NAME="E_L${L}${NAME_SUFFIX:-}"
ARM="$ROOT/outputs/mcf_early_write_seed1/$NAME"
HELD="$ROOT/outputs/mcf_heldout_generation_seed1"
mkdir -p "$ARM" "$HELD"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
ROUTER_SRC="${ROUTER_SRC:-$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router}"
test -f "$ROUTER_SRC/fact_association_embeddings.pt" || { echo "No seed-1 L19 router at $ROUTER_SRC" >&2; exit 2; }
echo "$NAME (write layer $L, subject span, extra: ${EXTRA_TRAIN_ARGS:-none}) | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

if [[ ! -f "$ARM/linear_global/training_report.json" ]]; then
  rm -rf "$ARM/linear_global"
  python -u scripts/train_mcf_jlens_feedback_rows.py --router-dir "$ROUTER_SRC" \
    --output-dir "$ARM/linear_global" --write-layer "$L" --write-mode subject_span \
    --lens-layer 24 --device cuda --local-files-only $EXTRA_TRAIN_ARGS
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
if [[ ! -f "$HELD/$NAME.json" ]]; then
  python -u scripts/mcf_heldout_generation.py run --method router --run-dir "$ARM/linear_global" \
    --label "early write L$L${NAME_SUFFIX:-}, subject span (50 vectors)" --out "$HELD/$NAME.json" --seed "$SEED" \
    --mcf-path "$MCF_PATH" --device cuda --local-files-only
fi
bash scripts/run_refusal_summaries.sh
echo "Finish: $(date)"
