#!/bin/bash
# Layer smoke test for another model (Qwen3-4B, Llama-3.1-8B): which block the classifier reads
# and which block Δe is written at, measured before the 4-dataset x 5-seed paper run.
# Submit through scripts/submit_layer_smoke.sh. One array task per (dataset[, write layer]),
# seed SEED (1), datasets DATASETS ("mcf"; e.g. DATASETS="mcf zsre"). Stages:
#   STAGE=router  fit the ours-recipe router at ROUTER_LAYER (default: the model's config value),
#                 cache the last-token state of every block for the router's eval / null /
#                 perturbed / train / aug prompts, refit the same classifier at EVERY block
#                 -> outputs/layer_smoke_<MODEL>/<ds>/seed<S>/layer_sweep/layer_choice.md
#   STAGE=base    router at ROUTER_LAYER (fit if missing) + base eval
#                 -> outputs/layer_smoke_<MODEL>_r<R>_base/<ds>/seed<S>/base
#   STAGE=write   rows at each of WRITE_LAYERS (comma list) under the ROUTER_LAYER router,
#                 official eval of ours and genie, base copied from the base stage
#                 -> outputs/layer_smoke_<MODEL>_r<R>_w<W>/<ds>/seed<S>/{ours,genie}
# Then: python scripts/layer_smoke_table.py MODEL
#SBATCH --job-name=layer_smoke
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/layer_smoke_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/layer_smoke_%A_%a.err

set -uo pipefail

if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi
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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
export MODEL="${MODEL:?MODEL=qwen3_4b|llama31_8b}"
STAGE="${STAGE:?STAGE=router|base|write}"
SEED="${SEED:-1}"
read -r -a DS_LIST <<< "${DATASETS:-mcf}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
export ONEPASS=1                      # the ours-recipe routers (20 / 20 prompts, fuzzy subject)

case "$STAGE" in
  router|base)
    DS="${DS_LIST[$TASK]:-}"; [[ -n "$DS" ]] || { echo "task $TASK out of range" >&2; exit 2; } ;;
  write)
    IFS=, read -r -a W_LIST <<< "${WRITE_LAYERS:?WRITE_LAYERS=w1,w2,...}"
    DS="${DS_LIST[$(( TASK / ${#W_LIST[@]} ))]:-}"; W="${W_LIST[$(( TASK % ${#W_LIST[@]} ))]:-}"
    [[ -n "$DS" && -n "$W" ]] || { echo "task $TASK out of range" >&2; exit 2; }
    export WRITE_LAYER="$W" ;;
  *) echo "STAGE must be router, base or write" >&2; exit 2 ;;
esac
source scripts/paper_model_config.sh || exit 2
paper_model_check || exit $?
R="$ROUTER_LAYER"
RL="$(printf '%02d' "$R")"
ROUTER_DIR="outputs/${DS}_${MODEL_TAG}_multiseed_ours_v1/seed$SEED/L$RL/router"

echo "layer smoke | $MODEL $STAGE | $DS seed $SEED | router L$R${WRITE_LAYER:+ write L$WRITE_LAYER} | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$STAGE" in
  router)
    OUT="outputs/layer_smoke_${MODEL}/$DS/seed$SEED"
    STOP_AFTER_ROUTER=1 TAG="layer_smoke_${MODEL}_router" bash scripts/run_l16_paper_one.sh "$DS" "$SEED" || exit $?
    test -f "$ROUTER_DIR/fact_association_embeddings.pt" || { echo "no router at $ROUTER_DIR" >&2; exit 2; }
    echo "===== cache: last-token state of every block | $(date)"
    python -u scripts/cache_router_states.py --dataset "$DS" --seed "$SEED" --router-recipe ours \
      --router-dir "$ROUTER_DIR" --out-dir "$OUT" --model-path "$MODEL_PATH" --layers all \
      --no-ppl --time-layers "" --local-files-only --device cuda || exit $?
    echo "===== sweep: the same classifier refit at every block | $(date)"
    python -u scripts/router_layer_sweep_cached.py --cache-dirs "$OUT" --out-root "outputs/layer_smoke_${MODEL}" \
      --layers all --variants baseline --robust-layers "$R" \
      --concat-layers "$EARLY_WRITE_LAYER,$R,$LENS_LAYER" --device cuda || exit $?
    echo "-> $OUT/layer_sweep/layer_choice.md"
    sed -n '/Decision rule: deployed/,/^$/p' "$OUT/layer_sweep/layer_choice.md" | head -60 ;;
  base)
    SKIP_OURS=1 BASELINES="" TAG="layer_smoke_${MODEL}_r${R}_base" bash scripts/run_l16_paper_one.sh "$DS" "$SEED" ;;
  write)
    BUILD_ROUTERS=0 BASE_FROM="layer_smoke_${MODEL}_r${R}_base" TAG="layer_smoke_${MODEL}_r${R}_w${WRITE_LAYER}" \
      bash scripts/run_l16_paper_one.sh "$DS" "$SEED" ;;
esac
STATUS=$?
echo "Finish $MODEL $STAGE $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
