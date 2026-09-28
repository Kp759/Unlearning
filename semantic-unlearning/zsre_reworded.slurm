#!/bin/bash
# ZsRE router fix: train the linear classifier on generated rewordings of each
# direct question (official rephrases never read), one array task per seed.
#   sbatch zsre_reworded.slurm
#   sbatch --export=ALL,LAYERS="19" zsre_reworded.slurm
#   v2 (answer-consistency + near-duplicate filters on the rewordings):
#   sbatch --export=ALL,REWORD_TAG=reworded_v2,GEN_ARGS="--consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4" zsre_reworded.slurm
# Per seed:
#   0. rewordings.json (base model, training-visible direct questions only)
#   1. per layer: prep + reworded router, then a PREVIEW: the existing trained
#      regular rows behind the new router (linear_global_swap) + official eval
#   2. per layer: the full method, rows retrained under the new router
#      (linear_global) + official eval
#   3. decomposition (router vs genie, miss categories, same-subject false fire)
# Outputs: outputs/zsre_multiseed_<REWORD_TAG>/seed<S>/...  (REWORD_TAG default reworded_v1)
#SBATCH --job-name=zsr_rew
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_rew_%A_s%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_rew_%A_s%a.err

set -uo pipefail

# Environment. On a compute node `module load Miniforge3` can fail after a
# purge (Lmod needs its parent module), so try the usual routes and fall back
# to putting the env's own bin/ first on PATH. Then stop at once if the env is
# not usable, instead of failing every layer in seconds.
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
python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting before any layer runs" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
export SEED="${SLURM_ARRAY_TASK_ID}"
REWORD_TAG="${REWORD_TAG:-reworded_v1}"
GEN_ARGS="${GEN_ARGS:-}"
export SWEEP_TAG=multiseed_$REWORD_TAG
export TRAINING_ROUTE=router
export NORM_SCALE=1
ROOT_OUT=outputs/zsre_multiseed_$REWORD_TAG/seed$SEED
REG=outputs/zsre_multiseed_regular_v1/seed$SEED
mkdir -p "$ROOT_OUT"
export REWORDINGS="$PWD/$ROOT_OUT/rewordings.json"
echo "ZsRE reworded router ($REWORD_TAG; gen args: ${GEN_ARGS:-none}) | seed $SEED | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
LAYER_LIST="${LAYERS:-19 23}"
FAILED=()

if [[ ! -f "$REWORDINGS" ]]; then
  echo "===== rewordings (seed $SEED) | $(date) ====="
  python -u scripts/zsre_router_rewordings.py generate --prep-dir "$REG/L19/prep" \
    --out "$REWORDINGS" --device cuda --local-files-only $GEN_ARGS || { echo "rewording generation failed"; exit 1; }
fi

for L in $LAYER_LIST; do
  LL=$(printf '%02d' "$L"); BASE="$ROOT_OUT/L$LL"
  echo "===== seed $SEED L$LL: prep + reworded router | $(date) ====="
  STOP_AFTER=router bash scripts/run_zsre_layer_sweep_one.sh "$L" || { FAILED+=("L$LL/router"); continue; }
  if [[ ! -f "$BASE/linear_global_swap/official_zsre_eval.json" ]]; then
    [[ -e "$BASE/linear_global_swap" ]] && mv "$BASE/linear_global_swap" "$BASE/linear_global_swap.incomplete_$(date +%s)"
    echo "===== seed $SEED L$LL: PREVIEW (regular rows, reworded router) | $(date) ====="
    python -u scripts/swap_router_rows.py --router-dir "$BASE/router" --rows-from "$REG/L$LL/linear_global" \
      --output-dir "$BASE/linear_global_swap" \
    && python -u scripts/evaluate_zsre_fact_association_embeddings_official.py --run-dir "$BASE/linear_global_swap" \
      --zsre-path data/zsre_mend_eval.json --wikidata-dir data/wikidata --seed "$SEED" \
      --device cuda --dtype bfloat16 --batch-size 8 --local-files-only \
    || FAILED+=("L$LL/swap")
  fi
done

for L in $LAYER_LIST; do
  LL=$(printf '%02d' "$L")
  echo "===== seed $SEED L$LL: FULL (rows retrained under reworded router) | $(date) ====="
  bash scripts/run_zsre_layer_sweep_one.sh "$L" || FAILED+=("L$LL/full")
done

for RUN in linear_global_swap linear_global; do
  echo "===== decomposition: $RUN ====="
  python -u scripts/evaluate_zsre_router_decomposition.py --run-dirs "$ROOT_OUT/L??/$RUN" \
    --zsre-path data/zsre_mend_eval.json --device cuda --dtype bfloat16 --batch-size 16 --local-files-only \
    || FAILED+=("decomposition/$RUN")
done
python scripts/make_router_threshold_variant.py --summarize --root outputs/zsre_multiseed_$REWORD_TAG || true
echo "Finish seed $SEED: $(date) | failed: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
