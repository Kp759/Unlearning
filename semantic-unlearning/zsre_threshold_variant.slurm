#!/bin/bash
# ZsRE: official eval of the TRAINED multiseed rows with only the router cutoff
# changed (no retraining). One array task per seed.
#   sbatch zsre_threshold_variant.slurm
#   sbatch --export=ALL,LAYERS="19",VARIANTS="-4 subject" zsre_threshold_variant.slurm
# Variants: a number moves the global cutoff by that many logits (negative = fires
# more); "subject" = subject gate (any prompt with a forget subject fires).
# Outputs: outputs/zsre_threshold_variant_v1/seed<S>/L<LL>/<variant>/official_zsre_eval.json
# Summary: python scripts/make_router_threshold_variant.py --summarize --root outputs/zsre_threshold_variant_v1
#SBATCH --job-name=zsr_thr
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_thr_%A_s%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_thr_%A_s%a.err

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
SEED="${SLURM_ARRAY_TASK_ID}"
SRC=outputs/zsre_multiseed_regular_v1/seed$SEED
DST=outputs/zsre_threshold_variant_v1/seed$SEED
echo "ZsRE cutoff variants | seed $SEED | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
FAILED=()
for L in ${LAYERS:-19 23}; do
  LL=$(printf '%02d' "$L")
  RUN="$SRC/L$LL/linear_global"
  [[ -f "$RUN/official_zsre_eval.json" ]] || { echo "missing trained run $RUN"; FAILED+=("L$LL"); continue; }
  for V in ${VARIANTS:--2 -4 -6 subject}; do
    OUT="$DST/L$LL/$V"
    [[ -f "$OUT/official_zsre_eval.json" ]] && { echo "done: $OUT"; continue; }
    if [[ -e "$OUT" ]]; then mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"; fi
    echo "===== seed $SEED L$LL variant $V | $(date) ====="
    if python -u scripts/make_router_threshold_variant.py --run-dir "$RUN" --variant="$V" --out-dir "$OUT" \
       && python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
            --run-dir "$OUT" --zsre-path data/zsre_mend_eval.json --wikidata-dir data/wikidata \
            --seed "$SEED" --device cuda --dtype bfloat16 --batch-size 8 --local-files-only; then
      :
    else
      FAILED+=("L$LL/$V")
    fi
  done
done
python scripts/make_router_threshold_variant.py --summarize --root outputs/zsre_threshold_variant_v1 || true
echo "Finish seed $SEED: $(date) | failed: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
