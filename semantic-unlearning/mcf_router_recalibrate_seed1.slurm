#!/bin/bash
# Re-pick the router cutoff (weights frozen), MCF seed 1, routing only.
# The L27 relation-aware router has audit AUC 0.99 (98% recall at 6.7% false fire), but
# its shipped cutoff (recall >= 98% on the calibration split alone) is p = 0.06, which
# fires on 46% of calibration negatives: a few hard calibration positives set it.
# Here the cutoff is chosen on the pooled validation prompts (calibration + audit; the
# router's own held-out families, never an official test prompt), macro over facts,
# under several rules. The specificity table is then computed on the official held-out
# paraphrases and the held-out ('eval') relations, which no rule ever sees.
#
#   sbatch mcf_router_recalibrate_seed1.slurm            (all variants below, one table)
#   then: cat outputs/mcf_router_recalibrate_seed1/summary.md
#   one variant as its own job (summary_<NAME>.md):
#   sbatch --export=ALL,NAME=X,SRC=<router dir>,ARGS="--objective min_recall --min-recall 0.90 --validation-splits calibration" mcf_router_recalibrate_seed1.slurm
#SBATCH --job-name=mcf_router_recal
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_recal_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_router_recal_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi  # /project env once built (scripts/build_env_project.sh)
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

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SWEEP="$ROOT/outputs/mcf_router_layer_sweep_seed1"
OUT="$ROOT/outputs/mcf_router_recalibrate_seed1"
mkdir -p "$OUT"
echo "router recalibration | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

# name | source router | recalibrate_router.py arguments
VARIANTS=(
  "L27relneg_constrained95_fpr10|$SWEEP/L27/relneg|--objective constrained --min-recall 0.95 --target-fpr 0.10"
  "L27relneg_constrained95_fpr05|$SWEEP/L27/relneg|--objective constrained --min-recall 0.95 --target-fpr 0.05"
  "L27relneg_balanced|$SWEEP/L27/relneg|--objective balanced"
  "L19relneg_constrained95_fpr10|$SWEEP/L19/relneg|--objective constrained --min-recall 0.95 --target-fpr 0.10"
  "L19plain_constrained95_fpr10|$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router|--objective constrained --min-recall 0.95 --target-fpr 0.10"
  # the shipped rule, only the recall floor lowered (98% -> 90%); calibration split as shipped, then pooled
  "L19plain_minrecall90_cal|$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router|--objective min_recall --min-recall 0.90 --validation-splits calibration"
  "L19plain_minrecall90_pooled|$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router|--objective min_recall --min-recall 0.90"
)
# One variant only (separate jobs): NAME, SRC and ARGS set -> that variant vs its source router.
if [[ -n "${NAME:-}" ]]; then
  VARIANTS=("$NAME|$SRC|$ARGS")
fi
RUNS=("L27 relneg (shipped cutoff)=$SWEEP/L27/relneg" "L19 plain (shipped cutoff)=$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router")
for V in "${VARIANTS[@]}"; do
  IFS='|' read -r NAME SRC ARGS <<< "$V"
  if [[ ! -f "$OUT/$NAME/fact_association_embeddings.pt" ]]; then
    rm -rf "$OUT/$NAME"
    # shellcheck disable=SC2086
    python -u scripts/recalibrate_router.py --router-dir "$SRC" --output-dir "$OUT/$NAME" \
      --macro fact --device cuda --local-files-only $ARGS
  fi
  python - "$OUT/$NAME/recalibration.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
keep = {k: v for k, v in r.items() if not isinstance(v, (dict, list)) or k == "cutoff_t"}
print(sys.argv[1].split("/")[-2], json.dumps(keep, default=str)[:400])
EOF
  RUNS+=("$NAME=$OUT/$NAME")
done

SUMMARY="$OUT/summary.md"
if [[ -n "${NAME:-}" ]]; then
  RUNS=("source (shipped cutoff)=$SRC" "$NAME=$OUT/$NAME")
  SUMMARY="$OUT/summary_$NAME.md"
fi
python -u scripts/mcf_router_specificity.py --runs "${RUNS[@]}" \
  --out "$SUMMARY" --device cuda --local-files-only
echo "Finish: $(date)"
