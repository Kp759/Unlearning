#!/bin/bash
# MCF seed 1, 50 facts, layer 19, setting A: train the Δe rows toward a REFUSAL target
# (ZeroUnlearn-style) instead of only pushing the answer away. Same router, no weight change.
#   R_eos   target = EOS                     (model stops right after the prompt)
#   R_idk   target = " I don't know." + EOS  (declines, then stops)
# Per arm: train rows -> official MCF eval -> routed greedy generation (24 tokens).
# Then one table + before/after examples against base, Δe, projection α=3 and ZeroUnlearn
# (needs outputs/zerounlearn_generation_leak_seed1 from mcf_zerounlearn_leak_seed1.slurm).
#
#   sbatch mcf_refusal_rows_seed1.slurm
#   cat outputs/mcf_refusal_rows_seed1/comparison.md
#SBATCH --job-name=mcf_refusal
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_refusal_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_refusal_%j.err

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
echo "python: $(command -v python)"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
OUT="$ROOT/outputs/mcf_refusal_rows_seed1"
mkdir -p "$OUT"
MCF_PATH="$(jq -r '.mcf_path' "$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json")"
TARGETS="${TARGETS:-eos idk}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

ROUTER_SRC="${ROUTER_SRC:-}"
if [[ -z "$ROUTER_SRC" ]]; then
  for CANDIDATE in "$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router" \
                   "$ROOT/outputs/mcf_layer_sweep_linear_regular_v1/L19/router"; do
    if [[ -f "$CANDIDATE/fact_association_embeddings.pt" ]]; then ROUTER_SRC="$CANDIDATE"; break; fi
  done
fi
test -n "$ROUTER_SRC" || { echo "No seed-1 L19 router found" >&2; exit 2; }
echo "MCF refusal rows | targets: $TARGETS | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
echo "router: $ROUTER_SRC"
nvidia-smi || true

EXTRA=()
for TARGET in $TARGETS; do
  ARM="$OUT/R_$TARGET"
  echo "===== [R_$TARGET] | $(date) ====="
  if [[ ! -f "$ARM/linear_global/training_report.json" ]]; then
    rm -rf "$ARM/linear_global"; mkdir -p "$ARM"
    python -u scripts/train_mcf_refusal_rows.py --router-dir "$ROUTER_SRC" \
      --output-dir "$ARM/linear_global" --target "$TARGET" --device cuda --local-files-only \
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
  EXTRA+=("Δe→$TARGET (refusal rows)=$ARM")
done

python scripts/zerounlearn_generation_leak.py compare \
  --zero-dir "$ROOT/outputs/zerounlearn_generation_leak_seed1" \
  --sweep-root "$ROOT/outputs/mcf_projection_alpha_sweep_seed1" --alphas 3 \
  --zero-official "$ROOT/outputs/zerounlearn_targettrue_parity_seeds1_10/seed1/zerounlearn_targettrue_seed1_official_eval.json" \
  --extra "${EXTRA[@]}" --examples 8 --out "$OUT/comparison.md"
echo "Finish: $(date)"
