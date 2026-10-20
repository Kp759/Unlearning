#!/bin/bash
# MCF seed 1, 50 facts, layer 19, setting A: alpha sweep of the projection edit
#     h' = h - alpha * U_i U_i^T h        (U_i = fact i's J-lens answer directions)
# alpha 1 = remove the component, 2 = reflect it, >2 = push further away.
#
# Reuses the J-lens directions from mcf_projection_seed1.slurm
# (outputs/mcf_projection_seed1/P_jlens_a1/projection_dirs.pt), so every alpha uses the
# SAME directions; builds them first if missing. Per alpha: artifact + official MCF eval
# + routed generation (answer leakage anywhere in a 24-token continuation, fluency =
# base-model NLL, repeated 3-grams). Also scores two reference rows the same way:
#   base       the seed-1 router with zero rows (= exact base model)
#   Δe         setting A with the trained rows (seed-1 multiseed regular run)
#
#   sbatch mcf_projection_alpha_sweep_seed1.slurm
#   sbatch --export=ALL,ALPHAS="2 3 4" mcf_projection_alpha_sweep_seed1.slurm
#   python scripts/mcf_projection_edit.py summarize --root outputs/mcf_projection_alpha_sweep_seed1
#SBATCH --job-name=mcf_proj_alpha
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_proj_alpha_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_proj_alpha_%j.err

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
echo "python: $(command -v python)"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
ALPHAS="${ALPHAS:-0.5 1 1.5 2 2.5 3 4 6}"
OUT="$ROOT/outputs/mcf_projection_alpha_sweep_seed1"
mkdir -p "$OUT"
SEED=1
MCF_PATH="$(jq -r '.mcf_path' "$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json")"
DELTA_E_RUN="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global"

ROUTER_SRC="${ROUTER_SRC:-}"
if [[ -z "$ROUTER_SRC" ]]; then
  for CANDIDATE in "$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router" \
                   "$ROOT/outputs/mcf_layer_sweep_linear_regular_v1/L19/router"; do
    if [[ -f "$CANDIDATE/fact_association_embeddings.pt" ]]; then ROUTER_SRC="$CANDIDATE"; break; fi
  done
fi
test -n "$ROUTER_SRC" || { echo "No seed-1 L19 router found" >&2; exit 2; }
echo "MCF projection alpha sweep | alphas: $ALPHAS | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
echo "router: $ROUTER_SRC"
nvidia-smi || true

official_eval() {  # run_dir
  if [[ ! -f "$1/official_mcf_eval.json" ]]; then
    python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
      --run-dir "$1" --mcf-path "$MCF_PATH" --wikidata-dir "$ROOT/data/wikidata" \
      --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only
  fi
}
generate() {  # run_dir out_json
  if [[ ! -f "$2" ]]; then
    python -u scripts/mcf_projection_edit.py routed-generate \
      --run-dir "$1" --out "$2" --device cuda --local-files-only
  fi
}

# J-lens directions (shared by every alpha).
DIRS="$ROOT/outputs/mcf_projection_seed1/P_jlens_a1/projection_dirs.pt"
if [[ ! -f "$DIRS" ]]; then
  echo "===== building J-lens directions | $(date) ====="
  python -u scripts/mcf_projection_edit.py build --router-dir "$ROUTER_SRC" \
    --output-dir "$ROOT/outputs/mcf_projection_seed1/P_jlens_a1" \
    --direction jlens --alpha 1 --device cuda --local-files-only
fi
echo "directions: $DIRS"

# Reference rows.
echo "===== [base] router with zero rows | $(date) ====="
if [[ ! -f "$OUT/base/linear_global/fact_association_embeddings.pt" ]]; then
  mkdir -p "$OUT/base"; rm -rf "$OUT/base/linear_global"
  cp -r "$ROUTER_SRC" "$OUT/base/linear_global"
fi
official_eval "$OUT/base/linear_global"
generate "$OUT/base/linear_global" "$OUT/base/routed_generation.json"
if [[ -f "$DELTA_E_RUN/fact_association_embeddings.pt" ]]; then
  echo "===== [Δe] reference generation | $(date) ====="
  generate "$DELTA_E_RUN" "$OUT/reference_delta_e/routed_generation.json"
fi

for ALPHA in $ALPHAS; do
  NAME="a${ALPHA}"
  ARM="$OUT/$NAME"
  echo "===== [$NAME] alpha=$ALPHA | $(date) ====="
  if [[ ! -f "$ARM/linear_global/fact_association_embeddings.pt" ]]; then
    python -u scripts/mcf_projection_edit.py build --router-dir "$ROUTER_SRC" \
      --output-dir "$ARM" --alpha "$ALPHA" --reuse-dirs "$DIRS" \
      --device cuda --local-files-only
  fi
  official_eval "$ARM/linear_global"
  generate "$ARM/linear_global" "$ARM/routed_generation.json"
done

python scripts/mcf_projection_edit.py summarize --root "$OUT"
echo "Finish: $(date)"
