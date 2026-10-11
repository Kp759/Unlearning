#!/bin/bash
# MCF seed 1, 50 facts, layer 19, setting A (last prompt token):
# PROJECTION edit instead of adding Δe. Same seed-1 router; no row training,
# no model weight change. When the router fires for fact i the hook does
#     h' = h - alpha * U_i U_i^T h      (U_i = fact i's answer directions)
#
# Arms (all in this one job; each is direction build + official MCF eval):
#   P_jlens_a1   J-lens directions (generic MCF prompts), alpha 1 = remove   <- main
#   P_local_a1   directions from the fact's own training views, alpha 1
#   P_jlens_a2   J-lens directions, alpha 2 = reflect the component
#   P_local_a2   local directions, alpha 2
# Reference row in the table: setting A with Δe (seed-1 multiseed regular run).
#
#   sbatch mcf_projection_seed1.slurm
#   python scripts/mcf_projection_edit.py summarize --root outputs/mcf_projection_seed1
#SBATCH --job-name=mcf_proj
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_proj_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_proj_%j.err

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
echo "python: $(command -v python)"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
OUT="$ROOT/outputs/mcf_projection_seed1"
mkdir -p "$OUT"
SEED=1
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"

ROUTER_SRC="${ROUTER_SRC:-}"
if [[ -z "$ROUTER_SRC" ]]; then
  for CANDIDATE in "$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router" \
                   "$ROOT/outputs/mcf_layer_sweep_linear_regular_v1/L19/router"; do
    if [[ -f "$CANDIDATE/fact_association_embeddings.pt" ]]; then ROUTER_SRC="$CANDIDATE"; break; fi
  done
fi
test -n "$ROUTER_SRC" || { echo "No seed-1 L19 router found" >&2; exit 2; }
echo "MCF projection | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
echo "router: $ROUTER_SRC"
nvidia-smi || true

run_arm() {  # name direction alpha [reuse-from]
  local NAME="$1" DIRECTION="$2" ALPHA="$3" REUSE="${4:-}"
  local ARM="$OUT/$NAME"
  if [[ ! -f "$ARM/linear_global/fact_association_embeddings.pt" ]]; then
    echo "===== [$NAME] BUILD directions=$DIRECTION alpha=$ALPHA | $(date) ====="
    python -u scripts/mcf_projection_edit.py build \
      --router-dir "$ROUTER_SRC" --output-dir "$ARM" \
      --direction "$DIRECTION" --alpha "$ALPHA" \
      ${REUSE:+--reuse-dirs "$OUT/$REUSE/projection_dirs.pt"} \
      --device cuda --local-files-only
  fi
  if [[ ! -f "$ARM/linear_global/official_mcf_eval.json" ]]; then
    echo "===== [$NAME] OFFICIAL MCF eval | $(date) ====="
    python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
      --run-dir "$ARM/linear_global" --mcf-path "$MCF_PATH" \
      --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
      --device cuda --dtype bfloat16 --local-files-only
  fi
  echo "===== [$NAME] COMPLETE | $(date) ====="
}

run_arm P_jlens_a1 jlens 1
run_arm P_local_a1 local 1
run_arm P_jlens_a2 jlens 2 P_jlens_a1
run_arm P_local_a2 local 2 P_local_a1

python scripts/mcf_projection_edit.py summarize --root "$OUT"
