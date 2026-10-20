#!/bin/bash
# Held-out generation check, MCF seed 1 (50 forget facts), greedy 24 tokens on the MCF
# records' own prompts: rewrite (50), paraphrase (100, held out), generation_prompts
# (same subject, other topics, ~150, held out), neighbourhood (500, held out).
# Methods: base, Δe (setting A), Δe→EOS, Δe→IDK (refusal rows), ZeroUnlearn (re-applied).
# Answers: leak on held-out paraphrases, and over-refusal on neighbourhood / same-subject
# prompts. Training-view leak is split canonical vs development from the earlier runs.
#
#   sbatch mcf_heldout_generation_seed1.slurm
#   cat outputs/mcf_heldout_generation_seed1/summary.md
#SBATCH --job-name=mcf_heldout
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_heldout_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_heldout_%j.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
OUT="$ROOT/outputs/mcf_heldout_generation_seed1"
mkdir -p "$OUT"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MODEL_PATH="$(jq -r '.model_path' "$REF")"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
ZERO_ROOT="$ROOT/../ZeroUnlearn"
REFUSAL="$ROOT/outputs/mcf_refusal_rows_seed1"
DELTA_E="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global"
echo "MCF held-out generation | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

gen() {  # stem method label [run_dir]
  local STEM="$1" METHOD="$2" LABEL="$3" RUN="${4:-}"
  if [[ -f "$OUT/$STEM.json" ]]; then echo "skip $STEM (exists)"; return; fi
  echo "===== [$STEM] $LABEL | $(date) ====="
  python -u scripts/mcf_heldout_generation.py run --method "$METHOD" --label "$LABEL" \
    --out "$OUT/$STEM.json" --seed "$SEED" --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
    --zero-unlearn-root "$ZERO_ROOT" \
    --hparams-path "$ZERO_ROOT/hparams/ZeroUnlearn/Llama-3.2-3B-Instruct.json" \
    ${RUN:+--run-dir "$RUN"} --device cuda --local-files-only
}

gen base        base        "base"
gen delta_e     router      "Δe (setting A)"   "$DELTA_E"
gen R_eos       router      "Δe→EOS"           "$REFUSAL/R_eos/linear_global"
gen R_idk       router      "Δe→IDK"           "$REFUSAL/R_idk/linear_global"
gen zerounlearn zerounlearn "ZeroUnlearn"

SWEEP="$ROOT/outputs/mcf_projection_alpha_sweep_seed1"
python scripts/mcf_heldout_generation.py summarize --root "$OUT" --seed "$SEED" --train-views \
  "base=$SWEEP/base/routed_generation.json" \
  "delta_e=$SWEEP/reference_delta_e/routed_generation.json" \
  "R_eos=$REFUSAL/R_eos/routed_generation.json" \
  "R_idk=$REFUSAL/R_idk/routed_generation.json" \
  "zerounlearn=$ROOT/outputs/zerounlearn_generation_leak_seed1/zerounlearn/routed_generation.json"
echo "Finish: $(date)"
