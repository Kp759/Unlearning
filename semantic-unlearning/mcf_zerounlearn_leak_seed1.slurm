#!/bin/bash
# ZeroUnlearn on MCF seed 1 (50 facts): does the forgotten answer still appear in free
# generation? Same prompts and leak check as the Δe / projection α-sweep runs.
#
#   1. base greedy continuations (24 tokens) on the canonical + development forget prompts
#   2. original ZeroUnlearn, target_true parity mapping (as in the parity multiseed run)
#   3. official MCF eval of the edited model (skipped if the parity run already has seed 1,
#      or with OFFICIAL=0)
#   4. edited greedy continuations; restore W_down; fluency/leak scoring
#   5. comparison table: base / Δe / projection α 2.5, 3 / ZeroUnlearn
#
#   sbatch mcf_zerounlearn_leak_seed1.slurm
#   cat outputs/zerounlearn_generation_leak_seed1/comparison.md
#SBATCH --job-name=mcf_zul_leak
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_zul_leak_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_zul_leak_%j.err

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
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
OUT="$ROOT/outputs/zerounlearn_generation_leak_seed1"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MODEL_PATH="${MODEL_PATH:-$(jq -r '.model_path' "$REF")}"
MCF_PATH="${MCF_PATH:-$(jq -r '.mcf_path' "$REF")}"
ZERO_ROOT="${ZERO_ROOT:-$ROOT/../ZeroUnlearn}"
test -d "$ZERO_ROOT/ZeroUnlearn" || { echo "ZeroUnlearn code not found at $ZERO_ROOT" >&2; exit 2; }
[[ -d "$ROOT/data/stats" ]] || echo "WARNING: $ROOT/data/stats missing; ZeroUnlearn will try to recompute mom2 stats"

# Official eval: reuse the parity run's seed-1 numbers if they exist.
PARITY="$ROOT/outputs/zerounlearn_targettrue_parity_seeds1_10/seed1/zerounlearn_targettrue_seed1_official_eval.json"
OFFICIAL="${OFFICIAL:-auto}"
EVAL_FLAG=""
if [[ "$OFFICIAL" == "1" || ( "$OFFICIAL" == "auto" && ! -f "$PARITY" ) ]]; then EVAL_FLAG="--official-eval"; fi
echo "ZeroUnlearn leak | seed $SEED | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
echo "model: $MODEL_PATH | zero root: $ZERO_ROOT | official eval: ${EVAL_FLAG:-reuse $PARITY}"
nvidia-smi || true

if [[ ! -f "$OUT/zerounlearn/routed_generation.json" ]]; then
  python -u scripts/zerounlearn_generation_leak.py run \
    --output-dir "$OUT" --seed "$SEED" --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
    --zero-unlearn-root "$ZERO_ROOT" \
    --hparams-path "$ZERO_ROOT/hparams/ZeroUnlearn/Llama-3.2-3B-Instruct.json" \
    --wikidata-dir "$ROOT/data/wikidata" --device cuda --local-files-only $EVAL_FLAG
fi

python scripts/zerounlearn_generation_leak.py compare --zero-dir "$OUT" \
  --sweep-root "$ROOT/outputs/mcf_projection_alpha_sweep_seed1" --alphas 2.5 3 \
  --zero-official "$PARITY"     # used only when this run has no official eval of its own
echo "Finish: $(date)"
