#!/bin/bash
# Factorized subject x relation router, MCF seed 1 (scripts/factorized_router.py).
# Both parts LEARNED: a subject head (forget subjects + none; typo / case / prefix
# augmented; near-miss names as negatives) and a relation head (34 MCF relations,
# learned from ~27k prompts of OTHER subjects). Fire fact f iff
# p_subj(subject_f) >= t_s and p_rel(relation_f) >= t_r (calibrated on validation).
#
#   1  fit at layers 19 23 27 (one feature pass), pick the best layer on validation
#   2  routing-only table vs the old router and the L27 relation-aware router:
#      recall, neighbour / same-subject / near-miss fire, typo + lower-case recall
#   3  attach the selected router to the trained question-side Δe rows
#      (E_L16_query_logit) and run the held-out leak eval (set WITH_GENERATION=0 to skip)
#
#   sbatch mcf_factorized_router_seed1.slurm
#   then: cat outputs/mcf_factorized_router_seed1/specificity.md
#         python -c "import json;d=json.load(open('outputs/mcf_factorized_router_seed1/summary.json'));print(d['selected_layer']);[print(L,v['calibration']) for L,v in d['layers'].items()]"
#SBATCH --job-name=mcf_fact_router
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_fact_router_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_fact_router_%j.err

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
OUT="$ROOT/outputs/mcf_factorized_router_seed1"
OLD="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router"
RELNEG27="$ROOT/outputs/mcf_router_layer_sweep_seed1/L27/relneg"
ROWS="$ROOT/outputs/mcf_early_write_seed1/E_L16_query_logit/linear_global"
HELD="$ROOT/outputs/mcf_heldout_generation_seed1"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
read -r -a LAYER_LIST <<< "${LAYERS:-19 23 27}"
echo "factorized router | layers ${LAYER_LIST[*]} | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

if [[ ! -f "$OUT/summary.json" ]]; then
  python -u scripts/fit_factorized_router.py --reference-dir "$OLD" --output-dir "$OUT" \
    --layers "${LAYER_LIST[@]}" --min-recall 0.95 --device cuda --local-files-only
fi
SEL="$(jq -r '.selected_layer' "$OUT/summary.json")"
SELDIR="$OUT/L$(printf '%02d' "$SEL")"
echo "selected layer: $SEL"

RUNS=("old L19 router=$OLD")
[[ -f "$RELNEG27/fact_association_embeddings.pt" ]] && RUNS+=("L27 relation-aware=$RELNEG27")
for L in "${LAYER_LIST[@]}"; do RUNS+=("factorized L$L=$OUT/L$(printf '%02d' "$L")"); done
if [[ ! -f "$OUT/specificity.json" ]]; then
  python -u scripts/mcf_router_specificity.py --runs "${RUNS[@]}" \
    --out "$OUT/specificity.md" --device cuda --local-files-only
fi

if [[ "${WITH_GENERATION:-1}" == "1" && -f "$ROWS/fact_association_embeddings.pt" ]]; then
  ATT="$OUT/attached_E_L16_query_logit_L$SEL"
  [[ -f "$ATT/fact_association_embeddings.pt" ]] || \
    python scripts/factorized_router.py attach --router-dir "$SELDIR" --rows-from "$ROWS" --output-dir "$ATT"
  if [[ ! -f "$HELD/F_L${SEL}_query_logit.json" ]]; then
    python -u scripts/mcf_heldout_generation.py run --method router --run-dir "$ATT" \
      --label "factorized router L$SEL + question-side Δe (E_L16_query_logit rows)" \
      --out "$HELD/F_L${SEL}_query_logit.json" --seed "$SEED" --mcf-path "$MCF_PATH" \
      --device cuda --local-files-only
  fi
  bash scripts/run_refusal_summaries.sh
fi
echo "Finish: $(date)"
