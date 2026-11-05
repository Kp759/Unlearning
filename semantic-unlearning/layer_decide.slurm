#!/bin/bash
# CPU steps of the automatic layer chain (scripts/submit_layer_auto.sh): no GPU needed.
#   STAGE=router  scripts/layer_decide.py MODEL router  -> decision.json: router_layer, write_candidates
#   STAGE=write   scripts/layer_decide.py MODEL write   -> decision.json: write_layer, tag
#   STAGE=table   the paper table of the finished full run (tag from decision.json)
#SBATCH --job-name=layer_decide
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/layer_decide_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/layer_decide_%j.err

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
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
MODEL="${MODEL:?MODEL=qwen3_4b|llama31_8b}"
STAGE="${STAGE:?STAGE=router|write|table}"
DECISION="outputs/layer_smoke_${MODEL}/decision.json"
echo "layer decide | $MODEL $STAGE | job ${SLURM_JOB_ID:-} | $(date) | commit $(git rev-parse HEAD)"
case "$STAGE" in
  router|write)
    python scripts/layer_decide.py "$MODEL" "$STAGE" || exit $?
    python scripts/layer_smoke_table.py "$MODEL" > /dev/null || true
    cat "$DECISION" | jq 'del(.write_table)' ;;
  table)
    TAG="$(jq -r '.tag' "$DECISION")"; R="$(jq -r '.router_layer' "$DECISION")"; W="$(jq -r '.write_layer' "$DECISION")"
    python scripts/l16_paper_table.py --root "outputs/$TAG" --ours-label "Ours (L$R router, Δe L$W)" ;;
  *) echo "STAGE must be router, write or table" >&2; exit 2 ;;
esac
STATUS=$?
echo "Finish $MODEL $STAGE: $(date) | exit $STATUS"
exit $STATUS
