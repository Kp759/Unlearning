#!/bin/bash
# Does more training data fix the router's same-subject misfires? MCF seed 1, gates off.
# 1. a bigger pool of base-model rewrites (positives) and model-written hard negatives
#    (build_router_rewrites.py, same quality checks as the paper's 6 + 3 / 8), written next to the
#    paper's prep dir, which is not touched
# 2. the paper's recipe refit at every (positives, negatives) cell; the shipped router is 20 / 20
# 3. each refit scored on the cached seed-1 prompts (router_data_grid_score.py): held-out
#    same-subject misfires, official forget recall, retain / no-association fires, with and
#    without the per-head bias
#   sbatch router_data_grid.slurm
#SBATCH --job-name=data_grid
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/data_grid_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/data_grid_%j.err
set -uo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning || exit 1
export PATH=/scratch/yl258/kp759/conda_envs/semantic_unlearning/bin:$PATH PYTHONPATH=scripts
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DS="${DS:-mcf}"; SEED="${SEED:-1}"
PREP="outputs/${DS}_multiseed_ours_v1/seed$SEED/L19/prep"
GRID="${GRID:-outputs/router_data_grid_v1}"
POOL="$GRID/$DS/seed$SEED/pool"
HARDNEG="${HARDNEG:-30}"
mkdir -p "$POOL"
echo "===== 1. bigger pools | $(date)"
if [[ ! -f "$POOL/router_rewrites.json" ]]; then
  python -u scripts/build_router_rewrites.py --prep-dir "$PREP" --output "$POOL/router_rewrites.json" \
    --n-para "${NPARA:-20}" --n-question "${NQUESTION:-10}" --hard-negatives "$HARDNEG" --device cuda --local-files-only || exit 2
fi
python - "$POOL" <<'EOF'
import json, sys, collections
from pathlib import Path
p = Path(sys.argv[1])
r = json.load(open(p / "router_rewrites.json")); h = json.load(open(p / "router_hard_negatives.json"))
per = collections.Counter(e["fact_id"] for e in r)
print(json.dumps({"rewrites": len(r), "rewrites_per_fact_median": sorted(per.values())[len(per)//2] if per else 0,
                  "hard_negatives": sum(len(v) for v in h.values()), "facts_with_hard_negatives": len(h)}))
EOF
FIT="--pca-dims 0 --subject-match fuzzy --max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5"
for CELL in ${CELLS:-20x20 20x40 20x60 40x20 40x40 40x60}; do
  P=${CELL%x*}; N=${CELL#*x}
  OUT="$GRID/$DS/seed$SEED/p${P}_n${N}/router"
  [[ -f "$OUT/fact_association_embeddings.pt" ]] && { echo "have $OUT"; continue; }
  echo "===== 2. refit positives=$P negatives=$N | $(date)"
  python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$OUT" --device cuda --local-files-only \
    --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1 $FIT \
    --positives-per-fact "$P" --negatives-per-fact "$N" \
    --views "$POOL/router_rewrites.json" --extra-negatives "$POOL/router_hard_negatives.json" || { echo "fit $CELL failed"; continue; }
done
echo "===== 3. score | $(date)"
python -u scripts/router_data_grid_score.py --dataset "$DS" --seed "$SEED" --grid "$GRID"
echo "exit $? $(date)"
