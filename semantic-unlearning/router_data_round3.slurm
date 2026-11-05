#!/bin/bash
# Lever 2 (C2): a third round of rewrites in the plain, conversational forms the classifier misses
# (router_rewrites._floor_ask, forced for every fact), merged with the round-1 pools, refit at the
# C cells, scored (router_data_grid_score.py). One task per dataset.
#SBATCH --job-name=round3
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/round3_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/round3_%A_%a.err
set -uo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning || exit 1
export PATH=/scratch/yl258/kp759/conda_envs/semantic_unlearning/bin:$PATH PYTHONPATH=scripts
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DSS=(mcf zsre mquake rwku); DS="${DSS[$SLURM_ARRAY_TASK_ID]}"; SEED=1
CELLS=(p40_n40 p20_n40 p40_n60 p40_n20); CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
PREP="outputs/${DS}_multiseed_ours_v1/seed$SEED/L19/prep"
R1="outputs/router_data_grid_v1/$DS/seed$SEED/pool"
GRID="outputs/router_data_grid_v3"; POOL="$GRID/$DS/seed$SEED/pool"; mkdir -p "$POOL"
echo "===== 1. round-3 plain rewrites | $DS | $(date)"
if [[ ! -f "$POOL/router_rewrites_plain.json" ]]; then
  python -u scripts/build_router_rewrites.py --prep-dir "$PREP" --output "$POOL/router_rewrites_plain.json" \
    --kinds paraphrase --n-para 1 --floor 999 --n-floor 12 --hard-negatives 0 --device cuda --local-files-only || exit 2
fi
python - "$R1" "$POOL" <<'PY'
import json, sys, collections
from pathlib import Path
r1, pool = Path(sys.argv[1]), Path(sys.argv[2])
a = json.load(open(r1 / "router_rewrites.json")); b = json.load(open(pool / "router_rewrites_plain.json"))
seen = {(e["fact_id"], " ".join(e["prompt"].casefold().split())) for e in a}
new = [e for e in b if (e["fact_id"], " ".join(e["prompt"].casefold().split())) not in seen]
json.dump(a + new, open(pool / "router_rewrites.json", "w"), indent=1, ensure_ascii=False)
import shutil; shutil.copy(r1 / "router_hard_negatives.json", pool / "router_hard_negatives.json")
per = collections.Counter(e["fact_id"] for e in a + new)
print(json.dumps({"round1": len(a), "round3_new": len(new), "total": len(a) + new.__len__(),
                  "per_fact_median": sorted(per.values())[len(per)//2]}))
PY
P=${CELL#p}; P=${P%%_*}; N=${CELL#*_n}
FIT="--pca-dims 0 --subject-match fuzzy --max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5"
GATE=""; [[ "$DS" == rwku ]] && GATE="--gate threshold"
for PP in "$P" $((P + 20)); do
  OUT="$GRID/$DS/seed$SEED/p${PP}_n${N}/router"
  [[ -f "$OUT/fact_association_embeddings.pt" ]] && continue
  echo "===== 2. refit positives=$PP negatives=$N | $(date)"
  python -u scripts/fit_linear_router.py --run-dir "$PREP" --output-dir "$OUT" --device cuda --local-files-only \
    --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1 $FIT $GATE \
    --positives-per-fact "$PP" --negatives-per-fact "$N" \
    --views "$POOL/router_rewrites.json" --extra-negatives "$POOL/router_hard_negatives.json" || echo "fit p${PP}_n${N} failed"
done
echo "===== 3. score | $(date)"
python -u scripts/router_data_grid_score.py --dataset "$DS" --seed "$SEED" --grid "$GRID"
echo "exit $? $(date)"
