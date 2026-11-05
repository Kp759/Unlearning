#!/bin/bash
# C2: round-3 refits (plain rewrites) + per-head bias, no gates. Build the four routers, then
# submit the seed-1 pipeline (outputs/l16_ours_data2_v1).
#SBATCH --job-name=build_c2
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/build_c2_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/build_c2_%j.err
set -uo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning || exit 1
export PATH=/scratch/yl258/kp759/conda_envs/semantic_unlearning/bin:$PATH PYTHONPATH=scripts OMP_NUM_THREADS=8
GRID="${GRID:-outputs/router_data_grid_v3}"; TAG="${TAG:-multiseed_ours_data2_v1}"
RECIPE="${RECIPE:-ours_data2}"; RUN_TAG="${RUN_TAG:-l16_ours_data2_v1}"
for pair in ${PAIRS:-mcf:p40_n40 zsre:p20_n40 mquake:p40_n60 rwku:p40_n20}; do
  ds=${pair%%:*}; cell=${pair#*:}
  python -u scripts/build_gated_router.py --datasets "$ds" --seed 1 --source-router "$GRID/$ds/seed1/$cell/router" \
    --no-relation-gate --no-combination-gate --out-tag "$TAG" --overwrite || exit 2
done
sbatch --parsable --time=08:00:00 --mem=64G --array=0,5,10,15 \
  --export=ALL,ONEPASS=1,WRITE_LAYER=16,ROUTER_RECIPE=$RECIPE,TAG=$RUN_TAG,BASE_FROM=l16_ours_v1,BUILD_ROUTERS=0,BASELINES= \
  l16_paper_multiseed.slurm | tee logs/submit_c2_pipeline.log
