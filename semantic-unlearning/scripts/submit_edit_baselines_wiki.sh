#!/usr/bin/env bash
# ZeroUnlearn, ROME and MEMIT on a real-Wikipedia covariance, 4 datasets x 5 seeds.
# The vendored loaders estimate covariance from data/wikidata, the 200-sentence PPL fixture
# (rank-deficient for an 8192-d down_proj input): ROME / MEMIT collapsed on it (PPL 1e5-1e6).
#
#   bash scripts/submit_edit_baselines_wiki.sh corpus   # 1. login node (internet), once: 100,020
#                                                       #    articles of wikimedia/wikipedia 20231101.en
#   bash scripts/submit_edit_baselines_wiki.sh submit   # 2. covariance job, then 60 baseline tasks
#
# Existing ZeroUnlearn / ROME / MEMIT results on another covariance are moved aside
# (*.superseded_<time>) and rerun by the driver.
set -euo pipefail
cd "$(dirname "$0")/.."
TEXT="${MOM2_TEXT:-/project/yl258/kp759/datasets/wikipedia_20231101_100k}"
NAME="${MOM2_DATASET:-wikipedia_20231101_100k}"
case "${1:-}" in
  corpus)
    if [[ -f "$TEXT/dataset_dict.json" ]]; then echo "corpus already at $TEXT"; exit 0; fi
    python -c "import datasets" 2>/dev/null || {
      echo "python ($(command -v python)) has no 'datasets': activate the env first, e.g." >&2
      echo "  module load Miniforge3; conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning" >&2
      exit 2; }
    python -u scripts/prepare_sure_wikipedia_corpus.py --output-dir "$TEXT"
    ;;
  submit)
    [[ -f "$TEXT/dataset_dict.json" ]] || { echo "no corpus at $TEXT: run '$0 corpus' on the login node first" >&2; exit 2; }
    P=$(sbatch --parsable --export=ALL,MOM2_TEXT="$TEXT",MOM2_DATASET="$NAME" edit_covariance.slurm)
    echo "covariance ($NAME): $P"
    ZERO_TASKS=$(seq -s, 4 5 99)          # l16_baselines.slurm: task = 25*ds + 5*(seed-1) + 4
    Z=$(sbatch --parsable --dependency=afterok:"$P" \
          --export=ALL,BASELINE_MOM2_TEXT="$TEXT",BASELINE_MOM2_DATASET="$NAME" \
          --array="$ZERO_TASKS" l16_baselines.slurm)
    echo "ZeroUnlearn, 20 tasks (after $P): $Z"
    E=$(sbatch --parsable --dependency=afterok:"$P" \
          --export=ALL,BASELINE_MOM2_TEXT="$TEXT",BASELINE_MOM2_DATASET="$NAME" \
          --array=0-39 l16_edit_baselines.slurm)
    echo "ROME + MEMIT, 40 tasks (after $P): $E"
    echo "then: python scripts/l16_paper_table.py --ours-from mcf=outputs/l16_paper_mcf_linear19"
    ;;
  *) echo "usage: $0 corpus|submit" >&2; exit 2 ;;
esac
