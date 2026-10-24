#!/usr/bin/env bash
# Scaling curve: the same model and seed with growing deletion sets (default 50 200 1000).
# Every size is its own split (knowledge filter re-run, so larger sets reach rarer facts).
#   bash scripts/run_scaling.sh MODEL_PATH [TAG]        (cluster: run inside a GPU job)
# Env: SIZES ("50 200 1000"), METHODS (ga,gd,npo,rmu), DEVICE (cuda), DTYPE (bf16), SEED (1)
set -euo pipefail
MODEL_PATH="${1:?model dir}"; TAG="${2:-$(basename "$MODEL_PATH")}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="results/scaling/$TAG"
for N in ${SIZES:-50 200 1000}; do
  D="$OUT/n$N"; mkdir -p "$D"
  [[ -f "$D/split.json" ]] || python -u -m scalpel build --model "$MODEL_PATH" --seed "${SEED:-1}" --n-forget "$N" \
      --n-disjoint "$(( N * 3 ))" --screen-max 200000 --out "$D/split.json" --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bf16}" \
      2>&1 | tail -3 || true
  python -u -m scalpel suite --split "$D/split.json" --model "$MODEL_PATH" --methods "${METHODS:-ga,gd,npo,rmu}" \
    --out "$D/runs" --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bf16}"
done
python -m scalpel report --runs "$OUT/n*/runs/*/results.json" --out "$OUT/leaderboard" --title "SCALPEL scaling — $TAG"
