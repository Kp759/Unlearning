#!/usr/bin/env bash
# Laptop pilot (CPU / Apple MPS, no GPU needed): a small open model on real MCF + zsRE + MQuAKE.
#   bash scripts/run_pilot_local.sh [HF model id]     (default Qwen/Qwen2.5-0.5B-Instruct)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HF_ID="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="$(basename "$HF_ID" | tr '[:upper:]' '[:lower:]')"
MODEL_PATH="$(python -c "from huggingface_hub import snapshot_download as s; print(s('$HF_ID', allow_patterns=['*.json','*.safetensors','*.txt']))")"
OUT="results/pilot/$TAG"; SPLIT="$OUT/split_dev_s1.json"
DEV="${DEVICE:-auto}"
export TOKENIZERS_PARALLELISM=false
[[ -f "$SPLIT" ]] || python -u -m scalpel build --model "$MODEL_PATH" --model-id "$HF_ID" --sources mcf,zsre,mquake \
  --out "$SPLIT" --n-forget "${N_FORGET:-50}" --device "$DEV" --dtype fp32 --batch-size 32
python -m scalpel validate --split "$SPLIT" > "$OUT/validation.json" || true
COMMON=(--split "$SPLIT" --model "$MODEL_PATH" --device "$DEV" --dtype fp32 --batch-size 16 --relearn-scope "${RELEARN_SCOPE:-mlp_down}")
python -u -m scalpel redteam "${COMMON[@]}" --out "$OUT/runs"
python -u -m scalpel suite "${COMMON[@]}" --methods "${METHODS:-ga,gd,npo,rmu}" --out "$OUT/runs"
python -m scalpel report --runs "$OUT/runs/*/results.json" --out "$OUT/leaderboard" --title "SCALPEL pilot — $HF_ID"
