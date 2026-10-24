#!/usr/bin/env bash
# One (model, seed, partition): build -> validate -> red-team -> baselines -> sequential -> report.
# Every stage is skipped once its output exists, so a stopped job can simply be resubmitted.
#
#   bash scripts/run_scalpel_one.sh MODEL SEED [dev|test]
#   MODEL: llama32_3b | llama31_8b | qwen3_4b | a local model directory
# Env: HF_CACHE (/project/yl258/kp759/hf), N_FORGET (50), METHODS (baselines), SURE (1 = add SURE),
#      SCALPEL_TEST_SALT (required for PARTITION=test), PRIVATE_TEMPLATES (test only), OUT_ROOT (results/cluster)
set -euo pipefail
MODEL="${1:?model}"; SEED="${2:?seed}"; PARTITION="${3:-dev}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
HF_CACHE="${HF_CACHE:-/project/yl258/kp759/hf}"
case "$MODEL" in
  llama32_3b) HF_ID=meta-llama/Llama-3.2-3B-Instruct; RMU_LAYER=7;  SCOPE=mlp ;;
  llama31_8b) HF_ID=meta-llama/Llama-3.1-8B-Instruct; RMU_LAYER=8;  SCOPE=mlp_down ;;   # fp32 Adam on all 8B MLPs does not fit 80 GB
  qwen3_4b)   HF_ID=Qwen/Qwen3-4B;                    RMU_LAYER=9;  SCOPE=mlp ;;
  *)          HF_ID=""; MODEL_PATH="$MODEL"; RMU_LAYER="${RMU_LAYER:-7}"; SCOPE="${SCOPE:-mlp}" ;;
esac
if [[ -n "$HF_ID" ]]; then
  MODEL_PATH=""
  for CFG in "$HF_CACHE/models--${HF_ID//\//--}"/snapshots/*/config.json; do
    [[ -f "$CFG" ]] && MODEL_PATH="$(dirname "$CFG")"
  done
  [[ -n "$MODEL_PATH" ]] || { echo "model $HF_ID not in $HF_CACHE (download it on a login node first)" >&2; exit 2; }
fi
TAG="$(basename "$MODEL")"
OUT="${OUT_ROOT:-results/cluster}/$TAG/$PARTITION/seed$SEED"
SPLIT="$OUT/split.json"
mkdir -p "$OUT"
SALT_ARGS=()
if [[ "$PARTITION" == test ]]; then
  [[ -n "${SCALPEL_TEST_SALT:-}" ]] || { echo "PARTITION=test needs SCALPEL_TEST_SALT" >&2; exit 2; }
  SALT_ARGS=(--partition test --salt "$SCALPEL_TEST_SALT")
  [[ -n "${PRIVATE_TEMPLATES:-}" ]] && SALT_ARGS+=(--private-templates "$PRIVATE_TEMPLATES")
fi
log() { echo "[$(date +%H:%M:%S)] [$TAG $PARTITION s$SEED] $*"; }

if [[ ! -f "$SPLIT" ]]; then
  log "build"
  python -u -m scalpel build --model "$MODEL_PATH" --model-id "${HF_ID:-$TAG}" --seed "$SEED" \
    --n-forget "${N_FORGET:-50}" --sources mcf,zsre,mquake --out "$SPLIT" --device cuda --dtype bf16 "${SALT_ARGS[@]}"
fi
python -m scalpel validate --split "$SPLIT" > "$OUT/validation.json" || log "validation reported failures (see validation.json)"

COMMON=(--split "$SPLIT" --model "$MODEL_PATH" --device cuda --dtype bf16 --batch-size 16)
log "red team (reference cheats)"
python -u -m scalpel redteam "${COMMON[@]}" --out "$OUT/runs"

METHODS="${METHODS:-ga,gd,npo,rmu}"
log "baselines: $METHODS"
python -u -m scalpel suite "${COMMON[@]}" --methods "$METHODS" --out "$OUT/runs" \
  --method-arg "ga.params=$SCOPE" --method-arg "gd.params=$SCOPE" --method-arg "npo.params=$SCOPE" \
  --method-arg "rmu.layer=$RMU_LAYER"
if [[ "${SURE:-0}" == 1 ]]; then
  log "SURE (repository method, trained on this split's requests only)"
  python -u -m scalpel run "${COMMON[@]}" --method sure --method-arg "model_path=$MODEL_PATH" \
    --method-arg "work_dir=$OUT/sure_work" --method-arg device=cuda --out "$OUT/runs/sure"
fi
if [[ ! -f "$OUT/sequential_gd/sequential.json" ]]; then
  log "sequential track (gd, 5 batches)"
  python -u -m scalpel sequential "${COMMON[@]}" --method gd --method-arg "params=$SCOPE" --steps 5 \
    --out "$OUT/sequential_gd" --no-gen --attacks ""
fi
python -m scalpel report --runs "$OUT/runs/*/results.json" --out "$OUT/leaderboard" \
  --title "SCALPEL — $TAG — $PARTITION seed $SEED"
log "done -> $OUT/leaderboard/leaderboard.md"
