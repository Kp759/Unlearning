#!/usr/bin/env bash
# Per-model settings for the paper table on models other than Llama-3.2-3B-Instruct.
# Sourced by paper_model_multiseed.slurm, paper_model_baselines.slurm and
# scripts/submit_paper_model.sh with MODEL set:
#
#   MODEL=llama31_8b   meta-llama/Llama-3.1-8B-Instruct   32 layers, hidden 4096, untied head
#   MODEL=qwen3_4b     Qwen/Qwen3-4B (no-thinking prompts)  36 layers, hidden 2560, tied head
#
# Layers are the 3B choices moved to the same relative depth (3B has 28 blocks):
#   write 16 -> 16/28, router 19, logit lens 24, v2.2 subject 6, v2.2 relation candidates
#   19 23 24 27 (concat 19,23,27; 27 = last block), RMU 7. Not re-tuned per model.
# ZeroUnlearn uses the upstream hparams file shipped for each model (no tuning by us).
#
# Exports: MODEL_PATH MODEL_TAG TAG ROUTER_LAYER WRITE_LAYER LENS_LAYER V22_SUBJECT_LAYERS
#          V22_RELATION_LAYERS V22_RELATION_CONCAT BASELINE_ARGS HF_ID N_LAYERS MIN_GPU_MIB
#          (+ CHAT_ENABLE_THINKING=0 for Qwen3)

HF_CACHE="${HF_CACHE:-/project/yl258/kp759/hf}"   # /project: /scratch purges ~30-day-old files
ZERO_HP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/ZeroUnlearn/hparams/ZeroUnlearn"

case "${MODEL:-}" in
  llama31_8b)
    HF_ID=meta-llama/Llama-3.1-8B-Instruct
    N_LAYERS=32
    ROUTER_LAYER=22; WRITE_LAYER=18; LENS_LAYER=27
    V22_SUBJECT_LAYERS=7; V22_RELATION_LAYERS="22 26 27 31"; V22_RELATION_CONCAT=22,26,31
    # GA/GD/NPO train in fp32 with AdamW: every MLP of an 8B model (5.6B params) needs
    # ~100 GB for weights + grads + Adam states, more than one 80 GB GPU. down_proj only
    # (1.9B params, ~55 GB) is the largest scope that fits.
    BASELINE_ARGS="--rmu-layer 8 --baseline-params mlp_down --zero-hparams $ZERO_HP/Llama-3.1-8B-Instruct.json"
    MIN_GPU_MIB=75000
    ;;
  qwen3_4b)
    HF_ID=Qwen/Qwen3-4B
    N_LAYERS=36
    ROUTER_LAYER=24; WRITE_LAYER=21; LENS_LAYER=31
    V22_SUBJECT_LAYERS=8; V22_RELATION_LAYERS="24 30 31 35"; V22_RELATION_CONCAT=24,30,35
    BASELINE_ARGS="--rmu-layer 9 --baseline-params mlp --zero-hparams $ZERO_HP/Qwen3-4B.json"
    MIN_GPU_MIB=40000
    export CHAT_ENABLE_THINKING=0   # RWKU chat prompts: answer directly, no <think> block
    ;;
  *)
    echo "MODEL must be llama31_8b or qwen3_4b (got '${MODEL:-}')" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# Snapshot dir in the shared HF cache (download: bash scripts/submit_paper_model.sh MODEL fetch)
# (glob loop, not `ls | head`: under `set -euo pipefail` a failing ls in the substitution
#  exits the caller silently before anything is downloaded)
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH=""
  for CFG in "$HF_CACHE/models--${HF_ID//\//--}"/snapshots/*/config.json; do
    if [[ -f "$CFG" ]]; then MODEL_PATH="${CFG%/config.json}"; break; fi
  done
fi
MODEL_TAG="$MODEL"
TAG="${TAG:-paper_${MODEL}_v1}"
export HF_ID N_LAYERS MODEL_PATH MODEL_TAG TAG ROUTER_LAYER WRITE_LAYER LENS_LAYER \
       V22_SUBJECT_LAYERS V22_RELATION_LAYERS V22_RELATION_CONCAT BASELINE_ARGS MIN_GPU_MIB

paper_model_check() {  # fail fast: weights present, layer count as configured, GPU big enough
  if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "$HF_ID not found under $HF_CACHE; run on the login node first:" >&2
    echo "  bash scripts/submit_paper_model.sh $MODEL fetch" >&2
    return 2
  fi
  local n
  n="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_hidden_layers"])' \
         "$MODEL_PATH/config.json")" || return 3
  [[ "$n" == "$N_LAYERS" ]] || { echo "$MODEL_PATH has $n layers, config expects $N_LAYERS" >&2; return 2; }
  if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local mib
    mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    [[ "${mib:-0}" -ge "$MIN_GPU_MIB" ]] \
      || { echo "GPU has ${mib} MiB, $MODEL needs >= $MIN_GPU_MIB MiB" >&2; return 3; }
  fi
  python - "$MODEL_PATH" <<'EOF' || return 3
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
print(f"model ok: {c.model_type}, {c.num_hidden_layers} layers, hidden {c.hidden_size}")
EOF
  echo "MODEL=$MODEL MODEL_PATH=$MODEL_PATH TAG=$TAG router L$ROUTER_LAYER write L$WRITE_LAYER lens L$LENS_LAYER"
}
