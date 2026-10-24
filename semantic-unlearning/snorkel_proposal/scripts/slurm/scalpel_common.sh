#!/usr/bin/env bash
# Shared by every SCALPEL job on Wulver: environment, paths and per-model settings.
# Sourced with MODEL set (llama32_3b | qwen3_4b | llama31_8b); PARTITION (dev) optional.
#
# Exports: ROOT SP MODEL_PATH HF_ID HP_NAME RMU_LAYER GRAD_SCOPE RELEARN_SCOPE SURE_LAYER MIN_GPU_MIB
#          MOM2_TEXT MOM2_DATASET EDIT_LAYERS PARTITION OUT_ROOT

ROOT="${ROOT:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
SP="$ROOT/snorkel_proposal"
HF_CACHE="${HF_CACHE:-/project/yl258/kp759/hf}"
MOM2_TEXT="${MOM2_TEXT:-/project/yl258/kp759/datasets/wikipedia_20231101_100k}"   # scripts/submit_edit_baselines_wiki.sh corpus
MOM2_DATASET="${MOM2_DATASET:-wikipedia_20231101_100k}"
PARTITION="${PARTITION:-dev}"
OUT_ROOT="${OUT_ROOT:-$SP/results/cluster}"

scalpel_env() {   # the repository's conda env (same activation as every other job here)
  if [[ -z "${ENV_DIR:-}" ]]; then
    ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning
    [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
  fi
  module purge 2>/dev/null || true
  module load wulver 2>/dev/null || true
  module load Miniforge3 2>/dev/null || true
  if command -v conda >/dev/null 2>&1; then
    set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"; set -u
  else
    export PATH="$ENV_DIR/bin:$PATH"; export CONDA_PREFIX="$ENV_DIR"
  fi
  export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
}

[[ "${SCALPEL_NO_ENV:-0}" == 1 ]] || scalpel_env

case "${MODEL:?set MODEL=llama32_3b|qwen3_4b|llama31_8b}" in
  llama32_3b)
    HF_ID=meta-llama/Llama-3.2-3B-Instruct; HP_NAME=Llama-3.2-3B-Instruct
    RMU_LAYER=7; GRAD_SCOPE=mlp; RELEARN_SCOPE=mlp; SURE_LAYER=19; EARLY_WRITE_LAYER=16; MIN_GPU_MIB=40000
    # the path the paper's routers and the existing Wikipedia covariance caches were built with,
    # so data/stats/<model>/wikipedia_20231101_100k_stats is reused as is
    REF="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router/association_manifest.json"
    if [[ -z "${MODEL_PATH:-}" && -f "$REF" ]]; then
      MODEL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$REF" 2>/dev/null || true)"
      [[ -f "${MODEL_PATH:-/nonexistent}/config.json" ]] || MODEL_PATH=""   # e.g. a purged /scratch copy
    fi ;;
  qwen3_4b)
    HF_ID=Qwen/Qwen3-4B; HP_NAME=Qwen3-4B
    RMU_LAYER=9; GRAD_SCOPE=mlp; RELEARN_SCOPE=mlp; SURE_LAYER=24; EARLY_WRITE_LAYER=21; MIN_GPU_MIB=40000 ;;
  llama31_8b)
    # fp32 Adam on every 8B MLP does not fit 80 GB (paper_model_config.sh): down_proj only
    HF_ID=meta-llama/Llama-3.1-8B-Instruct; HP_NAME=Llama-3.1-8B-Instruct
    RMU_LAYER=8; GRAD_SCOPE=mlp_down; RELEARN_SCOPE=mlp_down; SURE_LAYER=22; EARLY_WRITE_LAYER=18; MIN_GPU_MIB=75000 ;;
  *) echo "unknown MODEL=$MODEL" >&2; return 2 2>/dev/null || exit 2 ;;
esac
# overrides for smaller GPUs (scripts/aws/run_scalpel_aws.sh, one 24 GB A10G); Wulver never sets them
GRAD_SCOPE="${SCALPEL_GRAD_SCOPE:-$GRAD_SCOPE}"; RELEARN_SCOPE="${SCALPEL_RELEARN_SCOPE:-$RELEARN_SCOPE}"
MIN_GPU_MIB="${SCALPEL_MIN_GPU_MIB:-$MIN_GPU_MIB}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  for CFG in "$HF_CACHE/models--${HF_ID//\//--}"/snapshots/*/config.json; do
    if [[ -f "$CFG" ]]; then MODEL_PATH="${CFG%/config.json}"; break; fi
  done
fi
# layers whose covariance ROME / MEMIT / ZeroUnlearn need (from their hparams files)
EDIT_LAYERS="$(python - "$ROOT/../ZeroUnlearn/hparams" "$HP_NAME" <<'EOF' 2>/dev/null || true
import json, sys
from pathlib import Path
layers = set()
for m in ("ROME", "MEMIT", "ZeroUnlearn"):
    f = Path(sys.argv[1]) / m / f"{sys.argv[2]}.json"
    if f.is_file():
        layers |= set(json.loads(f.read_text())["layers"])
print(" ".join(str(x) for x in sorted(layers)))
EOF
)"
EDIT_LAYERS="$(echo $EDIT_LAYERS $SURE_LAYER | tr ' ' '\n' | sort -nu | tr '\n' ' ')"   # + SURE's layer (sure_r_bake)
export ROOT SP HF_CACHE MOM2_TEXT MOM2_DATASET PARTITION OUT_ROOT MODEL MODEL_PATH HF_ID HP_NAME RMU_LAYER \
       GRAD_SCOPE RELEARN_SCOPE SURE_LAYER EARLY_WRITE_LAYER MIN_GPU_MIB EDIT_LAYERS

scalpel_check() {  # fail fast: weights present and GPU large enough
  if [[ -z "${MODEL_PATH:-}" || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "no weights for $HF_ID under $HF_CACHE; on the login node: bash scripts/submit_scalpel_compare.sh fetch $MODEL" >&2
    return 2
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local mib
    mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    [[ "${mib:-0}" -ge "$MIN_GPU_MIB" ]] || { echo "GPU has ${mib} MiB; $MODEL needs >= $MIN_GPU_MIB" >&2; return 3; }
  fi
  echo "MODEL=$MODEL MODEL_PATH=$MODEL_PATH PARTITION=$PARTITION edit layers: ${EDIT_LAYERS:-?}"
}

split_dir() { echo "$OUT_ROOT/$MODEL/$PARTITION/seed$1"; }   # seed -> run directory

# method-specific flags (everything a method needs beyond the split and the model)
method_args() {
  case "$1" in
    ga|gd|npo) echo "--method-arg params=$GRAD_SCOPE${SCALPEL_GRAD_PRECISION:+ --method-arg precision=$SCALPEL_GRAD_PRECISION}" ;;
    rmu) echo "--method-arg layer=$RMU_LAYER" ;;
    rome|memit|zerounlearn)
      local d; case "$1" in rome) d=ROME;; memit) d=MEMIT;; zerounlearn) d=ZeroUnlearn;; esac
      echo "--method-arg hparams=$ROOT/../ZeroUnlearn/hparams/$d/$HP_NAME.json --method-arg mom2_dataset=$MOM2_DATASET --method-arg mom2_text=$MOM2_TEXT" ;;
    sure) echo "--method-arg model_path=$MODEL_PATH --method-arg layer=$SURE_LAYER --method-arg device=cuda --method-arg work_dir=$(split_dir "$SEED")/sure_work" ;;
    ours|ours_l16|sure_paper|sure_paper_l16)   # scalpel/methods/sure_paper.py: one pass at SURE's layer;
      # the _l16 ablations write at the paper's earlier layer (16 on 3B, paper_model_config.sh)
      echo "--method-arg model_path=$MODEL_PATH --method-arg layer=$SURE_LAYER --method-arg device=cuda --method-arg work_dir=$(split_dir "$SEED")/sure_work_$1$([[ $1 == *_l16 ]] && echo " --method-arg write_layer=$EARLY_WRITE_LAYER" || true)" ;;
    sure_r*)   # robust SURE variants (scalpel/methods/sure_robust.py); bake reads the Wikipedia covariance
      echo "--method-arg model_path=$MODEL_PATH --method-arg layer=$SURE_LAYER --method-arg device=cuda --method-arg work_dir=$(split_dir "$SEED")/sure_work_$1 --method-arg mom2_dataset=$MOM2_DATASET" ;;
    *) echo "" ;;
  esac
}
EVAL_ARGS=(--device cuda --dtype bf16 --batch-size "${SCALPEL_BATCH:-16}" --relearn-scope "$RELEARN_SCOPE")
