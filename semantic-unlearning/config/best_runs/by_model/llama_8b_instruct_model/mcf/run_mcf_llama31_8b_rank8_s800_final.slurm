#!/usr/bin/env bash
#SBATCH --job-name=mcf-l31-8b-r8s800-final
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_l31_8b_10s_%A_%a.out
#SBATCH --error=slurm_logs/mcf_l31_8b_10s_%A_%a.err

set -euo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

ENV=/scratch/yl258/kp759/conda_envs/semantic_unlearning
MODEL=/scratch/yl258/kp759/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

export PATH="$ENV/bin:$PATH"
export HF_HOME=/scratch/yl258/kp759/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SEED="${SLURM_ARRAY_TASK_ID}"
FORGET_NUM=50
RETAIN_NUM=1000

ROOT="outputs/crossmodel/llama31_8b/mcf_official_10seed_margin025_rank8_s800_final"
SEED_ROOT="${ROOT}/seed${SEED}"
SETTING_DIR="${SEED_ROOT}/setting5e"
REPAIR_DIR="${SEED_ROOT}/repair"
RESULT="${REPAIR_DIR}/official_eval_selected.json"

mkdir -p "$SETTING_DIR" "$REPAIR_DIR" slurm_logs

echo "Seed=${SEED}"
echo "Model=${MODEL}"
echo "Protocol=official, forget=${FORGET_NUM}, retain=${RETAIN_NUM}"
echo "Method=Setting5e-600 + active margin .25 rank8 s800"

if [[ -f "$RESULT" ]]; then
  if python - "$RESULT" <<'PY'
import json,sys
g=json.load(open(sys.argv[1])).get("post_reload_acceptance")
raise SystemExit(0 if isinstance(g,dict) and g.get("passed") else 1)
PY
  then
    echo "Seed ${SEED} already passed; skipping."
    exit 0
  fi
fi

test -f "$MODEL/config.json"
test -f data/multi_counterfact.json
test -d data/wikidata

echo "===== STAGE 1: SETTING 5e ====="
python scripts/gagd_compare.py \
  --dataset mcf \
  --model-path "$MODEL" \
  --mcf-cache-path data/multi_counterfact.json \
  --mcf-sample-mode official \
  --official-sample-mode official \
  --output-dir "$SETTING_DIR" \
  --mode emb_lm_all_restore_post_training_true \
  --forget-loss-type mcf_margin \
  --forget-margin 1.0 \
  --mcf-answer-field target_new \
  --forget-num "$FORGET_NUM" \
  --retain-num "$RETAIN_NUM" \
  --seed "$SEED" \
  --steps 600 \
  --batch-size 1 \
  --retain-batch-size 4 \
  --emb-lm-lr 1e-4 \
  --forget-weight 2.0 \
  --retain-weight 1.0 \
  --sampling-strategy epoch \
  --post-training-new-true-alpha 0.75 \
  --post-training-new-retain-alpha 0.50 \
  --post-training-new-true-retain-alpha 0.25 \
  --dtype bf16 \
  --device-map single \
  --wikidata-dir data/wikidata \
  --run-official-mcf-eval \
  --save-model

SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
SETTING_CONFIG="${SETTING_DIR}/config_used.json"
test -d "$SETTING_CKPT"
test -f "$SETTING_CONFIG"

echo "===== STAGE 2: PROTECTED ACTIVE REPAIR ====="
env \
  OUT_ROOT="$REPAIR_DIR" \
  MCF_PATH=data/multi_counterfact.json \
  WIKIDATA_DIR=data/wikidata \
  SAMPLE_MODE=official \
  SEED="$SEED" \
  FORGET_NUM="$FORGET_NUM" \
  RETAIN_NUM="$RETAIN_NUM" \
  ACTIVE_MARGIN=0.25 \
  DTYPE=bf16 \
  DEVICE_MAP=single \
  MARGIN_BATCH_SIZE=4 \
  REPAIR_STEPS=800 \
  REPAIR_LR=0.003 \
  REPAIR_OPTIMIZER=adamw \
  HINGE_WEIGHT=2.0 \
  DELTA_L2_LAMBDA=0.0001 \
  RETAIN_KL_MU=0.1 \
  RETAIN_CALIBRATION_NUM=200 \
  RETAIN_CALIBRATION_SEED=1729 \
  REPAIR_RANK=8 \
  PROJECT_AWAY_RETAIN_HIDDEN=1 \
  SKIP_PPL=0 \
  REQUIRE_POST_RELOAD_ZERO=1 \
  MIN_RELOADED_FORGET_MARGIN=0.10 \
  bash scripts/run_gagd_active_case_repair.sh \
    "$SETTING_CKPT" "$MODEL" "$SETTING_CONFIG"

echo "===== VERIFY FINAL RESULT ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
g=r.get("post_reload_acceptance")
if not isinstance(g,dict):
    raise SystemExit("missing post_reload_acceptance")
print(json.dumps(g,indent=2))
if not g.get("passed"):
    raise SystemExit("strict gate failed")
o=g.get("observed",{})
if float(o.get("forget_eff",999)) != 0.0:
    raise SystemExit("Eff != 0")
if float(o.get("forget_gen",999)) != 0.0:
    raise SystemExit("Gen != 0")
print("STRICT PASS: Eff=0, Gen=0")
PY

SELECTED_CKPT="$(python - "$REPAIR_DIR/selected_candidate.json" <<'PY'
import json,pathlib,sys
d=json.load(open(sys.argv[1]))
p=pathlib.Path(d["checkpoint"])
if not p.is_absolute():
    p=pathlib.Path.cwd()/p
print(p.resolve())
PY
)"

echo "Selected checkpoint: $SELECTED_CKPT"

# Capture hashes before any cleanup.
: > "$SEED_ROOT/selected_checkpoint_sha256.txt"
while IFS= read -r -d '' f; do
    sha256sum "$f" >> "$SEED_ROOT/selected_checkpoint_sha256.txt"
done < <(find "$SELECTED_CKPT" -maxdepth 1 -type f \
    \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -print0)

# 8B intermediate checkpoints are large. After a strict successful result,
# keep the selected checkpoint and remove only non-selected weight files.
# Submit with --export=ALL,KEEP_INTERMEDIATE=1 to keep every checkpoint.
if [[ "${KEEP_INTERMEDIATE:-0}" != "1" ]]; then
    echo "Cleaning non-selected 8B weight files..."
    while IFS= read -r -d '' d; do
        d_real="$(realpath "$d")"
        if [[ "$d_real" != "$SELECTED_CKPT" ]]; then
            find "$d" -maxdepth 1 -type f \
                \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) \
                -print -delete
        fi
    done < <(find "$SEED_ROOT" -type d -name checkpoint -print0)
fi

echo "Seed ${SEED} SUCCESS"
echo "Hash record: $SEED_ROOT/selected_checkpoint_sha256.txt"
