#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mquake_sure_active_lora_kd_dev.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
SEED="${MQUAKE_SEED:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
ROOT="${OUTPUT_ROOT:-outputs/aws_mquake_sure_active_lora_kd_dev/seed${SEED}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"

# Reuse the already-measured SURE Emb+LM Stage-1 checkpoint by default.
STAGE1_ROOT="${SURE_STAGE1_ROOT:-outputs/aws_mquake_zerounlearn_locked_3b/seed${SEED}}"
STAGE1_CKPT="${SURE_STAGE1_CKPT:-${STAGE1_ROOT}/setting5e_forget_only/emb_lm_all_restore_post_training_true/checkpoint}"
STAGE1_MANIFEST="${SURE_STAGE1_MANIFEST:-${STAGE1_ROOT}/protocol/split_manifest.json}"

NO_NEUTRAL_PROTOCOL="${ROOT}/protocol_no_neutral"
OUT="${ROOT}/sure_stage2_active_lora_kd"

mkdir -p "${ROOT}"
test -d "${STAGE1_CKPT}"
test -f "${STAGE1_MANIFEST}"

python scripts/build_mquake_zerounlearn_locked_no_neutral_split.py \
  --mquake-path "${MQUAKE}" \
  --output-dir "${NO_NEUTRAL_PROTOCOL}" \
  --seed "${SEED}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}"

# Hard guard: Stage 2 must use exactly the same sampled forget instances as Stage 1.
python - "${STAGE1_MANIFEST}" "${NO_NEUTRAL_PROTOCOL}/split_manifest.json" <<'PY'
import json, sys
old=json.load(open(sys.argv[1])); new=json.load(open(sys.argv[2]))
a=old["sampling"]["forget_source_indices"]
b=new["sampling"]["forget_source_indices"]
assert a == b, "Stage1/Stage2 forget samples differ"
print("Verified identical Stage1/Stage2 forget source indices:", len(a))
PY

rm -rf "${OUT}"
python scripts/mquake_sure_active_lora_kd.py \
  --model-path "${STAGE1_CKPT}" \
  --training-visible-path "${NO_NEUTRAL_PROTOCOL}/training_visible_forget.json" \
  --split-manifest "${NO_NEUTRAL_PROTOCOL}/split_manifest.json" \
  --output-dir "${OUT}" \
  --seed "${SEED}" \
  --forget-num "${FORGET_NUM}" \
  --rank "${LORA_RANK:-2}" \
  --alpha "${LORA_ALPHA:-4}" \
  --last-n-layers "${LORA_LAST_N_LAYERS:-1}" \
  --steps 0 \
  --lr "${LORA_LR:-0.00005}" \
  --margin "${LORA_MARGIN:-0.05}" \
  --active-steps "${LORA_ACTIVE_STEPS:-400}" \
  --active-lr "${LORA_ACTIVE_LR:-0.00005}" \
  --l2 "${LORA_L2:-0.000001}" \
  --target-eff-max "${LORA_TARGET_EFF_MAX:-20}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --dtype "${DTYPE:-bf16}" \
  --device-map "${DEVICE_MAP:-single}" \
  2>&1 | tee "${ROOT}/stage2_train.log"

python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${OUT}/checkpoint" \
  --mquake-path "${MQUAKE}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${ROOT}/official_eval_locked.json" \
  --method "SURE Emb+LM Stage1 + active-only rank2 contextual LoRA KD" \
  --unlearn-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}" \
  --seed "${SEED}" \
  --batch-size "${EVAL_BATCH_SIZE:-8}" \
  --dtype "${DTYPE:-bf16}" \
  --device-map "${DEVICE_MAP:-single}" \
  --skip-atomic-gen

echo "Done: ${ROOT}/official_eval_locked.json"
