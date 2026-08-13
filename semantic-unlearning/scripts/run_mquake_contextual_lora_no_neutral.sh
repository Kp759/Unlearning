#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mquake_contextual_lora_no_neutral.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
SEED="${MQUAKE_SEED:-1}"
ROOT="${OUTPUT_ROOT:-outputs/aws_mquake_contextual_lora_no_neutral_3b/seed${SEED}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"

python scripts/build_mquake_zerounlearn_locked_no_neutral_split.py \
  --mquake-path "$MQUAKE" \
  --output-dir "$ROOT/protocol" \
  --seed "$SEED" \
  --forget-num "${MQUAKE_FORGET_NUM:-50}" \
  --retain-num "${MQUAKE_RETAIN_EVAL_NUM:-1000}"

python scripts/mquake_forget_only_contextual_lora.py \
  --model-path "$MODEL" \
  --training-visible-path "$ROOT/protocol/training_visible_forget.json" \
  --split-manifest "$ROOT/protocol/split_manifest.json" \
  --output-dir "$ROOT/contextual_lora" \
  --seed "$SEED" \
  --forget-num "${MQUAKE_FORGET_NUM:-50}" \
  --rank "${LORA_RANK:-4}" \
  --alpha "${LORA_ALPHA:-8}" \
  --last-n-layers "${LORA_LAST_N_LAYERS:-2}" \
  --steps "${LORA_STEPS:-600}" \
  --lr "${LORA_LR:-0.0001}" \
  --margin "${LORA_MARGIN:-0.25}" \
  --active-steps "${LORA_ACTIVE_STEPS:-400}" \
  --active-lr "${LORA_ACTIVE_LR:-0.00005}" \
  --l2 "${LORA_L2:-0.000001}" \
  --target-eff-max "${LORA_TARGET_EFF_MAX:-20}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --dtype "${DTYPE:-bf16}" \
  --device-map "${DEVICE_MAP:-single}" \
  2>&1 | tee "$ROOT/contextual_lora_train.log"

python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "$ROOT/contextual_lora/checkpoint" \
  --mquake-path "$MQUAKE" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out "$ROOT/official_eval_locked.json" \
  --method "SURE contextual LoRA no-neutral min-change strict forget-only" \
  --unlearn-num "${MQUAKE_FORGET_NUM:-50}" \
  --retain-num "${MQUAKE_RETAIN_EVAL_NUM:-1000}" \
  --seed "$SEED" \
  --batch-size "${EVAL_BATCH_SIZE:-8}" \
  --dtype "${DTYPE:-bf16}" \
  --device-map "${DEVICE_MAP:-single}" \
  --skip-atomic-gen

echo "Done: $ROOT/official_eval_locked.json"
