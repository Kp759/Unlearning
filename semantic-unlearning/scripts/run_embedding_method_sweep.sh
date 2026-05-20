#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/config_3b_instruct_forget05.yaml}"
SEED="${SEED:-42}"

echo "[Sweep] Using config: $CONFIG"
echo "[Sweep] Seed: $SEED"

OUT_DIR=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['output']['dir'])")
FORGET_SPLIT=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['data']['forget_split'])")
RETAIN_SPLIT=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['data']['retain_split'])")

echo "[Sweep] output.dir=$OUT_DIR"
echo "[Sweep] forget_split=$FORGET_SPLIT"
echo "[Sweep] retain_split=$RETAIN_SPLIT"

echo
echo "========== Step 1: Identify tokens =========="
python scripts/identify_tokens.py --config "$CONFIG"

echo
echo "========== Step 2: Run non-noise methods =========="

for METHOD in zero mean sign_flip sample_retain; do
  echo
  echo "----- Running method: $METHOD -----"

  python scripts/erase_embeddings.py \
    --config "$CONFIG" \
    --method "$METHOD" \
    --seed "$SEED" \
    --skip-eval

  MODEL_DIR="${OUT_DIR}/unlearned_model_${METHOD}"

  python scripts/tofu_eval.py \
    --config "$CONFIG" \
    --model-dir "$MODEL_DIR" \
    --method "freq_${METHOD}_${FORGET_SPLIT}" \
    --forget-split "$FORGET_SPLIT" \
    --retain-split "$RETAIN_SPLIT" \
    --seed "$SEED"
done

echo
echo "========== Step 3: Run replace-noise sweep =========="

for SCALE in 0.25 0.5 1.0 2.0 5.0; do
  SCALE_DIR=$(echo "$SCALE" | sed 's/\./p/g')

  echo
  echo "----- Running method: noise scale=$SCALE -----"

  python scripts/erase_embeddings.py \
    --config "$CONFIG" \
    --method noise \
    --noise-scale "$SCALE" \
    --seed "$SEED" \
    --skip-eval

  MODEL_DIR="${OUT_DIR}/unlearned_model_noise_scale${SCALE_DIR}"

  python scripts/tofu_eval.py \
    --config "$CONFIG" \
    --model-dir "$MODEL_DIR" \
    --method "freq_noise_scale${SCALE_DIR}_${FORGET_SPLIT}" \
    --forget-split "$FORGET_SPLIT" \
    --retain-split "$RETAIN_SPLIT" \
    --seed "$SEED"
done

echo
echo "========== Step 4: Run additive-noise sweep =========="

for SCALE in 0.25 0.5 1.0 2.0 5.0; do
  SCALE_DIR=$(echo "$SCALE" | sed 's/\./p/g')

  echo
  echo "----- Running method: add_noise scale=$SCALE -----"

  python scripts/erase_embeddings.py \
    --config "$CONFIG" \
    --method add_noise \
    --noise-scale "$SCALE" \
    --seed "$SEED" \
    --skip-eval

  MODEL_DIR="${OUT_DIR}/unlearned_model_add_noise_scale${SCALE_DIR}"

  python scripts/tofu_eval.py \
    --config "$CONFIG" \
    --model-dir "$MODEL_DIR" \
    --method "freq_add_noise_scale${SCALE_DIR}_${FORGET_SPLIT}" \
    --forget-split "$FORGET_SPLIT" \
    --retain-split "$RETAIN_SPLIT" \
    --seed "$SEED"
done

echo
echo "[✓] Sweep complete."
echo "Check results/tofu/*summary.json"
