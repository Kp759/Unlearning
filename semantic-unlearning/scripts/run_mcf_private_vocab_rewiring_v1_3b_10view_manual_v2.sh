#!/usr/bin/env bash
set -euo pipefail

# V1.3b revision 2: leakage-safe 10-view ablation with subject-slot generation.
# Scientific change vs V1.3: 1 canonical + 9 additional training views.
# Generation-format repair vs V1.3b revision 1: Base emits [SUBJECT], which is
# deterministically replaced with the actual subject before scoring.
# Worst-1 objective, architecture, semantic thresholds, retain objective, and
# held-out leakage firewall are unchanged. Seed 1 remains development only.

OUTPUT_DIR=${1:?fresh V1.3b output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required only for split-builder process}
PROTO_TMP="${OUTPUT_DIR}.protocol_tmp"
LOG_TMP="${OUTPUT_DIR}.training.log.tmp"

for path in "$OUTPUT_DIR" "$PROTO_TMP" "$LOG_TMP"; do
  if [ -e "$path" ]; then
    echo "ERROR: V1.3b path already exists: $path" >&2
    exit 1
  fi
done

REGISTRY=protocols/mcf_private_vocab_rewiring_v1_3b_10view_registry.json

python -u scripts/build_mcf_biendpoint_nullspace_rewiring_v2_split.py \
  --mcf-path "$MCF_SOURCE" \
  --output-dir "$PROTO_TMP" \
  --seed 1 \
  --forget-num 50 \
  --official-retain-num 1000 \
  --protection-fit-num 2000 \
  --protection-development-num 500 \
  --protection-certification-num 1000

unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH MCF_SOURCE

VIEWS="$PROTO_TMP/training_visible_multiview_forget.json"
python -u scripts/build_mcf_private_vocab_rewiring_v1_3_training_views_v4.py \
  --model-path "$MODEL_PATH" \
  --forget-direct "$PROTO_TMP/training_visible_forget_direct.json" \
  --protection-fit-direct "$PROTO_TMP/training_visible_protection_fit_direct.json" \
  --out "$VIEWS" \
  --views-per-case 10 \
  --candidates-per-attempt 12 \
  --max-attempts 24 \
  --max-new-tokens 320 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 13131 \
  --dtype bf16 \
  --max-true-logprob-drop 3.0 \
  --max-margin-degradation 1.0

export MCF_V13_VIEW_CORPUS="$VIEWS"
export MCF_V13_VIEW_CHUNK=16

python -u scripts/run_mcf_private_vocab_rewiring_v1_3_multiview.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTO_TMP" \
  --experiment-registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 600 \
  --forget-batch-size 8 \
  --retain-batch-size 16 \
  --check-every 25 \
  --learning-rate 0.001 \
  --minimum-forget-margin 0.1 \
  --train-margin-target 0.1 \
  --retain-kl-weight 20.0 \
  --anchor-weight 0.001 \
  --relative-row-cap 0.5 \
  --topk 64 \
  --initial-equivalence-kl-max 0.0000001 \
  --initial-margin-drift-max 0.00001 \
  --retain-kl-mean-max 0.0001 \
  --nonclone-certification-prompts 64 \
  --save-model \
  2>&1 | tee "$LOG_TMP"

unset MCF_V13_VIEW_CORPUS MCF_V13_VIEW_CHUNK
mkdir -p "$OUTPUT_DIR/logs"
mv "$LOG_TMP" "$OUTPUT_DIR/logs/training.log"
mv "$PROTO_TMP" "$OUTPUT_DIR/protocol"

printf '\nV1.3b 10-view revision-2 experiment finished.\n'
printf 'Method report: %s\n' "$OUTPUT_DIR/method/private_vocab_rewiring_v1_3_multiview.json"
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Training-view corpus: %s\n' "$OUTPUT_DIR/protocol/training_visible_multiview_forget.json"
printf 'Final-certification status: DEVELOPMENT ONLY; final eval requires a new untouched seed.\n'
