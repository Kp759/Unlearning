#!/usr/bin/env bash
set -euo pipefail

# RSNR-V1A: oracle (subject, relation) routing + latent null adapter.
#
# Reuses the EXACT already-generated V1.3 seed1 5-view corpus so the comparison
# changes the intervention architecture, not the training prompts.
#
# IMPORTANT:
#   - Seed1 is DEVELOPMENT ONLY.
#   - No full MCF source is accepted by this launcher.
#   - No official paraphrase/neighborhood/eval-retain text is read.
#   - Oracle routing is supplied from known record metadata; learned gating is a
#     separate later experiment.
#   - Non-target queries have adapter gate=0 and must be bit/logit identical to Base.
#   - The strict entrypoint preserves failed-run diagnostics but exits nonzero
#     unless all 50 facts pass the registered joint 5-view training gate.
#
# Required environment:
#   MODEL_PATH=/home/ec2-user/models/Llama-3.2-3B-Instruct
#   SOURCE_V13_RUN=<existing successful V1.3 5-view output directory>
#
# Usage:
#   bash scripts/run_mcf_rsnr_v1a_oracle_manual.sh OUTPUT_DIR

OUTPUT_DIR=${1:?fresh RSNR-V1A output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SOURCE_V13_RUN=${SOURCE_V13_RUN:?SOURCE_V13_RUN is required}

PROTOCOL_DIR="$SOURCE_V13_RUN/protocol"
VIEWS="$PROTOCOL_DIR/training_visible_multiview_forget.json"
LOG_TMP="${OUTPUT_DIR}.training.log.tmp"

for path in "$OUTPUT_DIR" "$LOG_TMP"; do
  if [ -e "$path" ]; then
    echo "ERROR: RSNR-V1A path already exists: $path" >&2
    exit 1
  fi
done

for path in \
  "$PROTOCOL_DIR/split_manifest.json" \
  "$PROTOCOL_DIR/training_visible_forget_direct.json" \
  "$PROTOCOL_DIR/training_visible_protection_fit_direct.json" \
  "$VIEWS"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: required locked V1.3 artifact missing: $path" >&2
    exit 1
  fi
done

# Firewall: this run must not inherit handles to full/official evaluation data.
unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH

python -u scripts/run_mcf_rsnr_v1a_oracle_strict.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTOCOL_DIR" \
  --view-corpus "$VIEWS" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 800 \
  --case-batch-size 4 \
  --check-every 25 \
  --learning-rate 0.0002 \
  --weight-decay 0.0 \
  --adapter-rank 16 \
  --adapter-alpha 16 \
  --layer-index -4 \
  --abstain-weight 1.0 \
  --unlikelihood-weight 1.0 \
  --anchor-weight 0.0001 \
  --grad-clip 1.0 \
  --minimum-abstain-vs-true-margin 0.1 \
  --minimum-true-logprob-drop 2.0 \
  --gate-off-logit-drift-max 0.0 \
  2>&1 | tee "$LOG_TMP"

mkdir -p "$OUTPUT_DIR/logs"
mv "$LOG_TMP" "$OUTPUT_DIR/logs/training.log"

printf '\nRSNR-V1A oracle-null run finished with strict training gate.\n'
printf 'Method report: %s\n' "$OUTPUT_DIR/method/rsnr_v1a_oracle.json"
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Adapter: %s\n' "$OUTPUT_DIR/method/rsnr_oracle_null_adapter.pt"
printf 'Routing sidecar: %s\n' "$OUTPUT_DIR/method/relation_scoped_null_routing.json"
printf 'Final-certification status: DEVELOPMENT ONLY; seed1 aggregates are already consumed.\n'
