#!/usr/bin/env bash
set -euo pipefail

# MCF Setting 5e + neutral-row (target_new) active repair.
#
#   Stage 1: mcf_forget_only_setting5e.py -- tied emb+LM-head GA/GD over the
#            full vocabulary, ultra-aggressive, forget-only (no MCF retain
#            access during training). Same mechanism ZsRE's Setting 5e uses.
#   Stage 2: mcf_setting5e_neutral_row_active_repair.py (new) -- protected-
#            subspace LM-head repair restricted to TARGET_NEW's rows,
#            mirroring ZsRE's Unknown-row active repair. This is the MCF
#            analogue of the design that reached Eff=0.0000/Gen=0.0000 on all
#            10 ZsRE seeds (config/best_runs/.../zsre/setting5e_active_repair_
#            u1p20_ppl1p16_cal384_seeds1_10.md) -- untested on MCF until now.
#
# Stage 1's own seeded MCF sampling (gagd.sample_mcf_raw_records, sample_mode
# official) and the locked-split builder's sampling
# (build_sure_minimal_split.sample_records -> sample_official_mcf_records)
# both call the identical underlying function with the same seed, and forget
# records are drawn before retain records inside a freshly seeded RNG -- so
# they draw the SAME 50 forget case IDs regardless of each call's own
# retain_num. Verified by reading mcf_sampling.py, not assumed. This is what
# lets Stage 2 use the separately-built locked split's manifest against
# Stage 1's own checkpoint.
#
# Usage:
#   bash scripts/run_mcf_setting5e_neutral_repair.sh /path/to/model

MODEL_PATH=${1:?model path required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
RETAIN_NUM=${RETAIN_NUM:-1000}
MCF_PATH=${MCF_PATH:-data/multi_counterfact.json}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_setting5e_neutral_repair_seed${SEED}}

# Established ultra-aggressive Setting 5e controls (matches the ZsRE runner's
# defaults, since both call the same gagd.configure_trainable POST_TRAINING_
# RESTORE_MODE mechanism).
STAGE1_STEPS=${STAGE1_STEPS:-600}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-1}
STAGE1_EMB_LM_LR=${STAGE1_EMB_LM_LR:-1e-4}
STAGE1_FORGET_WEIGHT=${STAGE1_FORGET_WEIGHT:-2.0}
STAGE1_FORGET_MARGIN=${STAGE1_FORGET_MARGIN:-1.0}

REPAIR_STEPS=${REPAIR_STEPS:-800}
REPAIR_LR=${REPAIR_LR:-5e-3}
CONSTRAINT_MARGIN=${CONSTRAINT_MARGIN:-0.05}
REPAIR_L2=${REPAIR_L2:-1e-3}
PROTECTED_RANK=${PROTECTED_RANK:-256}
REPAIR_RANK=${REPAIR_RANK:-64}
PROTECTED_KL_MAX=${PROTECTED_KL_MAX:-0.5}
SYNTHETIC_PARAPHRASES_PER_RECORD=${SYNTHETIC_PARAPHRASES_PER_RECORD:-3}

WIKIDATA_DIR=${WIKIDATA_DIR:-data/wikidata}
GENERIC_PROTECTION_SAMPLES=${GENERIC_PROTECTION_SAMPLES:-5000}
GENERIC_PROTECTION_DOC_START=${GENERIC_PROTECTION_DOC_START:-20}

STAGE1_UNTIE_EMBEDDINGS=${STAGE1_UNTIE_EMBEDDINGS:-1}
STAGE1_TRAIN_SCOPE=${STAGE1_TRAIN_SCOPE:-lm_head}
# Shared by both stages: they must agree on which row is raised.
NEUTRAL_TARGET=${NEUTRAL_TARGET:-Unknown}
STAGE1_FREQUENCY_DOCS=${STAGE1_FREQUENCY_DOCS:-50000}
STAGE1_FREQUENCY_DOC_START=${STAGE1_FREQUENCY_DOC_START:-20}
STAGE1_FREQUENCY_CAP_ALPHA=${STAGE1_FREQUENCY_CAP_ALPHA:-0.5}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}
CANDIDATE_SCALES=${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}

SPLIT_ROOT="$OUT_ROOT/protocol"
STAGE1_OUT="$OUT_ROOT/stage1_setting5e"
STAGE2_OUT="$OUT_ROOT/stage2_neutral_repair"
VISIBLE="$SPLIT_ROOT/training_visible_target_aware_direct.json"
MANIFEST="$SPLIT_ROOT/split_manifest.json"

mkdir -p "$OUT_ROOT"

if [ ! -f "$VISIBLE" ] || [ ! -f "$MANIFEST" ]; then
  echo "Building locked MCF split (seed $SEED) for Stage-2 failure detection..."
  python -u scripts/build_mcf_sure_target_aware_direct_split.py \
    --mcf-path "$MCF_PATH" \
    --output-dir "$SPLIT_ROOT" \
    --seed "$SEED" \
    --forget-num "$FORGET_NUM" \
    --retain-eval-num "$RETAIN_NUM" \
    2>&1 | tee "$OUT_ROOT/split_build.log"
fi
test -f "$VISIBLE" || { echo "split builder produced no $VISIBLE"; exit 1; }
test -f "$MANIFEST" || { echo "split builder produced no $MANIFEST"; exit 1; }

# mcf_forget_only_setting5e.py refuses to run on the raw MCF file -- it
# explicitly requires a "repair-visible" copy with paraphrase_prompts and
# other evaluation-only fields stripped from every record ("Repair-visible
# MCF still exposes paraphrases during Stage 1"), so an accidental run can
# never train on data it should not see. build_mcf_zerounlearn_locked_split.py
# produces exactly that: a 1:1, order-preserving strip-only copy of the FULL
# dataset, written once regardless of --seeds. That script's own main()
# additionally asserts, for every seed, that sampling the raw data and the
# sanitized copy select the SAME forget/retain indices -- an independent,
# stronger confirmation of the same fact this wrapper already relies on for
# Stage 2 (mcf_sampling.py: forget is drawn before retain from a freshly
# seeded RNG, so stripping fields never changes which 50 records are drawn).
REPAIR_VISIBLE_ROOT="$OUT_ROOT/repair_visible"
REPAIR_VISIBLE_MCF="$REPAIR_VISIBLE_ROOT/repair_visible_mcf.json"
if [ ! -f "$REPAIR_VISIBLE_MCF" ]; then
  echo "Building repair-visible MCF copy (Stage-1 input)..."
  python -u scripts/build_mcf_zerounlearn_locked_split.py \
    --mcf-path "$MCF_PATH" \
    --output-dir "$REPAIR_VISIBLE_ROOT" \
    --seeds "$SEED" \
    --forget-num "$FORGET_NUM" \
    --retain-num "$RETAIN_NUM" \
    2>&1 | tee "$OUT_ROOT/repair_visible_build.log"
fi
test -f "$REPAIR_VISIBLE_MCF" || { echo "repair-visible builder produced no $REPAIR_VISIBLE_MCF"; exit 1; }

echo "Stage 1: Setting 5e tied emb+LM-head GA/GD (forget-only)..."
STAGE1_UNTIE_FLAG=--untie-embeddings
if [ "$STAGE1_UNTIE_EMBEDDINGS" = "0" ]; then
  STAGE1_UNTIE_FLAG=--no-untie-embeddings
fi
python -u scripts/mcf_forget_only_setting5e.py \
  --model-path "$MODEL_PATH" \
  --mcf-cache-path "$REPAIR_VISIBLE_MCF" \
  --output-dir "$STAGE1_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --emb-lm-lr "$STAGE1_EMB_LM_LR" \
  --forget-weight "$STAGE1_FORGET_WEIGHT" \
  --forget-margin "$STAGE1_FORGET_MARGIN" \
  "$STAGE1_UNTIE_FLAG" \
  --train-scope "$STAGE1_TRAIN_SCOPE" \
  --neutral-target "$NEUTRAL_TARGET" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --frequency-docs "$STAGE1_FREQUENCY_DOCS" \
  --frequency-doc-start "$STAGE1_FREQUENCY_DOC_START" \
  --frequency-cap-alpha "$STAGE1_FREQUENCY_CAP_ALPHA" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1_setting5e.log"

STAGE1_CKPT="$STAGE1_OUT/emb_lm_all_restore_post_training_true/checkpoint"
test -d "$STAGE1_CKPT" || { echo "Stage 1 produced no checkpoint at $STAGE1_CKPT"; exit 1; }

echo "Stage 2: neutral-row (target_new) protected-subspace active repair..."
python -u scripts/mcf_setting5e_neutral_row_active_repair.py \
  --model-path "$STAGE1_CKPT" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE2_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --constraint-margin "$CONSTRAINT_MARGIN" \
  --repair-l2 "$REPAIR_L2" \
  --protected-rank "$PROTECTED_RANK" \
  --repair-rank "$REPAIR_RANK" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --generic-protection-samples "$GENERIC_PROTECTION_SAMPLES" \
  --generic-protection-doc-start "$GENERIC_PROTECTION_DOC_START" \
  --neutral-target "$NEUTRAL_TARGET" \
  --protected-kl-max "$PROTECTED_KL_MAX" \
  --synthetic-paraphrases-per-record "$SYNTHETIC_PARAPHRASES_PER_RECORD" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2_neutral_repair.log"

echo "Official evaluation..."
python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "$STAGE2_OUT/checkpoint" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out "$OUT_ROOT/final_official_eval.json" \
  --unlearn-num "$FORGET_NUM" \
  --retain-num "$RETAIN_NUM" \
  --seed "$SEED" \
  --sample-mode official \
  --dtype bf16 \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/final_official_eval.log"

printf '\nFinished MCF Setting 5e + neutral-row active repair.\n'
printf 'Split   : %s\n' "$SPLIT_ROOT"
printf 'Stage 1 : %s\n' "$STAGE1_CKPT"
printf 'Stage 2 : %s\n' "$STAGE2_OUT/checkpoint"
printf 'Summary : %s\n' "$STAGE2_OUT/repair_summary.json"
printf 'Eval    : %s\n' "$OUT_ROOT/final_official_eval.json"
