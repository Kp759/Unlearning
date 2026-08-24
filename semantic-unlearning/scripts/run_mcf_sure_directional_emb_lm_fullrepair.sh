#!/usr/bin/env bash
set -euo pipefail

# Two-stage MCF SURE experiment:
#   Stage 1: untie Emb/LM, directional GA on target_true-sensitive rows.
#   Stage 2: direct+synthetic-failure protected-subspace sparse LM-head row
#            repair (delta restricted to a basis orthogonal to passing
#            cases' hidden states, not an unrestricted row edit).
#
# No LoRA. No rank sweep. No generated paraphrases.
# No official paraphrase/neighborhood/retain/PPL training visibility.
#
# Usage:
#   bash scripts/run_mcf_sure_directional_emb_lm_fullrepair.sh \
#     /path/to/model \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/training_visible_mcf_target_true.json \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/split_manifest.json

MODEL_PATH=${1:?model path required}
VISIBLE=${2:?training-visible JSON required}
MANIFEST=${3:?split manifest required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_directional_emb_lm_fullrepair_seed${SEED}}

# Defaults kept in sync with mcf_sure_directional_emb_lm_stage1.py's own
# argparse defaults (raised from 600/2/1 once the case pool grew ~4x with
# synthetic-paraphrase augmentation -- see SURE_MCF_DIRECTIONAL_EMB_LM_FULLREPAIR.md).
# NOTE: these env vars, if set in the calling shell, always override the
# fallback below -- unset them (or pass the new values) to pick up fixes to
# the Python script's own defaults.
STAGE1_STEPS=${STAGE1_STEPS:-1200}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-4}
STAGE1_CACHE_BATCH_SIZE=${STAGE1_CACHE_BATCH_SIZE:-8}
STAGE1_LR=${STAGE1_LR:-1e-4}
STAGE1_GA_WEIGHT=${STAGE1_GA_WEIGHT:-2.0}
STAGE1_KL_WEIGHT=${STAGE1_KL_WEIGHT:-1.0}
STAGE1_DELTA_L2=${STAGE1_DELTA_L2:-1e-6}
DIRECTION_RANK=${DIRECTION_RANK:-8}
CONSTRAINT_MARGIN=${CONSTRAINT_MARGIN:-0.05}
# Shared by both stages: Stage 1's fit is a no-op for single/first-token
# answers (decoder-row fallback is prompt-independent -- see
# mcf_sure_directional_emb_lm_stage1.py), but Stage 2's hidden-state-based
# repair is not, and is where synthetic-paraphrase generalization actually
# comes from.
SYNTHETIC_PARAPHRASES_PER_RECORD=${SYNTHETIC_PARAPHRASES_PER_RECORD:-3}

REPAIR_STEPS=${REPAIR_STEPS:-800}
REPAIR_LR=${REPAIR_LR:-0.005}
# Raised from 1e-6 -- see mcf_sure_fullrow_failure_repair.py's --repair-l2
# help text: at the old value this term was negligible next to the failure
# hinge once the wider direct+synthetic objective needed a much larger delta.
REPAIR_L2=${REPAIR_L2:-1e-3}
# Soft pass-guard-weight/distribution-kl-weight loss terms, and later a
# hard-gate-only design, were both replaced by protected-subspace projection
# (see --protected-rank help text): weight 1.0 left Spe collapsed (0.16);
# weight 10.0 overshot and broke Eff (0.0 -> 12.0); a hard gate alone at
# protected-kl-max 0.05 (tuned for protected_subspace's own already-rank-
# limited delta, not this one) rejected 796/800 steps and left Eff at 76.0.
# The repair delta is now geometrically restricted to a subspace orthogonal
# to the passing cases' hidden states; the KL/margin gate below is only a
# secondary backstop, so it can afford to be looser than 0.05.
REPAIR_PROTECTED_RANK=${REPAIR_PROTECTED_RANK:-32}
REPAIR_RANK=${REPAIR_RANK:-4}
# The ~26 in-sample passing records alone were not enough: a real run left
# PPL at 18.875 (every other run: ~10.9-11.1) and Spe collapsed at 0.68.
# Widen the protected subspace with hidden states from ordinary text so it
# reflects what general language use actually looks like. Doc range MUST
# stay disjoint from official PPL's hardcoded [:20] slice (enforced by the
# Python script's own argparse validation, doc-start >= 20) -- otherwise
# training would protect against the exact text the eval score is later
# measured on, contaminating the result.
REPAIR_WIKIDATA_DIR=${REPAIR_WIKIDATA_DIR:-data/wikidata}
REPAIR_GENERIC_PROTECTION_TOKENS=${REPAIR_GENERIC_PROTECTION_TOKENS:-300}
REPAIR_GENERIC_PROTECTION_DOC_START=${REPAIR_GENERIC_PROTECTION_DOC_START:-20}
REPAIR_GENERIC_PROTECTION_DOC_STOP=${REPAIR_GENERIC_PROTECTION_DOC_STOP:-40}
REPAIR_PROTECTED_KL_MAX=${REPAIR_PROTECTED_KL_MAX:-0.5}
REPAIR_BACKTRACK_SCALES=${REPAIR_BACKTRACK_SCALES:-1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625,0.00048828125,0.0}
REPAIR_BATCH_SIZE=${REPAIR_BATCH_SIZE:-8}
REPAIR_CHECK_EVERY=${REPAIR_CHECK_EVERY:-25}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}
CANDIDATE_SCALES=${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}

STAGE1_OUT="$OUT_ROOT/stage1"
STAGE2_OUT="$OUT_ROOT/stage2_fullrow_repair"

mkdir -p "$OUT_ROOT"

python -u scripts/mcf_sure_directional_emb_lm_stage1.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE1_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --cache-batch-size "$STAGE1_CACHE_BATCH_SIZE" \
  --lr "$STAGE1_LR" \
  --ga-weight "$STAGE1_GA_WEIGHT" \
  --distribution-kl-weight "$STAGE1_KL_WEIGHT" \
  --delta-l2 "$STAGE1_DELTA_L2" \
  --direction-rank "$DIRECTION_RANK" \
  --stage1-constraint-margin "$CONSTRAINT_MARGIN" \
  --synthetic-paraphrases-per-record "$SYNTHETIC_PARAPHRASES_PER_RECORD" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1.log"

python -u scripts/mcf_sure_fullrow_failure_repair.py \
  --model-path "$STAGE1_OUT/checkpoint" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE2_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --constraint-margin "$CONSTRAINT_MARGIN" \
  --repair-l2 "$REPAIR_L2" \
  --protected-rank "$REPAIR_PROTECTED_RANK" \
  --repair-rank "$REPAIR_RANK" \
  --wikidata-dir "$REPAIR_WIKIDATA_DIR" \
  --generic-protection-tokens "$REPAIR_GENERIC_PROTECTION_TOKENS" \
  --generic-protection-doc-start "$REPAIR_GENERIC_PROTECTION_DOC_START" \
  --generic-protection-doc-stop "$REPAIR_GENERIC_PROTECTION_DOC_STOP" \
  --protected-kl-max "$REPAIR_PROTECTED_KL_MAX" \
  --backtrack-scales "$REPAIR_BACKTRACK_SCALES" \
  --synthetic-paraphrases-per-record "$SYNTHETIC_PARAPHRASES_PER_RECORD" \
  --batch-size "$REPAIR_BATCH_SIZE" \
  --check-every "$REPAIR_CHECK_EVERY" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2_fullrow_repair.log"

printf '\nFinished directional Emb+LM -> protected-subspace full-row LM-head repair.\n'
printf 'Stage 1: %s\n' "$STAGE1_OUT/checkpoint"
printf 'Final  : %s\n' "$STAGE2_OUT/checkpoint"
printf 'Summary: %s\n' "$STAGE2_OUT/repair_summary.json"
