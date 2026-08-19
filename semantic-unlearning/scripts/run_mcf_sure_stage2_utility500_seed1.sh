#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/models/Llama-3.2-3B-Instruct-0cb88a4f764b7a12671c53f0838cd831a0843b95-runtime}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/mcf_sure_rome_target_true_r8m1}"
OUT_ROOT="${OUT_ROOT:-outputs/mcf_sure_rome_target_true_r8m1_u500}"
SEED="${SEED:-1}"
UTILITY_NUM="${UTILITY_NUM:-500}"
UTILITY_SEED="${UTILITY_SEED:-1729}"
UTILITY_EXCLUDE_FIRST="${UTILITY_EXCLUDE_FIRST:-20}"
UTILITY_WEIGHT="${UTILITY_WEIGHT:-1.0}"
UTILITY_MAX_LENGTH="${UTILITY_MAX_LENGTH:-100}"
UTILITY_BATCH_SIZE="${UTILITY_BATCH_SIZE:-4}"

STAGE1="$BASELINE_ROOT/seed${SEED}/setting5e_best_run_mirrored"
STAGE1_CKPT="$STAGE1/emb_lm_all_restore_post_training_true/checkpoint"
STAGE1_CONFIG="$STAGE1/config_used.json"
VISIBLE="$BASELINE_ROOT/protocol/repair_visible_mcf_target_true_sensitive.json"
MANIFEST="$BASELINE_ROOT/protocol/seed${SEED}_manifest.json"
BASE_EVAL="$BASELINE_ROOT/seed${SEED}/base_original_mcf_eval.json"
BASELINE_EVAL="$BASELINE_ROOT/seed${SEED}/target_true_sensitive_eval.json"

ROOT="$OUT_ROOT/seed${SEED}"
REPAIR="$ROOT/repair_utility500"
POST_EVAL="$ROOT/post_original_mcf_eval.json"
PAPER_EVAL="$ROOT/target_true_sensitive_eval.json"

for path in "$MODEL_PATH" "$STAGE1_CKPT" "$WIKIDATA_DIR"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
for path in "$MCF_PATH" "$STAGE1_CONFIG" "$VISIBLE" "$MANIFEST" "$BASE_EVAL" "$BASELINE_EVAL"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done

rm -rf "$ROOT"
mkdir -p "$ROOT"

printf '%s\n' \
  "===== MCF SURE STAGE 2 + EXTERNAL UTILITY GUARD =====" \
  "seed=$SEED" \
  "reusing Stage 1=$STAGE1_CKPT" \
  "utility_num=$UTILITY_NUM" \
  "utility_seed=$UTILITY_SEED" \
  "utility_weight=$UTILITY_WEIGHT" \
  "utility_source=$WIKIDATA_DIR train texts, excluding first $UTILITY_EXCLUDE_FIRST"

python scripts/mcf_forget_only_active_repair_utility.py \
  --model-path "$STAGE1_CKPT" \
  --base-model-path "$MODEL_PATH" \
  --experiment-config-path "$STAGE1_CONFIG" \
  --output-dir "$REPAIR" \
  --mcf-cache-path "$VISIBLE" \
  --sample-mode official \
  --seed "$SEED" \
  --forget-num 50 \
  --retain-num 0 \
  --repair-mode minimal_optimize \
  --active-margin 1.0 \
  --repair-steps 300 \
  --repair-lr 0.005 \
  --repair-optimizer adamw \
  --hinge-weight 2.0 \
  --delta-l2-lambda 0.0001 \
  --retain-kl-mu 0 \
  --retain-calibration-num 0 \
  --repair-rank 8 \
  --no-project-away-retain-hidden \
  --stop-when-all-satisfied \
  --utility-num "$UTILITY_NUM" \
  --utility-seed "$UTILITY_SEED" \
  --utility-exclude-first "$UTILITY_EXCLUDE_FIRST" \
  --utility-weight "$UTILITY_WEIGHT" \
  --utility-max-length "$UTILITY_MAX_LENGTH" \
  --utility-batch-size "$UTILITY_BATCH_SIZE" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --dtype bf16 \
  --device-map single \
  --margin-batch-size 4 \
  --save-model

test -d "$REPAIR/checkpoint"
test -f "$REPAIR/repair_summary.json"
test -f "$REPAIR/utility_manifest.json"

python scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "$REPAIR/checkpoint" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out "$POST_EVAL" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed "$SEED" \
  --sample-mode official \
  --dtype bf16 \
  --device-map single

python scripts/annotate_ppl_provenance.py \
  --eval-json "$POST_EVAL" \
  --model-dir "$REPAIR/checkpoint" \
  --wikidata-dir "$WIKIDATA_DIR"

python scripts/evaluate_mcf_target_true_sensitive.py \
  --base-eval-json "$BASE_EVAL" \
  --post-eval-json "$POST_EVAL" \
  --split-manifest "$MANIFEST" \
  --out "$PAPER_EVAL"

python - "$BASELINE_EVAL" "$PAPER_EVAL" "$REPAIR/repair_summary.json" <<'PY'
import json, sys
base=json.load(open(sys.argv[1]))
new=json.load(open(sys.argv[2]))
r=json.load(open(sys.argv[3]))

def get(m, key):
    x=m["metrics"][key]
    return x["mean"] if isinstance(x, dict) else x

print("\n" + "="*84)
print("MCF SEED 1 — BASELINE r8/m1 vs r8/m1 + 500 EXTERNAL UTILITY TEXTS")
print("="*84)
print(f"{'Metric':<30}{'Baseline':>16}{'Utility-500':>16}{'Delta':>16}")
print("-"*84)
for key,label in [
    ("Eff","Eff ↑"),
    ("Gen","Gen ↑"),
    ("Delta_Sensitive_NLL_direct","Δ Sensitive NLL ↑"),
    ("NLL_Separation_direct","NLL Separation ↑"),
    ("Spe_margin","Spe-Margin ↑"),
    ("Spe_success","Spe-Success ↑"),
    ("PPL","PPL ↓"),
]:
    a=float(get(base,key)); b=float(get(new,key))
    print(f"{label:<30}{a:>16.4f}{b:>16.4f}{b-a:>+16.4f}")
print("-"*84)
print("repair rank actual          :", r["repair_rank_actual"])
print("active before -> after      :", r["active_cases_before"], "->", r["active_cases_after"])
print("repair steps                :", r["optimization"]["steps_completed"])
print("all satisfied               :", r["optimization"]["all_satisfied"])
print("minimum final margin        :", r["minimum_margin_after"])
print("selected delta norm         :", r["selected_lm_head_delta_norm"])
print("utility row-drift RMS       :", r["utility"]["final_row_drift_rms"])
print("utility predictor states    :", r["utility"]["predictor_hidden_state_count"])
print("protocol                    :", r["protocol_status"])
print("="*84)
PY
