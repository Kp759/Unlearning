#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/models/Llama-3.2-3B-Instruct-0cb88a4f764b7a12671c53f0838cd831a0843b95-runtime}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/mcf_sure_rome_target_true_r8m1}"
OUT_ROOT="${OUT_ROOT:-outputs/mcf_sure_probability_context_protected_seed1}"
SEED="${SEED:-1}"

SENSITIVE_NLL_FLOOR="${SENSITIVE_NLL_FLOOR:-8.0}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-1.0}"
REPAIR_RANK="${REPAIR_RANK:-8}"
REPAIR_STEPS="${REPAIR_STEPS:-400}"
REPAIR_LR="${REPAIR_LR:-0.005}"
ABSOLUTE_WEIGHT="${ABSOLUTE_WEIGHT:-2.0}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-1.0}"
REFERENCE_ANCHOR_WEIGHT="${REFERENCE_ANCHOR_WEIGHT:-0.25}"
CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-1.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"
CONTEXT_PROTECTION="${CONTEXT_PROTECTION:-ridge}"
RETAIN_RIDGE_LAMBDA="${RETAIN_RIDGE_LAMBDA:-0.10}"
RETAIN_RANK_CAP="${RETAIN_RANK_CAP:-128}"
CALIBRATION_NUM="${CALIBRATION_NUM:-128}"

STAGE1="$BASELINE_ROOT/seed${SEED}/setting5e_best_run_mirrored"
STAGE1_CKPT="$STAGE1/emb_lm_all_restore_post_training_true/checkpoint"
STAGE1_CONFIG="$STAGE1/config_used.json"
VISIBLE="$BASELINE_ROOT/protocol/repair_visible_mcf_target_true_sensitive.json"
MANIFEST="$BASELINE_ROOT/protocol/seed${SEED}_manifest.json"

ROOT="$OUT_ROOT/seed${SEED}"
CALIB="$ROOT/context_calibration.json"
BASE_EVAL="$ROOT/base_original_mcf_eval.json"
STAGE1_EVAL="$ROOT/stage1_original_mcf_eval.json"
STAGE1_PAPER="$ROOT/stage1_target_true_sensitive_eval.json"
REPAIR="$ROOT/repair_probability_context_protected"
POST_EVAL="$ROOT/post_original_mcf_eval.json"
PAPER_EVAL="$ROOT/target_true_sensitive_eval.json"

for path in "$MODEL_PATH" "$STAGE1_CKPT" "$WIKIDATA_DIR"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
for path in "$MCF_PATH" "$STAGE1_CONFIG" "$VISIBLE" "$MANIFEST"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done

rm -rf "$ROOT"
mkdir -p "$ROOT"

echo "===== BUILD DISJOINT CONTEXT CALIBRATION ====="
python scripts/build_mcf_disjoint_context_calibration.py \
  --mcf-path "$MCF_PATH" \
  --seed-manifest "$MANIFEST" \
  --out "$CALIB" \
  --calibration-num "$CALIBRATION_NUM"

echo "===== BASE MODEL EVAL ====="
python scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out "$BASE_EVAL" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed "$SEED" \
  --sample-mode official \
  --dtype bf16 \
  --device-map single
python scripts/annotate_ppl_provenance.py \
  --eval-json "$BASE_EVAL" \
  --model-dir "$MODEL_PATH" \
  --wikidata-dir "$WIKIDATA_DIR"

echo "===== STAGE-1-ONLY EVAL (DIAGNOSE WHERE SPE DROPS) ====="
python scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "$STAGE1_CKPT" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out "$STAGE1_EVAL" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed "$SEED" \
  --sample-mode official \
  --dtype bf16 \
  --device-map single
python scripts/annotate_ppl_provenance.py \
  --eval-json "$STAGE1_EVAL" \
  --model-dir "$STAGE1_CKPT" \
  --wikidata-dir "$WIKIDATA_DIR"
python scripts/evaluate_mcf_target_true_sensitive.py \
  --base-eval-json "$BASE_EVAL" \
  --post-eval-json "$STAGE1_EVAL" \
  --split-manifest "$MANIFEST" \
  --out "$STAGE1_PAPER"

echo "===== ABSOLUTE SENSITIVE SUPPRESSION + CONTEXT-PROTECTED STAGE 2 ====="
python scripts/mcf_forget_only_probability_context_protected_repair.py \
  --model-path "$STAGE1_CKPT" \
  --experiment-config-path "$STAGE1_CONFIG" \
  --output-dir "$REPAIR" \
  --mcf-cache-path "$VISIBLE" \
  --calibration-json "$CALIB" \
  --seed "$SEED" \
  --forget-num 50 \
  --sample-mode official \
  --sensitive-nll-floor "$SENSITIVE_NLL_FLOOR" \
  --pairwise-margin "$PAIRWISE_MARGIN" \
  --repair-rank "$REPAIR_RANK" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --absolute-weight "$ABSOLUTE_WEIGHT" \
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --reference-anchor-weight "$REFERENCE_ANCHOR_WEIGHT" \
  --context-weight "$CONTEXT_WEIGHT" \
  --delta-l2-lambda "$DELTA_L2_LAMBDA" \
  --context-protection "$CONTEXT_PROTECTION" \
  --retain-ridge-lambda "$RETAIN_RIDGE_LAMBDA" \
  --retain-rank-cap "$RETAIN_RANK_CAP" \
  --dtype bf16 \
  --device-map single \
  --margin-batch-size 4 \
  --save-model

test -d "$REPAIR/checkpoint"
test -f "$REPAIR/repair_summary.json"
test -f "$REPAIR/geometry_report.json"

echo "===== FINAL ORIGINAL-UNSWAPPED MCF EVAL ====="
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

python - "$BASE_EVAL" "$STAGE1_PAPER" "$PAPER_EVAL" "$REPAIR/repair_summary.json" <<'PY'
import json, sys

base_raw=json.load(open(sys.argv[1]))
stage1=json.load(open(sys.argv[2]))
post=json.load(open(sys.argv[3]))
summary=json.load(open(sys.argv[4]))

def metric(obj, key):
    v=obj["metrics"][key]
    return v["mean"] if isinstance(v, dict) else v

base_forget=base_raw["forget"]
base_fs=float(base_forget["post_rewrite_success"][0])
base_gfs=float(base_forget["post_paraphrase_success"][0])
base_spe=float(base_forget["post_neighborhood_success"][0])
base_ppl=float(base_raw["forget_PPL"])

print("\n" + "="*100)
print("MCF SEED 1 — BASE vs STAGE 1 vs ABSOLUTE+CONTEXT-PROTECTED SURE")
print("="*100)
print(f"{'Metric':<34}{'Base':>18}{'Stage 1':>18}{'Final':>18}")
print("-"*100)
for key,label,base_val in [
    ("FS","FS ↑",base_fs),
    ("GFS","GFS ↑",base_gfs),
    ("Spe_success","Spe-Success ↑",base_spe),
    ("PPL","PPL ↓/stable",base_ppl),
]:
    print(f"{label:<34}{base_val:>18.4f}{float(metric(stage1,key)):>18.4f}{float(metric(post,key)):>18.4f}")
print("-"*100)
for key,label in [
    ("Delta_Sensitive_NLL_direct","Δ Sensitive NLL direct ↑"),
    ("Delta_Reference_NLL_direct","Δ Reference NLL direct audit"),
    ("NLL_Separation_direct","NLL separation direct ↑"),
]:
    print(f"{label:<34}{'--':>18}{float(metric(stage1,key)):>18.4f}{float(metric(post,key)):>18.4f}")
print("-"*100)
print("rank requested/actual              :", summary["repair_rank_requested"], "/", summary["repair_rank_actual"])
print("absolute failures after            :", summary["absolute_failures_after"])
print("pairwise failures after            :", summary["pairwise_failures_after"])
print("context selected-logit shift RMS   :", summary["optimization"]["context_selected_logit_shift_rms"])
print("selected LM-head delta norm         :", summary["selected_lm_head_delta_norm"])
print("evaluation probe leakage            :", summary["evaluation_probe_leakage"])
print("="*100)
PY

echo "Done: $ROOT"
