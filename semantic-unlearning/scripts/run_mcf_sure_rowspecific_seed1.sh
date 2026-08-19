#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/models/Llama-3.2-3B-Instruct-0cb88a4f764b7a12671c53f0838cd831a0843b95-runtime}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/mcf_sure_rome_target_true_r8m1}"
OUT_ROOT="${OUT_ROOT:-outputs/mcf_sure_rowspecific_sensitive_r8m1}"
SEED="${SEED:-1}"
ROW_SCOPE="${ROW_SCOPE:-sensitive_only}"
REPAIR_RANK="${REPAIR_RANK:-8}"
ACTIVE_MARGIN="${ACTIVE_MARGIN:-1.0}"
REPAIR_STEPS="${REPAIR_STEPS:-300}"
REPAIR_LR="${REPAIR_LR:-0.005}"
HINGE_WEIGHT="${HINGE_WEIGHT:-2.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"

STAGE1="$BASELINE_ROOT/seed${SEED}/setting5e_best_run_mirrored"
STAGE1_CKPT="$STAGE1/emb_lm_all_restore_post_training_true/checkpoint"
STAGE1_CONFIG="$STAGE1/config_used.json"
VISIBLE="$BASELINE_ROOT/protocol/repair_visible_mcf_target_true_sensitive.json"
MANIFEST="$BASELINE_ROOT/protocol/seed${SEED}_manifest.json"
BASE_EVAL="$BASELINE_ROOT/seed${SEED}/base_original_mcf_eval.json"
BASELINE_EVAL="$BASELINE_ROOT/seed${SEED}/target_true_sensitive_eval.json"

ROOT="$OUT_ROOT/seed${SEED}"
REPAIR="$ROOT/repair_rowspecific"
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
  "===== MCF SURE ROW-SPECIFIC CONTEXT STAGE 2 =====" \
  "seed=$SEED" \
  "reusing Stage 1=$STAGE1_CKPT" \
  "row_scope=$ROW_SCOPE" \
  "per-row rank cap=$REPAIR_RANK" \
  "active_margin=$ACTIVE_MARGIN" \
  "objective=canonical pairwise hinge + L2; no utility/retain/probes"

python scripts/mcf_forget_only_rowspecific_repair.py \
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
  --row-scope "$ROW_SCOPE" \
  --active-margin "$ACTIVE_MARGIN" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --repair-optimizer adamw \
  --hinge-weight "$HINGE_WEIGHT" \
  --delta-l2-lambda "$DELTA_L2_LAMBDA" \
  --retain-kl-mu 0 \
  --retain-calibration-num 0 \
  --repair-rank "$REPAIR_RANK" \
  --no-project-away-retain-hidden \
  --stop-when-all-satisfied \
  --dtype bf16 \
  --device-map single \
  --margin-batch-size 4 \
  --save-model

test -d "$REPAIR/checkpoint"
test -f "$REPAIR/repair_summary.json"
test -f "$REPAIR/row_specific_geometry.json"

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

def get(m, key, legacy=None):
    metrics=m["metrics"]
    use=key if key in metrics else legacy
    if use is None or use not in metrics:
        raise KeyError(f"Missing metric {key} (legacy={legacy})")
    x=metrics[use]
    return x["mean"] if isinstance(x, dict) else x

print("\n" + "="*96)
print("MCF SEED 1 — GLOBAL r8/m1 BASELINE vs ROW-SPECIFIC CONTEXT REPAIR")
print("FS/GFS are pairwise success rates; they are NOT ZeroUnlearn probability Eff/Gen.")
print("="*96)
print(f"{'Metric':<36}{'Global baseline':>18}{'Row-specific':>18}{'Delta':>18}")
print("-"*96)
for key,legacy,label in [
    ("FS","Eff","FS (Forget Success) ↑"),
    ("GFS","Gen","GFS (Generalized Forget Success) ↑"),
    ("Delta_Sensitive_NLL_direct",None,"Δ Sensitive NLL ↑"),
    ("Delta_Reference_NLL_direct",None,"Δ Reference NLL (audit)"),
    ("NLL_Separation_direct",None,"NLL Separation ↑"),
    ("Spe_margin",None,"Spe-Margin ↑"),
    ("Spe_success",None,"Spe-Success ↑"),
    ("PPL",None,"PPL ↓/stable"),
]:
    try:
        a=float(get(base,key,legacy)); b=float(get(new,key,legacy))
    except KeyError:
        continue
    print(f"{label:<36}{a:>18.4f}{b:>18.4f}{b-a:>+18.4f}")
print("-"*96)
print("row scope                   :", r["row_scope"])
print("selected rows               :", r["selected_lm_head_rows"])
print("row context ranks           :", r["row_context_ranks"])
print("mean row context rank       :", r["mean_row_context_rank"])
print("active before -> after      :", r["active_cases_before"], "->", r["active_cases_after"])
print("true ROME failures direct   :", r["true_rome_failure_count_after"])
print("repair steps                :", r["optimization"]["steps_completed"])
print("all target margins satisfied:", r["optimization"]["all_satisfied"])
print("minimum final margin        :", r["minimum_margin_after"])
print("selected delta norm         :", r["selected_lm_head_delta_norm"])
print("protocol                    :", r["protocol_status"])
print("="*96)
PY
