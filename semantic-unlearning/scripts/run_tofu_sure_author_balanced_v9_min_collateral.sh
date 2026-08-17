#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCRIPT_DIR}/run_tofu_sure_author_balanced_v7_sparsegagd_active.sh"
TMP="${SCRIPT_DIR}/.run_v9_min_collateral_${$}.sh"
trap 'rm -f -- "${TMP}"' EXIT

python - "${BASE}" "${TMP}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")

stage1_old = "python scripts/tofu_sure_sparse_lm_gagd_v7.py"
stage1_new = "python scripts/tofu_sure_sparse_lm_gagd_top3_bounded_v7.py"
stage2_old = "python scripts/tofu_sure_active_hidden_repair_v7.py"
stage2_new = "python scripts/tofu_sure_min_collateral_repair_v9.py"
root_old = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v7_sparsegagd_active_3b_test}"'
root_new = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v9_min_collateral_3b_test}"'
ranks_old = 'REPAIR_RANKS="${REPAIR_RANKS:-0 256}"'
ranks_new = 'REPAIR_RANKS="${REPAIR_RANKS:-1024}"'
for needle in (stage1_old, stage2_old, root_old, ranks_old):
    if needle not in text:
        raise SystemExit(f"base launcher no longer contains expected text: {needle}")
text = text.replace(stage1_old, stage1_new, 1)
text = text.replace(stage2_old, stage2_new, 1)
text = text.replace(root_old, root_new, 1)
text = text.replace(ranks_old, ranks_new, 1)
text = text.replace(
    "# Stage 1: sparse LM-head-only GA/GD on all answer-token rows from the 50 QAs.",
    "# Stage 1: unchanged bounded Top-3 sensitive-token suppression + same-prompt GD/KL.",
    1,
)
text = text.replace(
    "# Stage 2: residual active cases only. Rank 0 = unrestricted; rank 256 = all-50 hidden basis.",
    "# Stage 2: V9 minimum-collateral augmented-Lagrangian repair; default rank 1024.",
    1,
)

lines = text.splitlines()

def insert_after_exact(anchor: str, new_line: str) -> None:
    try:
        idx = lines.index(anchor)
    except ValueError as exc:
        raise SystemExit(f"missing launcher anchor: {anchor}") from exc
    lines.insert(idx + 1, new_line)

insert_after_exact(
    'STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"',
    'STAGE1_SUPPRESSION_FACTOR="${STAGE1_SUPPRESSION_FACTOR:-10}"',
)
insert_after_exact(
    'STAGE2_MATERIALIZATION_SAFETY_FRACTIONS="${STAGE2_MATERIALIZATION_SAFETY_FRACTIONS:-0.002,0.01,0.02,0.05,0.10}"',
    'STAGE2_FISHER_WEIGHT="${STAGE2_FISHER_WEIGHT:-0.05}"',
)
insert_after_exact(
    'STAGE2_FISHER_WEIGHT="${STAGE2_FISHER_WEIGHT:-0.05}"',
    'STAGE2_DUAL_RHO="${STAGE2_DUAL_RHO:-10}"',
)
insert_after_exact(
    'STAGE2_DUAL_RHO="${STAGE2_DUAL_RHO:-10}"',
    'STAGE2_DUAL_LR="${STAGE2_DUAL_LR:-0.25}"',
)
insert_after_exact(
    'STAGE2_DUAL_LR="${STAGE2_DUAL_LR:-0.25}"',
    'STAGE2_STAGE1_KL_MULTIPLIER="${STAGE2_STAGE1_KL_MULTIPLIER:-20}"',
)
insert_after_exact(
    'STAGE2_STAGE1_KL_MULTIPLIER="${STAGE2_STAGE1_KL_MULTIPLIER:-20}"',
    'STAGE2_KL_BUDGET_FLOOR="${STAGE2_KL_BUDGET_FLOOR:-0.02}"',
)
insert_after_exact(
    'STAGE2_KL_BUDGET_FLOOR="${STAGE2_KL_BUDGET_FLOOR:-0.02}"',
    'STAGE2_MAX_PROMOTION_ROUNDS="${STAGE2_MAX_PROMOTION_ROUNDS:-8}"',
)
insert_after_exact(
    'STAGE2_MAX_PROMOTION_ROUNDS="${STAGE2_MAX_PROMOTION_ROUNDS:-8}"',
    'STAGE2_PROMOTION_ROWS_PER_ROUND="${STAGE2_PROMOTION_ROWS_PER_ROUND:-25}"',
)
insert_after_exact(
    'STAGE2_PROMOTION_ROWS_PER_ROUND="${STAGE2_PROMOTION_ROWS_PER_ROUND:-25}"',
    'STAGE2_PROMOTION_QA_COUNT="${STAGE2_PROMOTION_QA_COUNT:-20}"',
)
insert_after_exact(
    'STAGE2_PROMOTION_QA_COUNT="${STAGE2_PROMOTION_QA_COUNT:-20}"',
    'STAGE2_POST_FEASIBLE_STEPS="${STAGE2_POST_FEASIBLE_STEPS:-250}"',
)

# Bounded Stage1 suppression factor.
stage1_cmd = next(i for i, line in enumerate(lines) if stage1_new in line)
stage2_cmd = next(i for i, line in enumerate(lines) if stage2_new in line)
stage1_target = next(
    (i for i in range(stage1_cmd, stage2_cmd) if '--target-forget-answer-probability "${TARGET_FORGET_PROB}"' in lines[i]),
    None,
)
if stage1_target is None:
    raise SystemExit("missing Stage1 target-probability argument")
lines.insert(stage1_target + 1, '      --suppression-factor "${STAGE1_SUPPRESSION_FACTOR}" \\')

# V9 Stage2 controls.
stage2_cmd = next(i for i, line in enumerate(lines) if stage2_new in line)
rank_line = next(
    (i for i in range(stage2_cmd, len(lines)) if '--repair-rank "${RANK}"' in lines[i]),
    None,
)
if rank_line is None:
    raise SystemExit("missing Stage2 repair-rank argument")
extras = [
    '        --fisher-weight "${STAGE2_FISHER_WEIGHT}" \\',
    '        --dual-rho "${STAGE2_DUAL_RHO}" \\',
    '        --dual-lr "${STAGE2_DUAL_LR}" \\',
    '        --stage1-kl-budget-multiplier "${STAGE2_STAGE1_KL_MULTIPLIER}" \\',
    '        --kl-budget-floor "${STAGE2_KL_BUDGET_FLOOR}" \\',
    '        --max-promotion-rounds "${STAGE2_MAX_PROMOTION_ROUNDS}" \\',
    '        --promotion-rows-per-round "${STAGE2_PROMOTION_ROWS_PER_ROUND}" \\',
    '        --promotion-qa-count "${STAGE2_PROMOTION_QA_COUNT}" \\',
    '        --post-feasible-steps "${STAGE2_POST_FEASIBLE_STEPS}" \\',
]
for offset, line in enumerate(extras, start=1):
    lines.insert(rank_line + offset, line)
text = "\n".join(lines) + "\n"

text = text.replace(
    'echo "===== SURE-TOFU V7 SEED ${SEED}: STAGE1 SPARSE LM-HEAD GA/GD ====="',
    'echo "===== SURE-TOFU V9 SEED ${SEED}: BOUNDED-TOP3 STAGE1 ====="',
    1,
)
text = text.replace(
    'echo "===== V7 STAGE2 RANK ${RANK}: ACTIVE CASES ONLY ====="',
    'echo "===== V9 STAGE2 RANK ${RANK}: MINIMUM-COLLATERAL CONSTRAINED REPAIR ====="',
    1,
)
text = text.replace("locked_eval_v7_", "locked_eval_v9_")
text = text.replace(
    'print("===== V7 STAGE1 / ACTIVE REPAIR COMPARISON =====")',
    'print("===== V9 BOUNDED-TOP3 / MIN-COLLATERAL REPAIR =====")',
    1,
)
text = text.replace(
    'root/"comparison_v7_stage1_active_repairs.json"',
    'root/"comparison_v9_min_collateral.json"',
)
text = text.replace(
    '"active_rows",geom["selected_active_lm_head_row_count"],',
    '"active_rows",geom["selected_active_lm_head_row_count"],\n        "promotion_rounds",geom.get("promotion_round_count_used"),\n        "base_KL",geom.get("same_prompt_non_target_kl_from_base"),\n        "KL_budget",geom.get("same_prompt_non_target_kl_budget"),',
)
text = text.replace(
    "SURE-TOFU V7 sparse GA/GD + active-case rank0/rank256 comparison complete.",
    "SURE-TOFU V9 bounded-top3 + minimum-collateral constrained repair complete.",
)
text = text.replace("V7 CLEANUP", "V9 CLEANUP")
text = text.replace("protected Full-TOFU checkpoint missing after V7 cleanup", "protected Full-TOFU checkpoint missing after V9 cleanup")

dst.write_text(text, encoding="utf-8")
PY

chmod +x "${TMP}"
bash "${TMP}" "$@"
