#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCRIPT_DIR}/run_tofu_sure_author_balanced_v7_sparsegagd_active.sh"
TMP="${SCRIPT_DIR}/.run_v8_bounded_top3_progressive_${$}.sh"
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
stage2_new = "python scripts/tofu_sure_progressive_active_hidden_repair_v8.py"
root_old = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v7_sparsegagd_active_3b_test}"'
root_new = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v8_bounded_top3_progressive_3b_test}"'
for needle in (stage1_old, stage2_old, root_old):
    if needle not in text:
        raise SystemExit(f"base launcher no longer contains expected text: {needle}")
text = text.replace(stage1_old, stage1_new, 1)
text = text.replace(stage2_old, stage2_new, 1)
text = text.replace(root_old, root_new, 1)
text = text.replace(
    "# Stage 1: sparse LM-head-only GA/GD on all answer-token rows from the 50 QAs.",
    "# Stage 1: top-3 rare/content rows with bounded 10x suppression + same-prompt GD/KL.",
    1,
)
text = text.replace(
    "# Stage 2: residual active cases only. Rank 0 = unrestricted; rank 256 = all-50 hidden basis.",
    "# Stage 2: progressive row promotion. Rank 0 = unrestricted; rank 256 = all-50 hidden basis.",
    1,
)

lines = text.splitlines()

def insert_after_exact(anchor: str, new_line: str) -> None:
    try:
        index = lines.index(anchor)
    except ValueError as exc:
        raise SystemExit(f"missing launcher anchor: {anchor}") from exc
    lines.insert(index + 1, new_line)

insert_after_exact(
    'STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"',
    'STAGE1_SUPPRESSION_FACTOR="${STAGE1_SUPPRESSION_FACTOR:-10}"',
)
insert_after_exact(
    'STAGE2_MATERIALIZATION_SAFETY_FRACTIONS="${STAGE2_MATERIALIZATION_SAFETY_FRACTIONS:-0.002,0.01,0.02,0.05,0.10}"',
    'STAGE2_MAX_PROMOTION_ROUNDS="${STAGE2_MAX_PROMOTION_ROUNDS:-5}"',
)
insert_after_exact(
    'STAGE2_MAX_PROMOTION_ROUNDS="${STAGE2_MAX_PROMOTION_ROUNDS:-5}"',
    'STAGE2_PROMOTIONS_PER_RESIDUAL_QA="${STAGE2_PROMOTIONS_PER_RESIDUAL_QA:-1}"',
)

stage1_cmd = next(i for i, line in enumerate(lines) if stage1_new in line)
stage2_cmd = next(i for i, line in enumerate(lines) if stage2_new in line)
stage1_target = next(
    (i for i in range(stage1_cmd, stage2_cmd) if '--target-forget-answer-probability "${TARGET_FORGET_PROB}"' in lines[i]),
    None,
)
if stage1_target is None:
    raise SystemExit("missing Stage1 target probability argument")
lines.insert(stage1_target + 1, '      --suppression-factor "${STAGE1_SUPPRESSION_FACTOR}" \\')

stage2_cmd = next(i for i, line in enumerate(lines) if stage2_new in line)
rank_line = next(
    (i for i in range(stage2_cmd, len(lines)) if '--repair-rank "${RANK}"' in lines[i]),
    None,
)
if rank_line is None:
    raise SystemExit("missing Stage2 repair-rank argument")
lines.insert(rank_line + 1, '        --max-promotion-rounds "${STAGE2_MAX_PROMOTION_ROUNDS}" \\')
lines.insert(rank_line + 2, '        --promotions-per-residual-qa "${STAGE2_PROMOTIONS_PER_RESIDUAL_QA}" \\')
text = "\n".join(lines) + "\n"

text = text.replace(
    'echo "===== SURE-TOFU V7 SEED ${SEED}: STAGE1 SPARSE LM-HEAD GA/GD ====="',
    'echo "===== SURE-TOFU V8 SEED ${SEED}: BOUNDED-TOP3 STAGE1 ====="',
    1,
)
text = text.replace(
    'echo "===== V7 STAGE2 RANK ${RANK}: ACTIVE CASES ONLY ====="',
    'echo "===== V8 STAGE2 RANK ${RANK}: PROGRESSIVE ACTIVE-ROW REPAIR ====="',
    1,
)
text = text.replace("locked_eval_v7_", "locked_eval_v8_")
text = text.replace(
    'print("===== V7 STAGE1 / ACTIVE REPAIR COMPARISON =====")',
    'print("===== V8 BOUNDED-TOP3 / PROGRESSIVE REPAIR COMPARISON =====")',
    1,
)
text = text.replace(
    '        "active_rows",geom["selected_active_lm_head_row_count"],\n        "actual_rank",geom["repair_rank_actual"],',
    '        "active_rows",geom["selected_active_lm_head_row_count"],\n        "promotion_rounds",geom.get("promotion_round_count_used"),\n        "actual_rank",geom["repair_rank_actual"],',
    1,
)
text = text.replace(
    'root/"comparison_v7_stage1_active_repairs.json"',
    'root/"comparison_v8_bounded_top3_progressive_repairs.json"',
)
text = text.replace(
    "SURE-TOFU V7 sparse GA/GD + active-case rank0/rank256 comparison complete.",
    "SURE-TOFU V8 bounded-top3 + progressive rank0/rank256 comparison complete.",
)
dst.write_text(text, encoding="utf-8")
PY

chmod +x "${TMP}"
bash "${TMP}" "$@"
