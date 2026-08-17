#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCRIPT_DIR}/run_tofu_sure_author_balanced_v7_sparsegagd_active.sh"
TMP="${SCRIPT_DIR}/.run_v7_top3_bounded_active_${$}.sh"
trap 'rm -f -- "${TMP}"' EXIT

python - "${BASE}" "${TMP}" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
old_script = "python scripts/tofu_sure_sparse_lm_gagd_v7.py"
new_script = "python scripts/tofu_sure_sparse_lm_gagd_top3_bounded_v7.py"
old_root = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v7_sparsegagd_active_3b_test}"'
new_root = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v7_top3_bounded_active_3b_test}"'
if old_script not in text:
    raise SystemExit(f"base launcher no longer contains expected Stage1 invocation: {old_script}")
if old_root not in text:
    raise SystemExit("base launcher no longer contains expected V7 default output root")
text = text.replace(old_script, new_script, 1)
text = text.replace(old_root, new_root, 1)
text = text.replace(
    "# Stage 1: sparse LM-head-only GA/GD on all answer-token rows from the 50 QAs.",
    "# Stage 1: top-3 rare/content rows with bounded 10x sensitive-token suppression + same-prompt GD/KL.",
    1,
)
text = text.replace(
    'echo "===== SURE-TOFU V7 SEED ${SEED}: STAGE1 SPARSE LM-HEAD GA/GD ====="',
    'echo "===== SURE-TOFU V7 BOUNDED-TOP3 SEED ${SEED}: STAGE1 CONSERVATIVE GA/GD ====="',
    1,
)
# Rename comparison artifact/heading so it cannot be confused with the old V7 run.
text = text.replace(
    'print("===== V7 STAGE1 / ACTIVE REPAIR COMPARISON =====")',
    'print("===== V7 BOUNDED-TOP3 / ACTIVE REPAIR COMPARISON =====")',
    1,
)
text = text.replace(
    'root/"comparison_v7_stage1_active_repairs.json"',
    'root/"comparison_v7_bounded_top3_active_repairs.json"',
)
dst.write_text(text, encoding="utf-8")
PY

chmod +x "${TMP}"
bash "${TMP}" "$@"
