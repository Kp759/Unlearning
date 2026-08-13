#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

ROOT="${ZSRE_STAGEWISE_ROOT:-outputs/zsre_stagewise_ppl_diag_seeds1_10}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SEEDS_TEXT="${ZSRE_DIAG_SEEDS:-1 10}"
DTYPE="${DTYPE:-bf16}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  SROOT="${ROOT}/seed${SEED}"
  STAGE1="${SROOT}/stage1/emb_lm_all_restore_post_training_true/checkpoint"
  STAGE2="${SROOT}/stage2/checkpoint"
  OUT="${SROOT}/unknown_row_ppl_mechanism.json"

  test -d "${STAGE1}"
  test -d "${STAGE2}"
  echo
  echo "===== seed ${SEED}: Unknown-row PPL mechanism ====="
  python scripts/zsre_unknown_row_ppl_mechanism.py \
    --stage1-checkpoint "${STAGE1}" \
    --stage2-checkpoint "${STAGE2}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${OUT}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}"
done

python - "${ROOT}" "${SEEDS_TEXT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
rows = []
for seed in seeds:
    p = json.loads((root / f"seed{seed}/unknown_row_ppl_mechanism.json").read_text())
    rows.append({
        "seed": seed,
        "stage1_ppl": p["ppl"]["stage1"],
        "stage2_ppl": p["ppl"]["stage2"],
        "restored_unknown_ppl": p["ppl"]["stage2_with_unknown_logit_restored_to_stage1"],
        "injected_unknown_ppl": p["ppl"]["stage1_with_stage2_unknown_logit_injected"],
        "explained_fraction": p["ppl"]["unknown_row_explained_fraction_of_log_ppl_increase"],
        "delta_norm": p["unknown_row"]["delta_norm"],
        "delta_logit_mean": p["unknown_logit_delta"]["mean"],
        "delta_logit_p95": p["unknown_logit_delta"]["p95"],
        "delta_logit_max": p["unknown_logit_delta"]["max"],
        "unknown_top1_stage1": p["unknown_is_top1_fraction"]["stage1"],
        "unknown_top1_stage2": p["unknown_is_top1_fraction"]["stage2"],
        "max_non_unknown_change": p["non_unknown_logit_change_validation"]["max_abs"],
    })

summary = {
    "schema_version": 1,
    "kind": "zsre_unknown_row_ppl_mechanism_summary",
    "seeds": seeds,
    "rows": rows,
    "diagnostic_only": True,
    "heldout_zsre_probe_access": False,
}
(root / "unknown_row_ppl_mechanism_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# ZsRE Unknown-row PPL mechanism diagnostic",
    "",
    "| Seed | Stage-1 PPL | Stage-2 PPL | S2 w/ Unknown restored | S1 w/ S2 Unknown injected | Explained log-PPL frac. | ΔUnknown mean | p95 | max | Unknown top1 S1→S2 | Max non-Unknown Δlogit |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['seed']} | {r['stage1_ppl']:.4f} | {r['stage2_ppl']:.4f} | "
        f"{r['restored_unknown_ppl']:.4f} | {r['injected_unknown_ppl']:.4f} | "
        f"{r['explained_fraction']:.4f} | {r['delta_logit_mean']:.4f} | "
        f"{r['delta_logit_p95']:.4f} | {r['delta_logit_max']:.4f} | "
        f"{r['unknown_top1_stage1']:.3f}→{r['unknown_top1_stage2']:.3f} | "
        f"{r['max_non_unknown_change']:.6g} |"
    )
lines += [
    "",
    "Diagnostic only. Uses the fixed Wikidata PPL text and existing Stage-1/Stage-2 checkpoints; no ZsRE rephrases, locality probes, or benchmark-retain records are loaded.",
]
(root / "unknown_row_ppl_mechanism_summary.md").write_text("\n".join(lines) + "\n")

print("\n===== AGGREGATE UNKNOWN-ROW DIAGNOSTIC =====")
print((root / "unknown_row_ppl_mechanism_summary.md").read_text())
PY

echo "JSON: ${ROOT}/unknown_row_ppl_mechanism_summary.json"
echo "MD:   ${ROOT}/unknown_row_ppl_mechanism_summary.md"
