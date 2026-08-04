#!/usr/bin/env bash
set -euo pipefail

# Publish ZsRE Setting 5e + active LM-head repair results and checkpoints.
#
# This mirrors the MCF/TOFU publication pattern:
#   1. commit lightweight authoritative results/configuration to Git;
#   2. archive lightweight per-seed artifacts as release metadata;
#   3. split each full candidate model.safetensors below GitHub's 2 GiB
#      per-asset limit and upload it to a tagged GitHub Release.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/outputs/zsre_cal384_uniform_unknown_seeds1_10}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${PROJECT_ROOT}/outputs/zsre_cal384_repair_candidates_seeds1_10}"
REPO="${REPO:-Kp759/Unlearning}"
TARGET_BRANCH="${TARGET_BRANCH:-claude/setup-project-structure-JQ7fN}"
TAG="${TAG:-zsre-setting5e-active-repair-seeds1-10}"
RELEASE_TITLE="${RELEASE_TITLE:-ZsRE Setting 5e + active LM-head repair (seeds 1-10)}"
PART_SIZE="${PART_SIZE:-1900M}"
STAGING_ROOT="${STAGING_ROOT:-${PROJECT_ROOT}/outputs/release_staging/${TAG}}"
TRACKED_DIR="${TRACKED_DIR:-${PROJECT_ROOT}/config/best_runs/zsre}"
PYTHON="${PYTHON:-python}"

required_commands=(gh git split sha256sum tar find cp stat awk)
for command_name in "${required_commands[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -d "${PROJECT_ROOT}/scripts" ]]; then
  echo "Run from semantic-unlearning or set PROJECT_ROOT." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated." >&2
  echo "Run: gh auth login --hostname github.com --git-protocol ssh --web" >&2
  exit 1
fi

current_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
if [[ "${current_branch}" != "${TARGET_BRANCH}" ]]; then
  echo "Expected branch ${TARGET_BRANCH}, found ${current_branch}." >&2
  exit 1
fi

mkdir -p "${TRACKED_DIR}"

SUMMARY_JSON="${TRACKED_DIR}/official_setting5e_active_repair_seeds1_10.json"
SUMMARY_MD="${TRACKED_DIR}/official_setting5e_active_repair_seeds1_10.md"
SUMMARY_CSV="${TRACKED_DIR}/official_setting5e_active_repair_seeds1_10.csv"
CONFIG_JSON="${TRACKED_DIR}/official_setting5e_active_repair_seeds1_10_config.json"

"${PYTHON}" - \
  "${SOURCE_ROOT}" \
  "${CANDIDATE_ROOT}" \
  "${SUMMARY_JSON}" \
  "${SUMMARY_MD}" \
  "${SUMMARY_CSV}" \
  "${CONFIG_JSON}" <<'PY'
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

source_root = Path(sys.argv[1]).resolve()
candidate_root = Path(sys.argv[2]).resolve()
summary_json = Path(sys.argv[3]).resolve()
summary_md = Path(sys.argv[4]).resolve()
summary_csv = Path(sys.argv[5]).resolve()
config_json = Path(sys.argv[6]).resolve()

seeds = list(range(1, 11))


def get_path(value: Any, *path: str, default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def metric(block: dict[str, Any], section: str, name: str) -> Any:
    value = get_path(block, section, name)
    if value is not None:
        return value
    aliases = {
        ("forget", "PPL"): ("PPL",),
        ("forget", "ppl"): ("PPL",),
    }
    alias = aliases.get((section, name))
    if alias:
        return get_path(block, *alias)
    return None


def candidate_ppl(block: dict[str, Any]) -> Any:
    for key in ("PPL", "forget_PPL", "ppl"):
        if key in block:
            return block[key]
    return get_path(block, "forget", "PPL")


rows: list[dict[str, Any]] = []
for seed in seeds:
    seed_dir = candidate_root / f"seed{seed}"
    result_path = seed_dir / "zsre_results.json"
    weight_path = seed_dir / "active_candidate_checkpoint" / "model.safetensors"
    provenance_path = (
        seed_dir
        / "active_candidate_checkpoint"
        / "candidate_provenance.json"
    )

    for required in (result_path, weight_path, provenance_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required seed-{seed} artifact: {required}")

    result = json.loads(result_path.read_text())
    repair = result["repair"]
    setting5 = result["setting5e"]
    candidate = result["active_candidate"]
    selected = result["selected"]

    checks = get_path(repair, "official_metric_gates", "checks", default={}) or {}
    failed_gates = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and not bool(check.get("passed"))
    ]

    row = {
        "seed": seed,
        "candidate_checkpoint": str(weight_path),
        "candidate_scale": repair.get("candidate_scale"),
        "selected_scale": repair.get("selected_scale"),
        "candidate_accepted": bool(repair.get("candidate_accepted")),
        "selection_reason": repair.get("selection_reason"),
        "failed_gates": ",".join(failed_gates),
        "setting5_Eff": metric(setting5, "forget", "Eff"),
        "setting5_Gen": metric(setting5, "forget", "Gen"),
        "setting5_Spe": metric(setting5, "forget", "Spe"),
        "setting5_PPL": candidate_ppl(setting5),
        "candidate_Eff": metric(candidate, "forget", "Eff"),
        "candidate_Gen": metric(candidate, "forget", "Gen"),
        "candidate_Spe": metric(candidate, "forget", "Spe"),
        "candidate_PPL": candidate_ppl(candidate),
        "candidate_retain_Eff": metric(candidate, "retain", "Eff"),
        "candidate_retain_Gen": metric(candidate, "retain", "Gen"),
        "candidate_retain_Spe": metric(candidate, "retain", "Spe"),
        "selected_Eff": metric(selected, "forget", "Eff"),
        "selected_Gen": metric(selected, "forget", "Gen"),
        "selected_Spe": metric(selected, "forget", "Spe"),
        "selected_PPL": candidate_ppl(selected),
    }
    rows.append(row)


def numeric_values(key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def aggregate_metric(key: str) -> dict[str, float | int | None]:
    values = numeric_values(key)
    if not values:
        return {"count": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }

aggregate = {
    key: aggregate_metric(key)
    for key in (
        "candidate_Eff",
        "candidate_Gen",
        "candidate_Spe",
        "candidate_PPL",
        "candidate_retain_Eff",
        "candidate_retain_Gen",
        "candidate_retain_Spe",
        "setting5_Eff",
        "setting5_Gen",
        "setting5_Spe",
        "setting5_PPL",
    )
}

config = {
    "schema_version": 1,
    "dataset": "ZsRE",
    "method": "Setting 5e plus exact-BF16 active LM-head repair candidate",
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "seeds": seeds,
    "forget_num": 50,
    "retain_num": 1000,
    "setting5e": {
        "steps": 600,
        "embedding_lr": 1e-4,
        "lm_head_lr": 1e-4,
        "forget_loss_weight": 2.0,
        "retain_loss_weight": 1.0,
        "margin": 1.0,
        "post_training_vocabulary_restoration": True,
    },
    "active_lm_head_repair": {
        "neutral_target": "Unknown",
        "neutral_token_id": 14109,
        "repair_steps": 800,
        "repair_lr": 5e-3,
        "repair_optimizer": "adamw",
        "active_logit_margin": 0.25,
        "selection_logit_margin": 0.05,
        "repair_rank": 0,
        "repair_l2_lambda": 1e-6,
        "retain_calibration_num": 384,
        "retain_calibration_seed": 1729,
        "project_away_protected_hidden": True,
        "candidate_scales": [
            1.0,
            0.875,
            0.75,
            0.625,
            0.5,
            0.375,
            0.25,
            0.1875,
            0.125,
            0.09375,
            0.0625,
            0.046875,
            0.03125,
            0.015625,
            0.0078125,
            0.0,
        ],
    },
    "evaluation": {
        "dtype": "bfloat16",
        "eval_batch_size": 8,
        "cache_batch_size": 4,
        "device_map": "single",
        "utility_drop_tolerance_percentage_points": 0.10,
        "maximum_ppl_ratio": 1.02,
        "target_eff_max": 0.0,
        "target_gen_max": 0.0,
        "strict_utility_gates": True,
    },
    "source_roots": {
        "setting5e": str(source_root),
        "repair_candidates": str(candidate_root),
    },
}
config_json.write_text(json.dumps(config, indent=2) + "\n")

record = {
    "schema_version": 1,
    "record_id": "zsre-official-setting5e-active-repair-seeds1-10",
    "status": "BEHAVIORAL_FORGETTING_SUCCESS_STRICT_UTILITY_GATE_FAILED",
    "dataset": "ZsRE",
    "protocol": {
        "forget_num": 50,
        "retain_num": 1000,
        "seeds": seeds,
        "checkpoint_seed_alignment": "candidate checkpoint seed N evaluated on ZsRE split seed N",
    },
    "method": {
        "name": "Setting 5e plus exact-BF16 active LM-head repair",
        "candidate_checkpoint_family": (
            "outputs/zsre_cal384_repair_candidates_seeds1_10/"
            "seed{seed}/active_candidate_checkpoint"
        ),
        "selected_checkpoint_family": (
            "outputs/zsre_cal384_repair_candidates_seeds1_10/"
            "seed{seed}/selected_checkpoint"
        ),
        "configuration": str(config_json),
    },
    "aggregate": {
        "evaluated_seeds": len(rows),
        "zero_eff_gen_seeds": sum(
            float(row["candidate_Eff"]) == 0.0
            and float(row["candidate_Gen"]) == 0.0
            for row in rows
        ),
        "strict_accepted_seeds": sum(bool(row["candidate_accepted"]) for row in rows),
        "metrics_mean_sample_sd": aggregate,
    },
    "per_seed": rows,
    "scientific_conclusion": (
        "The raw active LM-head repair candidate reached Eff=0 and Gen=0 on "
        "all ten ZsRE seeds. None passed every predefined strict relative "
        "utility gate, so behavioral forgetting success and strict utility "
        "acceptance are reported separately."
    ),
}
summary_json.write_text(json.dumps(record, indent=2) + "\n")

fieldnames = list(rows[0].keys())
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)

md: list[str] = []
md.append("# ZsRE Setting 5e + active LM-head repair — seeds 1–10")
md.append("")
md.append("## Raw candidate result")
md.append("")
md.append(
    "The full repaired candidate checkpoint for each seed was evaluated before "
    "strict-gate rollback. Eff and Gen reached zero on all ten seeds."
)
md.append("")
md.append("| Seed | Strict gate | Scale | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ | Failed gates |")
md.append("|---:|:---:|---:|---:|---:|---:|---:|:---|")
for row in rows:
    md.append(
        "| {seed} | {gate} | {scale} | {eff} | {gen} | {spe} | {ppl} | {failed} |".format(
            seed=row["seed"],
            gate="PASS" if row["candidate_accepted"] else "FAIL",
            scale=fmt(row["candidate_scale"], 6),
            eff=fmt(row["candidate_Eff"], 6),
            gen=fmt(row["candidate_Gen"], 6),
            spe=fmt(row["candidate_Spe"], 6),
            ppl=fmt(row["candidate_PPL"], 4),
            failed=row["failed_gates"] or "none",
        )
    )
md.append("")
md.append("## Aggregate: mean ± sample SD")
md.append("")
md.append("| Metric | Mean ± SD |")
md.append("|---|---:|")
for key, label, digits in (
    ("candidate_Eff", "Eff ↓", 6),
    ("candidate_Gen", "Gen ↓", 6),
    ("candidate_Spe", "Spe ↑", 6),
    ("candidate_PPL", "PPL ↓", 6),
    ("candidate_retain_Eff", "Retain Eff", 6),
    ("candidate_retain_Gen", "Retain Gen", 6),
    ("candidate_retain_Spe", "Retain Spe", 6),
):
    stats = aggregate[key]
    if stats["mean"] is not None:
        md.append(
            f"| {label} | **{stats['mean']:.{digits}f} ± {stats['sample_sd']:.{digits}f}** |"
        )
md.append("")
md.append("## Interpretation")
md.append("")
md.append("- Raw behavioral forgetting success: **10/10 seeds** (`Eff=0`, `Gen=0`).")
md.append("- Strict relative-utility acceptance: **0/10 seeds**.")
md.append("- Every final `selected_checkpoint` is the Setting 5e fallback.")
md.append("- Every `active_candidate_checkpoint` is the preserved full repaired candidate.")
md.append("")
md.append("The result supports reproducible behavioral forgetting on the official ZsRE Eff/Gen metrics. Strict utility-gate acceptance is reported separately and was not achieved.")
md.append("")
md.append("The machine-readable authoritative record is `config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.json`.")
summary_md.write_text("\n".join(md) + "\n")

print(json.dumps(record["aggregate"], indent=2))
PY

# Commit lightweight results and the exact patched evaluator/repair runner.
git -C "${REPO_ROOT}" add \
  "semantic-unlearning/config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.json" \
  "semantic-unlearning/config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.md" \
  "semantic-unlearning/config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.csv" \
  "semantic-unlearning/config/best_runs/zsre/official_setting5e_active_repair_seeds1_10_config.json" \
  "semantic-unlearning/scripts/zsre_bf16_safe_active_repair_v2.py"

if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
  git -C "${REPO_ROOT}" commit -m \
    "results(zsre): record active LM-head repair seeds 1-10"
  git -C "${REPO_ROOT}" push origin "${TARGET_BRANCH}"
else
  echo "No new lightweight result changes to commit."
fi

rm -rf "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}/metadata" "${STAGING_ROOT}/parts"

cp "${SUMMARY_JSON}" "${STAGING_ROOT}/metadata/"
cp "${SUMMARY_MD}" "${STAGING_ROOT}/metadata/"
cp "${SUMMARY_CSV}" "${STAGING_ROOT}/metadata/"
cp "${CONFIG_JSON}" "${STAGING_ROOT}/metadata/"

# Preserve all lightweight per-seed artifacts while excluding model weights.
tar \
  --exclude='model.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  -czf "${STAGING_ROOT}/zsre-setting5e-active-repair-seeds1-10-metadata.tar.gz" \
  -C "${PROJECT_ROOT}" \
  "outputs/zsre_cal384_uniform_unknown_seeds1_10" \
  "outputs/zsre_cal384_repair_candidates_seeds1_10"

MANIFEST="${STAGING_ROOT}/checkpoint_manifest.tsv"
PART_SUMS="${STAGING_ROOT}/release_part_sha256sums.txt"
RECONSTRUCT="${STAGING_ROOT}/RECONSTRUCT.md"
METADATA_SUMS="${STAGING_ROOT}/metadata_sha256sums.txt"

printf 'seed\tcheckpoint_path\tweight_bytes\tweight_sha256\tpart_name\tpart_bytes\tpart_sha256\n' > "${MANIFEST}"
: > "${PART_SUMS}"

cat > "${RECONSTRUCT}" <<'EOF'
# Reconstruct ZsRE active-repair candidate checkpoints

Each seed's full `model.safetensors` was split into release assets below
GitHub's per-asset size limit. Download all parts for one seed, then concatenate
in lexical order. Example for seed 1:

```bash
cat seed1.active_candidate.model.safetensors.part-* > model.safetensors
sha256sum model.safetensors
```

Compare the reconstructed hash against `checkpoint_manifest.tsv`.

Place the reconstructed file beside the corresponding small checkpoint files
from the metadata archive at:

```text
outputs/zsre_cal384_repair_candidates_seeds1_10/
  seedN/active_candidate_checkpoint/model.safetensors
```

These assets are the raw repaired candidates with `Eff=0` and `Gen=0`. They are
not the strict selected checkpoints; strict utility acceptance was 0/10.
EOF

(
  cd "${STAGING_ROOT}/metadata"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${METADATA_SUMS}"

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  gh release create "${TAG}" \
    --repo "${REPO}" \
    --target "${TARGET_BRANCH}" \
    --title "${RELEASE_TITLE}" \
    --notes-file "${SUMMARY_MD}"
else
  echo "Using existing release ${TAG}."
fi

gh release upload "${TAG}" \
  "${STAGING_ROOT}/zsre-setting5e-active-repair-seeds1-10-metadata.tar.gz" \
  "${SUMMARY_JSON}" \
  "${SUMMARY_MD}" \
  "${SUMMARY_CSV}" \
  "${CONFIG_JSON}" \
  "${METADATA_SUMS}" \
  "${RECONSTRUCT}" \
  --repo "${REPO}" \
  --clobber

for seed in {1..10}; do
  checkpoint_dir="${CANDIDATE_ROOT}/seed${seed}/active_candidate_checkpoint"
  weight="${checkpoint_dir}/model.safetensors"

  [[ -f "${weight}" ]] || {
    echo "Missing candidate checkpoint: ${weight}" >&2
    exit 1
  }

  part_dir="${STAGING_ROOT}/parts/seed${seed}"
  rm -rf "${part_dir}"
  mkdir -p "${part_dir}"

  prefix="seed${seed}.active_candidate.model.safetensors.part-"
  weight_bytes="$(stat -c '%s' "${weight}")"
  weight_sha="$(sha256sum "${weight}" | awk '{print $1}')"

  echo "Splitting candidate seed ${seed}: ${weight}"
  split -b "${PART_SIZE}" -d -a 2 \
    "${weight}" \
    "${part_dir}/${prefix}"

  shopt -s nullglob
  parts=("${part_dir}/${prefix}"*)
  shopt -u nullglob
  [[ ${#parts[@]} -gt 0 ]] || {
    echo "No split parts created for seed ${seed}." >&2
    exit 1
  }

  seed_sum_file="${part_dir}/seed${seed}.parts.sha256"
  (
    cd "${part_dir}"
    sha256sum "${prefix}"* > "$(basename "${seed_sum_file}")"
  )
  cat "${seed_sum_file}" >> "${PART_SUMS}"

  for part in "${parts[@]}"; do
    part_name="$(basename "${part}")"
    part_bytes="$(stat -c '%s' "${part}")"
    part_sha="$(sha256sum "${part}" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${seed}" \
      "outputs/zsre_cal384_repair_candidates_seeds1_10/seed${seed}/active_candidate_checkpoint" \
      "${weight_bytes}" \
      "${weight_sha}" \
      "${part_name}" \
      "${part_bytes}" \
      "${part_sha}" >> "${MANIFEST}"
  done

  echo "Uploading ${#parts[@]} checkpoint parts for seed ${seed}."
  gh release upload "${TAG}" \
    "${parts[@]}" \
    "${seed_sum_file}" \
    --repo "${REPO}" \
    --clobber

  rm -rf "${part_dir}"
done

gh release upload "${TAG}" \
  "${MANIFEST}" \
  "${PART_SUMS}" \
  --repo "${REPO}" \
  --clobber

sha256sum \
  "${STAGING_ROOT}/zsre-setting5e-active-repair-seeds1-10-metadata.tar.gz" \
  "${MANIFEST}" \
  "${PART_SUMS}" \
  "${RECONSTRUCT}" \
  > "${STAGING_ROOT}/RELEASE_SHA256SUMS"

gh release upload "${TAG}" \
  "${STAGING_ROOT}/RELEASE_SHA256SUMS" \
  --repo "${REPO}" \
  --clobber

echo
echo "Published ZsRE release:"
gh release view "${TAG}" --repo "${REPO}"
