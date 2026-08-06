#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
REPO_ROOT="${REPO_ROOT:-/scratch/yl258/kp759/Unlearning}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/semantic-unlearning}"
BRANCH="${BRANCH:-claude/setup-project-structure-JQ7fN}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/outputs/zsre_cal384_uniform_unknown_seeds1_10}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${PROJECT_ROOT}/outputs/zsre_cal384_repair_candidates_seeds1_10}"
REPRO_ROOT="${REPRO_ROOT:-${PROJECT_ROOT}/outputs/zsre_cal384_repair_candidates_seeds1_10_repro}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash fix_zsre_reproducibility.sh patch
  bash fix_zsre_reproducibility.sh audit
  bash fix_zsre_reproducibility.sh rerun-repair
  bash fix_zsre_reproducibility.sh finalize

Optional environment:
  REPO_ROOT, PROJECT_ROOT, BRANCH, SOURCE_ROOT, CANDIDATE_ROOT,
  REPRO_ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, FINAL_CANDIDATE_ROOT
EOF
}

require_project() {
  [[ -d "${PROJECT_ROOT}/scripts" ]] || {
    echo "Missing semantic-unlearning project: ${PROJECT_ROOT}" >&2
    exit 2
  }
}

patch_repo() {
  require_project
  cd "${REPO_ROOT}"

  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository is dirty. Commit or stash changes before applying the ZsRE patch." >&2
    git status --short
    exit 2
  fi

  git fetch origin
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
  cd "${PROJECT_ROOT}"

  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

ROOT = Path.cwd()

def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"already patched: {relative}")
            return
        raise SystemExit(f"Patch context not found in {relative}: {old!r}")
    if count != 1:
        raise SystemExit(f"Expected one patch context in {relative}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {relative}")

# The published record used retain calibration 384. Make all defaults and the
# official registry reproduce that exact method rather than silently using 128.
for relative in (
    "scripts/run_zsre_gagd_setting5e_active_repair.sh",
    "scripts/run_zsre_bf16_safe_active_repair_v2.sh",
):
    replace_once(
        relative,
        'RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-128}"',
        'RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-384}"',
    )

for relative in (
    "scripts/zsre_gagd_setting5e_active_repair.py",
    "scripts/zsre_bf16_safe_active_repair_v2.py",
):
    replace_once(
        relative,
        'parser.add_argument("--retain-calibration-num", type=int, default=128)',
        'parser.add_argument("--retain-calibration-num", type=int, default=384)',
    )

replace_once(
    "config/official_benchmarks/registry.json",
    '"retain_calibration_num": 128',
    '"retain_calibration_num": 384',
)

# Stable wrapper: use the published calibration and allow the known rejected
# candidates to finish all seeds so final Setting-5e rollbacks can be reported.
replace_once(
    "scripts/run_three_benchmark_experiments.sh",
    '''    DEVICE_MAP="${DEVICE_MAP}" \\
    bash scripts/run_zsre_gagd_setting5e_active_repair.sh "${MODEL_PATH}"''',
    '''    DEVICE_MAP="${DEVICE_MAP}" \\
    RETAIN_CALIBRATION_NUM="${ZSRE_RETAIN_CALIBRATION_NUM:-384}" \\
    FAIL_IF_TARGET_MISSED="${ZSRE_FAIL_IF_TARGET_MISSED:-0}" \\
    bash scripts/run_zsre_gagd_setting5e_active_repair.sh "${MODEL_PATH}"''',
)

# The archived table is labeled sample SD. Use ddof=1 consistently.
replace_once(
    "scripts/aggregate_zsre_gagd_results.py",
    '''    array = np.array(values, dtype=np.float64)
    if not np.isfinite(array).any():
        return None, None
    return float(np.nanmean(array)), float(np.nanstd(array))
''',
    '''    array = np.array(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    sample_sd = 0.0 if finite.size == 1 else float(np.std(finite, ddof=1))
    return float(np.mean(finite)), sample_sd
''',
)

# Keep strict publication behavior by default, but add an explicit mode that
# aggregates transparently selected Setting-5e fallbacks.
replace_once(
    "scripts/aggregate_zsre_gagd_results.py",
    '''    parser.add_argument("--require-selected-eff-max", type=float, default=0.0)
    parser.add_argument("--require-selected-gen-max", type=float, default=0.0)
    args = parser.parse_args()
''',
    '''    parser.add_argument("--require-selected-eff-max", type=float, default=0.0)
    parser.add_argument("--require-selected-gen-max", type=float, default=0.0)
    parser.add_argument(
        "--allow-fallback-selected",
        action="store_true",
        help=(
            "Aggregate selected Setting-5e fallbacks even when the raw active "
            "candidate failed strict gates. Candidate rows remain diagnostic."
        ),
    )
    args = parser.parse_args()
''',
)

replace_once(
    "scripts/aggregate_zsre_gagd_results.py",
    '''    require_selected_targets(
        results,
        eff_max=args.require_selected_eff_max,
        gen_max=args.require_selected_gen_max,
    )
''',
    '''    if not args.allow_fallback_selected:
        require_selected_targets(
            results,
            eff_max=args.require_selected_eff_max,
            gen_max=args.require_selected_gen_max,
        )
''',
)

replace_once(
    "scripts/aggregate_zsre_gagd_results.py",
    '''        "# ZsRE Setting 5e + active LM-head repair aggregate",
        "",
''',
    '''        "# ZsRE Setting 5e + active LM-head repair aggregate",
        "",
        (
            "**Reporting rule:** `Setting 5e + active candidate` is a raw "
            "diagnostic candidate. `Selected` is the algorithm output after "
            "strict-gate rollback and may therefore equal Setting 5e."
        ),
        "",
''',
)

for relative in (
    "scripts/run_zsre_gagd_setting5e_active_repair.sh",
    "scripts/run_zsre_bf16_safe_active_repair_v2.sh",
):
    replace_once(
        relative,
        '''"${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \\
  "${RESULT_ARGS[@]}" \\
  --output-dir "${OUT_ROOT}/aggregate"
''',
        '''AGGREGATE_ARGS=()
if [[ "${FAIL_IF_TARGET_MISSED}" == "0" ]]; then
  AGGREGATE_ARGS+=(--allow-fallback-selected)
fi

"${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \\
  "${RESULT_ARGS[@]}" \\
  --output-dir "${OUT_ROOT}/aggregate" \\
  "${AGGREGATE_ARGS[@]}"
''',
    )

# Capture exact repair-only runtime config and sampled case identities.
replace_once(
    "scripts/zsre_bf16_safe_active_repair_v2.py",
    '''    repair_dir = output_dir / "active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args)
''',
    '''    repair_dir = output_dir / "active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = {
        "method": "zsre_bf16_safe_active_repair_v2",
        **vars(args),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    gagd.write_json(output_dir / "config_used.json", runtime_config)

    model, tok = load_model(args)
''',
)

replace_once(
    "scripts/zsre_bf16_safe_active_repair_v2.py",
    '''    records = (forget_records, retain_records)

    print("Evaluating saved 600-step Setting 5e checkpoint")
''',
    '''    records = (forget_records, retain_records)
    gagd.write_json(
        output_dir / "sampled_case_ids.json",
        {
            "seed": args.seed,
            "forget_case_ids": [record["case_id"] for record in forget_records],
            "retain_case_ids": [record["case_id"] for record in retain_records],
            "zsre_sha256": zsre.file_sha256(zsre_path),
        },
    )

    print("Evaluating saved 600-step Setting 5e checkpoint")
''',
)

replace_once(
    "scripts/zsre_bf16_safe_active_repair_v2.py",
    '''            "method": "zsre_bf16_safe_active_repair_v2",
            "dataset": "ZsRE",
''',
    '''            "method": "zsre_bf16_safe_active_repair_v2",
            "protocol_status": summary["protocol_status"],
            "protocol_status_reason": summary["protocol_status_reason"],
            "dataset": "ZsRE",
''',
)

# Hash the actual repair-only and aggregation implementation in official
# manifests, not only the older combined runner.
replace_once(
    "src/official_benchmarks/provenance.py",
    '''    "scripts/zsre_gagd_setting5e_active_repair.py",
    "scripts/run_zsre_gagd_setting5e_active_repair.sh",
''',
    '''    "scripts/zsre_gagd_setting5e_active_repair.py",
    "scripts/run_zsre_gagd_setting5e_active_repair.sh",
    "scripts/zsre_bf16_safe_active_repair_v2.py",
    "scripts/run_zsre_bf16_safe_active_repair_v2.sh",
    "scripts/aggregate_zsre_gagd_results.py",
''',
)

old_claim = (
    "The result supports reproducible behavioral forgetting on the official "
    "ZsRE Eff/Gen metrics. Strict utility-gate acceptance is reported separately "
    "and was not achieved."
)
new_claim = (
    "The raw candidates are diagnostic evaluation-conditioned repairs: they "
    "reached zero official Eff/Gen, but none passed the predefined strict "
    "relative-utility gates. The final selected outputs are the Setting 5e "
    "fallbacks and must be reported separately."
)
replace_once(
    "config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.md",
    old_claim,
    new_claim,
)
replace_once(
    "scripts/publish_zsre_active_repair_checkpoints_release.sh",
    f'md.append("{old_claim}")',
    f'md.append("{new_claim}")',
)

replace_once(
    "tests/test_zsre_gagd_setting5e_active.py",
    '            self.assertEqual(selected["forget_Eff_down_std"], 1.0)',
    '            self.assertAlmostEqual(selected["forget_Eff_down_std"], 2 ** 0.5, places=6)',
)

print("ZsRE patch applied.")
PY

  "${PYTHON_BIN}" -m unittest discover -s tests -p 'test_zsre_gagd_setting5e_active.py'
  "${PYTHON_BIN}" -m unittest discover -s tests -p 'test_official_benchmarks.py'
  git diff --check

  echo
  git status --short
  echo
  echo "Review the diff, then commit:"
  echo 'git add semantic-unlearning && git commit -m "fix(zsre): separate rejected candidates from selected fallbacks"'
}

audit_artifacts() {
  require_project
  cd "${PROJECT_ROOT}"

  SOURCE_ROOT="${SOURCE_ROOT}" CANDIDATE_ROOT="${CANDIDATE_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os

source = Path(os.environ["SOURCE_ROOT"])
candidate = Path(os.environ["CANDIDATE_ROOT"])
missing = []
errors = []
dataset_hashes = set()
needs_repair_config_capture = False

for seed in range(1, 11):
    s = source / f"seed{seed}"
    c = candidate / f"seed{seed}"
    required = [
        s / "setting5e" / "checkpoint" / "model.safetensors",
        s / "setting5e" / "checkpoint" / "zsre_neutral_target.json",
        c / "zsre_results.json",
        c / "active_repair" / "repair_summary.json",
        c / "active_candidate_checkpoint" / "model.safetensors",
        c / "active_candidate_checkpoint" / "candidate_provenance.json",
        c / "selected_checkpoint" / "model.safetensors",
    ]
    for path in required:
        if not path.is_file():
            missing.append(str(path))

    result_path = c / "zsre_results.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        dataset_hashes.add(result.get("zsre_sha256"))
        repair = result.get("repair", {})
        candidate_metrics = result.get("active_candidate", {}).get("forget", {})
        selected_metrics = result.get("selected", {}).get("forget", {})
        setting5_metrics = result.get("setting5e", {}).get("forget", {})
        if repair.get("candidate_accepted") is not False:
            errors.append(f"seed {seed}: candidate_accepted is not false")
        if float(repair.get("selected_scale", -1)) != 0.0:
            errors.append(f"seed {seed}: selected_scale is not 0")
        if float(candidate_metrics.get("Eff", -1)) != 0.0:
            errors.append(f"seed {seed}: candidate Eff is not 0")
        if float(candidate_metrics.get("Gen", -1)) != 0.0:
            errors.append(f"seed {seed}: candidate Gen is not 0")
        for metric in ("Eff", "Gen", "Spe"):
            if selected_metrics.get(metric) != setting5_metrics.get(metric):
                errors.append(f"seed {seed}: selected {metric} != Setting5 {metric}")

    if not (c / "config_used.json").is_file():
        needs_repair_config_capture = True
    if not (c / "sampled_case_ids.json").is_file() and not (
        s / "sampled_case_ids.json"
    ).is_file():
        needs_repair_config_capture = True

print(f"SOURCE_ROOT={source}")
print(f"CANDIDATE_ROOT={candidate}")
print(f"dataset_hashes={sorted(str(x) for x in dataset_hashes)}")
print(f"missing_required_artifacts={len(missing)}")
for item in missing:
    print(f"MISSING {item}")
print(f"scientific_errors={len(errors)}")
for item in errors:
    print(f"ERROR {item}")
print(f"NEED_REPAIR_ONLY_RERUN={int(needs_repair_config_capture)}")

if missing or errors or len(dataset_hashes) != 1:
    raise SystemExit(1)
PY
}

rerun_repair() {
  require_project
  cd "${PROJECT_ROOT}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  mkdir -p "${REPRO_ROOT}"

  for seed in 1 2 3 4 5 6 7 8 9 10; do
    checkpoint="${SOURCE_ROOT}/seed${seed}/setting5e/checkpoint"
    [[ -d "${checkpoint}" ]] || {
      echo "Missing Stage-1 Setting 5e checkpoint: ${checkpoint}" >&2
      exit 2
    }

    "${PYTHON_BIN}" scripts/zsre_bf16_safe_active_repair_v2.py \
      --setting5-checkpoint "${checkpoint}" \
      --output-dir "${REPRO_ROOT}/seed${seed}" \
      --zsre-path data/zsre_mend_eval.json \
      --wikidata-dir data/wikidata \
      --seed "${seed}" \
      --forget-num 50 \
      --retain-num 1000 \
      --repair-steps 800 \
      --repair-lr 0.005 \
      --repair-optimizer adamw \
      --active-logit-margin 0.25 \
      --selection-logit-margin 0.05 \
      --repair-rank 0 \
      --repair-l2-lambda 1e-6 \
      --retain-calibration-num 384 \
      --retain-calibration-seed 1729 \
      --project-away-protected-hidden \
      --candidate-scales "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0" \
      --utility-drop-tolerance 0.10 \
      --max-ppl-ratio 1.02 \
      --target-eff-max 0.0 \
      --target-gen-max 0.0 \
      --eval-batch-size 8 \
      --cache-batch-size 4 \
      --dtype bf16 \
      --device-map single \
      --save-candidate-checkpoint \
      --save-selected-checkpoint \
      --no-fail-if-target-missed
  done

  result_args=()
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    result_args+=(--result "${REPRO_ROOT}/seed${seed}/zsre_results.json")
  done

  "${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \
    "${result_args[@]}" \
    --allow-fallback-selected \
    --output-dir "${REPRO_ROOT}/aggregate"

  echo "Repair-only reproducibility run complete: ${REPRO_ROOT}"
}

finalize_record() {
  require_project
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local finalizer="${FINALIZER:-${script_dir}/finalize_zsre_record.py}"
  [[ -f "${finalizer}" ]] || {
    echo "Missing finalizer: ${finalizer}" >&2
    exit 2
  }

  local candidate_for_final="${FINAL_CANDIDATE_ROOT:-${REPRO_ROOT}}"
  "${PYTHON_BIN}" "${finalizer}" \
    --project-root "${PROJECT_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --candidate-root "${candidate_for_final}" \
    --output-prefix "${PROJECT_ROOT}/config/best_runs/zsre/official_setting5e_active_repair_seeds1_10_v2"
}

case "${MODE}" in
  patch) patch_repo ;;
  audit) audit_artifacts ;;
  rerun-repair) rerun_repair ;;
  finalize) finalize_record ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
