#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SETTING5E_CHECKPOINT BASE_MODEL [EXPERIMENT_CONFIG]"
  exit 2
fi

INPUT_MODEL="$1"
BASE_MODEL="$2"
EXPERIMENT_CONFIG="${3:-${EXPERIMENT_CONFIG:-}}"

OUT_ROOT="${OUT_ROOT:-outputs/gagd_active_case_repair}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SAMPLE_MODE="${SAMPLE_MODE:-official}"
SEED="${SEED:-0}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
ACTIVE_MARGIN="${ACTIVE_MARGIN:-0.1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
MARGIN_BATCH_SIZE="${MARGIN_BATCH_SIZE:-4}"

REPAIR_STEPS="${REPAIR_STEPS:-50}"
REPAIR_LR="${REPAIR_LR:-1e-2}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
HINGE_WEIGHT="${HINGE_WEIGHT:-1.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-1e-4}"
RETAIN_KL_MU="${RETAIN_KL_MU:-0.1}"
RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-32}"
RETAIN_CALIBRATION_SEED="${RETAIN_CALIBRATION_SEED:-1729}"
REPAIR_RANK="${REPAIR_RANK:-1}"
PROJECT_AWAY_RETAIN_HIDDEN="${PROJECT_AWAY_RETAIN_HIDDEN:-1}"
SKIP_PPL="${SKIP_PPL:-0}"

if [[ "${SAMPLE_MODE}" != "official" && "${SAMPLE_MODE}" != "first" ]]; then
  echo "The staged runner requires SAMPLE_MODE=official or first for final evaluation."
  exit 2
fi

mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL}"
  --mcf-cache-path "${MCF_PATH}"
  --sample-mode "${SAMPLE_MODE}"
  --seed "${SEED}"
  --forget-num "${FORGET_NUM}"
  --retain-num "${RETAIN_NUM}"
  --active-margin "${ACTIVE_MARGIN}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --margin-batch-size "${MARGIN_BATCH_SIZE}"
  --save-model
)

if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  COMMON_ARGS+=(--experiment-config-path "${EXPERIMENT_CONFIG}")
fi

active_count() {
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(int(d.get("active_prompt_instances_after", d.get("active_cases_after", 0))))' "$1"
}

select_best_checkpoint() {
  python - "$@" <<'PY'
import json
import pathlib
import sys

def score(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    active = int(
        data.get(
            "active_prompt_instances_after",
            data.get("active_cases_after", 0),
        )
    )
    active_parents = int(data.get("active_parent_records_after", active))
    minimum = data.get(
        "minimum_official_compatible_margin_after",
        data.get("minimum_margin_after"),
    )
    minimum = float("-inf") if minimum is None else float(minimum)
    delta = float(data.get("selected_lm_head_delta_norm", 0.0))
    return (active, active_parents, -minimum, delta)

best = min(sys.argv[1:], key=score)
print(pathlib.Path(best).parent / "checkpoint")
PY
}

echo "Stage 1: active-only target-true LM-head scale 1.50"
TRUE_DIR="${OUT_ROOT}/true_scale_150"
python scripts/gagd_active_case_repair.py \
  --model-path "${INPUT_MODEL}" \
  --output-dir "${TRUE_DIR}" \
  --repair-mode true_scale \
  --target-true-scale 1.50 \
  "${COMMON_ARGS[@]}"

SUMMARY_PATHS=("${TRUE_DIR}/repair_summary.json")
BEST_CHECKPOINT="${TRUE_DIR}/checkpoint"

if [[ "$(active_count "${TRUE_DIR}/repair_summary.json")" -gt 0 ]]; then
  echo "Stage 2: active-only target-new gamma sweep"
  for gamma in 1.10 1.25 1.50; do
    gamma_tag="${gamma/./}"
    GAMMA_DIR="${OUT_ROOT}/gamma_${gamma_tag}"
    python scripts/gagd_active_case_repair.py \
      --model-path "${TRUE_DIR}/checkpoint" \
      --output-dir "${GAMMA_DIR}" \
      --repair-mode extrapolate_delta \
      --target-new-gamma "${gamma}" \
      "${COMMON_ARGS[@]}"
    SUMMARY_PATHS+=("${GAMMA_DIR}/repair_summary.json")
    if [[ "$(active_count "${GAMMA_DIR}/repair_summary.json")" -eq 0 ]]; then
      break
    fi
  done
  BEST_CHECKPOINT="$(select_best_checkpoint "${SUMMARY_PATHS[@]}")"
fi

BEST_SUMMARY="$(dirname "${BEST_CHECKPOINT}")/repair_summary.json"
if [[ "$(active_count "${BEST_SUMMARY}")" -gt 0 ]]; then
  echo "Stage 3: sparse minimal optimization"
  MINIMAL_DIR="${OUT_ROOT}/minimal_optimize"
  MINIMAL_ARGS=(
    --model-path "${BEST_CHECKPOINT}"
    --reference-model-path "${INPUT_MODEL}"
    --output-dir "${MINIMAL_DIR}"
    --repair-mode minimal_optimize
    --repair-steps "${REPAIR_STEPS}"
    --repair-lr "${REPAIR_LR}"
    --repair-optimizer "${REPAIR_OPTIMIZER}"
    --hinge-weight "${HINGE_WEIGHT}"
    --delta-l2-lambda "${DELTA_L2_LAMBDA}"
    --retain-kl-mu "${RETAIN_KL_MU}"
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM}"
    --retain-calibration-seed "${RETAIN_CALIBRATION_SEED}"
    --repair-rank "${REPAIR_RANK}"
    --stop-when-all-satisfied
  )
  if [[ "${PROJECT_AWAY_RETAIN_HIDDEN}" == "1" ]]; then
    MINIMAL_ARGS+=(--project-away-retain-hidden)
  fi
  python scripts/gagd_active_case_repair.py \
    "${MINIMAL_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
  SUMMARY_PATHS+=("${MINIMAL_DIR}/repair_summary.json")
  BEST_CHECKPOINT="$(select_best_checkpoint "${SUMMARY_PATHS[@]}")"
fi

python - "${BEST_CHECKPOINT}" "${OUT_ROOT}/selected_candidate.json" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
summary = json.loads(
    (checkpoint.parent / "repair_summary.json").read_text(encoding="utf-8")
)
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "checkpoint": str(checkpoint),
            "repair_summary": str(checkpoint.parent / "repair_summary.json"),
            "active_prompt_instances_after": summary.get(
                "active_prompt_instances_after",
                summary.get("active_cases_after", 0),
            ),
            "active_parent_records_after": summary.get(
                "active_parent_records_after",
                summary.get("active_cases_after", 0),
            ),
            "minimum_official_compatible_margin_after": summary.get(
                "minimum_official_compatible_margin_after",
                summary.get("minimum_margin_after"),
            ),
            "selected_lm_head_delta_norm": summary.get(
                "selected_lm_head_delta_norm",
                0.0,
            ),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "Stage 4: official evaluation of selected candidate only"
EVAL_ARGS=(
  --model-dir "${BEST_CHECKPOINT}"
  --mcf-path "${MCF_PATH}"
  --wikidata-dir "${WIKIDATA_DIR}"
  --out "${OUT_ROOT}/official_eval_selected.json"
  --unlearn-num "${FORGET_NUM}"
  --retain-num "${RETAIN_NUM}"
  --seed "${SEED}"
  --sample-mode "${SAMPLE_MODE}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
)
if [[ "${SKIP_PPL}" == "1" ]]; then
  EVAL_ARGS+=(--skip-ppl)
fi
python scripts/mcf_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"

python - "${BEST_CHECKPOINT}" "${OUT_ROOT}/official_eval_selected.json" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
summary = json.loads(
    (checkpoint.parent / "repair_summary.json").read_text(encoding="utf-8")
)
official = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
forget = official.get("forget", official)
eff = float(forget.get("Eff", official.get("Eff", 0.0)))
gen = float(forget.get("Gen", official.get("Gen", 0.0)))
active_before = int(
    summary.get(
        "active_prompt_instances_before",
        summary.get("active_cases_before", 0),
    )
)
if active_before == 0 and (eff > 0 or gen > 0):
    raise SystemExit(
        "Refusing selected zero-row/no-op candidate: local official-compatible "
        f"active prompts before repair=0, but final Eff={eff}, Gen={gen}."
    )
PY

echo "Selected checkpoint: ${BEST_CHECKPOINT}"
echo "Official result: ${OUT_ROOT}/official_eval_selected.json"
