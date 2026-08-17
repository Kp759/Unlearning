#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mquake_sure_v7_rank0_rank256_locked.sh MODEL_PATH [MQUAKE_PATH]

SURE-MQuAKE V7 locked protocol:
  Split:    ZeroUnlearn-style 50 forget instances / 1000 retain eval instances.
  Stage 1:  direct rewrite tokens only; sparse sensitive LM-head rows; GA +
            same-prompt non-target KL; transformer/input/non-sensitive rows Base.
  Stage 2:  residual active sensitive rows only; Rank 256 and Rank 0 variants;
            basis for Rank 256 comes from ALL visible direct forget token states.
  Audit:    exact BF16 all-visible direct-token margin audit before evaluation.
  Eval:     only after checkpoint freeze: Eff + retain diagnostic + Wikidata PPL.
            AtomicGen is optional and post-selection only.

Useful smoke test:
  MQUAKE_SEEDS="1" REPAIR_RANKS="256 0" SKIP_PPL=1 \
    bash scripts/run_mquake_sure_v7_rank0_rank256_locked.sh MODEL_PATH
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
ORIGINAL_MQUAKE="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v7_sparsegagd_active_locked_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
REPAIR_RANKS_TEXT="${REPAIR_RANKS:-256 0}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
EVAL_RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_PPL="${SKIP_PPL:-0}"
RUN_ATOMIC_GEN_EXTENSION="${RUN_ATOMIC_GEN_EXTENSION:-0}"
CLEANUP_AFTER_EVAL="${CLEANUP_AFTER_EVAL:-0}"

# Stage 1: minimal sparse GA/GD.  Stop as soon as every official sensitive token
# loses by a small direct-token margin, limiting global row displacement.
STAGE1_STEPS="${STAGE1_STEPS:-600}"
STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE1_GA_WEIGHT="${STAGE1_GA_WEIGHT:-2.0}"
STAGE1_GD_WEIGHT="${STAGE1_GD_WEIGHT:-1.0}"
STAGE1_L2="${STAGE1_L2:-0.000001}"
STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"
STAGE1_TARGET_MARGIN="${STAGE1_TARGET_MARGIN:-0.05}"

# Stage 2: stronger official-token safety margin.  Optimization reaches an
# additional BF16 buffer, then exact materialization only has to retain the
# target margin itself.
STAGE2_TARGET_MARGIN="${STAGE2_TARGET_MARGIN:-0.25}"
STAGE2_BF16_BUFFER="${STAGE2_BF16_BUFFER:-0.05}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE2_LR="${STAGE2_LR:-0.005}"
STAGE2_OPTIMIZER="${STAGE2_OPTIMIZER:-adamw}"
STAGE2_HINGE_WEIGHT="${STAGE2_HINGE_WEIGHT:-100.0}"
STAGE2_HARDEST_WEIGHT="${STAGE2_HARDEST_WEIGHT:-25.0}"
STAGE2_L2="${STAGE2_L2:-0.000001}"
STAGE2_GRAD_CLIP="${STAGE2_GRAD_CLIP:-1.0}"
STAGE2_BISECTION_STEPS="${STAGE2_BISECTION_STEPS:-30}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
read -r -a REPAIR_RANKS <<< "${REPAIR_RANKS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 || "${#REPAIR_RANKS[@]}" -eq 0 ]]; then
  echo "MQUAKE_SEEDS and REPAIR_RANKS must be non-empty" >&2
  exit 2
fi

if [[ "${FORGET_NUM}" != "50" ]]; then
  echo "WARNING: publication protocol uses 50 forget instances; got ${FORGET_NUM}." >&2
fi
if [[ "${SKIP_PPL}" != "1" ]]; then
  test -d "${WIKIDATA_DIR}"
fi

safe_cleanup_checkpoint() {
  local target="$1"
  [[ "${CLEANUP_AFTER_EVAL}" == "1" ]] || return 0
  [[ -d "${target}" ]] || return 0
  [[ "$(basename "${target}")" == "checkpoint" ]] || {
    echo "REFUSE cleanup of non-checkpoint path: ${target}" >&2
    return 1
  }
  local absolute
  absolute="$(realpath "${target}")"
  case "${absolute}" in
    */outputs/mquake_sure_v7_*/*/checkpoint) ;;
    *)
      echo "REFUSE cleanup outside MQuAKE V7 outputs: ${absolute}" >&2
      return 1
      ;;
  esac
  local protected
  protected="$(realpath "${MODEL_PATH}")"
  if [[ "${absolute}" == "${protected}" || "${absolute}" == "${protected}"/* ]]; then
    echo "REFUSE cleanup overlapping protected Base: ${absolute}" >&2
    return 1
  fi
  echo "Deleting evaluated temporary checkpoint weights: ${absolute}"
  rm -rf -- "${absolute}"
}

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${ROOT}/protocol"
  REPAIR_VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  STAGE1_DIR="${ROOT}/stage1_sparse_sensitive_gagd"
  STAGE1_CKPT="${STAGE1_DIR}/checkpoint"
  SEED_MANIFEST="${ROOT}/run_manifest_v7.json"

  mkdir -p "${ROOT}" "${PROTOCOL_DIR}"

  echo
  echo "===== SEED ${SEED}: BUILD LOCKED MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${ORIGINAL_MQUAKE}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${EVAL_RETAIN_NUM}"
  test -f "${REPAIR_VISIBLE}"
  test -f "${SPLIT_MANIFEST}"

  echo "===== SEED ${SEED}: V7 STAGE1 SPARSE SENSITIVE-ROW GA/GD ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1_CKPT}/config.json" ]]; then
    rm -rf "${STAGE1_CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_sparse_lm_gagd_v7.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${REPAIR_VISIBLE}" \
      --split-manifest "${SPLIT_MANIFEST}" \
      --output-dir "${STAGE1_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1_STEPS}" \
      --lr "${STAGE1_LR}" \
      --ga-weight "${STAGE1_GA_WEIGHT}" \
      --gd-weight "${STAGE1_GD_WEIGHT}" \
      --delta-l2-lambda "${STAGE1_L2}" \
      --grad-clip "${STAGE1_GRAD_CLIP}" \
      --target-logit-margin "${STAGE1_TARGET_MARGIN}" \
      --batch-size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1_CKPT}"
  fi
  test -f "${STAGE1_DIR}/repair_summary.json"
  test -d "${STAGE1_CKPT}"

  PASSING_RANKS=()
  EVAL_PATHS=()
  for RANK in "${REPAIR_RANKS[@]}"; do
    STAGE2_DIR="${ROOT}/stage2_active_r${RANK}"
    STAGE2_CKPT="${STAGE2_DIR}/checkpoint"
    FINAL_EVAL="${ROOT}/official_eval_v7_r${RANK}.json"
    FINAL_SPLIT_MANIFEST="${ROOT}/final_eval_split_manifest_r${RANK}.json"
    ATOMIC_EVAL="${ROOT}/atomic_gen_postselection_v7_r${RANK}.json"

    echo
    echo "===== SEED ${SEED}: V7 STAGE2 RANK ${RANK} ====="
    STAGE2_OK=1
    if [[ "${SKIP_EXISTING}" == "1" && -f "${STAGE2_DIR}/repair_summary.json" && -d "${STAGE2_CKPT}" ]]; then
      echo "Reusing ${STAGE2_CKPT}"
    else
      rm -rf "${STAGE2_CKPT}"
      if ! "${PYTHON_BIN}" scripts/mquake_sure_active_hidden_repair_v7.py \
        --model-path "${STAGE1_CKPT}" \
        --reference-model-path "${MODEL_PATH}" \
        --repair-visible-path "${REPAIR_VISIBLE}" \
        --split-manifest "${SPLIT_MANIFEST}" \
        --output-dir "${STAGE2_DIR}" \
        --seed "${SEED}" \
        --forget-num "${FORGET_NUM}" \
        --target-logit-margin "${STAGE2_TARGET_MARGIN}" \
        --bf16-buffer-margin "${STAGE2_BF16_BUFFER}" \
        --repair-rank "${RANK}" \
        --repair-steps "${STAGE2_STEPS}" \
        --repair-lr "${STAGE2_LR}" \
        --repair-optimizer "${STAGE2_OPTIMIZER}" \
        --forget-hinge-weight "${STAGE2_HINGE_WEIGHT}" \
        --hardest-forget-hinge-weight "${STAGE2_HARDEST_WEIGHT}" \
        --delta-l2-lambda "${STAGE2_L2}" \
        --grad-clip "${STAGE2_GRAD_CLIP}" \
        --boundary-bisection-steps "${STAGE2_BISECTION_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}"; then
        STAGE2_OK=0
      fi
    fi

    if [[ "${STAGE2_OK}" != "1" || ! -d "${STAGE2_CKPT}" ]]; then
      echo "Seed ${SEED} rank ${RANK}: Stage2 failed; preserving diagnostics and continuing." >&2
      continue
    fi
    PASSING_RANKS+=("${RANK}")

    echo "===== SEED ${SEED}: LOCKED FINAL EVAL RANK ${RANK} ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${FINAL_EVAL}" ]]; then
      EVAL_ARGS=(
        --model-dir "${STAGE2_CKPT}"
        --mquake-path "${ORIGINAL_MQUAKE}"
        --wikidata-dir "${WIKIDATA_DIR}"
        --out "${FINAL_EVAL}"
        --split-manifest "${FINAL_SPLIT_MANIFEST}"
        --method "SURE-MQuAKE V7 rank ${RANK}"
        --unlearn-num "${FORGET_NUM}"
        --retain-num "${EVAL_RETAIN_NUM}"
        --seed "${SEED}"
        --batch-size "${EVAL_BATCH_SIZE}"
        --dtype "${DTYPE}"
        --device-map "${DEVICE_MAP}"
        --skip-atomic-gen
      )
      if [[ "${SKIP_PPL}" == "1" ]]; then
        EVAL_ARGS+=(--skip-ppl)
      fi
      "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"
    else
      echo "Reusing ${FINAL_EVAL}"
    fi
    EVAL_PATHS+=("${FINAL_EVAL}")

    if [[ "${RUN_ATOMIC_GEN_EXTENSION}" == "1" ]]; then
      echo "===== SEED ${SEED}: POST-SELECTION ATOMICGEN RANK ${RANK} ====="
      "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py \
        --model-dir "${STAGE2_CKPT}" \
        --mquake-path "${ORIGINAL_MQUAKE}" \
        --wikidata-dir "${WIKIDATA_DIR}" \
        --out "${ATOMIC_EVAL}" \
        --method "SURE-MQuAKE V7 rank ${RANK} post-selection AtomicGen" \
        --unlearn-num "${FORGET_NUM}" \
        --retain-num "${EVAL_RETAIN_NUM}" \
        --seed "${SEED}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}" \
        --skip-ppl
    fi

    safe_cleanup_checkpoint "${STAGE2_CKPT}"
  done

  # Pre-specified method preference uses training geometry only: Rank 256 when
  # it passes the locked forget-only Stage2 audit, otherwise Rank 0.  Final
  # retain/PPL results never choose the checkpoint.
  PREFERRED_RANK=""
  for candidate in 256 0; do
    for passed in "${PASSING_RANKS[@]:-}"; do
      if [[ "${passed}" == "${candidate}" ]]; then
        PREFERRED_RANK="${candidate}"
        break 2
      fi
    done
  done

  "${PYTHON_BIN}" - "${SEED_MANIFEST}" "${SPLIT_MANIFEST}" "${STAGE1_DIR}" \
    "${PREFERRED_RANK}" "${PASSING_RANKS[*]:-}" "${EVAL_PATHS[*]:-}" <<'PY'
import json, pathlib, sys
out=pathlib.Path(sys.argv[1])
split=json.loads(pathlib.Path(sys.argv[2]).read_text())
payload={
    "schema_version": 1,
    "protocol": "mquake_zerounlearn_forget_only_locked_probes",
    "method_family": "SURE-MQuAKE-v7 sparse sensitive-row GA/GD + active hidden repair",
    "seed": split["seed"],
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "stage1_dir": str(pathlib.Path(sys.argv[3]).resolve()),
    "passing_stage2_ranks": [int(x) for x in sys.argv[5].split() if x],
    "preferred_rank_pre_evaluation": (int(sys.argv[4]) if sys.argv[4] else None),
    "preferred_rank_policy": "rank256 if its forget-only BF16 Stage2 audit passes; otherwise rank0; retain/PPL never select",
    "official_eval_paths": [str(pathlib.Path(x).resolve()) for x in sys.argv[6].split() if x],
    "data_firewall": {
        "stage1_stage2_visible": "direct requested_rewrite token positions from sampled forget instances only",
        "retain_instances_during_training_or_repair": 0,
        "atomic_questions_during_training_or_repair": 0,
        "multihop_questions_during_training_or_repair": 0,
        "counterfactual_target_new_during_training_or_repair": 0,
        "retain_and_ppl_first_used": "after checkpoint freeze in final official evaluation",
    },
    "sampling": split["sampling"],
}
out.write_text(json.dumps(payload, indent=2)+"\n")
PY

  if [[ "${#PASSING_RANKS[@]}" -eq 0 ]]; then
    echo "Seed ${SEED}: no Stage2 rank passed the forget-only BF16 audit." >&2
  else
    echo "Seed ${SEED}: passing ranks=${PASSING_RANKS[*]}; pre-eval preferred rank=${PREFERRED_RANK:-none}."
  fi

done

echo
echo "SURE-MQuAKE V7 locked run complete."
echo "Results: ${OUTPUT_ROOT}/seed*/official_eval_v7_r*.json"
echo "Rank preference is determined before retain/PPL evaluation: Rank256 PASS -> Rank0 fallback."
