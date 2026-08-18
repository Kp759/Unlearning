#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mquake_sure_canonical.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_canonical_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_INSTANCES="${MQUAKE_FORGET_NUM:-50}"
RETAIN_INSTANCES="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# Exact shared canonical Stage 1 defaults used by MCF/ZsRE.
STEPS="${SURE_STAGE1_STEPS:-600}"
BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
EMB_LM_LR="${SURE_STAGE1_LR:-0.0001}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"

# Exact shared canonical Stage 2 defaults used by MCF/ZsRE.
# MQuAKE uses the same target_true/top-1 direct constraint semantics as ZsRE.
CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_REPAIR_LR:-0.005}"
REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${SURE_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"
CONSTRAINT_MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"
EVAL_BATCH_SIZE="${MQUAKE_EVAL_BATCH_SIZE:-8}"
RUN_ATOMIC_GEN="${MQUAKE_RUN_ATOMIC_GEN:-0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MQUAKE}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_gagd"
  STAGE2="${ROOT}/stage2_sparse_row"
  FINAL="${ROOT}/official_eval_locked.json"
  mkdir -p "${ROOT}"

  echo "===== MQUAKE SEED ${SEED}: CANONICAL LOCKED SPLIT ====="
  rm -rf "${PROTOCOL}"
  python scripts/build_mquake_sure_canonical_split.py \
    --mquake-path "${MQUAKE}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_INSTANCES}" --retain-num "${RETAIN_INSTANCES}"

  ATOMIC_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"
  echo "MQuAKE canonical adapter: ${FORGET_INSTANCES} source instances -> ${ATOMIC_COUNT} direct atomic facts"
  echo "Shared engine semantics: ZsRE-style target_true sensitive token / top-1 margin"

  echo "===== MQUAKE SEED ${SEED}: COMMON STAGE 1 GA/GD ====="
  rm -rf "${STAGE1}"
  # Important: dataset=zsre selects the canonical target_true-sensitive adapter.
  # No ZsRE data are used; VISIBLE is the locked MQuAKE direct-only artifact.
  python scripts/sure_stage1_gagd.py \
    --dataset zsre --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE1}" --seed "${SEED}" --forget-num "${ATOMIC_COUNT}" \
    --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" --emb-lm-lr "${EMB_LM_LR}" \
    --ga-weight "${GA_WEIGHT}" --gd-weight "${GD_WEIGHT}" \
    --optimizer adamw --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== MQUAKE SEED ${SEED}: COMMON STAGE 2 SPARSE SENSITIVE ROWS ====="
  echo "AtomicGen/multihop/target_new/retain/PPL have not been opened for selection."
  rm -rf "${STAGE2}"
  python scripts/sure_stage2_sparse_repair.py \
    --dataset zsre --model-path "${STAGE1}/checkpoint" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${ATOMIC_COUNT}" \
    --candidate-ranks "${CANDIDATE_RANKS}" --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" --constraint-margin "${CONSTRAINT_MARGIN}" \
    --repair-l2 "${REPAIR_L2}" --batch-size "${REPAIR_BATCH_SIZE}" \
    --check-every "${REPAIR_CHECK_EVERY}" --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  # Preserve transparent benchmark/adapter metadata next to shared-engine outputs.
  python - "${ROOT}" "${SEED}" "${FORGET_INSTANCES}" "${RETAIN_INSTANCES}" "${ATOMIC_COUNT}" "${CONSTRAINT_MARGIN}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
payload={
    "schema_version": 1,
    "benchmark_dataset": "mquake",
    "canonical_shared_engine": "MCF/ZsRE SURE-LM",
    "shared_engine_adapter": "zsre",
    "reason": "MQuAKE and ZsRE both forget original target_true tokens by requiring the sensitive token to lose top-1 by a margin",
    "seed": int(sys.argv[2]),
    "forget_source_instances": int(sys.argv[3]),
    "retain_eval_instances": int(sys.argv[4]),
    "training_visible_atomic_facts": int(sys.argv[5]),
    "constraint_margin": float(sys.argv[6]),
    "stage1_entrypoint": "scripts/sure_stage1_gagd.py",
    "stage2_entrypoint": "scripts/sure_stage2_sparse_repair.py",
    "candidate_ranks": [2,8,0],
    "selection_uses_heldout": False,
}
(root/"canonical_adapter.json").write_text(json.dumps(payload,indent=2)+"\n")
PY

  echo "===== MQUAKE SEED ${SEED}: FINAL OFFICIAL EVAL ====="
  EVAL_ARGS=(
    --model-dir "${STAGE2}/checkpoint"
    --mquake-path "${MQUAKE}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL}"
    --split-manifest "${MANIFEST}"
    --method "SURE-LM canonical shared architecture (MQuAKE target_true adapter)"
    --unlearn-num "${FORGET_INSTANCES}"
    --retain-num "${RETAIN_INSTANCES}"
    --seed "${SEED}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  if [[ "${RUN_ATOMIC_GEN}" != "1" ]]; then EVAL_ARGS+=(--skip-atomic-gen); fi
  python scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL}" --model-dir "${STAGE2}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

done

echo "Canonical MQuAKE complete: ${OUTPUT_ROOT}"
