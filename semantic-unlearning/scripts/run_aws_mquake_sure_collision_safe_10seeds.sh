#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?MODEL required}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_mquake_sure_collision_safe_seeds1_10_3b}"
SEEDS="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

mkdir -p "${OUTPUT_ROOT}"

python - "${OUTPUT_ROOT}" "${MODEL}" "${MQUAKE}" "${WIKIDATA_DIR}" "${SEEDS}" "$(git rev-parse HEAD)" <<'PY'
import json,sys
from pathlib import Path
root,model,mquake,wikidata,seeds,git_sha=sys.argv[1:]
payload={
  "schema_version":1,
  "status":"frozen before seeds1-10 execution",
  "git_sha":git_sha,
  "model":model,
  "mquake":mquake,
  "wikidata_dir":wikidata,
  "seeds":[int(x) for x in seeds.split()],
  "split":{
    "dataset_instances":3000,
    "retain_pool":"[0,1500)",
    "forget_pool":"[1500,3000)",
    "forget_instances_per_seed":50,
    "retain_eval_instances_per_seed":1000,
    "sampling":"one random.Random(seed); forget sampled first, retain second; flatten requested_rewrite after instance sampling"
  },
  "stage1":{
    "script":"scripts/mquake_no_neutral_stage1_collision_safe.py",
    "steps":600,"batch_size":1,"cache_batch_size":8,
    "emb_lr":0.0001,"sensitive_row_lr":0.0005,
    "ga_weight":2.0,"gd_weight":1.0,"active_margin":0.05,
    "context_weight":1.0,"context_batch_size":64,
    "row_l2":0.0001,"grad_clip":1.0,
    "transformer":"frozen","lm_head":"untied before training; base head frozen",
    "trainable_lm_component":"sparse FP32 sensitive-row delta",
    "input_embeddings":"temporarily trainable; fully restored to base after Stage1",
    "selection_data":"same direct forget facts only"
  },
  "stage2":{
    "script":"scripts/mquake_forget_only_no_neutral.py",
    "stage1_steps_inside_driver":0,
    "steps":800,"lr":0.005,"margin":0.05,"l2":0.000001,
    "batch_size":8,
    "candidate_scales":"1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0",
    "selection_data":"same direct forget facts only",
    "mechanism":"sparse LM-head repair on residual active sensitive token decisions"
  },
  "training_visibility":{
    "retain":0,"PPL":False,"AtomicGen":False,"multihop":False,
    "target_new":False,"Unknown":False,"IDK":False
  }
}
Path(root,"frozen_run_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
PY

for seed in ${SEEDS}; do
  echo
  echo "=================================================================="
  echo "MQuAKE COLLISION-SAFE FROZEN RUN: SEED ${seed}"
  echo "=================================================================="
  mkdir -p "${OUTPUT_ROOT}/seed${seed}"
  MQUAKE_SEED="${seed}" \
  MQUAKE_FORGET_NUM=50 \
  MQUAKE_RETAIN_EVAL_NUM=1000 \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  WIKIDATA_DIR="${WIKIDATA_DIR}" \
  DTYPE="${DTYPE}" \
  DEVICE_MAP="${DEVICE_MAP}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  bash scripts/run_mquake_sure_collision_safe_seed1.sh "${MODEL}" "${MQUAKE}" \
    2>&1 | tee "${OUTPUT_ROOT}/seed${seed}/run.log"

  for required in \
    "${OUTPUT_ROOT}/seed${seed}/stage1_collision_safe/config_used.json" \
    "${OUTPUT_ROOT}/seed${seed}/stage2_sensitive_row_repair/summary.json" \
    "${OUTPUT_ROOT}/seed${seed}/official_eval_stage1_posthoc.json" \
    "${OUTPUT_ROOT}/seed${seed}/official_eval_locked.json" \
    "${OUTPUT_ROOT}/seed${seed}/multihop_eval_final.json"; do
    test -s "${required}" || { echo "ERROR: missing ${required}" >&2; exit 2; }
  done

  if [[ "${KEEP_CHECKPOINTS}" != "1" ]]; then
    rm -rf \
      "${OUTPUT_ROOT}/seed${seed}/stage1_collision_safe/checkpoint" \
      "${OUTPUT_ROOT}/seed${seed}/stage2_sensitive_row_repair/checkpoint"
    echo "seed ${seed}: metrics preserved; checkpoints removed (KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS})"
  fi

done

python scripts/aggregate_mquake_collision_safe_10seeds.py \
  --root "${OUTPUT_ROOT}" \
  --seeds "$(echo "${SEEDS}" | tr ' ' ',')" \
  --out "${OUTPUT_ROOT}/aggregate_seeds1_10.json"

echo
echo "===== COMPLETE ====="
echo "root: ${OUTPUT_ROOT}"
echo "aggregate: ${OUTPUT_ROOT}/aggregate_seeds1_10.json"
echo "manifest: ${OUTPUT_ROOT}/frozen_run_manifest.json"
