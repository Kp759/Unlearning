#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_sure_canonical_target_true.sh MODEL [MCF_JSON]

Purpose:
  Canonical SURE-LM MCF experiment with ORIGINAL target_true as sensitive.
  The locked training adapter maps original target_true into canonical
  target_new (the sensitive slot) and original target_new into canonical
  target_true (the counterfactual-reference slot). Final evaluation always
  uses the ORIGINAL UNSWAPPED MCF.

Default:
  seed 1 only; set MCF_SEEDS="1 2 ..." to run more seeds.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="$1"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_canonical_target_true_sensitive}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
HASH_CHECKPOINT="${SURE_HASH_CHECKPOINT:-1}"

# Exact canonical SURE Stage-1 settings.
STEPS="${SURE_STAGE1_STEPS:-600}"
BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
EMB_LM_LR="${SURE_STAGE1_LR:-0.0001}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"

# Exact canonical SURE Stage-2 settings.
CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_REPAIR_LR:-0.005}"
REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${SURE_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"
CONSTRAINT_MARGIN="${MCF_SURE_CONSTRAINT_MARGIN:-0.25}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "MCF_SEEDS resolved to an empty list" >&2
  exit 2
fi

test -f "${MCF}"
test -d "${WIKIDATA_DIR}"
test -d "${MODEL}"
mkdir -p "${OUTPUT_ROOT}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_gagd"
  STAGE2="${ROOT}/stage2_sparse_row"
  BASE_EVAL="${ROOT}/base_original_mcf_eval.json"
  POST_EVAL="${ROOT}/post_original_mcf_eval.json"
  PAPER_EVAL="${ROOT}/target_true_sensitive_eval.json"
  RUN_MANIFEST="${ROOT}/run_manifest.json"

  # Paper track never silently reuses old checkpoints/configs. A rerun is clean.
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo
  echo "===== MCF SEED ${SEED}: TARGET-TRUE-SENSITIVE CANONICAL LOCKED SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mcf_sure_target_true_canonical_split.py \
    --mcf-path "${MCF}" \
    --output-dir "${PROTOCOL}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  test -f "${VISIBLE}"
  test -f "${MANIFEST}"

  echo "===== MCF SEED ${SEED}: CANONICAL STAGE 1 GA/GD ====="
  echo "Sensitive slot = canonical target_new = ORIGINAL target_true."
  "${PYTHON_BIN}" scripts/sure_stage1_gagd.py \
    --dataset mcf \
    --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE1}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --emb-lm-lr "${EMB_LM_LR}" \
    --ga-weight "${GA_WEIGHT}" \
    --gd-weight "${GD_WEIGHT}" \
    --optimizer adamw \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  test -d "${STAGE1}/checkpoint"
  test -f "${STAGE1}/vocabulary_restoration.json"

  echo "===== MCF SEED ${SEED}: CANONICAL STAGE 2 SENSITIVE-ROW-ONLY REPAIR ====="
  echo "Only canonical target_new rows are editable; these are ORIGINAL target_true rows."
  echo "All counterfactual-reference rows remain frozen during Stage 2."
  "${PYTHON_BIN}" scripts/sure_stage2_sparse_repair.py \
    --dataset mcf \
    --model-path "${STAGE1}/checkpoint" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE2}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --candidate-ranks "${CANDIDATE_RANKS}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --constraint-margin "${CONSTRAINT_MARGIN}" \
    --repair-l2 "${REPAIR_L2}" \
    --batch-size "${REPAIR_BATCH_SIZE}" \
    --check-every "${REPAIR_CHECK_EVERY}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  test -d "${STAGE2}/checkpoint"
  test -f "${STAGE2}/repair_summary.json"

  echo "===== MCF SEED ${SEED}: BASE EVAL ON ORIGINAL UNSWAPPED MCF ====="
  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" \
    --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" \
    --sample-mode official \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== MCF SEED ${SEED}: POST EVAL ON ORIGINAL UNSWAPPED MCF ====="
  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2}/checkpoint" \
    --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${POST_EVAL}" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" \
    --sample-mode official \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  # Record the exact fixed-PPL fixture provenance for both base and post model.
  "${PYTHON_BIN}" scripts/annotate_ppl_provenance.py \
    --eval-json "${BASE_EVAL}" \
    --model-dir "${MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}"
  "${PYTHON_BIN}" scripts/annotate_ppl_provenance.py \
    --eval-json "${POST_EVAL}" \
    --model-dir "${STAGE2}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

  echo "===== MCF SEED ${SEED}: PAPER-FACING TARGET-TRUE-SENSITIVE METRICS ====="
  "${PYTHON_BIN}" scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${POST_EVAL}" \
    --split-manifest "${MANIFEST}" \
    --out "${PAPER_EVAL}"

  echo "===== MCF SEED ${SEED}: IMMUTABLE RUN MANIFEST ====="
  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${MODEL}" "${MCF}" "${VISIBLE}" "${MANIFEST}" \
    "${STAGE1}/config_used.json" "${STAGE2}/repair_summary.json" \
    "${STAGE2}/checkpoint" "${BASE_EVAL}" "${POST_EVAL}" "${PAPER_EVAL}" \
    "${HASH_CHECKPOINT}" "${SEED}" "${FORGET_NUM}" "${RETAIN_NUM}" \
    "${STEPS}" "${EMB_LM_LR}" "${GA_WEIGHT}" "${GD_WEIGHT}" \
    "${CANDIDATE_RANKS}" "${REPAIR_STEPS}" "${REPAIR_LR}" \
    "${REPAIR_L2}" "${CONSTRAINT_MARGIN}" "${CANDIDATE_SCALES}" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

(
    manifest_out,
    model_path,
    mcf_path,
    visible_path,
    split_manifest_path,
    stage1_config_path,
    stage2_summary_path,
    checkpoint_dir,
    base_eval_path,
    post_eval_path,
    paper_eval_path,
    hash_checkpoint_text,
    seed,
    forget_num,
    retain_num,
    steps,
    lr,
    ga_weight,
    gd_weight,
    candidate_ranks,
    repair_steps,
    repair_lr,
    repair_l2,
    margin,
    candidate_scales,
) = sys.argv[1:]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def optional_sha(path: pathlib.Path):
    return sha256_file(path) if path.exists() and path.is_file() else None


def git_value(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

split = json.loads(pathlib.Path(split_manifest_path).read_text(encoding="utf-8"))
effective_config = {
    "seed": int(seed),
    "forget_num": int(forget_num),
    "retain_eval_num": int(retain_num),
    "target_semantics": "original_target_true_sensitive",
    "stage1_steps": int(steps),
    "stage1_lr": float(lr),
    "ga_weight": float(ga_weight),
    "gd_weight": float(gd_weight),
    "candidate_ranks": candidate_ranks,
    "repair_steps": int(repair_steps),
    "repair_lr": float(repair_lr),
    "repair_l2": float(repair_l2),
    "constraint_margin": float(margin),
    "candidate_scales": candidate_scales,
}
config_bytes = json.dumps(effective_config, sort_keys=True, separators=(",", ":")).encode()

checkpoint_files = []
ckpt = pathlib.Path(checkpoint_dir)
for p in sorted(x for x in ckpt.rglob("*") if x.is_file()):
    item = {
        "path": str(p.relative_to(ckpt)),
        "bytes": p.stat().st_size,
    }
    if hash_checkpoint_text == "1":
        item["sha256"] = sha256_file(p)
    checkpoint_files.append(item)

model_dir = pathlib.Path(model_path)
payload = {
    "schema_version": 2,
    "protocol": "mcf_sure_canonical_target_true_sensitive_v1",
    "metric_schema": "mcf_target_true_sensitive_v2",
    "seed": int(seed),
    "target_semantics": {
        "sensitive": "original target_true",
        "counterfactual_reference": "original target_new",
        "canonical_training_sensitive_slot": "target_new",
        "canonical_training_reference_slot": "target_true",
    },
    "source_mcf": str(pathlib.Path(mcf_path).resolve()),
    "source_mcf_sha256": split.get("source_sha256"),
    "training_visible": str(pathlib.Path(visible_path).resolve()),
    "training_visible_sha256": split.get("training_visible_sha256"),
    "ordered_forget_case_ids": split.get("sampling", {}).get("forget_case_ids", []),
    "ordered_retain_case_ids": split.get("sampling", {}).get("retain_case_ids", []),
    "model_path": str(model_dir.resolve()),
    "model_config_sha256": optional_sha(model_dir / "config.json"),
    "tokenizer_config_sha256": optional_sha(model_dir / "tokenizer_config.json"),
    "effective_config": effective_config,
    "effective_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    "stage1_config_sha256": sha256_file(pathlib.Path(stage1_config_path)),
    "stage2_summary_sha256": sha256_file(pathlib.Path(stage2_summary_path)),
    "base_eval_sha256": sha256_file(pathlib.Path(base_eval_path)),
    "post_eval_sha256": sha256_file(pathlib.Path(post_eval_path)),
    "paper_eval_sha256": sha256_file(pathlib.Path(paper_eval_path)),
    "checkpoint_file_manifest": checkpoint_files,
    "checkpoint_content_hashed": hash_checkpoint_text == "1",
    "code_commit": git_value(["git", "rev-parse", "HEAD"]),
    "code_status_porcelain": git_value(["git", "status", "--porcelain"]),
}
path = pathlib.Path(manifest_out)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("run manifest:", path.resolve())
print("checkpoint files hashed:", hash_checkpoint_text == "1")
PY

  echo "Seed ${SEED} complete: ${PAPER_EVAL}"
done

"${PYTHON_BIN}" scripts/aggregate_mcf_target_true_sensitive.py \
  --root "${OUTPUT_ROOT}" \
  --seeds "${SEEDS[@]}"

echo
echo "Canonical target-true-sensitive MCF complete: ${OUTPUT_ROOT}"
echo "Final paper metric files: ${OUTPUT_ROOT}/seed*/target_true_sensitive_eval.json"
