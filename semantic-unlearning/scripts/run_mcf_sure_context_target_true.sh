#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_sure_context_target_true.sh MODEL [MCF_JSON]

Context-conditioned canonical SURE-LM MCF track:
  original target_true = sensitive
  original target_new  = non-sensitive/counterfactual reference

Stage 1:
  sensitive GA + explicit reference-answer GD + non-sensitive-distribution KL
  only row-specific forget-context-projected sensitive LM-head deltas train
  input embeddings and transformer stay exactly Base

Stage 2:
  sensitive rows only; row-specific context ranks 2 -> 8 -> full-context
  explicit reference-answer GD; direct-only scale selection

Default: seed 1 only.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_context_target_true_sensitive}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
HASH_CHECKPOINT="${SURE_HASH_CHECKPOINT:-1}"

# Stage 1: context-conditioned GA/GD.
STEPS="${SURE_STAGE1_STEPS:-600}"
BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
REFERENCE_BATCH_SIZE="${SURE_REFERENCE_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
LR="${SURE_STAGE1_LR:-0.0001}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
REFERENCE_GD_WEIGHT="${SURE_REFERENCE_GD_WEIGHT:-1.0}"
DISTRIBUTION_KL_WEIGHT="${SURE_DISTRIBUTION_KL_WEIGHT:-1.0}"
STAGE1_DELTA_L2="${SURE_STAGE1_DELTA_L2:-0.0}"
CONTEXT_RANK="${SURE_STAGE1_CONTEXT_RANK:-2}"
STAGE1_MARGIN="${SURE_STAGE1_CONSTRAINT_MARGIN:-0.0}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

# Stage 2: direct-only context repair.
CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_REPAIR_LR:-0.005}"
REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${SURE_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"
CONSTRAINT_MARGIN="${MCF_SURE_CONSTRAINT_MARGIN:-0.25}"
STAGE2_REFERENCE_GD_WEIGHT="${SURE_STAGE2_REFERENCE_GD_WEIGHT:-1.0}"

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
  STAGE1="${ROOT}/stage1_context_gagd"
  STAGE2="${ROOT}/stage2_context_sparse_row"
  BASE_EVAL="${ROOT}/base_original_mcf_eval.json"
  POST_EVAL="${ROOT}/post_original_mcf_eval.json"
  PAPER_EVAL="${ROOT}/target_true_sensitive_eval.json"
  RUN_MANIFEST="${ROOT}/run_manifest.json"

  # Never silently reuse stale state for the paper track.
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo
  echo "===== MCF SEED ${SEED}: LOCKED TARGET-TRUE-SENSITIVE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mcf_sure_target_true_canonical_split.py \
    --mcf-path "${MCF}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

  echo "===== MCF SEED ${SEED}: CONTEXT-CONDITIONED STAGE 1 GA/GD ====="
  echo "GA: ORIGINAL target_true sensitive answer."
  echo "GD: ORIGINAL target_new non-sensitive/reference answer + distribution KL."
  echo "Editable parameters: context-projected sensitive LM-head rows only."
  "${PYTHON_BIN}" scripts/sure_stage1_context_gagd.py \
    --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE1}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
    --reference-batch-size "${REFERENCE_BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" --emb-lm-lr "${LR}" \
    --ga-weight "${GA_WEIGHT}" --reference-gd-weight "${REFERENCE_GD_WEIGHT}" \
    --distribution-kl-weight "${DISTRIBUTION_KL_WEIGHT}" \
    --delta-l2 "${STAGE1_DELTA_L2}" --context-rank "${CONTEXT_RANK}" \
    --stage1-constraint-margin "${STAGE1_MARGIN}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  test -d "${STAGE1}/checkpoint"
  test -f "${STAGE1}/config_used.json"

  echo "===== MCF SEED ${SEED}: CONTEXT-CONDITIONED STAGE 2 ====="
  echo "Only sensitive rows and direct forget contexts remain visible."
  "${PYTHON_BIN}" scripts/sure_stage2_context_sparse_repair.py \
    --model-path "${STAGE1}/checkpoint" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --candidate-ranks "${CANDIDATE_RANKS}" --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" --constraint-margin "${CONSTRAINT_MARGIN}" \
    --reference-gd-weight "${STAGE2_REFERENCE_GD_WEIGHT}" \
    --repair-l2 "${REPAIR_L2}" --batch-size "${REPAIR_BATCH_SIZE}" \
    --check-every "${REPAIR_CHECK_EVERY}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  test -d "${STAGE2}/checkpoint"
  test -f "${STAGE2}/repair_summary.json"

  echo "===== MCF SEED ${SEED}: BASE + POST EVAL ON ORIGINAL UNSWAPPED MCF ====="
  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" --mcf-path "${MCF}" --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2}/checkpoint" --mcf-path "${MCF}" --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${POST_EVAL}" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  "${PYTHON_BIN}" scripts/annotate_ppl_provenance.py \
    --eval-json "${BASE_EVAL}" --model-dir "${MODEL}" --wikidata-dir "${WIKIDATA_DIR}"
  "${PYTHON_BIN}" scripts/annotate_ppl_provenance.py \
    --eval-json "${POST_EVAL}" --model-dir "${STAGE2}/checkpoint" --wikidata-dir "${WIKIDATA_DIR}"

  echo "===== MCF SEED ${SEED}: PAPER TARGET-TRUE-SENSITIVE METRICS ====="
  "${PYTHON_BIN}" scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" --post-eval-json "${POST_EVAL}" \
    --split-manifest "${MANIFEST}" --out "${PAPER_EVAL}"

  echo "===== MCF SEED ${SEED}: IMMUTABLE RUN MANIFEST ====="
  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${MODEL}" "${MCF}" "${VISIBLE}" "${MANIFEST}" \
    "${STAGE1}/config_used.json" "${STAGE2}/repair_summary.json" \
    "${STAGE2}/checkpoint" "${BASE_EVAL}" "${POST_EVAL}" "${PAPER_EVAL}" \
    "${HASH_CHECKPOINT}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

(
    out_path, model_path, mcf_path, visible_path, split_path,
    stage1_path, stage2_path, checkpoint_dir, base_eval, post_eval,
    paper_eval, hash_checkpoint,
) = sys.argv[1:]


def sha(path):
    p = pathlib.Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

split = json.loads(pathlib.Path(split_path).read_text(encoding="utf-8"))
stage1 = json.loads(pathlib.Path(stage1_path).read_text(encoding="utf-8"))
stage2 = json.loads(pathlib.Path(stage2_path).read_text(encoding="utf-8"))
files = []
ckpt = pathlib.Path(checkpoint_dir)
for p in sorted(x for x in ckpt.rglob("*") if x.is_file()):
    item = {"path": str(p.relative_to(ckpt)), "bytes": p.stat().st_size}
    if hash_checkpoint == "1":
        item["sha256"] = sha(p)
    files.append(item)

payload = {
    "schema_version": 1,
    "protocol": "mcf_sure_context_target_true_sensitive_v1",
    "metric_schema": "mcf_target_true_sensitive_v2",
    "target_semantics": split.get("target_semantics"),
    "source_mcf": str(pathlib.Path(mcf_path).resolve()),
    "source_mcf_sha256": split.get("source_sha256"),
    "training_visible": str(pathlib.Path(visible_path).resolve()),
    "training_visible_sha256": split.get("training_visible_sha256"),
    "ordered_forget_case_ids": split.get("sampling", {}).get("forget_case_ids", []),
    "ordered_retain_case_ids": split.get("sampling", {}).get("retain_case_ids", []),
    "model_path": str(pathlib.Path(model_path).resolve()),
    "stage1_config": stage1,
    "stage1_config_sha256": sha(stage1_path),
    "stage2_summary": stage2,
    "stage2_summary_sha256": sha(stage2_path),
    "base_eval_sha256": sha(base_eval),
    "post_eval_sha256": sha(post_eval),
    "paper_eval_sha256": sha(paper_eval),
    "checkpoint_file_manifest": files,
    "checkpoint_content_hashed": hash_checkpoint == "1",
    "code_commit": git_value(["git", "rev-parse", "HEAD"]),
    "code_status_porcelain": git_value(["git", "status", "--porcelain"]),
}
path = pathlib.Path(out_path)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("run manifest:", path.resolve())
PY

  echo "Seed ${SEED} complete: ${PAPER_EVAL}"
done

"${PYTHON_BIN}" scripts/aggregate_mcf_target_true_sensitive.py \
  --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo
echo "Context-conditioned target-true-sensitive MCF complete: ${OUTPUT_ROOT}"
echo "Paper metrics: ${OUTPUT_ROOT}/seed*/target_true_sensitive_eval.json"
