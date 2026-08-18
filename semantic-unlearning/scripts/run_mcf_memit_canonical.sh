#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_memit_canonical.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_memit_canonical_3b}"
SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
HPARAMS="${MEMIT_HPARAMS:-../ZeroUnlearn/hparams/MEMIT/Llama-3.2-3B-Instruct.json}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MCF}"
test -d "${WIKIDATA_DIR}"
test -f "${HPARAMS}"

python scripts/verify_rome_memit_baseline_source.py

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  EDIT="${ROOT}/memit_edit"
  FINAL="${ROOT}/official_eval_locked.json"
  mkdir -p "${ROOT}"

  echo "===== MCF SEED ${SEED}: CANONICAL LOCKED SPLIT ====="
  python scripts/build_mcf_sure_canonical_split.py \
    --mcf-path "${MCF}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

  echo "===== MCF SEED ${SEED}: MEMIT ON SAME ${FORGET_NUM} LOCKED FACTS ====="
  rm -rf "${EDIT}"
  python scripts/run_model_editing_canonical_mcf_memit_compat.py \
    --algorithm MEMIT --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --hparams-path "${HPARAMS}" --output-dir "${EDIT}" \
    --seed "${SEED}" --dtype "${DTYPE}"

  echo "===== MCF SEED ${SEED}: SAME FINAL OFFICIAL EVAL ====="
  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${EDIT}/checkpoint" --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" --out "${FINAL}" \
    --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL}" --model-dir "${EDIT}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"
done

python scripts/aggregate_model_editing_canonical.py \
  --method MEMIT --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo "Canonical MEMIT MCF complete: ${OUTPUT_ROOT}"
