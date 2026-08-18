#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="${1:?Usage: bash scripts/run_sure_canonical_base_eval.sh {mcf|zsre} MODEL_PATH [DATA_JSON]}"
MODEL="${2:?MODEL_PATH required}"
DATA_JSON="${3:-}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
FORGET_NUM="${SURE_FORGET_NUM:-50}"
RETAIN_NUM="${SURE_RETAIN_EVAL_NUM:-1000}"

case "${DATASET}" in
  mcf)
    DATA_JSON="${DATA_JSON:-data/multi_counterfact.json}"
    SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_base_canonical_3b}"
    ;;
  zsre)
    DATA_JSON="${DATA_JSON:-data/zsre_mend_eval.json}"
    SEEDS_TEXT="${ZSRE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_base_canonical_3b}"
    ;;
  *)
    echo "DATASET must be mcf or zsre" >&2
    exit 2
    ;;
esac

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${DATA_JSON}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  OUT="${ROOT}/official_eval_locked.json"
  mkdir -p "${ROOT}"
  if [[ "${DATASET}" == "mcf" ]]; then
    python scripts/mcf_zero_unlearn_official_eval.py \
      --model-dir "${MODEL}" --mcf-path "${DATA_JSON}" \
      --wikidata-dir "${WIKIDATA_DIR}" --out "${OUT}" \
      --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" \
      --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    python scripts/zsre_zero_unlearn_official_eval.py \
      --model-dir "${MODEL}" --zsre-path "${DATA_JSON}" \
      --wikidata-dir "${WIKIDATA_DIR}" --out "${OUT}" \
      --method "Base canonical" --unlearn-num "${FORGET_NUM}" \
      --retain-num "${RETAIN_NUM}" --seed "${SEED}" --batch-size 8 \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  fi
  python scripts/annotate_ppl_provenance.py \
    --eval-json "${OUT}" --model-dir "${MODEL}" --wikidata-dir "${WIKIDATA_DIR}"
done

python scripts/aggregate_sure_canonical.py \
  --dataset "${DATASET}" --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo "Canonical ${DATASET} base evaluation complete: ${OUTPUT_ROOT}"
