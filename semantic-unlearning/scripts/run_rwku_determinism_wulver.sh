#!/usr/bin/env bash
# Run from the checkout root or anywhere on Wulver.
set -euo pipefail
ROOT="/scratch/yl258/kp759/Unlearning"
cd "$ROOT"

# The determinism source was committed as a gzipped, split patch.
if [[ ! -f semantic-unlearning/scripts/check_rwku_generation_determinism.py ]]; then
  git switch feat/router-ordered-fix-plan
  git pull --ff-only origin feat/router-ordered-fix-plan
  cat semantic-unlearning/patches/rwku-determinism-combined.patch.gz.part00 \
      semantic-unlearning/patches/rwku-determinism-combined.patch.gz.part01 \
    | gzip -dc > /tmp/determinism-and-combined-residual.patch
  echo '70ef89b3494dd073c6c57934ad9c810ccbf8825d22dc5d434e8b82a0b4351637  /tmp/determinism-and-combined-residual.patch' \
    | sha256sum --check
  git apply --check /tmp/determinism-and-combined-residual.patch
  git apply /tmp/determinism-and-combined-residual.patch
fi

cd semantic-unlearning
RUN="outputs/rwku_fact_assoc_router_v2_seed1_direct"
MATRIX="$RUN/cross_person_control/suppression_matrix.jsonl"
DECOMP=""
for trydir in "$RUN/decomposition_generation" "$RUN/decomposition"; do
  if [[ -s "$trydir/rwku_genie_subject_candidates.json" && -s "$trydir/rwku_router_decomposition_rows.json" ]]; then
    DECOMP="$trydir"
    break
  fi
done
if [[ -z "$DECOMP" ]]; then
  echo 'Could not find generation-genie candidates + decomposition rows under the direct run.' >&2
  exit 1
fi
[[ -s "$MATRIX" ]] || { echo "Missing $MATRIX" >&2; exit 1; }
mkdir -p logs "$RUN/determinism"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Artifact sha256:\n'
sha256sum "$RUN/fact_association_embeddings.pt" "$MATRIX" \
  "$DECOMP/rwku_genie_subject_candidates.json" \
  "$DECOMP/rwku_router_decomposition_rows.json"
python -u scripts/check_rwku_generation_determinism.py \
  --run-dir "$RUN" --data-root data/rwku \
  --matrix "$MATRIX" \
  --candidates "$DECOMP/rwku_genie_subject_candidates.json" \
  --decomposition-rows "$DECOMP/rwku_router_decomposition_rows.json" \
  --output-dir "$RUN/determinism" \
  --local-files-only --no-download \
  2>&1 | tee logs/rwku_generation_determinism.log
