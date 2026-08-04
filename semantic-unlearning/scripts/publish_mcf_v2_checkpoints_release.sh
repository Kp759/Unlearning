#!/usr/bin/env bash
set -euo pipefail

# Publish the MCF protected-v2 official 10-seed artifacts as a GitHub Release.
#
# Why split files: every model.safetensors is about 7.21 GB, while a GitHub
# Release asset must be smaller than 2 GiB. This script uses 1900 MiB parts.
# It uploads one seed at a time and deletes temporary parts after upload so the
# full set is never duplicated locally.

REPO="${REPO:-Kp759/Unlearning}"
TARGET_BRANCH="${TARGET_BRANCH:-claude/setup-project-structure-JQ7fN}"
TAG="${TAG:-mcf-protected-v2-official-seeds0-9}"
TITLE="${TITLE:-MCF protected-v2 official seeds 0-9}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/outputs/gagd_5e_ultra_protected_v2_10seeds}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/mcf_v2_official_seeds0_9}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/outputs/release_staging/${TAG}}"
PART_SIZE="${PART_SIZE:-1900M}"

if [[ ! -d "${PROJECT_ROOT}/scripts" ]]; then
  echo "Run from semantic-unlearning or set PROJECT_ROOT." >&2
  exit 2
fi

for cmd in gh split sha256sum tar; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Missing required command: ${cmd}" >&2
    exit 2
  }
done

gh auth status >/dev/null

mkdir -p "${WORK_ROOT}/metadata" "${WORK_ROOT}/tmp"

# Capture exact lightweight artifacts. Never infer training settings from names.
cp -f "${RESULT_ROOT}/aggregate.json" "${WORK_ROOT}/metadata/"
cp -f "${RESULT_ROOT}/aggregate.md" "${WORK_ROOT}/metadata/"
cp -f "${RESULT_ROOT}/per_seed_results.csv" "${WORK_ROOT}/metadata/"

# Archive all lightweight experiment/config/evaluation files while excluding
# model weights. This includes tokenizer/config files when present.
tar \
  --exclude='model.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  -czf "${WORK_ROOT}/mcf-protected-v2-metadata.tar.gz" \
  -C "${PROJECT_ROOT}" \
  "outputs/gagd_5e_ultra_protected_v2_10seeds" \
  "outputs/mcf_v2_official_seeds0_9"

MANIFEST="${WORK_ROOT}/checkpoint_manifest.tsv"
printf 'seed\toriginal_bytes\toriginal_sha256\tpart_name\tpart_bytes\tpart_sha256\n' > "${MANIFEST}"

cat > "${WORK_ROOT}/RECONSTRUCT.md" <<'EOF'
# Reconstruct an MCF protected-v2 checkpoint

Download every part for a seed into one directory, then concatenate in lexical
order. Example for seed 0:

```bash
cat seed0.model.safetensors.part-* > model.safetensors
sha256sum model.safetensors
```

Compare the resulting SHA-256 with `checkpoint_manifest.tsv`. The checkpoint
belongs at:

```text
outputs/gagd_5e_ultra_protected_v2_10seeds/seedN/checkpoint/model.safetensors
```

The official evaluation used checkpoint seed N on official MCF split seed N.
EOF

cat > "${WORK_ROOT}/release_notes.md" <<'EOF'
# MCF protected-v2 official seeds 0–9

- Eff = 0 and Gen = 0 on 10/10 seeds.
- Strict post-reload gate passed on 9/10 seeds.
- Seed 7 reached minimum margin 0.09375, below the frozen 0.10 threshold.
- Aggregate Spe: 13.114000 ± 1.941621.
- Aggregate PPL: 11.381250 ± 0.523286.

The release contains:

- exact lightweight training/evaluation artifacts in `mcf-protected-v2-metadata.tar.gz`;
- split checkpoint assets for seeds 0–9;
- SHA-256 and size information in `checkpoint_manifest.tsv`;
- reconstruction instructions in `RECONSTRUCT.md`.
EOF

if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "Using existing release ${TAG}"
else
  gh release create "${TAG}" \
    --repo "${REPO}" \
    --target "${TARGET_BRANCH}" \
    --title "${TITLE}" \
    --notes-file "${WORK_ROOT}/release_notes.md"
fi

gh release upload "${TAG}" \
  "${WORK_ROOT}/mcf-protected-v2-metadata.tar.gz" \
  "${WORK_ROOT}/RECONSTRUCT.md" \
  --repo "${REPO}" \
  --clobber

for seed in {0..9}; do
  weight="${CHECKPOINT_ROOT}/seed${seed}/checkpoint/model.safetensors"
  [[ -f "${weight}" ]] || {
    echo "Missing checkpoint: ${weight}" >&2
    exit 3
  }

  tmp_seed="${WORK_ROOT}/tmp/seed${seed}"
  rm -rf "${tmp_seed}"
  mkdir -p "${tmp_seed}"

  original_bytes="$(stat -c '%s' "${weight}")"
  original_sha="$(sha256sum "${weight}" | awk '{print $1}')"

  echo "Splitting seed ${seed}: ${weight}"
  split \
    -b "${PART_SIZE}" \
    -d \
    -a 2 \
    --additional-suffix='.bin' \
    "${weight}" \
    "${tmp_seed}/seed${seed}.model.safetensors.part-"

  shopt -s nullglob
  parts=("${tmp_seed}"/seed${seed}.model.safetensors.part-*.bin)
  shopt -u nullglob
  [[ ${#parts[@]} -gt 0 ]] || {
    echo "No parts created for seed ${seed}" >&2
    exit 4
  }

  for part in "${parts[@]}"; do
    part_name="$(basename "${part}")"
    part_bytes="$(stat -c '%s' "${part}")"
    part_sha="$(sha256sum "${part}" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${seed}" "${original_bytes}" "${original_sha}" \
      "${part_name}" "${part_bytes}" "${part_sha}" >> "${MANIFEST}"
  done

  echo "Uploading ${#parts[@]} parts for seed ${seed}"
  gh release upload "${TAG}" "${parts[@]}" --repo "${REPO}" --clobber
  rm -rf "${tmp_seed}"
done

gh release upload "${TAG}" "${MANIFEST}" --repo "${REPO}" --clobber

sha256sum \
  "${WORK_ROOT}/mcf-protected-v2-metadata.tar.gz" \
  "${WORK_ROOT}/checkpoint_manifest.tsv" \
  "${WORK_ROOT}/RECONSTRUCT.md" \
  > "${WORK_ROOT}/RELEASE_SHA256SUMS"

gh release upload "${TAG}" \
  "${WORK_ROOT}/RELEASE_SHA256SUMS" \
  --repo "${REPO}" \
  --clobber

echo "Published release: ${TAG}"
echo "Manifest: ${MANIFEST}"
