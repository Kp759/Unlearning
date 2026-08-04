#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/tofu_successful_splits_eval_20260804}"
REPO="${REPO:-Kp759/Unlearning}"
TARGET_BRANCH="${TARGET_BRANCH:-claude/setup-project-structure-JQ7fN}"
TAG="${TAG:-tofu-successful-splits-seed42}"
RELEASE_TITLE="${RELEASE_TITLE:-TOFU successful Forget01/05/10 checkpoints (seed 42)}"
PART_SIZE="${PART_SIZE:-1900M}"
STAGING_ROOT="${STAGING_ROOT:-${PROJECT_ROOT}/outputs/release_staging/${TAG}}"

required_commands=(gh split sha256sum tar find cp)
for command_name in "${required_commands[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login --hostname github.com --git-protocol ssh --web" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

NAMES=(forget01 forget05 forget10)
RUN_DIRS=(
  "outputs/tofu_forget01_setting3_restore_repair_seed42/lm_head_repair_unique100_overlap000"
  "outputs/tofu_setting3_5e_ultra_seed42/lm_head_repair_alpha000"
  "outputs/tofu_forget10_setting3_5e_repair_seed42/lm_head_repair_unique050_overlap000"
)
EVAL_FILES=(
  "${EVAL_ROOT}/forget01/probability_ratio_eval.json"
  "${EVAL_ROOT}/forget05/probability_ratio_eval.json"
  "${EVAL_ROOT}/forget10/probability_ratio_eval.json"
)

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  run_dir="${PROJECT_ROOT}/${RUN_DIRS[$index]}"
  checkpoint="${run_dir}/checkpoint"
  config="${run_dir}/config_used.json"
  weight="${checkpoint}/model.safetensors"
  eval_file="${EVAL_FILES[$index]}"

  [[ -d "${checkpoint}" ]] || { echo "Missing checkpoint directory: ${checkpoint}" >&2; exit 1; }
  [[ -f "${weight}" ]] || { echo "Missing checkpoint weight: ${weight}" >&2; exit 1; }
  [[ -f "${config}" ]] || { echo "Missing exact config: ${config}" >&2; exit 1; }
  [[ -f "${eval_file}" ]] || { echo "Missing fresh evaluation: ${eval_file}" >&2; exit 1; }
done

for required_eval in \
  "${EVAL_ROOT}/tofu_successful_probability_ratios.json" \
  "${EVAL_ROOT}/tofu_successful_probability_ratios.csv" \
  "${EVAL_ROOT}/tofu_successful_probability_ratios.md"; do
  [[ -f "${required_eval}" ]] || { echo "Missing combined evaluation artifact: ${required_eval}" >&2; exit 1; }
done

rm -rf "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}/metadata/evaluation"

cp "${EVAL_ROOT}/tofu_successful_probability_ratios.json" \
   "${STAGING_ROOT}/metadata/evaluation/"
cp "${EVAL_ROOT}/tofu_successful_probability_ratios.csv" \
   "${STAGING_ROOT}/metadata/evaluation/"
cp "${EVAL_ROOT}/tofu_successful_probability_ratios.md" \
   "${STAGING_ROOT}/metadata/evaluation/"

MANIFEST="${STAGING_ROOT}/checkpoint_manifest.tsv"
PART_SUMS="${STAGING_ROOT}/release_part_sha256sums.txt"
RECONSTRUCT="${STAGING_ROOT}/RECONSTRUCT.md"
METADATA_SUMS="${STAGING_ROOT}/metadata_sha256sums.txt"

printf 'split\tcheckpoint_path\tweight_bytes\tweight_sha256\tconfig_path\teval_path\tpart_prefix\tpart_size\n' > "${MANIFEST}"
: > "${PART_SUMS}"

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  relative_run="${RUN_DIRS[$index]}"
  run_dir="${PROJECT_ROOT}/${relative_run}"
  checkpoint="${run_dir}/checkpoint"
  weight="${checkpoint}/model.safetensors"
  eval_file="${EVAL_FILES[$index]}"
  metadata_dir="${STAGING_ROOT}/metadata/${name}"

  mkdir -p "${metadata_dir}/run" "${metadata_dir}/checkpoint"
  cp "${eval_file}" "${metadata_dir}/probability_ratio_eval.json"

  for filename in \
    config_used.json \
    candidate_local_metrics.json \
    baseline_local_metrics.json \
    repair_summary.json \
    selected_lm_head_rows.json \
    selected_rows.json \
    repair_experiment_config.json \
    active_cases_before.json \
    active_cases_after.json; do
    if [[ -f "${run_dir}/${filename}" ]]; then
      cp "${run_dir}/${filename}" "${metadata_dir}/run/${filename}"
    fi
  done

  while IFS= read -r -d '' small_file; do
    cp "${small_file}" "${metadata_dir}/checkpoint/$(basename "${small_file}")"
  done < <(
    find "${checkpoint}" -maxdepth 1 -type f \
      ! -name 'model.safetensors' \
      ! -name '*.bin' \
      -size -100M -print0
  )

  bytes="$(stat -c '%s' "${weight}")"
  weight_sha="$(sha256sum "${weight}" | awk '{print $1}')"
  part_prefix="${name}.model.safetensors.part-"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" \
    "${relative_run}/checkpoint" \
    "${bytes}" \
    "${weight_sha}" \
    "${relative_run}/config_used.json" \
    "${eval_file#${PROJECT_ROOT}/}" \
    "${part_prefix}" \
    "${PART_SIZE}" >> "${MANIFEST}"
done

(
  cd "${STAGING_ROOT}/metadata"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${METADATA_SUMS}"

tar -C "${STAGING_ROOT}" -czf \
  "${STAGING_ROOT}/tofu-successful-splits-seed42-metadata.tar.gz" \
  metadata

cat > "${RECONSTRUCT}" <<'EOF'
# Reconstruct TOFU successful checkpoints

Each checkpoint weight was split into release assets below GitHub's per-asset limit.
The exact configurations, fresh probability-ratio evaluations, tokenizer/configuration
files, and checkpoint SHA-256 hashes are included in the metadata archive and manifest.

For one split, download all matching parts and reconstruct with:

```bash
cat forget01.model.safetensors.part-* > model.safetensors
sha256sum model.safetensors
```

Compare the resulting hash with `checkpoint_manifest.tsv`.

Repeat with `forget05` or `forget10` for the other checkpoints. Place the reconstructed
`model.safetensors` beside the corresponding small checkpoint files from the metadata
archive.
EOF

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  gh release create "${TAG}" \
    --repo "${REPO}" \
    --target "${TARGET_BRANCH}" \
    --title "${RELEASE_TITLE}" \
    --notes-file "${RECONSTRUCT}"
fi

gh release upload "${TAG}" \
  "${STAGING_ROOT}/tofu-successful-splits-seed42-metadata.tar.gz" \
  "${MANIFEST}" \
  "${METADATA_SUMS}" \
  "${RECONSTRUCT}" \
  --repo "${REPO}" \
  --clobber

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  run_dir="${PROJECT_ROOT}/${RUN_DIRS[$index]}"
  weight="${run_dir}/checkpoint/model.safetensors"
  part_dir="${STAGING_ROOT}/parts/${name}"
  mkdir -p "${part_dir}"

  echo "Splitting and uploading ${name}: ${weight}"
  split -b "${PART_SIZE}" -d -a 2 \
    "${weight}" \
    "${part_dir}/${name}.model.safetensors.part-"

  split_sum_file="${part_dir}/${name}.parts.sha256"
  (
    cd "${part_dir}"
    sha256sum "${name}.model.safetensors.part-"* > "$(basename "${split_sum_file}")"
  )
  cat "${split_sum_file}" >> "${PART_SUMS}"

  gh release upload "${TAG}" \
    "${part_dir}/${name}.model.safetensors.part-"* \
    "${split_sum_file}" \
    --repo "${REPO}" \
    --clobber

  rm -rf "${part_dir}"
done

gh release upload "${TAG}" \
  "${PART_SUMS}" \
  --repo "${REPO}" \
  --clobber

echo
echo "Published release:"
gh release view "${TAG}" --repo "${REPO}"
