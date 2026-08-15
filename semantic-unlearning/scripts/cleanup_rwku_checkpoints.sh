#!/usr/bin/env bash
set -euo pipefail

mode="preview"
root="outputs"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_rwku_checkpoints.sh [--delete] [OUTPUT_ROOT]

By default this only previews RWKU checkpoint directories and their sizes.
Pass --delete to remove only checkpoint/checkpoints/checkpoint-* directories
whose full path contains "rwku" (case-insensitive). JSON metrics, manifests,
logs, and checkpoint receipt files are preserved.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --delete) mode="delete" ;;
    -h|--help) usage; exit 0 ;;
    *) root="$arg" ;;
  esac
done

if [[ ! -d "$root" ]]; then
  echo "No output directory: $root"
  exit 0
fi

mapfile -d '' candidates < <(
  find "$root" -type d \
    \( -iname checkpoint -o -iname checkpoints -o -iname 'checkpoint-*' \) \
    -ipath '*rwku*' -print0 2>/dev/null
)

if (( ${#candidates[@]} == 0 )); then
  echo "No RWKU checkpoint directories found under $root"
  exit 0
fi

echo "RWKU checkpoint directories under $root:"
for path in "${candidates[@]}"; do
  du -sh "$path" 2>/dev/null || printf '?\t%s\n' "$path"
done

echo
printf 'Combined checkpoint size: '
du -sch "${candidates[@]}" 2>/dev/null | tail -1 || true

if [[ "$mode" != "delete" ]]; then
  echo
  echo "Preview only. To delete exactly these checkpoint directories:"
  echo "  bash scripts/cleanup_rwku_checkpoints.sh --delete '$root'"
  exit 0
fi

echo
for path in "${candidates[@]}"; do
  echo "Deleting: $path"
  rm -rf -- "$path"
done

echo "RWKU checkpoint cleanup complete. Metrics/logs/manifests were preserved."
