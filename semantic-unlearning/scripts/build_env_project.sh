#!/usr/bin/env bash
# Rebuild the semantic_unlearning conda env on /project (backed up, never purged) with the
# pip versions scripts/scratch_purge_check.sh saved from the purged /scratch env.
#
#   module load Miniforge3
#   bash scripts/build_env_project.sh            # -> /project/yl258/kp759/conda_envs/semantic_unlearning
#
# On success it writes $ENV/.ready; from then on every job script uses this env instead of
# the /scratch one (no edits needed). Override the location with the first argument.
set -euo pipefail
E="${1:-/project/yl258/kp759/conda_envs/semantic_unlearning}"
PINS="${PINS:-$HOME/semantic_unlearning_pip_versions.txt}"
command -v conda >/dev/null || { echo "run 'module load Miniforge3' first" >&2; exit 2; }
[[ -f "$PINS" ]] || { echo "no $PINS: run scripts/scratch_purge_check.sh first" >&2; exit 2; }
[[ -e "$E" ]] && { echo "$E exists; remove it first (rm -rf $E) or pass another path" >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -p "$E" python=3.10 jq
set +u; conda activate "$E"; set -u
grep -viE '^(pip|setuptools|wheel)==' "$PINS" > "$E/pip_pins.txt"
echo "installing $(wc -l < "$E/pip_pins.txt") pinned pip packages (no cache: home quota is 50 GB)"
if ! pip install --no-cache-dir -r "$E/pip_pins.txt"; then
  # a pin that is not on PyPI (e.g. a local editable install) breaks the bulk install:
  # install the rest one by one (pins are a full set, so --no-deps is consistent) and list misses
  echo "bulk install failed; installing pins one by one" >&2
  : > "$E/pip_failed.txt"
  while read -r p; do
    [[ -z "$p" ]] && continue
    pip install --no-cache-dir --no-deps "$p" >/dev/null 2>&1 || echo "$p" >> "$E/pip_failed.txt"
  done < "$E/pip_pins.txt"
  echo "not installed (see $E/pip_failed.txt):"; sed 's/^/  /' "$E/pip_failed.txt"
fi
python - <<'EOF'
import datasets, numpy, sklearn, torch, transformers
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) | transformers {transformers.__version__} "
      f"| numpy {numpy.__version__} | datasets {datasets.__version__}")
EOF
command -v jq >/dev/null && echo "jq ok"
touch "$E/.ready"
echo "ready: $E  (job scripts now pick it up; test on a GPU node: python -c 'import torch; print(torch.cuda.is_available())')"
