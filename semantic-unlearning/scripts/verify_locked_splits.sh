#!/usr/bin/env bash
# Rebuild every seed's locked ZsRE/MQuAKE split in a scratch dir and compare it
# byte-for-byte with the installed one (the builders are deterministic).
# Needs the project env (activate it first, or run inside a job):
#   conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning
#   bash scripts/verify_locked_splits.sh            # seeds 2-5, both datasets
set -uo pipefail
python -c "import torch, transformers" 2>/dev/null || {
  echo "The project Python env is not active (torch/transformers not importable)." >&2
  echo "Run: conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning" >&2
  exit 3
}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CHECK="$(mktemp -d "$ROOT/outputs/.split_check.XXXX")"
trap 'rm -rf "$CHECK"' EXIT
STATUS=0
for SEED in ${SEEDS:-2 3 4 5}; do
  for DS in zsre mquake; do
    INSTALLED="$ROOT/outputs/${DS}_locked_split_seed$SEED"
    if [[ ! -d "$INSTALLED" ]]; then echo "seed $SEED $DS: not built (skip)"; continue; fi
    LOG="$CHECK/$DS$SEED.log"
    if [[ $DS == zsre ]]; then
      python -u scripts/build_zsre_zerounlearn_locked_no_neutral_split.py --zsre-path data/zsre_mend_eval.json \
        --output-dir "$CHECK/$DS$SEED" --seed "$SEED" --forget-num 50 --retain-num 1000 >"$LOG" 2>&1
    else
      python -u scripts/build_mquake_zerounlearn_locked_no_neutral_split.py --mquake-path data/MQuAKE-CF-3k-v2.json \
        --output-dir "$CHECK/$DS$SEED" --seed "$SEED" --forget-num 50 --retain-num 1000 >"$LOG" 2>&1
    fi
    if [[ $? -ne 0 || ! -s "$CHECK/$DS$SEED/split_manifest.json" ]]; then
      echo "seed $SEED $DS: REBUILD FAILED (not a mismatch); last lines of the builder log:"
      tail -n 5 "$LOG" | sed 's/^/    /'
      STATUS=2
      continue
    fi
    for f in training_visible_forget.json split_manifest.json; do
      a="$(sha256sum < "$INSTALLED/$f" | cut -c1-16)"; b="$(sha256sum < "$CHECK/$DS$SEED/$f" | cut -c1-16)"
      if [[ "$a" == "$b" ]]; then echo "seed $SEED $DS $f: OK"
      else
        # manifests may record the output path; compare them without it
        if [[ $f == split_manifest.json ]] && python - "$INSTALLED/$f" "$CHECK/$DS$SEED/$f" <<'EOF'
import json, sys
def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if "path" not in k and "dir" not in k}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x
a, b = (strip(json.load(open(p))) for p in sys.argv[1:3])
sys.exit(0 if a == b else 1)
EOF
        then echo "seed $SEED $DS $f: OK (paths differ only)"
        else echo "seed $SEED $DS $f: MISMATCH"; STATUS=1; fi
      fi
    done
  done
done
exit $STATUS
