#!/usr/bin/env bash
set -euo pipefail

# V1.3c full-fit optimization ablation.
# Reuses the exact already-built V1.3 five-view protocol/corpus so the only
# experimental change is optimization (curriculum + hard-case oversampling).
# Seed 1 remains development only.

OUTPUT_DIR=${1:?fresh V1.3c output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SOURCE_V13_RUN=${SOURCE_V13_RUN:-outputs/mcf_private_vocab_rewiring_v1_3_multiview_seed1_3b}
SOURCE_PROTOCOL="$SOURCE_V13_RUN/protocol"
VIEWS="$SOURCE_PROTOCOL/training_visible_multiview_forget.json"
REGISTRY=protocols/mcf_private_vocab_rewiring_v1_3c_fullfit_registry.json
LOG_TMP="${OUTPUT_DIR}.training.log.tmp"

for path in "$OUTPUT_DIR" "$LOG_TMP"; do
  if [ -e "$path" ]; then
    echo "ERROR: V1.3c path already exists: $path" >&2
    exit 1
  fi
done

for path in \
  "$SOURCE_PROTOCOL/training_visible_forget_direct.json" \
  "$SOURCE_PROTOCOL/training_visible_protection_fit_direct.json" \
  "$SOURCE_PROTOCOL/split_manifest.json" \
  "$VIEWS"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: required locked V1.3 source file missing: $path" >&2
    exit 1
  fi
done

python - "$VIEWS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
assert p.get('views_per_case') == 5, p.get('views_per_case')
assert len(p.get('cases', [])) == 50, len(p.get('cases', []))
assert p.get('leakage_contract', {}).get('official_paraphrase_prompts_read') is False
assert p.get('leakage_contract', {}).get('official_neighborhood_prompts_read') is False
print({
    'locked_source_cases': len(p['cases']),
    'locked_source_views_per_case': p['views_per_case'],
    'heldout_probe_text_read': p.get('heldout_probe_text_read', False),
})
PY

unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH ADVERSARIAL_EVAL_PATH

export MCF_V13_VIEW_CORPUS="$VIEWS"
export MCF_V13_VIEW_CHUNK=16

python -u scripts/run_mcf_private_vocab_rewiring_v1_3c_fullfit.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$SOURCE_PROTOCOL" \
  --experiment-registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 1500 \
  --forget-batch-size 8 \
  --retain-batch-size 16 \
  --check-every 25 \
  --learning-rate 0.001 \
  --minimum-forget-margin 0.1 \
  --train-margin-target 0.1 \
  --retain-kl-weight 20.0 \
  --anchor-weight 0.001 \
  --relative-row-cap 0.5 \
  --topk 64 \
  --initial-equivalence-kl-max 0.0000001 \
  --initial-margin-drift-max 0.00001 \
  --retain-kl-mean-max 0.0001 \
  --nonclone-certification-prompts 64 \
  --save-model \
  2>&1 | tee "$LOG_TMP"

unset MCF_V13_VIEW_CORPUS MCF_V13_VIEW_CHUNK
mkdir -p "$OUTPUT_DIR/logs"
mv "$LOG_TMP" "$OUTPUT_DIR/logs/training.log"

# Preserve provenance of the exact source corpus without copying held-out data.
mkdir -p "$OUTPUT_DIR/provenance"
sha256sum "$VIEWS" > "$OUTPUT_DIR/provenance/source_5view_corpus.sha256"
printf '%s\n' "$SOURCE_V13_RUN" > "$OUTPUT_DIR/provenance/source_v13_run.txt"

printf '\nV1.3c full-fit experiment finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Target: all 50 cases pass worst-of-5 margin >= 0.1.\n'
printf 'Final-certification status: DEVELOPMENT ONLY.\n'
