#!/usr/bin/env bash
set -euo pipefail

# V1.4 relation-scoped private-null routing.
# Reuses the exact leakage-safe V1.3 seed-1 protocol and five-view corpus.
# The only architectural change is relation-aware gating + natural null behavior.
# Seed 1 remains development only.

OUTPUT_DIR=${1:?fresh V1.4 output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SOURCE_V13_RUN=${SOURCE_V13_RUN:-outputs/mcf_private_vocab_rewiring_v1_3_multiview_seed1_3b}
SOURCE_PROTOCOL="$SOURCE_V13_RUN/protocol"
VIEWS="$SOURCE_PROTOCOL/training_visible_multiview_forget.json"
REGISTRY=protocols/mcf_private_vocab_rewiring_v1_4_relation_null_registry.json
LOG_TMP="${OUTPUT_DIR}.training.log.tmp"

for path in "$OUTPUT_DIR" "$LOG_TMP"; do
  if [ -e "$path" ]; then
    echo "ERROR: V1.4 path already exists: $path" >&2
    exit 1
  fi
done

for path in \
  "$SOURCE_PROTOCOL/training_visible_forget_direct.json" \
  "$SOURCE_PROTOCOL/training_visible_protection_fit_direct.json" \
  "$SOURCE_PROTOCOL/split_manifest.json" \
  "$VIEWS" \
  "$REGISTRY"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: required V1.4 source file missing: $path" >&2
    exit 1
  fi
done

python - "$VIEWS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
assert p.get('views_per_case') == 5, p.get('views_per_case')
assert len(p.get('cases', [])) == 50, len(p.get('cases', []))
leak = p.get('leakage_contract', {})
for key in (
    'full_mcf_path_accepted',
    'official_paraphrase_prompts_read',
    'official_neighborhood_prompts_read',
    'official_generation_prompts_read',
    'official_retain_records_read',
    'generator_received_target_true',
    'generator_received_target_new',
):
    assert leak.get(key) is False, (key, leak.get(key))
print({
    'locked_source_cases': len(p['cases']),
    'locked_source_views_per_case': p['views_per_case'],
    'heldout_probe_text_read': p.get('heldout_probe_text_read', False),
})
PY

# The learner must not have any accidental path to official/held-out evaluation data.
unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH ADVERSARIAL_EVAL_PATH

python -u scripts/run_mcf_private_vocab_rewiring_v1_4_relation_null.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$SOURCE_PROTOCOL" \
  --view-corpus "$VIEWS" \
  --experiment-registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --gate-negatives-per-case 16 \
  --gate-steps 600 \
  --gate-learning-rate 0.03 \
  --gate-weight-decay 0.01 \
  --gate-feature-batch-size 16 \
  --steps 800 \
  --forget-batch-size 8 \
  --check-every 25 \
  --learning-rate 0.001 \
  --minimum-abstention-margin 0.1 \
  --minimum-true-suppression 2.0 \
  --true-suppression-weight 1.0 \
  --anchor-weight 0.001 \
  --relative-row-cap 0.5 \
  --topk 64 \
  --initial-equivalence-kl-max 0.0000001 \
  --save-model \
  2>&1 | tee "$LOG_TMP"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/provenance"
mv "$LOG_TMP" "$OUTPUT_DIR/logs/training.log"
sha256sum "$VIEWS" > "$OUTPUT_DIR/provenance/source_5view_corpus.sha256"
sha256sum "$SOURCE_PROTOCOL/training_visible_forget_direct.json" \
  > "$OUTPUT_DIR/provenance/source_forget_direct.sha256"
sha256sum "$SOURCE_PROTOCOL/training_visible_protection_fit_direct.json" \
  > "$OUTPUT_DIR/provenance/source_protection_fit_direct.sha256"
printf '%s\n' "$SOURCE_V13_RUN" > "$OUTPUT_DIR/provenance/source_v13_run.txt"

printf '\nV1.4 relation-null development run finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Model routing sidecar: %s\n' "$OUTPUT_DIR/model/relation_null_routing.json"
printf 'Behavioral target: natural abstention; target_new is not optimized.\n'
printf 'Final-certification status: DEVELOPMENT ONLY; final eval requires a new untouched seed.\n'
