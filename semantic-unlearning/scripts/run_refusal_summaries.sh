#!/usr/bin/env bash
# Rebuild both seed-1 refusal tables from whatever arms exist (CPU, seconds):
#   outputs/mcf_refusal_rows_seed1/comparison.md      training views + official Eff/Gen/Spe/PPL
#   outputs/mcf_heldout_generation_seed1/summary.md   held-out leak + over-refusal
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REFUSAL=outputs/mcf_refusal_rows_seed1
HELD=outputs/mcf_heldout_generation_seed1
SWEEP=outputs/mcf_projection_alpha_sweep_seed1
CANNED=outputs/mcf_canned_idk_seed1
mkdir -p "$REFUSAL"

EXTRA=()
TRAIN=(
  "base=$SWEEP/base/routed_generation.json"
  "delta_e=$SWEEP/reference_delta_e/routed_generation.json"
  "zerounlearn=outputs/zerounlearn_generation_leak_seed1/zerounlearn/routed_generation.json")
for ARM in R_eos R_idk S_idk; do
  if [[ -f "$REFUSAL/$ARM/routed_generation.json" ]]; then
    case "$ARM" in
      R_eos) LABEL="Δe→EOS (50 vectors)";; R_idk) LABEL="Δe→IDK (50 vectors)";;
      S_idk) LABEL="shared IDK (1 vector)";;
    esac
    EXTRA+=("$LABEL=$REFUSAL/$ARM"); TRAIN+=("$ARM=$REFUSAL/$ARM/routed_generation.json")
  fi
done
for ARM in J_A J_C; do
  DIR=outputs/mcf_jlens_feedback_seed1/$ARM
  if [[ -f "$DIR/routed_generation.json" ]]; then
    case "$ARM" in J_A) LABEL="J-lens feedback, last token (50 vectors)";;
                   J_C) LABEL="J-lens feedback, subject span (50 vectors)";; esac
    EXTRA+=("$LABEL=$DIR"); TRAIN+=("$ARM=$DIR/routed_generation.json")
  fi
done
for DIR in outputs/mcf_early_write_seed1/E_L*; do
  [[ -f "$DIR/routed_generation.json" ]] || continue
  NAME="$(basename "$DIR")"
  EXTRA+=("early write ${NAME#E_}, subject span (50 vectors)=$DIR")
  TRAIN+=("$NAME=$DIR/routed_generation.json")
done
if [[ -f "$CANNED/routed_generation.json" ]]; then
  EXTRA+=("router + fixed IDK (0 vectors)=$CANNED"); TRAIN+=("C_idk=$CANNED/routed_generation.json")
fi

python scripts/zerounlearn_generation_leak.py compare \
  --zero-dir outputs/zerounlearn_generation_leak_seed1 --sweep-root "$SWEEP" --alphas 3 \
  --zero-official outputs/zerounlearn_targettrue_parity_seeds1_10/seed1/zerounlearn_targettrue_seed1_official_eval.json \
  --extra "${EXTRA[@]}" --examples 8 --out "$REFUSAL/comparison.md" > /dev/null
echo "wrote $REFUSAL/comparison.md"
if ls "$HELD"/*.json >/dev/null 2>&1; then
  python scripts/mcf_heldout_generation.py summarize --root "$HELD" --train-views "${TRAIN[@]}" > /dev/null
  echo "wrote $HELD/summary.md"
fi
