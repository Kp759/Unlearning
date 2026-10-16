#!/bin/bash
# Factorized router v2, MCF seed 1: subject read at ITS OWN tokens (multiple-instance,
# token + next token, mid layer), relation at the last token (L27). Timeboxed against the
# L27 relation-aware (exact-match) router.
#
#   0  recall-site table (where facts are recalled), if mcf_recall_site_seed1 ran
#   1  probe heatmap: subject identifiable at the subject token vs the last token, per layer
#   2  fit v2 at subject layers 8 / 12 / 16 (one feature pass); calibration with a hard
#      validation false-fire cap (10%); exit 4 = no deployable layer, nothing shipped
#   3  routing-only table: old L19, L27 relation-aware, v1 (last-token subject), v2 per
#      layer (typo / lower-case / near-miss rows per layer, not only the selected one)
#   4  verdict: v2 ships only with recall >= 89%, neighbour fire <= 1%, near-miss <= 2%;
#      if it passes, attach it to the question-side Δe rows and run the held-out leak eval
#
#   sbatch mcf_factorized_router_v2_seed1.slurm
#   then: cat outputs/mcf_factorized_router_v2_seed1/{heatmap.md,specificity.md,verdict.md}
#SBATCH --job-name=mcf_fact_router_v2
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_fact_router_v2_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_fact_router_v2_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u  # conda's activate scripts reference unset variables
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
  set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$ENV_DIR"
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
SEED=1
OUT="$ROOT/outputs/mcf_factorized_router_v2_seed1"
OLD="$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router"
RELNEG27="$ROOT/outputs/mcf_router_layer_sweep_seed1/L27/relneg"
V1="$ROOT/outputs/mcf_factorized_router_seed1/L27"
ROWS="$ROOT/outputs/mcf_early_write_seed1/E_L16_query_logit/linear_global"
HELD="$ROOT/outputs/mcf_heldout_generation_seed1"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
MCF_PATH="$(jq -r '.mcf_path' "$REF")"
read -r -a SUBJECT_LAYERS <<< "${SUBJECT_LAYERS:-8 12 16}"
RELATION_LAYER="${RELATION_LAYER:-27}"
mkdir -p "$OUT"
echo "factorized router v2 | subject layers ${SUBJECT_LAYERS[*]} | relation L$RELATION_LAYER | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"

echo "===== 0. recall site (where facts are recalled at the subject tokens)"
cat "$ROOT/outputs/mcf_recall_site_seed1/summary.md" 2>/dev/null || echo "(mcf_recall_site_seed1 not run)"

echo "===== 1. probe heatmap"
if [[ ! -f "$OUT/heatmap.json" ]]; then
  python -u scripts/mcf_subject_probe_heatmap.py --reference-dir "$OLD" --out-dir "$OUT" \
    --device cuda --local-files-only
fi
cat "$OUT/heatmap.md"

echo "===== 2. fit v2"
FIT_STATUS=0
if [[ ! -f "$OUT/summary.json" ]]; then
  python -u scripts/fit_factorized_router_v2.py --reference-dir "$OLD" --output-dir "$OUT" \
    --subject-layers "${SUBJECT_LAYERS[@]}" --relation-layer "$RELATION_LAYER" \
    --min-recall 0.95 --max-false-fire 0.10 --device cuda --local-files-only || FIT_STATUS=$?
fi
python - "$OUT/summary.json" "$OUT/heatmap.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); h = json.load(open(sys.argv[2]))
print("selected:", s["selected_dir"], s["selected_status"], "| heatmap best subject-token layer:",
      h["best_subject_token_layer"])
for L, r in s["layers"].items():
    c = r["calibration"]
    print(f"  S{L}: {c['status']}  t_s={c['t_subject']:.4g} t_r={c['t_relation']:.4g} "
          f"recall={c['recall']:.3f} ff_same={c['false_fire_same_subject']:.3f} ff_other={c['false_fire_other']:.3f} "
          f"| detection {json.dumps({k: round(v, 3) for k, v in r['subject_detection_val'].items()})}")
EOF
if [[ "$FIT_STATUS" == "4" ]]; then echo "No deployable v2 layer (see above)."
elif [[ "$FIT_STATUS" != "0" ]]; then echo "v2 fit failed (exit $FIT_STATUS)" >&2; exit "$FIT_STATUS"; fi

echo "===== 3. routing-only table"
RUNS=("old L19 router=$OLD")
[[ -f "$RELNEG27/fact_association_embeddings.pt" ]] && RUNS+=("L27 relation-aware=$RELNEG27")
[[ -f "$V1/fact_association_embeddings.pt" ]] && RUNS+=("v1 last-token subject L27=$V1")
CANDS=()
for L in "${SUBJECT_LAYERS[@]}"; do
  D="$OUT/S$(printf '%02d' "$L")_R$(printf '%02d' "$RELATION_LAYER")"
  if [[ -f "$D/fact_association_embeddings.pt" ]]; then
    STATUS="$(jq -r '.calibration.status' "$D/factorized_router_report.json")"
    RUNS+=("v2 S$L ($STATUS)=$D"); CANDS+=("v2 S$L ($STATUS)")
  fi
done
if [[ ! -f "$OUT/specificity.json" ]]; then
  python -u scripts/mcf_router_specificity.py --runs "${RUNS[@]}" \
    --out "$OUT/specificity.md" --device cuda --local-files-only
fi
cat "$OUT/specificity.md" | sed -n 1,12p

echo "===== 4. verdict"
VERDICT=1
python scripts/router_v2_verdict.py "$OUT/specificity.json" --baseline "L27 relation-aware" \
  --candidates "${CANDS[@]}" | tee "$OUT/verdict.md" || true
if grep -q "^BEST=" "$OUT/verdict.md"; then VERDICT=0; fi
if [[ "$VERDICT" == "0" && -f "$ROWS/fact_association_embeddings.pt" ]]; then
  BEST="$(grep '^BEST=' "$OUT/verdict.md" | sed 's/^BEST=v2 S\([0-9]*\).*/\1/')"
  SEL="$OUT/S$(printf '%02d' "$BEST")_R$(printf '%02d' "$RELATION_LAYER")"
  ATT="$OUT/attached_E_L16_query_logit_S$BEST"
  [[ -f "$ATT/fact_association_embeddings.pt" ]] || \
    python scripts/factorized_router.py attach --router-dir "$SEL" --rows-from "$ROWS" --output-dir "$ATT"
  if [[ ! -f "$HELD/F2_S${BEST}_query_logit.json" ]]; then
    python -u scripts/mcf_heldout_generation.py run --method router --run-dir "$ATT" \
      --label "v2 router (subject S$BEST, relation L$RELATION_LAYER) + question-side Δe" \
      --out "$HELD/F2_S${BEST}_query_logit.json" --seed "$SEED" --mcf-path "$MCF_PATH" \
      --device cuda --local-files-only
  fi
  bash scripts/run_refusal_summaries.sh
else
  echo "Held-out leak eval skipped (no v2 layer passed the verdict)."
fi
echo "Finish: $(date)"
