#!/bin/bash
# Router diagnostics step by step: one task per benchmark (0 mcf, 1 zsre, 2 mquake, 3 rwku),
# seed SEED (1). Same scripts and outputs as router_diag.slurm, run in the order of the to-do
# list, with a check after the steps everything else depends on:
#   1. cache       last-token states at every layer + L19 router outputs           (task 1)
#                  -> stop unless hook parity is 0
#   2. report      L19 per-example errors, confusion, RWKU, no-association, UMAP   (tasks 2-6)
#   3. sweep       same router refit at every layer; L19 vs penultimate            (tasks 7-8)
#                  -> stop unless the L19 refit reproduces the shipped router (MIN_AGREEMENT)
#   4. downstream  only if routing improves at the penultimate layer: the Eff/Gen/Spe/PPL
#                  command is written to layer_sweep/downstream_penultimate.sh, and submitted
#                  as its own 24 h job (this file, DOWNSTREAM_SCRIPT mode) when RUN_DOWNSTREAM=1
#   5. robustness  baseline / hardneg / aug / hardneg_aug / concat / unmasked at L19 and the
#                  penultimate layer, exact / fuzzy / no subject eligibility        (task 9)
#   6. combine     ROUTER_LAYER_CHOICE.md over every finished benchmark
# Finished work is skipped (cache: manifest.json; sweep and variants: sweep.json), so a
# resubmit resumes. Every step prints a banner and its key result in the log.
#
#   cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
#   sbatch router_diag_staged.slurm                                   # all 4, seed 1
#   sbatch --export=ALL,RUN_DOWNSTREAM=1 router_diag_staged.slurm     # + step 4 where needed
#   sbatch --array=3 router_diag_staged.slurm                         # RWKU only
#   tail -f logs/rdiag_s_<jobid>_0.out                                # 0 mcf 1 zsre 2 mquake 3 rwku
# Do not run it next to router_diag.slurm (job name rdiag): both write outputs/router_diag_v1.
#SBATCH --job-name=rdiag_s
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rdiag_s_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rdiag_s_%A_%a.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi  # /project env once built (scripts/build_env_project.sh)
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
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
cd "$REPO_DIR" || { echo "cannot cd to $REPO_DIR" >&2; exit 2; }
mkdir -p logs

# Step 4's downstream job: this same file, submitted with DOWNSTREAM_SCRIPT set.
if [[ -n "${DOWNSTREAM_SCRIPT:-}" ]]; then
  command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }
  echo "router downstream | $DOWNSTREAM_SCRIPT | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD 2>/dev/null)"
  cat "$DOWNSTREAM_SCRIPT" || exit 2
  bash "$DOWNSTREAM_SCRIPT"
  STATUS=$?
  echo "Finish downstream: $(date) | exit $STATUS"
  exit $STATUS
fi

python -c "import umap" 2>/dev/null \
  || echo "WARNING umap-learn not installed in $ENV_DIR: the L16/L19 projection falls back to PCA (pip install umap-learn)"
# Steps 3 and 5 are separate calls of the sweep; without this fix the robustness 'baseline'
# rows (fuzzy / no eligibility) are skipped as already done.
grep -q have_sweep scripts/router_layer_sweep_cached.py \
  || { echo "scripts/router_layer_sweep_cached.py lacks the resume-key fix: git pull origin feat/mcf-layer-sweep" >&2; exit 2; }
if command -v squeue >/dev/null 2>&1 && [[ "${FORCE:-0}" != 1 ]]; then
  OTHER="$(squeue -h -u "${USER:-$(id -un)}" -n rdiag -o '%i %T' 2>/dev/null || true)"
  [[ -z "$OTHER" ]] || { echo "router_diag.slurm job(s) queued/running, same outputs: $OTHER -- scancel first (FORCE=1 to ignore)" >&2; exit 2; }
fi

DATASETS=(mcf zsre mquake rwku)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$TASK]:-}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-3)" >&2; exit 2; }
SEED="${SEED:-1}"
ROOT_OUT="${ROOT_OUT:-outputs/router_diag_v1}"
OUT="$ROOT_OUT/$DS/seed$SEED"
MIN_AGREEMENT="${MIN_AGREEMENT:-0.99}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-0}"
VARIANTS="${VARIANTS:-baseline,hardneg,aug,hardneg_aug,concat,unmasked}"
declare -A ROUTER=(
  [mcf]="outputs/mcf_multiseed_regular_v1/seed$SEED/L19/router"
  [zsre]="outputs/zsre_multiseed_reworded_v2/seed$SEED/L19/router"
  [mquake]="outputs/mquake_multiseed_regular_v1/seed$SEED/L19/router"
  [rwku]="outputs/rwku_multiseed_subject_v1/seed$SEED/L19/router"
)
declare -A OURS=(
  [mcf]="outputs/l16_paper_mcf_linear19/mcf/seed$SEED/ours"
  [zsre]="outputs/l16_paper_v1/zsre/seed$SEED/ours"
  [mquake]="outputs/l16_paper_v1/mquake/seed$SEED/ours"
  [rwku]="outputs/l16_paper_rwku_subject_v1/rwku/seed$SEED/ours"
)
OURS_ARGS=()
[[ -f "${OURS[$DS]}/fact_association_embeddings.pt" ]] && OURS_ARGS=(--ours-dir "${OURS[$DS]}")

STATUS=0
START=$SECONDS
step() {
  echo
  echo "================ STEP $1/6 | $DS seed $SEED | $2 | $(date '+%F %T') | +$(( (SECONDS - START) / 60 )) min"
}
stop() {
  echo
  echo "STOPPED at step $1 ($DS seed $SEED): $2"
  echo "Finish $DS seed $SEED: $(date) | exit 1"
  exit 1
}
# Checks between steps; JSON only. Exit 0 pass, 1 fail, 2 input missing.
gate() {
  python - "$1" "$OUT" "$MIN_AGREEMENT" <<'PY'
import ast, json, sys
from pathlib import Path

check, out, minimum = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])

def load(path):
    if not path.is_file():
        print(f"MISSING {path}")
        sys.exit(2)
    return json.loads(path.read_text())

m = load(out / "manifest.json")
if check == "parity":
    parity = m.get("hook_parity") or {}
    mism = {k: v.get("route_mismatches") for k, v in parity.items()}
    if parity and all(v == 0 for v in mism.values()):
        print(f"PASS hook parity: route mismatches {mism}")
        sys.exit(0)
    print(f"FAIL hook parity: route mismatches {mism or 'not run'} (must all be 0)")
    for k, v in parity.items():
        for ex in (v.get("examples") or [])[:3]:
            print(f"  {k}: {json.dumps(ex)[:300]}")
    sys.exit(1)
if check == "agreement":
    sweep = load(out / "layer_sweep" / "sweep.json")
    L = m["router_layer"]
    a = next((e.get("agreement_with_shipped_router") for e in sweep.get("sweep", [])
              if e.get("rule") == "deployed" and e.get("layers") == [L]), None)
    if a is None:
        print(f"FAIL no deployed-rule refit at L{L} in sweep.json")
        sys.exit(1)
    if a >= 1.0 - 1e-12:
        print(f"PASS L{L} refit makes the shipped router's decision on 100% of eval prompts")
        sys.exit(0)
    if a >= minimum:
        print(f"WARN L{L} refit agrees with the shipped router on {a:.2%} of eval prompts (>= {minimum:.0%}); continuing")
        sys.exit(0)
    print(f"FAIL L{L} refit agrees with the shipped router on {a:.2%} of eval prompts (< {minimum:.0%})")
    sys.exit(1)
if check == "downstream":
    verdict = load(out / "layer_sweep" / "penultimate_vs_L19.json").get("deployed")
    L, pen = m["router_layer"], m["n_layers"] - 2
    if not verdict:
        print(f"NO_VERDICT L{L} vs L{pen} could not be compared; downstream run not needed")
        sys.exit(1)
    if not verdict.get("routing_improves"):
        print(f"KEEP_L{L}: routing does not improve at L{pen}; downstream run not needed")
        sys.exit(1)
    # DOWNSTREAM from router_layer_sweep_cached.py, read without importing it (no torch)
    tree = ast.parse(Path("scripts/router_layer_sweep_cached.py").read_text())
    templates = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", None) == "DOWNSTREAM" for t in n.targets))
    print(templates[m["dataset"]].format(L=pen, S=m["seed"]))
    sys.exit(0)
sys.exit(f"unknown check {check}")
PY
}

echo "router diagnostics (staged) | $DS seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD 2>/dev/null)"
echo "out $OUT | variants $VARIANTS | min agreement $MIN_AGREEMENT | run downstream $RUN_DOWNSTREAM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
[[ -f "${ROUTER[$DS]}/fact_association_embeddings.pt" || -f "$OUT/manifest.json" ]] \
  || echo "WARNING $DS router missing: ${ROUTER[$DS]} (step 1 will fail)"

# 1 ---------------------------------------------------------------------------------------
step 1 "cache: last-token states at every layer + L19 router outputs"
python -u scripts/cache_router_states.py --dataset "$DS" --seed "$SEED" --out-dir "$OUT" \
  --device cuda --local-files-only "${OURS_ARGS[@]}" || stop 1 "cache_router_states.py failed"
gate parity || stop 1 "hook parity failed: the cached decisions are not the deployed router's"

# 2 ---------------------------------------------------------------------------------------
step 2 "report: per-example errors, confusion, RWKU, no-association prompts, UMAP L16/L19"
if python -u scripts/router_diag_report.py --cache-dir "$OUT"; then
  echo "-> $OUT/report/summary.md"
  grep -E '^## ' "$OUT/report/summary.md" || true
else
  echo "router_diag_report.py failed (the sweep reads the cache, so it still runs)"
  STATUS=1
fi

# 3 ---------------------------------------------------------------------------------------
step 3 "layer sweep: same router refit at every layer; L19 vs penultimate"
python -u scripts/router_layer_sweep_cached.py --cache-dirs "$OUT" --out-root "$OUT/layer_sweep" \
  --device cuda --variants "" || stop 3 "router_layer_sweep_cached.py failed"
gate agreement || stop 3 "the L19 refit does not reproduce the shipped router; robustness variants not run"
echo "-> $OUT/layer_sweep/layer_choice.md, errors_vs_layer.png"
grep -E '^Best F1|vs penultimate' "$OUT/layer_sweep/layer_choice.md" || true

# 4 ---------------------------------------------------------------------------------------
step 4 "downstream Eff/Gen/Spe/PPL at the penultimate layer (only if routing improves there)"
if CMD="$(gate downstream)"; then
  SCRIPT="$OUT/layer_sweep/downstream_penultimate.sh"
  [[ "$SCRIPT" == /* ]] || SCRIPT="$PWD/$SCRIPT"
  printf '#!/bin/bash\n# %s seed %s: routing improves at the penultimate layer (router_diag_staged.slurm, %s)\nset -euo pipefail\ncd %q\n%s\n' \
    "$DS" "$SEED" "$(date '+%F %T')" "$REPO_DIR" "$CMD" > "$SCRIPT"
  echo "routing improves at the penultimate layer -> $SCRIPT"
  echo "  $CMD"
  SUBMIT="sbatch --array=0 --time=24:00:00 --job-name=rdiag_down --export=ALL,DOWNSTREAM_SCRIPT=$SCRIPT router_diag_staged.slurm"
  MARK="${SCRIPT%.sh}.submitted"
  if [[ -f "$MARK" ]]; then
    echo "already submitted: $(cat "$MARK")"
  elif [[ "$RUN_DOWNSTREAM" == 1 && -f "$REPO_DIR/router_diag_staged.slurm" ]]; then
    if JID="$(sbatch --parsable --array=0 --time=24:00:00 --job-name=rdiag_down \
                --export=ALL,DOWNSTREAM_SCRIPT="$SCRIPT",REPO_DIR="$REPO_DIR" "$REPO_DIR/router_diag_staged.slurm")"; then
      echo "job $JID ($(date '+%F %T'))" > "$MARK"
      echo "submitted downstream job $JID"
    else
      echo "sbatch failed; submit from the login node: $SUBMIT"
      STATUS=1
    fi
  else
    echo "not submitted (RUN_DOWNSTREAM=$RUN_DOWNSTREAM). To run it: $SUBMIT"
  fi
else
  echo "$CMD"
fi

# 5 ---------------------------------------------------------------------------------------
step 5 "robustness: $VARIANTS at L19 and the penultimate layer"
if python -u scripts/router_layer_sweep_cached.py --cache-dirs "$OUT" --out-root "$OUT/layer_sweep" \
     --device cuda --variants "$VARIANTS"; then
  echo "-> $OUT/layer_sweep/robustness.md"
else
  echo "robustness variants failed"
  STATUS=1
fi

# 6 ---------------------------------------------------------------------------------------
# Cross-benchmark write-up from every finished sweep (nothing is refit); the last task to
# finish writes the complete $ROOT_OUT/ROUTER_LAYER_CHOICE.md.
step 6 "combine: ROUTER_LAYER_CHOICE.md over every finished benchmark"
DONE=()
for D in "${DATASETS[@]}"; do
  [[ -f "$ROOT_OUT/$D/seed$SEED/layer_sweep/sweep.json" ]] && DONE+=("$ROOT_OUT/$D/seed$SEED")
done
if (( ${#DONE[@]} )); then
  python -u scripts/router_layer_sweep_cached.py --cache-dirs "${DONE[@]}" --out-root "$ROOT_OUT" \
    --combine-only || STATUS=1
  echo "-> $ROOT_OUT/ROUTER_LAYER_CHOICE.md (${#DONE[@]} of ${#DATASETS[@]} benchmarks so far)"
fi
echo
echo "Finish $DS seed $SEED: $(date) | +$(( (SECONDS - START) / 60 )) min | exit $STATUS"
exit $STATUS
