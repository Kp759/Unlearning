#!/usr/bin/env bash
# SCALPEL on Wulver: every method vs ours (SURE), chained with SLURM dependencies.
# Run from snorkel_proposal/ on the login node.
#
#   bash scripts/submit_scalpel_compare.sh fetch  llama32_3b|qwen3_4b|llama31_8b|all   # 1. internet: weights + MQuAKE + RWKU
#   bash scripts/submit_scalpel_compare.sh smoke  llama32_3b                           # 2. seed 1, every method
#   bash scripts/submit_scalpel_compare.sh full   llama32_3b|qwen3_4b|llama31_8b|all   # 3. seeds 1-5
#   bash scripts/submit_scalpel_compare.sh robust llama32_3b                          # robust SURE variants only
#   bash scripts/submit_scalpel_compare.sh agent  llama32_3b                          # agentic leakage track
#   bash scripts/submit_scalpel_compare.sh report llama32_3b ...                       # re-collect leaderboards
#
# Per model the chain is:
#   covariance (ROME/MEMIT/ZeroUnlearn layers, skipped if cached) ─┐
#   build splits + base reference (array over seeds) ──────────────┼─> methods (array: seed x method)
#                                                                   ├─> SURE (array: seeds, 24 h)
#                                                                   └─> sequential track (array: seeds)
#   report (gate per seed + leaderboards) after everything above, whatever its exit status.
# robust = SURE-R ablation on the same splits (finished runs are skipped): SURE_VARIANTS
#          (default: ours_l16 sure_paper sure_r sure_r_noviews sure_r_exact sure_r_hardneg sure_r_free sure_r_idk sure_r_bake).
# agent  = agentic leakage track (scalpel/agentic.py) on splits that exist: AGENT_METHODS (default:
#          noop,idk_all,subject_blocklist_idk,answer_ban,ga,gd,npo,rmu,rome,memit,zerounlearn,sure,ours).
#          SURE methods reuse their finished core run's router, so run agent after smoke / robust.
# Env: METHODS (default below), SURE_VARIANTS (smoke/full: sure ours), SEEDS, PARTITION (dev | test + SCALPEL_TEST_SALT), N_FORGET (50),
#      SEQ_METHODS (gd,npo,memit), SKIP_SURE=1, SKIP_SEQ=1, MOM2_TEXT / MOM2_DATASET (scalpel_common.sh).
# Finished runs are skipped, so re-submitting after a failure only redoes what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: $0 fetch|smoke|full|report MODEL...}"; shift
USER_MODEL_PATH="${MODEL_PATH:-}"   # a caller's explicit path; never let one model's resolved path leak to the next
MODELS=("$@"); [[ ${#MODELS[@]} -gt 0 ]] || { echo "name at least one model (or all)" >&2; exit 2; }
[[ "${MODELS[0]}" == all ]] && MODELS=(llama32_3b qwen3_4b llama31_8b)
DEFAULT_METHODS="noop,idk_all,prompt_cache_idk,subject_blocklist_idk,keyword_router_idk,answer_ban,ga,gd,npo,rmu,rome,memit,zerounlearn"
METHODS="${METHODS:-$DEFAULT_METHODS}"
[[ "$MODE" == agent ]] || METHODS="$(echo "$METHODS" | tr ',' '\n' | grep -v -E '^sure' | paste -sd, - || true)"   # SURE variants: own array
if [[ "$MODE" == agent ]]; then
  METHODS="${AGENT_METHODS:-noop,idk_all,subject_blocklist_idk,answer_ban,ga,gd,npo,rmu,rome,memit,zerounlearn,sure,ours}"
fi
if [[ "$MODE" == robust ]]; then
  METHODS=""
  SURE_VARIANTS="${SURE_VARIANTS:-ours_l16 sure_paper sure_r sure_r_noviews sure_r_exact sure_r_hardneg sure_r_free sure_r_idk sure_r_bake}"
fi
SURE_VARIANTS="${SURE_VARIANTS:-sure ours}"     # ours: SURE-R router + one-pass L19 Δe (scalpel/methods/sure_paper.py)
SURE_LIST="$(echo $SURE_VARIANTS | tr ' ' ',')"; N_SURE=$(wc -w <<< "$SURE_VARIANTS" | tr -d " ")
case "$MODE" in smoke|agent) SEEDS="${SEEDS:-1}";; *) SEEDS="${SEEDS:-1 2 3 4 5}";; esac
SEED_ARRAY="$(echo $SEEDS | tr ' ' ',')"
N_SEEDS=$(wc -w <<< "$SEEDS" | tr -d " "); N_METHODS=0
[[ -n "$METHODS" ]] && N_METHODS=$(tr ',' '\n' <<< "$METHODS" | wc -l | tr -d ' ')
SL=scripts/slurm
mkdir -p /scratch/yl258/kp759/Unlearning/semantic-unlearning/logs

if [[ "$MODE" == fetch ]]; then
  mkdir -p data_cache/mquake
  for f in MQuAKE-CF-3k-v2.json MQuAKE-CF.json; do
    [[ -f "data_cache/mquake/$f" ]] || curl -fsSL -o "data_cache/mquake/$f" \
      "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/$f"
  done
  for MODEL in "${MODELS[@]}"; do
    export MODEL; MODEL_PATH="$USER_MODEL_PATH"; SCALPEL_NO_ENV=0 source $SL/scalpel_common.sh
    if [[ -f "${MODEL_PATH:-/x}/config.json" ]]; then echo "$MODEL: weights at $MODEL_PATH"; continue; fi
    echo "$MODEL: downloading $HF_ID into $HF_CACHE"
    # Llama is gated: accept the licence on huggingface.co and `hf auth login` once
    # the job environment sets HF_HUB_OFFLINE=1; the fetch step is the one place that goes online
    if command -v hf >/dev/null; then HF_HUB_OFFLINE=0 hf download "$HF_ID" --cache-dir "$HF_CACHE" --exclude "original/*"
    else HF_HUB_OFFLINE=0 huggingface-cli download "$HF_ID" --cache-dir "$HF_CACHE" --exclude "original/*"; fi
  done
  # RWKU supplies U's MMLU / TriviaQA questions; downloads only missing files, checked against the pinned hashes
  python "$ROOT/scripts/rwku_data.py" --data-root "$ROOT/data/rwku" --manifest "$SP/data_cache/rwku_manifest_check.json"
  [[ -f "${MOM2_TEXT}/dataset_dict.json" ]] || echo "NOTE: no Wikipedia corpus at $MOM2_TEXT; run (once): cd .. && bash scripts/submit_edit_baselines_wiki.sh corpus"
  exit 0
fi

submit() { sbatch --parsable "$@"; }
REPORT_DEPS=()
for MODEL in "${MODELS[@]}"; do
  export MODEL; MODEL_PATH="$USER_MODEL_PATH"; source $SL/scalpel_common.sh
  if [[ "$MODE" == report ]]; then continue; fi
  [[ -f "${MODEL_PATH:-/x}/config.json" ]] || { echo "$MODEL: no weights; run '$0 fetch $MODEL' first" >&2; exit 2; }
  # values go through the environment (--export=ALL): sbatch --export=A=x,y would split "x,y"
  export MODEL PARTITION N_FORGET="${N_FORGET:-50}" SEEDS
  if [[ "$MODE" == agent ]]; then   # splits, covariance and SURE's routers already exist
    AG=$(METHODS="$METHODS" submit --export=ALL --array=0-$(( N_SEEDS * N_METHODS - 1 )) $SL/scalpel_agent.slurm)
    echo "$MODEL agent track ($N_SEEDS seeds x $N_METHODS: $METHODS): $AG"; REPORT_DEPS+=("$AG")
    continue
  fi
  COV=""
  if grep -qE '(^|,)(rome|memit|zerounlearn)(,|$)' <<< "$METHODS" || grep -q sure_r_bake <<< "$SURE_VARIANTS"; then
    COV=$(submit --export=ALL $SL/scalpel_cov.slurm); echo "$MODEL covariance: $COV"
  fi
  BUILD=$(submit --export=ALL --array="$SEED_ARRAY" $SL/scalpel_build.slurm); echo "$MODEL build (seeds $SEEDS): $BUILD"
  if [[ "$N_METHODS" -gt 0 ]]; then
    MAIN=$(METHODS="$METHODS" submit --export=ALL --dependency="afterany:$BUILD${COV:+:$COV}" \
           --array=0-$(( N_SEEDS * N_METHODS - 1 )) $SL/scalpel_method.slurm)
    echo "$MODEL methods ($N_SEEDS seeds x $N_METHODS: $METHODS): $MAIN"
    REPORT_DEPS+=("$MAIN")
  fi
  if [[ "${SKIP_SURE:-0}" != 1 ]]; then
    SURE=$(METHODS="$SURE_LIST" MAX_UNLEARN_SECONDS=36000 submit --export=ALL --dependency="afterany:$BUILD${COV:+:$COV}" \
           --array=0-$(( N_SEEDS * N_SURE - 1 )) --time=24:00:00 --job-name=sc_sure $SL/scalpel_method.slurm)
    echo "$MODEL SURE ($N_SEEDS seeds x $N_SURE: $SURE_LIST): $SURE"; REPORT_DEPS+=("$SURE")
  fi
  if [[ "${SKIP_SEQ:-0}" != 1 && "$MODE" != robust ]]; then
    SEQ=$(SEQ_METHODS="${SEQ_METHODS:-gd,npo,memit}" submit --export=ALL --dependency="afterany:$BUILD${COV:+:$COV}" \
          --array="$SEED_ARRAY" $SL/scalpel_seq.slurm)
    echo "$MODEL sequential: $SEQ"; REPORT_DEPS+=("$SEQ")
  fi
done
DEP_ARG=(); [[ ${#REPORT_DEPS[@]} -gt 0 ]] && DEP_ARG=(--dependency="afterany:$(IFS=:; echo "${REPORT_DEPS[*]}")")
R=$(MODELS="${MODELS[*]}" PARTITION="${PARTITION:-dev}" submit --export=ALL "${DEP_ARG[@]}" $SL/scalpel_report.slurm)
echo "report: $R  ->  results/cluster/<model>/${PARTITION:-dev}/leaderboard/leaderboard.md"
echo "watch: squeue -u \$USER ; logs in /scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_*"
