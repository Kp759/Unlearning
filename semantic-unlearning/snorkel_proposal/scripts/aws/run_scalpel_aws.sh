#!/usr/bin/env bash
# SCALPEL on one GPU without SLURM (any GPU node: AWS, an HPC allocation; written for a 24 GB A10G,
# with SCALPEL_GRAD_SCOPE=mlp SCALPEL_RELEARN_SCOPE=mlp SCALPEL_GRAD_PRECISION=fp32 on >= 40 GB). The same chain
# as the Wulver jobs (scripts/slurm/*), one stage after another. Every step skips finished work, so
# after a stop, a crash or a reboot, rerun the same command. Run from anywhere.
#
#   bash scripts/aws/run_scalpel_aws.sh setup        # once: Python env, data, Wikipedia corpus, tests
#   bash scripts/aws/run_scalpel_aws.sh core         # covariance, splits, 13 methods + sure + ours, gate, report
#   bash scripts/aws/run_scalpel_aws.sh agent        # agentic leakage track (after core)
#   bash scripts/aws/run_scalpel_aws.sh robust       # ablations of ours: ours_l16, sure_paper, sure_r + 6 SURE-R
#   bash scripts/aws/run_scalpel_aws.sh sequential   # sequential deletion (gd, npo, memit)
#   bash scripts/aws/run_scalpel_aws.sh report       # leaderboards again
#   bash scripts/aws/run_scalpel_aws.sh status       # what is done / missing / failed
#   bash scripts/aws/run_scalpel_aws.sh all          # core, agent, robust, sequential
# Long stages survive a closed SSH session when started detached:
#   nohup bash scripts/aws/run_scalpel_aws.sh core > ~/scalpel_core.log 2>&1 &     tail -f ~/scalpel_core.log
#
# Settings (environment variables; defaults fit this instance):
#   MODEL_PATH      ~/models/Llama-3.2-3B-Instruct        MODEL  llama32_3b
#   DATASETS        "merged mcf mquake zsre"   one split each; merged = mcf+zsre+mquake (the headline split)
#   AGENT_DATASETS  "merged"     ROBUST_DATASETS "merged"     SEQ_DATASETS "merged"
#   SEEDS "1"   N_FORGET 50   METHODS / AGENT_METHODS / ROBUST_METHODS / SEQ_METHODS (lists below)
#   MOM2_TEXT       ~/datasets/wikipedia_20231101_100k (setup builds it: the corpus the Wulver runs use)
#   ENV_DIR         ~/envs/scalpel   (Python 3.11, torch 2.5.1 cu124, transformers 4.57.1; /opt/pytorch untouched)
#   METHOD_SECONDS 14400  SURE_SECONDS 14400 (unlearning time budget per run)
#   DRY_RUN=1       print the commands instead of running them
# 24 GB GPU (recorded in every results.json via method_args / eval_config): GA, GD, NPO train the MLP
# down-projections in mixed precision, and the relearning attack fine-tunes the down-projections.
set -euo pipefail
STAGE="${1:?usage: $0 setup|core|agent|robust|sequential|report|status|all}"

SP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"            # snorkel_proposal
export ROOT="$(cd "$SP/.." && pwd)"                                 # semantic-unlearning
export MODEL="${MODEL:-llama32_3b}"
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Llama-3.2-3B-Instruct}"
export HF_CACHE="${HF_CACHE:-${HF_HUB_CACHE:-$HOME/hf-cache/hub}}"
export MOM2_TEXT="${MOM2_TEXT:-$HOME/datasets/wikipedia_20231101_100k}"
export MOM2_DATASET="${MOM2_DATASET:-wikipedia_20231101_100k}"
export PARTITION=dev
ENV_DIR="${ENV_DIR:-$HOME/envs/scalpel}"
AWS_OUT="${AWS_OUT:-$SP/results/aws}"
LOGS="$AWS_OUT/logs"
DATASETS="${DATASETS:-merged mcf mquake zsre}"
AGENT_DATASETS="${AGENT_DATASETS:-merged}"
ROBUST_DATASETS="${ROBUST_DATASETS:-merged}"
SEQ_DATASETS="${SEQ_DATASETS:-merged}"
SEEDS="${SEEDS:-1}"
N_FORGET="${N_FORGET:-50}"
METHODS="${METHODS:-noop idk_all prompt_cache_idk subject_blocklist_idk keyword_router_idk answer_ban ga gd npo rmu rome memit zerounlearn sure ours}"
AGENT_METHODS="${AGENT_METHODS:-noop idk_all subject_blocklist_idk answer_ban ga gd npo rmu rome memit zerounlearn sure ours}"
# ablations of ours, one component each: ours_l16 (Δe at 16), sure_paper (plain router), sure_r (SURE-R's rows)
ROBUST_METHODS="${ROBUST_METHODS:-ours_l16 sure_paper sure_r sure_r_noviews sure_r_exact sure_r_hardneg sure_r_free sure_r_idk sure_r_bake}"
SEQ_METHODS="${SEQ_METHODS:-gd npo memit}"
METHOD_SECONDS="${METHOD_SECONDS:-14400}"
SURE_SECONDS="${SURE_SECONDS:-14400}"
export SCALPEL_GRAD_SCOPE="${SCALPEL_GRAD_SCOPE:-mlp_down}"
export SCALPEL_GRAD_PRECISION="${SCALPEL_GRAD_PRECISION:-mixed}"
export SCALPEL_RELEARN_SCOPE="${SCALPEL_RELEARN_SCOPE:-mlp_down}"
export SCALPEL_MIN_GPU_MIB="${SCALPEL_MIN_GPU_MIB:-20000}"
export SCALPEL_BATCH="${SCALPEL_BATCH:-16}"
mkdir -p "$LOGS"
FAILED=()

say() { echo "$(date '+%F %T') | $*"; }
run() {   # run (or print) a command; its output goes to the log named first
  local log="$1"; shift
  if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY: $* > $log"; return 0; fi
  "$@" > "$log" 2>&1
}

activate() {
  [[ -f "$ENV_DIR/.ready" ]] || { echo "no environment at $ENV_DIR: run '$0 setup' first" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
  export SCALPEL_NO_ENV=1 OUT_ROOT="$AWS_OUT/merged"
  # shellcheck disable=SC1091
  source "$SP/scripts/slurm/scalpel_common.sh"     # per-model settings, method_args, EVAL_ARGS, EDIT_LAYERS
  [[ -f "$MODEL_PATH/config.json" ]] || { echo "no model at $MODEL_PATH (set MODEL_PATH)" >&2; exit 2; }
}

use_dataset() {   # results/aws/<dataset>/<model>/dev/seed<S>; split_dir and method_args follow OUT_ROOT
  export OUT_ROOT="$AWS_OUT/$1"
  case "$1" in
    merged) SOURCES=mcf,zsre,mquake ;;
    mcf|zsre|mquake) SOURCES="$1" ;;
    *) echo "unknown dataset $1 (merged | mcf | zsre | mquake)" >&2; exit 2 ;;
  esac
}

# ------------------------------------------------------------------------------------------ setup
stage_setup() {
  if [[ ! -f "$ENV_DIR/.ready" ]]; then
    say "creating $ENV_DIR (Python 3.11, tested package versions)"
    local UVV="$HOME/.venvs/uvtool"
    [[ -x "$UVV/bin/uv" ]] || { python3 -m venv "$UVV" && "$UVV/bin/pip" install -q uv; }
    "$UVV/bin/uv" venv --python 3.11 "$ENV_DIR"
    "$UVV/bin/uv" pip install --python "$ENV_DIR/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    "$UVV/bin/uv" pip install --python "$ENV_DIR/bin/python" -r "$SP/scripts/aws/requirements-aws.txt"
    "$ENV_DIR/bin/python" -c "import torch, transformers; gpu = torch.cuda.is_available(); \
print('torch', torch.__version__, '| transformers', transformers.__version__, '|', \
      torch.cuda.get_device_name(0) if gpu else 'no GPU here (fine on a login node: core needs one)')"
    touch "$ENV_DIR/.ready"
  fi
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  say "data: MQuAKE, zsRE, RWKU (pinned hashes)"
  mkdir -p "$SP/data_cache/mquake"
  for f in MQuAKE-CF-3k-v2.json MQuAKE-CF.json; do
    [[ -f "$SP/data_cache/mquake/$f" ]] || curl -fsSL -o "$SP/data_cache/mquake/$f" \
      "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/$f"
  done
  [[ -f "$ROOT/data/zsre_mend_eval.json" ]] || curl -fsSL -o "$ROOT/data/zsre_mend_eval.json" \
    https://memit.baulab.info/data/dsets/zsre_mend_eval.json
  [[ -f "$ROOT/data/multi_counterfact.json" ]] || { echo "missing $ROOT/data/multi_counterfact.json (it is in git: git pull)" >&2; exit 2; }
  python "$ROOT/scripts/rwku_data.py" --data-root "$ROOT/data/rwku" --manifest "$SP/data_cache/rwku_manifest_check.json"
  if [[ ! -f "$MOM2_TEXT/dataset_dict.json" ]]; then
    say "Wikipedia corpus for the ROME / MEMIT / ZeroUnlearn covariance -> $MOM2_TEXT (20-40 min, streamed)"
    (cd "$ROOT" && python -u scripts/prepare_sure_wikipedia_corpus.py --output-dir "$MOM2_TEXT")
  fi
  [[ -f "$MODEL_PATH/config.json" ]] || { echo "no model at $MODEL_PATH" >&2; exit 2; }
  say "tests (CPU, about a minute)"
  (cd "$SP" && python -m pytest -q tests/ -p no:cacheprovider -W ignore)
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible here"
  df -h "$HOME" | tail -1
  say "setup done"
}

# ------------------------------------------------------------------------------------------- steps
covariance() {   # once per model: the layers ROME / MEMIT / ZeroUnlearn edit and SURE's (sure_r_bake)
  [[ -f "$MOM2_TEXT/dataset_dict.json" || "${DRY_RUN:-0}" == 1 ]] || { echo "no corpus at $MOM2_TEXT: run '$0 setup'" >&2; exit 2; }
  local log="$LOGS/covariance_$MODEL.log"
  say "covariance: layers $EDIT_LAYERS ($MOM2_DATASET; existing caches are skipped) -> $log"
  (cd "$ROOT" && run "$log" python -u scripts/build_edit_covariance.py --model-path "$MODEL_PATH" \
     --text-dir "$MOM2_TEXT" --dataset-name "$MOM2_DATASET" --layers $EDIT_LAYERS --device cuda --local-files-only) \
    || { FAILED+=("covariance"); say "covariance FAILED (see $log): rome/memit/zerounlearn/sure_r_bake will fail"; }
}

build() {   # build <dataset> <seed>
  use_dataset "$1"; local DIR SPLIT; DIR="$(split_dir "$2")"; SPLIT="$DIR/split.json"; mkdir -p "$DIR"
  if [[ ! -f "$SPLIT" ]]; then
    local log="$LOGS/$1_seed$2_build.log"
    say "$1 seed $2: building the split ($SOURCES) -> $log"
    (cd "$SP" && run "$log" python -u -m scalpel build --model "$MODEL_PATH" --model-id "$HF_ID" --seed "$2" \
       --n-forget "$N_FORGET" --sources "$SOURCES" --name "scalpel-$1" \
       --mquake "$SP/data_cache/mquake/MQuAKE-CF-3k-v2.json,$SP/data_cache/mquake/MQuAKE-CF.json" \
       --out "$SPLIT" --device cuda --dtype bf16) || { FAILED+=("$1/seed$2/build"); say "build FAILED: $log"; return 1; }
  fi
  [[ "${DRY_RUN:-0}" == 1 ]] || (cd "$SP" && python -m scalpel validate --split "$SPLIT" > "$DIR/validation.json") \
    || say "$1 seed $2: split validation reported failures ($DIR/validation.json)"
}

method() {   # method <dataset> <seed> <method>: the core track, as scripts/slurm/scalpel_method.slurm
  use_dataset "$1"; export SEED="$2"; local M="$3" DIR SPLIT OUT
  DIR="$(split_dir "$2")"; SPLIT="$DIR/split.json"; OUT="$DIR/runs/$M"
  [[ -f "$OUT/results.json" ]] && return 0
  [[ -f "$SPLIT" || "${DRY_RUN:-0}" == 1 ]] || { say "$1 seed $2 $M: no split, skipped"; return 0; }
  [[ -e "$OUT" ]] && mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
  local secs="$METHOD_SECONDS"; [[ "$M" == sure* ]] && secs="$SURE_SECONDS"
  local log="$LOGS/$1_seed$2_$M.log"; local t0=$SECONDS
  say "$1 seed $2: $M -> $log"
  # shellcheck disable=SC2046
  if (cd "$SP" && run "$log" python -u -m scalpel run --split "$SPLIT" --model "$MODEL_PATH" --method "$M" \
        $(method_args "$M") --out "$OUT" --max-unlearn-seconds "$secs" "${EVAL_ARGS[@]}"); then
    say "$1 seed $2: $M done in $(( (SECONDS - t0) / 60 )) min: $(grep -h "SCALPEL(tf)" "$log" 2>/dev/null | tail -1 | sed 's/.*SCALPEL/SCALPEL/')"
  else
    FAILED+=("$1/seed$2/$M"); say "$1 seed $2: $M FAILED (last lines of $log):"; tail -5 "$log" 2>/dev/null || true
  fi
}

agent() {   # agent <dataset> <seed> <method>: as scripts/slurm/scalpel_agent.slurm
  use_dataset "$1"; export SEED="$2"; local M="$3" DIR SPLIT OUT
  DIR="$(split_dir "$2")"; SPLIT="$DIR/split.json"; OUT="$DIR/agent/$M"
  [[ -f "$OUT/agent.json" ]] && return 0
  [[ -f "$SPLIT" || "${DRY_RUN:-0}" == 1 ]] || { say "$1 seed $2 agent $M: no split, skipped"; return 0; }
  if [[ "$M" == sure* && ! -f "$DIR/runs/$M/results.json" && "${DRY_RUN:-0}" != 1 ]]; then
    say "$1 seed $2 agent $M: its core run is missing (the trained router is reused), skipped"; return 0
  fi
  [[ -e "$OUT" ]] && mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
  local log="$LOGS/$1_seed$2_agent_$M.log"; local t0=$SECONDS
  say "$1 seed $2: agent $M -> $log"
  # shellcheck disable=SC2046
  if (cd "$SP" && run "$log" python -u -m scalpel agent --split "$SPLIT" --model "$MODEL_PATH" --method "$M" \
        $(method_args "$M") --out "$OUT" --max-unlearn-seconds "$SURE_SECONDS" --device cuda --dtype bf16 \
        --batch-size "${AGENT_BATCH:-16}" ${AGENT_ARGS:-}); then
    say "$1 seed $2: agent $M done in $(( (SECONDS - t0) / 60 )) min: $(grep -h "SCALPEL_agent" "$log" 2>/dev/null | tail -1 | sed 's/.*SCALPEL_agent/SCALPEL_agent/')"
  else
    FAILED+=("$1/seed$2/agent_$M"); say "$1 seed $2: agent $M FAILED (last lines of $log):"; tail -5 "$log" 2>/dev/null || true
  fi
}

sequential() {   # sequential <dataset> <seed> <method>
  use_dataset "$1"; export SEED="$2"; local M="$3" DIR SPLIT
  DIR="$(split_dir "$2")"; SPLIT="$DIR/split.json"
  [[ -f "$DIR/sequential/$M/sequential.json" ]] && return 0
  local log="$LOGS/$1_seed$2_seq_$M.log"
  say "$1 seed $2: sequential $M -> $log"
  # shellcheck disable=SC2046
  (cd "$SP" && run "$log" python -u -m scalpel sequential --split "$SPLIT" --model "$MODEL_PATH" --method "$M" \
     $(method_args "$M") --steps 5 --out "$DIR/sequential/$M" --no-gen --attacks "" "${EVAL_ARGS[@]}") \
    || { FAILED+=("$1/seed$2/seq_$M"); say "sequential $M FAILED: $log"; }
}

gate() {   # red-team gate of one split: every reference cheat must score below 35
  use_dataset "$1"; local DIR; DIR="$(split_dir "$2")"
  [[ -f "$DIR/split.json" && "${DRY_RUN:-0}" != 1 ]] || return 0
  (cd "$SP" && python -m scalpel gate --split "$DIR/split.json" --runs "$DIR/runs" > /dev/null 2>&1) || true
  [[ -f "$DIR/runs/RED_TEAM_REPORT.md" ]] && say "$1 seed $2: $(grep -m1 "Overall" "$DIR/runs/RED_TEAM_REPORT.md")"
  return 0
}

stage_report() {
  [[ "${DRY_RUN:-0}" == 1 ]] && { echo "DRY: reports"; return 0; }
  local d S
  for d in $DATASETS; do
    use_dataset "$d"
    compgen -G "$AWS_OUT/$d/$MODEL/dev/seed*/runs/*/results.json" > /dev/null || continue
    (cd "$SP" && python -m scalpel report --runs "$AWS_OUT/$d/$MODEL/dev/seed*/runs/*/results.json" \
       --out "$AWS_OUT/$d/$MODEL/dev/leaderboard" --exclude sure ours_early --title "LETHE-Bench — $MODEL — $d (AWS A10G)" > /dev/null)
    for S in $SEEDS; do
      local DIR; DIR="$(split_dir "$S")"
      compgen -G "$DIR/agent/*/agent.json" > /dev/null && (cd "$SP" && python -m scalpel agent-report \
        --runs "$DIR/agent/*/agent.json" --out "$DIR/agent" --title "LETHE-Bench agent track — $MODEL — $d — seed $S" > /dev/null)
    done
  done
  if compgen -G "$AWS_OUT/*/$MODEL/dev/seed*/runs/*/results.json" > /dev/null; then
    (cd "$SP" && python -m scalpel report --runs "$AWS_OUT/*/$MODEL/dev/seed*/runs/*/results.json" \
       --out "$AWS_OUT/leaderboard_$MODEL" --exclude sure ours_early --title "LETHE-Bench — $MODEL — every dataset (AWS A10G)" > /dev/null)
    say "leaderboards: $AWS_OUT/leaderboard_$MODEL/leaderboard.md (every dataset) and $AWS_OUT/<dataset>/$MODEL/dev/leaderboard/"
  fi
}

stage_status() {
  local d S M DIR
  for d in $DATASETS; do
    use_dataset "$d"
    for S in $SEEDS; do
      DIR="$(split_dir "$S")"
      [[ -f "$DIR/split.json" ]] || { echo "$d seed $S: no split yet"; continue; }
      local done_=() todo=() agent_=()
      for M in $METHODS $ROBUST_METHODS; do
        if [[ -f "$DIR/runs/$M/results.json" ]]; then done_+=("$M"); else todo+=("$M"); fi
      done
      for M in $AGENT_METHODS; do [[ -f "$DIR/agent/$M/agent.json" ]] && agent_+=("$M"); done
      echo "$d seed $S: core done (${#done_[@]}): ${done_[*]:-}"
      echo "$d seed $S: core missing (${#todo[@]}): ${todo[*]:-}"
      echo "$d seed $S: agent done (${#agent_[@]}): ${agent_[*]:-}"
    done
  done
  grep -l "Traceback" "$LOGS"/*.log 2>/dev/null | sed 's/^/log with an error: /' || true
}

stage_core() {
  say "core: datasets [$DATASETS], seeds [$SEEDS], methods [$METHODS]"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  if grep -qE '(^| )(rome|memit|zerounlearn)( |$)' <<< "$METHODS"; then covariance; fi
  local d S M
  for d in $DATASETS; do
    for S in $SEEDS; do
      build "$d" "$S" || continue
      for M in $METHODS; do method "$d" "$S" "$M"; done
      gate "$d" "$S"
    done
  done
  stage_report
}

stage_agent() {
  local d S M
  for d in $AGENT_DATASETS; do for S in $SEEDS; do for M in $AGENT_METHODS; do agent "$d" "$S" "$M"; done; done; done
  stage_report
}

stage_robust() {
  grep -q sure_r_bake <<< "$ROBUST_METHODS" && covariance
  local d S M
  for d in $ROBUST_DATASETS; do for S in $SEEDS; do for M in $ROBUST_METHODS; do method "$d" "$S" "$M"; done; done; done
  stage_report
}

stage_sequential() {
  local d S M
  for d in $SEQ_DATASETS; do for S in $SEEDS; do for M in $SEQ_METHODS; do sequential "$d" "$S" "$M"; done; done; done
}

summary() {
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "finished with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    say "rerun the same command after a fix: finished runs are skipped"
    return 1
  fi
  say "finished without failures"
}

case "$STAGE" in
  setup) stage_setup ;;
  core) activate; stage_core; summary ;;
  agent) activate; stage_agent; summary ;;
  robust) activate; stage_robust; summary ;;
  sequential) activate; stage_sequential; summary ;;
  report) activate; stage_report ;;
  status) activate; stage_status ;;
  all) activate; stage_core; stage_agent; stage_robust; stage_sequential; summary ;;
  *) echo "usage: $0 setup|core|agent|robust|sequential|report|status|all" >&2; exit 2 ;;
esac
