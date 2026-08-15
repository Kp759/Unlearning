#!/usr/bin/env bash
#SBATCH --job-name=mq-sure-r0-s0
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mq_sure_r0_s0_%j.out
#SBATCH --error=slurm_logs/mq_sure_r0_s0_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning_mquake_sure_seed0/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL="${MODEL_PATH:?submit with MODEL_PATH=/path/to/Llama-3.2-3B-Instruct}"
MQUAKE="${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUT="${OUTPUT_ROOT:-outputs/mquake_sure_zul50_r0_restore_seed0}/seed0"
SEED=0

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUT}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf '===== SURE MQuAKE seed0 =====\n'
printf 'host=%s job=%s\nmodel=%s\nMQuAKE=%s\nout=%s\n' "$(hostname)" "${SLURM_JOB_ID}" "${MODEL}" "${MQUAKE}" "${OUT}"
python --version
nvidia-smi

test -d "${MODEL}"
test -f "${MQUAKE}"
test -d "${WIKIDATA_DIR}"

PROTOCOL="${OUT}/protocol"
VISIBLE="${PROTOCOL}/training_visible_forget.json"
MANIFEST="${PROTOCOL}/split_manifest.json"
STAGE1="${OUT}/stage1_gagd"
SURE="${OUT}/sure_r0_restore"

rm -rf "${STAGE1}" "${SURE}"

# Exact public-code-reproducible ZeroUnlearn-style data split: instance-level
# sampling from second-half forget pool, 50 forget first then 1000 retain.
python scripts/build_mquake_zerounlearn_locked_no_neutral_split.py \
  --mquake-path "${MQUAKE}" \
  --output-dir "${PROTOCOL}" \
  --seed "${SEED}" \
  --forget-num 50 \
  --retain-num 1000

test -f "${VISIBLE}"
test -f "${MANIFEST}"

# Stage 1A: no-neutral GA/GD.  Only the same 50 direct forget instances are
# visible.  GD preserves the frozen Base non-sensitive distribution on these
# same contexts; benchmark retain/AtomicGen/PPL are not opened.
python scripts/mquake_no_neutral_stage1_gagd.py \
  --model-path "${MODEL}" \
  --training-visible-path "${VISIBLE}" \
  --split-manifest "${MANIFEST}" \
  --output-dir "${STAGE1}" \
  --seed "${SEED}" \
  --forget-num 50 \
  --steps 600 \
  --batch-size 1 \
  --cache-batch-size 8 \
  --emb-lm-lr 0.0001 \
  --ga-weight 2.0 \
  --gd-weight 1.0 \
  --grad-clip 1.0 \
  --restoration-mode sensitive_both \
  --dtype bf16 \
  --device-map single

test -d "${STAGE1}/checkpoint"

# Stage 1B + Stage 2.  Rank-0 uses ONE pre-registered margin (0.25); there is
# no adaptive margin sweep.  Rank-64 and rank-128 independently branch from
# the exact same rank-0 rows.  Candidate scale selection sees direct forget
# constraints only.
python scripts/mquake_sure_gagd_rank0_restore.py \
  --stage1-model-path "${STAGE1}/checkpoint" \
  --base-model-path "${MODEL}" \
  --training-visible-path "${VISIBLE}" \
  --split-manifest "${MANIFEST}" \
  --output-dir "${SURE}" \
  --seed "${SEED}" \
  --forget-num 50 \
  --rank0-margin-schedule 0.25 \
  --rank0-steps-per-margin 600 \
  --rank0-lr 0.005 \
  --rank0-hard-weight 2.0 \
  --rank0-l2 0.0001 \
  --selection-margin 0.10 \
  --restore-ranks 64,128 \
  --restore-scales 1,.875,.75,.625,.5,.375,.25,.1875,.125,.0625,0 \
  --cache-batch-size 8 \
  --eval-batch-size 8 \
  --dtype bf16 \
  --device-map single \
  --save-model

for RANK in 64 128; do
  test -d "${SURE}/rank${RANK}/checkpoint"
done

# The evaluation firewall opens only now, after both SURE candidates are
# frozen.  Evaluate Base, Stage1 post-hoc, and both SURE candidates with the
# same locked local evaluator and same seed0 50/1000 split.
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${MODEL}" \
  --mquake-path "${MQUAKE}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUT}/base_eval.json" \
  --split-manifest "${OUT}/base_eval_split_manifest.json" \
  --method "Base Llama-3.2-3B-Instruct seed0" \
  --unlearn-num 50 --retain-num 1000 --seed "${SEED}" \
  --batch-size 8 --dtype bf16 --device-map single

python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${STAGE1}/checkpoint" \
  --mquake-path "${MQUAKE}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUT}/stage1_gagd_posthoc_eval.json" \
  --split-manifest "${OUT}/stage1_gagd_posthoc_split_manifest.json" \
  --method "SURE MQuAKE GA/GD Stage1 posthoc seed0" \
  --unlearn-num 50 --retain-num 1000 --seed "${SEED}" \
  --batch-size 8 --dtype bf16 --device-map single

for RANK in 64 128; do
  python scripts/mquake_zero_unlearn_official_eval.py \
    --model-dir "${SURE}/rank${RANK}/checkpoint" \
    --mquake-path "${MQUAKE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${OUT}/sure_r0_r${RANK}_eval.json" \
    --split-manifest "${OUT}/sure_r0_r${RANK}_split_manifest.json" \
    --method "SURE MQuAKE GA/GD rank0 restore-rank${RANK} seed0" \
    --unlearn-num 50 --retain-num 1000 --seed "${SEED}" \
    --batch-size 8 --dtype bf16 --device-map single

  python scripts/mquake_multihop_unlearning_eval.py \
    --model-dir "${SURE}/rank${RANK}/checkpoint" \
    --mquake-path "${MQUAKE}" \
    --split-manifest "${OUT}/sure_r0_r${RANK}_split_manifest.json" \
    --out "${OUT}/sure_r0_r${RANK}_multihop.json" \
    --mode both --batch-size 4 --dtype bf16 --device-map single
done

python - "${OUT}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
rows={}
for label,fn in (
    ("Base","base_eval.json"),
    ("Stage1_GAGD","stage1_gagd_posthoc_eval.json"),
    ("SURE_R0_R64","sure_r0_r64_eval.json"),
    ("SURE_R0_R128","sure_r0_r128_eval.json"),
):
    x=json.loads((root/fn).read_text())
    rows[label]={
        "F_Eff":x["forget"]["Eff"],
        "F_AtomicGen":x["forget"].get("AtomicGen"),
        "R_Eff":x["retain"]["Eff"],
        "R_AtomicGen":x["retain"].get("AtomicGen"),
        "PPL":x["forget_PPL"],
    }
for rank in (64,128):
    m=json.loads((root/f"sure_r0_r{rank}_multihop.json").read_text())
    rows[f"SURE_R0_R{rank}"]["multihop"]={
        mode:{
            "MHLeak_exact_any":v["MHLeak_exact_any"],
            "MHLeak_contains_any":v["MHLeak_contains_any"],
        } for mode,v in m["results"].items()
    }
(root/"local_comparison_seed0.json").write_text(json.dumps(rows,indent=2)+"\n")
print("===== LOCAL LOCKED EVALUATOR SEED0 =====")
for k,v in rows.items():
    print(k, "F-Eff",v["F_Eff"],"F-AtomicGen",v["F_AtomicGen"],"R-Eff",v["R_Eff"],"R-AtomicGen",v["R_AtomicGen"],"PPL",v["PPL"])
print("wrote",root/"local_comparison_seed0.json")
PY

# Keep the three temporary checkpoints for the dependent official ZeroUnlearn
# comparison job.  That job removes them only after official-evaluator scoring.
echo "===== SURE seed0 complete; temporary checkpoints retained for official comparison ====="
echo "OUT=${OUT}"
