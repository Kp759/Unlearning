#!/usr/bin/env bash
#SBATCH --job-name=mq-zul-base-s0
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mq_zul_base_s0_%j.out
#SBATCH --error=slurm_logs/mq_zul_base_s0_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning_mquake_sure_seed0/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/zerounlearn_official}"
ZU_DIR="${ZERO_UNLEARN_DIR:-/scratch/yl258/kp759/ZeroUnlearn_official}"
ZU_COMMIT="deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"
MODEL="${MODEL_PATH:?submit with MODEL_PATH=/path/to/Llama-3.2-3B-Instruct}"
MQUAKE="${MQUAKE_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/data/wikidata}"
SURE_OUT="${SURE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs/mquake_sure_zul50_r0_restore_seed0/seed0}"
OUT="${BASELINE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs/mquake_zerounlearn_public_seed0}"
ZU_CKPT="${OUT}/checkpoint"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUT}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf '===== PUBLIC-CODE ZEROUnlearn BASELINE seed0 =====\n'
printf 'host=%s job=%s\nZeroUnlearn=%s\nSURE_OUT=%s\nOUT=%s\n' \
  "$(hostname)" "${SLURM_JOB_ID}" "${ZU_DIR}" "${SURE_OUT}" "${OUT}"
python --version
python - <<'PY'
import torch, transformers, datasets, scipy, sklearn, einops, hydra, higher
print("torch",torch.__version__)
print("transformers",transformers.__version__)
print("datasets",datasets.__version__)
PY
nvidia-smi

test -x "${ENV_DIR}/bin/python"
test -d "${ZU_DIR}/.git"
test "$(git -C "${ZU_DIR}" rev-parse HEAD)" = "${ZU_COMMIT}"
test -d "${MODEL}"
test -f "${MQUAKE}"
test -d "${WIKIDATA_DIR}"
test -f "${SURE_OUT}/protocol/split_manifest.json"
test -f "${SURE_OUT}/base_eval.json"
test -f "${SURE_OUT}/sure_r0_r64_eval.json"
test -f "${SURE_OUT}/sure_r0_r128_eval.json"
test -d "${SURE_OUT}/sure_r0_restore/rank64/checkpoint"
test -d "${SURE_OUT}/sure_r0_restore/rank128/checkpoint"

# ZeroUnlearn's published code computes/caches Wikipedia KKT/VKT statistics
# under its own DATA_DIR.  Reuse the exact local HF dataset rather than permit a
# network download from a compute node.  Keep data/stats persistent for later
# seeds so the expensive statistics can be reused.
mkdir -p "${ZU_DIR}/data" "${ZU_DIR}/data/stats"
if [[ -e "${ZU_DIR}/data/wikidata" && ! -L "${ZU_DIR}/data/wikidata" ]]; then
  echo "Refusing to replace a real ZeroUnlearn data/wikidata directory" >&2
  exit 2
fi
ln -sfn "${WIKIDATA_DIR}" "${ZU_DIR}/data/wikidata"

rm -rf "${ZU_CKPT}"

# Call the authors' pinned ZeroUnlearn algorithm/hparams directly on EXACTLY
# the same 50/1000 raw-v2 instance IDs in SURE's manifest.  No unpublished
# mquake_data_saved_split.json is assumed.
python scripts/run_official_zerounlearn_seed0_save.py \
  --zero-unlearn-dir "${ZU_DIR}" \
  --model-path "${MODEL}" \
  --mquake-path "${MQUAKE}" \
  --split-manifest "${SURE_OUT}/protocol/split_manifest.json" \
  --save-checkpoint "${ZU_CKPT}" \
  --seed 0 --unlearn-num 50 --retain-num 1000 --edit-layer-nums 3

test -d "${ZU_CKPT}"
test -f "${OUT}/official_zerounlearn_seed0_manifest.json"

# Common evaluator: the public-code ZeroUnlearn model is scored under the exact
# same locked evaluator already used for Base and SURE.  This is the primary
# seed0 comparison: same model, same raw-v2 split, same evaluator.
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${ZU_CKPT}" \
  --mquake-path "${MQUAKE}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUT}/zerounlearn_eval.json" \
  --split-manifest "${OUT}/zerounlearn_eval_split_manifest.json" \
  --method "ZeroUnlearn public algorithm deff011; common locked evaluator seed0" \
  --unlearn-num 50 --retain-num 1000 --seed 0 \
  --batch-size 8 --dtype bf16 --device-map single

python scripts/mquake_multihop_unlearning_eval.py \
  --model-dir "${ZU_CKPT}" \
  --mquake-path "${MQUAKE}" \
  --split-manifest "${OUT}/zerounlearn_eval_split_manifest.json" \
  --out "${OUT}/zerounlearn_multihop.json" \
  --mode both --batch-size 4 --dtype bf16 --device-map single

# Fail closed if the final evaluator's split manifest is not the same locked
# source-index sample used by SURE.
python - "${SURE_OUT}/protocol/split_manifest.json" \
           "${OUT}/zerounlearn_eval_split_manifest.json" \
           "${OUT}/split_parity.json" <<'PY'
import json, pathlib, sys
a=json.loads(pathlib.Path(sys.argv[1]).read_text())
b=json.loads(pathlib.Path(sys.argv[2]).read_text())
def ids(x,key):
    s=x.get("sampling",x)
    for k in (key, key.replace("_source_indices","_indices")):
        if k in s: return [int(v) for v in s[k]]
    return None
af=ids(a,"forget_source_indices"); ar=ids(a,"retain_source_indices")
bf=ids(b,"forget_source_indices"); br=ids(b,"retain_source_indices")
p={"forget_match":af==bf,"retain_match":ar==br,"sure_forget":af,"eval_forget":bf}
pathlib.Path(sys.argv[3]).write_text(json.dumps(p,indent=2)+"\n")
print("split parity",p["forget_match"],p["retain_match"])
if bf is not None and af != bf: raise SystemExit("forget split mismatch")
if br is not None and ar != br: raise SystemExit("retain split mismatch")
PY

# Compact same-evaluator comparison.  AtomicGen and retain are extensions; Eff
# and PPL are the primary MQuAKE comparison columns.
python - "${SURE_OUT}" "${OUT}" <<'PY'
import json, pathlib, sys
sure=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
def ext(path):
 x=json.loads(path.read_text())
 return {
  "F_Eff":x["forget"]["Eff"],
  "F_AtomicGen":x["forget"].get("AtomicGen"),
  "R_Eff":x["retain"]["Eff"],
  "R_AtomicGen":x["retain"].get("AtomicGen"),
  "PPL":x["forget_PPL"],
 }
rows={
 "Base":ext(sure/"base_eval.json"),
 "Stage1_GAGD":ext(sure/"stage1_gagd_posthoc_eval.json"),
 "SURE_R0_R64":ext(sure/"sure_r0_r64_eval.json"),
 "SURE_R0_R128":ext(sure/"sure_r0_r128_eval.json"),
 "ZeroUnlearn_public":ext(out/"zerounlearn_eval.json"),
}
(out/"comparison_seed0.json").write_text(json.dumps(rows,indent=2)+"\n")
print("===== MQuAKE SEED0 SAME-EVALUATOR COMPARISON =====")
for k,v in rows.items():
 print(k,"F-Eff",v["F_Eff"],"F-AtomicGen",v["F_AtomicGen"],"R-Eff",v["R_Eff"],"R-AtomicGen",v["R_AtomicGen"],"PPL",v["PPL"])
PY

# Durable metrics/manifests stay; temporary trained models do not.
rm -rf "${ZU_CKPT}"
rm -rf "${SURE_OUT}/stage1_gagd/checkpoint"
rm -rf "${SURE_OUT}/sure_r0_restore/rank64/checkpoint"
rm -rf "${SURE_OUT}/sure_r0_restore/rank128/checkpoint"

echo "===== seed0 SURE vs ZeroUnlearn complete; checkpoints removed ====="
cat "${OUT}/comparison_seed0.json"
