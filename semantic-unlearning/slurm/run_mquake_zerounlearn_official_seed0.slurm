#!/usr/bin/env bash
#SBATCH --job-name=mq-zul-base-s0
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mq_zul_base_s0_%j.out
#SBATCH --error=slurm_logs/mq_zul_base_s0_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning_mquake_sure_seed0/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
ZU_DIR="${ZERO_UNLEARN_DIR:-/scratch/yl258/kp759/ZeroUnlearn_official}"
ZU_COMMIT="deff011c3df367b700b9ad0aa0f5d7aad0cca9b9"
MODEL="${MODEL_PATH:?submit with MODEL_PATH=/path/to/Llama-3.2-3B-Instruct}"
MQUAKE="${MQUAKE_PATH:-${PROJECT_DIR}/data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-${PROJECT_DIR}/data/wikidata}"
SURE_OUT="${SURE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs/mquake_sure_zul50_r0_restore_seed0/seed0}"
OUT="${BASELINE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs/mquake_zerounlearn_official_seed0}"
MODEL_LINK_ROOT="${OUT}/model_links"
ZU_CKPT="${OUT}/checkpoint"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUT}" "${MODEL_LINK_ROOT}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf '===== PINNED OFFICIAL ZEROUnlearn MQuAKE seed0 =====\n'
printf 'host=%s job=%s\nZeroUnlearn=%s\nSURE_OUT=%s\n' "$(hostname)" "${SLURM_JOB_ID}" "${ZU_DIR}" "${SURE_OUT}"
python --version
nvidia-smi

test -d "${ZU_DIR}/.git"
test "$(git -C "${ZU_DIR}" rev-parse HEAD)" = "${ZU_COMMIT}"
test -d "${MODEL}"
test -f "${MQUAKE}"
test -d "${WIKIDATA_DIR}"
test -d "${SURE_OUT}/sure_r0_restore/rank64/checkpoint"
test -d "${SURE_OUT}/sure_r0_restore/rank128/checkpoint"
test -f "${SURE_OUT}/protocol/split_manifest.json"

# The authors' evaluator joins model_path_dir/Llama-3.2-3B-Instruct.
BASE_PARENT="${MODEL_LINK_ROOT}/base"
mkdir -p "${BASE_PARENT}"
ln -sfn "${MODEL}" "${BASE_PARENT}/Llama-3.2-3B-Instruct"

# Force the pinned official loader to adapt the exact local v2 source, never a
# stale saved split.  Remove cached seed0 sample files so this run regenerates
# them using the authors' random.seed(0) -> forget sample -> retain sample path.
mkdir -p "${ZU_DIR}/data"
cp -f "${MQUAKE}" "${ZU_DIR}/data/MQuAKE-CF-3k-v2.json"
rm -f "${ZU_DIR}/data/mquake_data_saved_split.json"
rm -f "${ZU_DIR}/data/mquake_unlearn_ds_num_50_seed0.json"
rm -f "${ZU_DIR}/data/mquake_retain_ds_num_1000_seed0.json"
rm -rf "${ZU_DIR}/results/ZeroUnlearn_SURECompareSeed0"
rm -rf "${ZU_CKPT}"

# Run the authors' exact ZeroUnlearn algorithm/hparams.  The wrapper only saves
# the returned edited model; it does not change the algorithm.
python scripts/run_official_zerounlearn_seed0_save.py \
  --zero-unlearn-dir "${ZU_DIR}" \
  --model-parent "${BASE_PARENT}" \
  --save-checkpoint "${ZU_CKPT}" \
  --seed 0 --unlearn-num 50 --retain-num 1000 --edit-layer-nums 3 \
  --dir-name ZeroUnlearn_SURECompareSeed0

test -d "${ZU_CKPT}"
test -f "${ZU_DIR}/data/mquake_unlearn_ds_num_50_seed0.json"
test -f "${ZU_DIR}/data/mquake_retain_ds_num_1000_seed0.json"

# Hard parity check: the official authors' seed0 random sample must be exactly
# the same 50/1000 instance IDs as SURE's locked manifest.  Abort comparison if
# even one ID/order differs.
python - "${SURE_OUT}/protocol/split_manifest.json" \
           "${ZU_DIR}/data/mquake_unlearn_ds_num_50_seed0.json" \
           "${ZU_DIR}/data/mquake_retain_ds_num_1000_seed0.json" \
           "${OUT}/split_parity.json" <<'PY'
import json, pathlib, sys
manifest=json.loads(pathlib.Path(sys.argv[1]).read_text())
zu_f=json.loads(pathlib.Path(sys.argv[2]).read_text())
zu_r=json.loads(pathlib.Path(sys.argv[3]).read_text())
ours_f=[int(x) for x in manifest["sampling"]["forget_source_indices"]]
ours_r=[int(x) for x in manifest["sampling"]["retain_source_indices"]]
zu_f_ids=[int(x["case_id"]) for x in zu_f]
zu_r_ids=[int(x["case_id"]) for x in zu_r]
payload={
  "forget_exact_order_match":ours_f==zu_f_ids,
  "retain_exact_order_match":ours_r==zu_r_ids,
  "forget_set_match":set(ours_f)==set(zu_f_ids),
  "retain_set_match":set(ours_r)==set(zu_r_ids),
  "our_forget":ours_f,"zerounlearn_forget":zu_f_ids,
  "our_retain":ours_r,"zerounlearn_retain":zu_r_ids,
}
pathlib.Path(sys.argv[4]).write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({k:v for k,v in payload.items() if not isinstance(v,list)},indent=2))
if not all(payload[k] for k in ("forget_exact_order_match","retain_exact_order_match")):
    raise SystemExit("FAIL: SURE and official ZeroUnlearn seed0 splits are not identical")
PY

# Evaluate the saved official ZeroUnlearn checkpoint under OUR locked evaluator,
# giving the same Eff/AtomicGen/1000-retain/PPL metrics used for SURE.
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${ZU_CKPT}" \
  --mquake-path "${MQUAKE}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUT}/zerounlearn_local_eval.json" \
  --split-manifest "${OUT}/zerounlearn_local_split_manifest.json" \
  --method "Official ZeroUnlearn pinned deff011 seed0; locked local evaluator" \
  --unlearn-num 50 --retain-num 1000 --seed 0 \
  --batch-size 8 --dtype bf16 --device-map single

# Also score frozen SURE R64/R128 under the AUTHORS' exact MQuAKE evaluator.
# BASE means 'do not edit this already-frozen checkpoint'.  The dataset cache
# from the ZeroUnlearn run is intentionally reused, guaranteeing identical IDs.
for RANK in 64 128; do
  PARENT="${MODEL_LINK_ROOT}/sure_r${RANK}"
  mkdir -p "${PARENT}"
  ln -sfn "${SURE_OUT}/sure_r0_restore/rank${RANK}/checkpoint" \
          "${PARENT}/Llama-3.2-3B-Instruct"
  rm -rf "${ZU_DIR}/results/BASE_SURE_R${RANK}_Seed0"
  (
    cd "${ZU_DIR}"
    python experiments/evaluate.py \
      --alg_name BASE \
      --ablation "_SURE_R${RANK}_Seed0" \
      --model_name Llama-3.2-3B-Instruct \
      --hparams_fname Llama-3.2-3B-Instruct.json \
      --ds_name mquake \
      --ratio_or_num \
      --unlearn_num 50 \
      --retain_num 1000 \
      --model_path_dir "${PARENT}" \
      --edit_layer_nums 3 \
      --seed 0 \
      --eval_retain \
      --skip_generation_tests
  )
done

# Copy authors' summary artifacts into one durable comparison root.  These are
# direct-rewrite metrics; our local evaluator supplies AtomicGen/PPL extensions.
python - "${ZU_DIR}" "${OUT}" <<'PY'
import json, pathlib, shutil, sys
zu=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
blocks={
  "ZeroUnlearn_official":"ZeroUnlearn_SURECompareSeed0",
  "SURE_R64_official":"BASE_SURE_R64_Seed0",
  "SURE_R128_official":"BASE_SURE_R128_Seed0",
}
index={}
for label,dname in blocks.items():
    root=zu/"results"/dname
    runs=sorted([p for p in root.glob("*") if p.is_dir()])
    if len(runs)!=1:
        raise SystemExit(f"expected exactly one run under {root}, got {runs}")
    run=runs[0]
    copied={}
    for name in ("forget_summarize_results.json","retain_summarize_results.json"):
        src=run/name
        if src.exists():
            dst=out/f"{label}_{name}"
            shutil.copy2(src,dst); copied[name]=str(dst)
    index[label]={"official_run_dir":str(run),"copied":copied}
(out/"official_evaluator_index.json").write_text(json.dumps(index,indent=2)+"\n")
print(json.dumps(index,indent=2))
PY

# One compact local-evaluator table: Base + SURE R64/R128 + official ZeroUnlearn.
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
 "SURE_R0_R64":ext(sure/"sure_r0_r64_eval.json"),
 "SURE_R0_R128":ext(sure/"sure_r0_r128_eval.json"),
 "ZeroUnlearn_official":ext(out/"zerounlearn_local_eval.json"),
}
(out/"local_evaluator_comparison_seed0.json").write_text(json.dumps(rows,indent=2)+"\n")
print("===== SAME LOCKED LOCAL EVALUATOR =====")
for k,v in rows.items():
 print(k,"F-Eff",v["F_Eff"],"F-AtomicGen",v["F_AtomicGen"],"R-Eff",v["R_Eff"],"R-AtomicGen",v["R_AtomicGen"],"PPL",v["PPL"])
PY

# All requested evaluations are durable now; trained-model directories are not.
rm -rf "${ZU_CKPT}"
rm -rf "${SURE_OUT}/stage1_gagd/checkpoint"
rm -rf "${SURE_OUT}/sure_r0_restore/rank64/checkpoint"
rm -rf "${SURE_OUT}/sure_r0_restore/rank128/checkpoint"
rm -rf "${MODEL_LINK_ROOT}"

echo "===== seed0 apples-to-apples comparison complete; checkpoints removed ====="
echo "SURE_OUT=${SURE_OUT}"
echo "BASELINE_OUT=${OUT}"
cat "${OUT}/local_evaluator_comparison_seed0.json"
