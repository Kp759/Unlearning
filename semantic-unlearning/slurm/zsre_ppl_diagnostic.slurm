#!/usr/bin/env bash
#SBATCH --job-name=zsre-ppl-diag
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/zsre_ppl_diag_%j.out
#SBATCH --error=slurm_logs/zsre_ppl_diag_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
ROOT="${ZSRE_ROOT:-outputs/zsre_zerounlearn_forget_only_locked_3b}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python - <<'PY'
import json
import math
import os
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

ROOT = Path(os.environ.get("ZSRE_ROOT", "outputs/zsre_zerounlearn_forget_only_locked_3b"))
MODEL = os.environ.get(
    "MODEL_PATH",
    "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
)

print(f"ROOT={ROOT}")
print(f"MODEL={MODEL}")

tok = AutoTokenizer.from_pretrained(MODEL)

# Exact text used by the official PPL evaluator.
ds = load_from_disk("data/wikidata")
ppl_text = " ".join(ds["train"]["text"][:20])
ppl_ids = tok(ppl_text, add_special_tokens=False)["input_ids"]
ppl_freq = Counter(int(x) for x in ppl_ids)

rows = []

print(
    f"{'seed':>4} {'PPL':>9} {'sensRows':>8} "
    f"{'hitTypes':>8} {'hitTok':>8} "
    f"{'sensNLL':>10} {'neutralNLL':>10} {'grad':>9}"
)

for s in range(1, 11):
    seed = ROOT / f"seed{s}"

    ev = json.loads((seed / "official_eval_locked.json").read_text())
    ppl = float(ev["forget_PPL"])

    policy_path = (
        seed
        / "setting5e_forget_only"
        / "emb_lm_all_restore_post_training_true"
        / "post_training_row_policy.json"
    )
    policy = json.loads(policy_path.read_text())

    sensitive_ids = set(
        int(x) for x in policy["token_ids"]["unique_target_new"]
    )

    overlap = sensitive_ids & set(ppl_freq)
    hit_types = len(overlap)
    hit_tokens = sum(ppl_freq[x] for x in overlap)

    log_path = (
        seed
        / "setting5e_forget_only"
        / "emb_lm_all_restore_post_training_true"
        / "train_log.jsonl"
    )
    lines = [
        json.loads(x)
        for x in log_path.read_text().splitlines()
        if x.strip()
    ]
    last = lines[-1]

    sens_nll = float(last["sensitive_target_nll"])
    neutral_nll = float(last["neutral_target_nll"])
    grad = float(last["gradient_norm_before_clip"])

    rows.append({
        "seed": s,
        "ppl": ppl,
        "logppl": math.log(ppl),
        "nrows": len(sensitive_ids),
        "hit_types": hit_types,
        "hit_tokens": hit_tokens,
        "sens_nll": sens_nll,
        "neutral_nll": neutral_nll,
        "grad": grad,
        "overlap": overlap,
    })

    print(
        f"{s:4d} {ppl:9.2f} {len(sensitive_ids):8d} "
        f"{hit_types:8d} {hit_tokens:8d} "
        f"{sens_nll:10.4f} {neutral_nll:10.4f} "
        f"{grad:9.4f}"
    )


def corr(key):
    x = np.array([r[key] for r in rows], dtype=float)
    y = np.array([r["logppl"] for r in rows], dtype=float)
    if np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


print("\n===== CORRELATION WITH log(PPL) =====")
for k in [
    "nrows",
    "hit_types",
    "hit_tokens",
    "sens_nll",
    "neutral_nll",
    "grad",
]:
    print(f"{k:12s}: {corr(k):+.4f}")

print("\n===== PPL-CORPUS OVERLAP TOKENS =====")
for r in rows:
    vals = sorted(
        [(ppl_freq[t], t, tok.decode([t])) for t in r["overlap"]],
        reverse=True,
    )
    print(f"\nseed {r['seed']}  PPL={r['ppl']}")
    print(vals[:20])
PY
