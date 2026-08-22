# Reproduction commands

These commands reproduce the v3.2 training protocol from the frozen v3.2 commit and, separately, the already-opened held-out diagnostic.

## 1. Frozen v3.2 training

```bash
source /opt/pytorch/bin/activate

cd /home/ec2-user/workspace/Unlearning
git fetch origin

git worktree add --detach \
  /home/ec2-user/workspace/rwku-hidden-v32-kl \
  dc30b024ca092583f40a7e2ee2b15f236e9f449d

cd /home/ec2-user/workspace/rwku-hidden-v32-kl/semantic-unlearning

python -m py_compile scripts/rwku_sure_hidden_direction_v32_kl_w1k.py
bash -n scripts/run_rwku_sure_hidden_direction_v32_kl_w1k.sh

MODEL=/home/ec2-user/models/Llama-3.2-3B-Instruct
CORPUS=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1
REAL_WIKI=/home/ec2-user/workspace/Unlearning/semantic-unlearning/data/wikipedia_sure_100020
SOURCE_RUN=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_h_w1k_ab12969_fullheadfix/rwku-h-w1k-stephen-king-atomic-seed0-v1

export RWKU_H_W1K_UTILITY_CACHE=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/sure_wikipedia_stats/rwku_h_w1k_ab12969/Llama-3.2-3B-Instruct_rwku_stephen_king_excluded_docs1000_candidates100000_v1.pt
export RWKU_HD_V32_KL_W1K_OUTPUT_ROOT=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_hd_v32_kl_w1k

set -o pipefail
bash scripts/run_rwku_sure_hidden_direction_v32_kl_w1k.sh \
  "$MODEL" \
  "$CORPUS" \
  "$REAL_WIKI" \
  "$SOURCE_RUN" \
  2>&1 | tee /home/ec2-user/workspace/Unlearning/semantic-unlearning/rwku_hidden_direction_v32_kl_seed0.log
rc=${PIPESTATUS[0]}
echo "hidden-direction v3.2 KL rc=$rc"
```

Expected protocol behavior for the recorded run: training succeeds, but final selection exits fail-closed with `rc=1` because no selected physical candidate satisfies every gate, specifically the 1% representation-norm limit.

## 2. Held-out utility diagnostic

The held-out utility set is already opened. This diagnostic is for reproduction/reporting only, not for tuning.

```bash
source /opt/pytorch/bin/activate

cd /home/ec2-user/workspace/Unlearning
git fetch origin

git worktree add --detach \
  /home/ec2-user/workspace/rwku-v32-heldout-diag \
  b228b48fdb4dc3fb1f161536bd098915a41746c5

cd /home/ec2-user/workspace/rwku-v32-heldout-diag/semantic-unlearning

python -m py_compile scripts/rwku_v32_heldout_utility_diagnostic.py
bash -n scripts/run_rwku_v32_heldout_utility_diagnostic.sh

MODEL=/home/ec2-user/models/Llama-3.2-3B-Instruct
CORPUS=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1
REAL_WIKI=/home/ec2-user/workspace/Unlearning/semantic-unlearning/data/wikipedia_sure_100020
SOURCE_RUN=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_h_w1k_ab12969_fullheadfix/rwku-h-w1k-stephen-king-atomic-seed0-v1

export RWKU_H_W1K_UTILITY_CACHE=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/sure_wikipedia_stats/rwku_h_w1k_ab12969/Llama-3.2-3B-Instruct_rwku_stephen_king_excluded_docs1000_candidates100000_v1.pt
export RWKU_V32_HELDOUT_DIAG_OUTPUT_ROOT=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_v32_heldout_diag

set -o pipefail
bash scripts/run_rwku_v32_heldout_utility_diagnostic.sh \
  "$MODEL" \
  "$CORPUS" \
  "$REAL_WIKI" \
  "$SOURCE_RUN" \
  2>&1 | tee /home/ec2-user/workspace/Unlearning/semantic-unlearning/rwku_v32_heldout_utility_diagnostic.log
rc=${PIPESTATUS[0]}
echo "v3.2 held-out diagnostic rc=$rc"
```

Expected recorded diagnostic:

```text
candidate scale: 1.000000
true relative Frobenius: 0.014117
held-out KL mean/p95/max: 0.000386 / 0.001657 / 0.036320
No checkpoint was accepted or frozen.
rc=0
```

## 3. Inspect the diagnostic JSON

```bash
cat /home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_v32_heldout_diag/rwku-h-w1k-stephen-king-hidden-direction-seed0-v32-kl/heldout_utility_diagnostic.json
```
