# Fix5o — data-only augmentation of the exact-name target-local router

Fix5o freezes the successful Fix5k/Fix5n-v3 architecture and tests one variable:
**router training coverage**.

## What changes

Two fresh linear heads are trained under the same frozen-Llama mean-pooled exact-name
target-local representation:

1. `baseline_exact_name` — original semantic fit rows only.
2. `augmented_exact_name` — the same rows plus fit-only relation formulations and
   same-subject/different-relation semantic contrasts.

The model, tokenizer, selector, pooling, linear-head architecture, class inventory,
optimizer, calibration/preservation manifests, decoder, token supports, and `-12`
penalty are unchanged.

Official Seed-1 MCF paraphrases are not used to construct augmentation, train either
head, choose eta, filter rows, or select the model. They remain previously inspected
development evidence only.

## Augmentation families

Fit-only families:

- `aug_relation_report`
- `aug_relation_focus`
- `aug_relation_identify`
- `aug_relation_value`

Held-out authored probe families:

- `aug_holdout_relation_answer`
- `aug_holdout_relation_request`

The fit and held-out family sets are disjoint. Augmentation uses only relation labels
from `scripts/mcf_relation_contracts_fix5.json` and training-visible subjects already
present in the original semantic fit manifest. No answer values are used.

Same-subject alternate-relation examples retain the alternate relation as their class;
they are never relabeled `NONE` simply because that subject–relation pair is outside
the forget bank.

## Leakage guards

After actual target-local preprocessing:

- conflicting duplicate selected texts fail closed;
- exact overlap between augmented fit and calibration/validation/held-out probe is
  removed;
- near overlap at token Jaccard >= `0.90` is removed by default and audited;
- filtering must preserve at least one augmentation row for every relation that had
  candidates, otherwise the run stops.

## Fetch on AWS

From `semantic-unlearning`:

```bash
git fetch origin

for p in \
  scripts/mcf_target_local_augmented_relation_router_fix5o_seed1.py \
  scripts/mcf_target_local_augmented_relation_router_fix5o_v2_seed1.py \
  scripts/run_mcf_target_local_augmented_relation_router_fix5o_seed1.sh \
  tests/test_target_local_augmented_relation_router_fix5o.py \
  README_target_local_augmented_relation_router_fix5o.md
do
  git show origin/feat/retain-anchored-context-quotient-head:semantic-unlearning/$p > "$p"
done
```

## CPU checks first

```bash
python -m py_compile \
  scripts/mcf_target_local_augmented_relation_router_fix5o_seed1.py \
  scripts/mcf_target_local_augmented_relation_router_fix5o_v2_seed1.py

bash -n scripts/run_mcf_target_local_augmented_relation_router_fix5o_seed1.sh

python -m pytest -q tests/test_target_local_augmented_relation_router_fix5o.py
```

## Required paths

Reuse the same completed Fix5f directory used by Fix5k:

```bash
export TARGET_REPRESENTATION_OUT_DIR="<completed Fix5f output directory>"
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export MCF_PATH="/home/ec2-user/workspace/Unlearning/semantic-unlearning/data/multi_counterfact.json"
export FIX5O_OUT_DIR="$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_augmented_relation_router_fix5o"
```

If you previously exported `TARGET_REPRESENTATION_OUT_DIR` for Fix5k, reuse that exact
value. Do not point this at Fix5k/Fix5n output directories.

## Run

```bash
set -o pipefail
bash scripts/run_mcf_target_local_augmented_relation_router_fix5o_seed1.sh \
  2>&1 | tee seed1_target_local_augmented_relation_router_fix5o.log
```

The launcher archives a previous Fix5o output directory instead of overwriting it.

## What to compare

Primary preservation/validation gate for the augmented head remains the existing
Fix5k pilot contract:

- calibration status `ACCEPTABLE_OPERATING_POINT`;
- validation relation accuracy >= 70%;
- correct forbidden binding acceptance >= 60%;
- wrong forbidden binding acceptance <= 2%;
- route-level permitted false activation <= 2%;
- candidate-present permitted false activation <= 2%;
- whole-query permitted false activation <= 2%;
- mixed permitted-companion false activation <= 2%;
- every declared hard-negative family <= 2%.

The authored held-out formulation probe is semantic recognition evidence, not a policy
benchmark.

After both heads and their own etas are frozen, report the previously inspected
official Seed-1 paraphrase development metrics:

- relation accuracy;
- correct forbidden-binding acceptance.

Current historical baseline is approximately 47% relation accuracy and 25% correct
forbidden acceptance. A 60%+ correct-accept rate is a development target, not a fresh
independent-test threshold.

## Decision

Continue only if the augmented head:

1. passes the same original preservation/validation pilot;
2. preserves the leakage guards;
3. improves unfamiliar-wording recognition on the authored held-out probe; and
4. improves the official Seed-1 paraphrase development evidence without expanding
   permitted activation.

If it passes, freeze the new head and eta, then create a **new route manifest** for the
same fixed query bank before evaluating with the unchanged Fix5n-v3 decoder. The
current Fix5n-v3 records contain old saved routes and therefore cannot test a new
router by themselves.

No stronger `-12`, quotient, output-controller change, or Seeds 2–10 should be mixed
into Fix5o.
