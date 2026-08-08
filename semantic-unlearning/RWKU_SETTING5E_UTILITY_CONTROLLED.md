# RWKU Setting 5e-UC with protected row-wise repair

This directory contains a new RWKU target-only method extension:

> **Setting 5e-UC + protected row-wise LM-head repair**

Its protocol status is
`rwku_target_only_utility_controlled_setting5e_method_extension`.
It is not unchanged Setting 5e, and it does not relabel or modify the completed
600-step `rwku-s5e-repair-sk-v3atomic-seed0-v1` experiment.

The implementation is isolated in:

- `scripts/rwku_setting5e_utility_controlled.py`
- `scripts/rwku_rowwise_active_repair.py`
- `scripts/validate_rwku_setting5e_utility_run.py`

No legacy RWKU, MCF, ZsRE, TOFU, representation-unlearning, ZeroUnlearn, or
original Setting 5e entrypoint is changed.

## Protocol boundary

The state sequence is:

```text
PREPARED
  -> TRAINING
  -> CANDIDATES_EVALUATED
  -> CHECKPOINT_FROZEN
  -> OFFICIAL_EVALUATION_OPENED
  -> EVALUATION_COMPLETE
```

If no checkpoint passes every fixed gate, the terminal state is
`NO_FEASIBLE_CANDIDATE`. In that state the run has a complete candidate report
but no selected checkpoint and no checkpoint receipt.

Before `CHECKPOINT_FROZEN`, only the generated bundle and generator receipt,
target-independent MCF partitions, matched optimization/protection artifacts,
the target-independent protection vocabulary, Wikidata proxy-PPL text, and the
pinned Base model/tokenizer are visible. Official Level 1/2/3, MIA, neighbor,
utility, fluency, and prior-output paths are rejected.

The official descriptor created by `prepare` contains pinned hashes and row
counts only. `evaluate` atomically opens the checkpoint receipt before loading
any official rows. Evaluation is read-only.

## Training and selection

The transformer is frozen. A tied LM head is cloned into a distinct output
matrix, and logits are checked for bitwise equality before training starts.
Only declared subject/alias input rows and safe content-bearing sensitive-answer
output rows receive gradients. Hooks zero all other gradients, and immutable
Base copies restore every nonselected row after every optimizer step.

The default objective is:

```text
2.0  * forget_margin_loss
+ 4.0  * retain_answer_ce
+ 10.0 * top_k_plus_tail_base_teacher_kl
+ 20.0 * protected_answer_hinge
+ 1e-4 * selected_row_delta_l2
```

The default input-row LR is `5e-6`; the output-row LR is `2e-5`. Teacher
probabilities are detached and KL arithmetic is FP32.

For `K` facts, candidates are evaluated after 2, 4, 6, 8, 10, 12, 15, and 20
balanced exposures per fact. For 13 facts this is 26, 52, 78, 104, 130, 156,
195, and 260 steps. Candidate scales 0.25, 0.50, 0.75, and 1.00 are always
materialized as `Base + scale * checkpoint_delta`; scaling is never cumulative.

Candidate gates are fixed in code and bound into the configuration manifest,
candidate report, and checkpoint receipt. No gate is relaxed automatically.
Eligible candidates are ordered by smallest selected-row delta norm, earliest
checkpoint, then smallest interpolation scale.

## Row-wise repair

Repair active points have the source label
`target_generated_entity_fact_views`. Every point binds its fact, view, prompt
style, answer alias, source-record hash, and generated-bundle hash. Official
Level 1/2/3 and evaluation artifacts are mechanically rejected.

Punctuation, whitespace, numeric-only, single-character, special,
high-retain-frequency, and protected-overlap output rows remain unsupported.
Each eligible row receives an independent scale selected from:

```text
1, .875, .75, .625, .5, .375, .25, .1875,
.125, .09375, .0625, .046875, .03125,
.015625, .0078125, 0
```

Every trial runs the complete disjoint protection bank under no gradient. The
combined edit is checked again, with a scale-back pass if interaction effects
violate a gate. The all-zero repair is mandatory, and a failed repair is never
reported as selected.

## Development seed-0 commands

Run from `/scratch/yl258/kp759/Unlearning/semantic-unlearning`. These commands
use the existing frozen v3 atomic corpus and create a new output directory.

```bash
MODEL=/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
CORPUS=outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1
RUN_ID=rwku-s5e-uc-rowwise-sk-v3atomic-seed0-v1
OUT=outputs/rwku_setting5e_utility_controlled
```

Prepare:

```bash
python scripts/rwku_setting5e_utility_controlled.py \
  --stage prepare \
  --experiment-id "${RUN_ID}" \
  --seed 0 \
  --model-path "${MODEL}" \
  --model-revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --generated-entity-fact-bundle "${CORPUS}/generated_training_bundle.json" \
  --generator-receipt "${CORPUS}/generator_receipt.json" \
  --output-root "${OUT}" \
  --data-root data/rwku \
  --wikidata-dir data/wikidata \
  --dtype bf16 \
  --development \
  --no-download
```

Build disjoint protection partitions:

```bash
python scripts/rwku_setting5e_utility_controlled.py \
  --stage protection \
  --experiment-id "${RUN_ID}" \
  --seed 0 \
  --model-path "${MODEL}" \
  --model-revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --generated-entity-fact-bundle "${CORPUS}/generated_training_bundle.json" \
  --generator-receipt "${CORPUS}/generator_receipt.json" \
  --output-root "${OUT}" \
  --data-root data/rwku \
  --wikidata-dir data/wikidata \
  --mcf-path data/multi_counterfact.json \
  --protection-source data/multi_counterfact.json \
  --protection-vocabulary config/rwku/protection_vocabulary_v1.json \
  --dtype bf16 \
  --development \
  --no-download
```

Train, select, and repair:

```bash
python scripts/rwku_setting5e_utility_controlled.py \
  --stage train \
  --experiment-id "${RUN_ID}" \
  --seed 0 \
  --model-path "${MODEL}" \
  --model-revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --generated-entity-fact-bundle "${CORPUS}/generated_training_bundle.json" \
  --generator-receipt "${CORPUS}/generator_receipt.json" \
  --output-root "${OUT}" \
  --data-root data/rwku \
  --wikidata-dir data/wikidata \
  --mcf-path data/multi_counterfact.json \
  --dtype bf16 \
  --development \
  --no-download
```

Evaluate only after `CHECKPOINT_FROZEN`:

```bash
python scripts/rwku_setting5e_utility_controlled.py \
  --stage evaluate \
  --experiment-id "${RUN_ID}" \
  --seed 0 \
  --model-path "${MODEL}" \
  --model-revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --generated-entity-fact-bundle "${CORPUS}/generated_training_bundle.json" \
  --generator-receipt "${CORPUS}/generator_receipt.json" \
  --output-root "${OUT}" \
  --data-root data/rwku \
  --wikidata-dir data/wikidata \
  --dtype bf16 \
  --eval-batch-size 4 \
  --development \
  --no-download
```

Validate without changing the run:

```bash
python scripts/validate_rwku_setting5e_utility_run.py \
  --run-dir "${OUT}/${RUN_ID}"
```

## Untouched-seed confirmatory template

Replace `SEED`, `CORPUS_DIR`, and `RUN_ID` with an untouched target. The frozen
seed-0 development manifest fixes every optimization parameter and threshold.
Use the same four stages above, replacing `--development` with:

```bash
--confirmatory \
--frozen-development-config \
  outputs/rwku_setting5e_utility_controlled/rwku-s5e-uc-rowwise-sk-v3atomic-seed0-v1/configuration_manifest.json
```

The prepare stage fails closed if any confirmatory configuration or threshold
differs from that manifest. Seed 0 cannot be invoked as confirmatory.

## Strict evaluation JSON

The final result stores Base, selected pre-repair Setting 5e-UC, and selected
post-repair results separately. Complete official RWKU evaluation is delegated
to `rwku_eval.evaluate_rwku`, including forced-prefix, aliases, multiple choice,
open-ended recovery, and the frozen Base-head probe. Additional Level 1/2/3
records include generation recovery, paper-code-compatible Eff, literal
full-sequence answer probability, and first-token probability.

NaN and infinity are converted to JSON `null`, never zero. Every replacement
is recorded by RFC-6901 path and original classification before strict
`allow_nan=false` serialization.
