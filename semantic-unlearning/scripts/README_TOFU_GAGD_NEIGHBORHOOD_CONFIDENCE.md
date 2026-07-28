# TOFU GA/GD + neighborhood-confidence comparison

This experiment is isolated from the older MCF/TOFU runners. It evaluates:

1. unchanged full-TOFU model;
2. optional original ZeroUnlearn;
3. retain-only retraining oracle;
4. full model, all answer tokens;
5. full model, selective answer tokens;
6. input embedding + LM head, all answer tokens;
7. input embedding + LM head, selective answer tokens;
8. TOFU Setting 5e restoration;
9. Setting 5e plus active forget-case repair;
10. Setting 5e plus active and neighborhood-confidence repair;
11. prompt-conditional isolated input repair for the extreme clean protocol.

The recommended extreme-target method is prompt-conditional isolated input
repair. The GA/GD chain remains available as a scientific baseline, but its
global vocabulary-row repair is not expected to satisfy the new retain budget.
A final method is accepted only when:

- clean forget answer probability is at most `0.00002`; and
- retain answer probability preserves at least `99.99998%` of the unchanged
  full-TOFU model on the identical evaluation sample.

## Run

From `semantic-unlearning/`:

```bash
bash scripts/run_tofu_prompt_conditional_input_repair.sh
```

The older nine-method GA/GD comparison is still available:

```bash
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

After those framework evaluations exist, run the original closed-form
ZeroUnlearn method on the identical TOFU protocol with:

```bash
bash scripts/run_zerounlearn_tofu.sh
```

On the scratch filesystem used by the 3B experiments:

```bash
MODEL_PATH=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/finetuned_model_3B_instruct \
FRAMEWORK_OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_gagd_targeted_2e5 \
OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/zerounlearn_tofu \
bash scripts/run_zerounlearn_tofu.sh
```

The ZeroUnlearn runner does not modify the vendored algorithm. Because
upstream ZeroUnlearn has no TOFU dataset adapter, it maps each clean TOFU
question to the method's subject slot, reproduces the exact chat-formatted
prompt used by `tofu_eval.py`, puts the clean answer in the sensitive
`target_true` slot, and uses tokenizer EOS as the neutral `target_new`. It
loads the full-TOFU checkpoint in BF16, applies the original edit in FP32 as in
the reviewed MCF runner, returns to BF16, and evaluates the edited model in
memory. The fixed run uses seed 42, all 200 `forget05` examples, and the same
1,000 sampled `retain95` examples as the framework.

Its primary artifacts are:

- `outputs/zerounlearn_tofu/evaluation/original_zerounlearn_summary.json`;
- `outputs/zerounlearn_tofu/zerounlearn_tofu_provenance.json`;
- `outputs/zerounlearn_tofu/comparison/comparison_tofu.md`.

The comparison includes all framework summaries found in
`FRAMEWORK_OUTPUT_ROOT/evaluation`. ZeroUnlearn is an optional table row, so
the existing GA/GD pipeline remains runnable before this separate baseline is
available.

For the scratch model path used by the 3B experiments:

```bash
MODEL_PATH=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/finetuned_model_3B_instruct \
OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_prompt_conditional_2e5 \
bash scripts/run_tofu_prompt_conditional_input_repair.sh
```

## Extreme-target prompt-conditional repair

The strict runner starts directly from the unchanged full-TOFU checkpoint. It:

- deterministically selects one protected-exclusive phrase from each of the
  200 clean forget questions;
- renames 200 dormant Llama-3 reserved token IDs without expanding the
  vocabulary or changing the softmax;
- verifies that token IDs are identical for all 1,000 retain answers and the
  complete real-author and world-fact sets;
- clones the tied LM head and freezes it;
- optimizes sparse deltas only for the 200 reserved input-embedding rows;
- re-scores every forget and retain answer after BF16 materialization;
- refuses to save unless every forget record is at or below `2e-5` and the
  retain/Base answer-probability ratio is at least `0.9999998`.

This construction makes the protected retain computation invariant: protected
prompts never contain a trigger ID, and no shared model row changes. It is also
explicitly evaluation-protocol-conditional. The generated `trigger_plan.json`
lists every phrase, and `repair_summary.json` sets
`semantic_generalization_claimed` to false. Report the full official TOFU
metrics and held-out/paraphrased behavior separately; do not describe a clean
question pass as general semantic unlearning.

The default protocol is seed 42, `forget05`, `retain95`, 200 forget examples,
and 1,000 retain examples. The four-setting source training uses 200 steps:
one pass over forget05 at batch size 1 and one pass over all 1,000 sampled
retain records at batch size 5. The retain loss weight of 5 compensates for
batch averaging, instead of giving each forget record roughly 20 times the
effective pressure of a retain record.
The four-setting front end also uses the same chat-template prompt and leading
plain-text prompt. TOFU GA/GD does not append EOS to the answer objective,
because `tofu_eval.py` does not score EOS.

Existing checkpoints can be reused:

```bash
RUN_FOUR_SETTINGS=0 bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

The independently reusable phase switches are:

- `RUN_SETTING5_RESTORE`
- `RUN_ACTIVE_REPAIR`
- `RUN_NEIGHBORHOOD_REPAIR`
- `RUN_RETAIN_ONLY_ORACLE`
- `RUN_EVAL`

For an enabled stage, set its switch to `0` only when its expected output
already exists. `RUN_NEIGHBORHOOD_REPAIR` is the exception: it is optional and
`0` by default because the requested final edit is the active-pair LM-head
repair.

## Setting 5e restoration

TOFU has no MCF `target_new`/`target_true` pair. Its 5e analogue unties the
output head and uses answer-token groups:

- unique forget LM-head rows keep the learned GA/GD update;
- rows shared with the protected corpus are restored completely by default;
- protected-only and unrelated LM-head rows return to the full-TOFU base;
- every input embedding row returns to the full-TOFU base.

The protected row corpus includes the complete paired retain split plus
`real_authors` and `world_facts`, not only the 1,000-record evaluation sample.
Answer rows are extracted at the exact answer positions and truncation boundary
used by `tofu_eval.py`, rather than by tokenizing answer strings in isolation.
Restoring the complete input matrix is required because this model begins with
tied input/output embeddings; otherwise an answer-row edit also changes prompt
representations everywhere that token occurs.

## Absolute active-case repair

For each clean forget record, `tofu_eval.py` reports:

```text
answer_probability = exp(-mean_answer_nll)
```

The active repair requires every record to reach an answer NLL of at least
`-log(2e-5)`, plus a BF16 safety buffer. It:

- selects rows only from initially failing forget answers;
- removes every row shared with protected utility answers;
- freezes the transformer and input embeddings;
- restricts each row delta to a low-rank basis of active answer-position hidden
  states;
- optimizes every forget record simultaneously so failures cannot migrate;
- protects all 1,000 protocol-matched retain answers plus deterministic
  real-author and world-fact answers against the original full-TOFU model;
- can project sparse LM-head updates away from utility hidden directions;
- enforces a `0.9999998` aggregate answer-probability ratio separately for
  retain, real-authors, and world-facts, matching the evaluator's aggregation;
- keeps the former per-example NLL ceilings behind the explicit
  `--utility-constraint-mode per-example` diagnostic setting;
- adds a hardest-forget hinge so a few high-probability failures are not
  diluted by the 200-record mean;
- ranks utility-safe candidates before candidates with stronger forgetting;
- refuses to save a normal candidate unless every materialized forget answer
  and every utility gate pass.

An LM-head vocabulary row is global, not intrinsically pair-specific. Here,
"active-pair repair" means that eligible rows come only from initially failing
pairs and their deltas are restricted to those pairs' hidden-state directions.
If every required row overlaps protected answers, the runner fails safely
instead of making a global destructive edit.

`--save-best-effort` exists only for diagnostics and is not used by the shell
runner.

## Optional neighborhood-confidence repair

The optional neighborhood stage starts from the target-qualified active checkpoint,
never directly from a weak GA/GD checkpoint. Its utility neighborhood contains:

- the paired retain split;
- `real_authors`;
- `world_facts`.

It restores their answer confidence toward the original full-TOFU model while:

- freezing the complete transformer;
- freezing input embeddings;
- changing only selected LM-head rows;
- excluding every row that occurs in sampled forget answers;
- keeping every sampled forget answer in a per-example NLL protection
  objective;
- optionally projecting the update away from forget hidden-state directions.

The stage rejects weak input checkpoints and rechecks the absolute `2e-5`
target after BF16 materialization.

Best-effort saving is disabled by default. The full `tofu_eval.py` evaluation
runs only after sparse candidate selection.

## Retain-only oracle

The oracle is not an unlearning method. It reads
`finetune_metadata.json` from the full-TOFU checkpoint, reloads the exact
pre-TOFU model revision, and retrains only on the complete paired retain split
using the recovered epoch, batch-size, and learning-rate settings.

To reuse a previously trained oracle:

```bash
RETAIN_ONLY_ORACLE_PATH=/path/to/retain95/checkpoint \
RUN_RETAIN_ONLY_ORACLE=0 \
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

Its forget truth-ratio distribution is reused as the KS reference for every
comparison row.

## Results

The fixed-order comparison is written to:

- `comparison/comparison_tofu.csv`
- `comparison/comparison_tofu.md`
- `comparison/comparison_tofu.json`

The table reports TOFU-native metrics:

- forget answer probability and forget ROUGE-L (lower is better);
- retain answer probability and ROUGE-L (higher is better);
- real-author and world-fact normalized answer probability and ROUGE-L
  (higher is better);
- truth-ratio metrics;
- KS p-value against the retain-only oracle;
- an explicit pass/fail column for the `0.00002` forgetting target.
- retain probability relative to Base;
- explicit retain and joint-target pass/fail columns.

Forget truth ratio is not a standalone monotonic score. Interpret it together
with the retain-only oracle KS comparison.

For quick smoke tests, set `N_FORGET_EVAL`, `N_RETAIN_EVAL`,
`N_REAL_AUTHORS_EVAL`, `N_WORLD_FACTS_EVAL`, and `N_PERTURBED_EVAL` to small
values. Do not use those truncated runs as final benchmark results.

The comparison generator exits nonzero when the selected final checkpoint
exceeds the forget target or falls below the `0.9999998` retain/Base ratio. This
prevents either ineffective forgetting or destructive utility loss from being
silently presented as success.

The retain target is relative, not an absolute answer probability of
`0.9999998`.
The tracked unchanged model has retain answer probability about `0.9980`, so
an absolute `0.9999998` requirement would demand improving beyond the starting
model rather than preserving it.
