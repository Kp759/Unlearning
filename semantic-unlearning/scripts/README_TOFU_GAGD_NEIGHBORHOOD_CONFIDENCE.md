# TOFU GA/GD + neighborhood-confidence comparison

This experiment is isolated from the older MCF/TOFU runners. It evaluates:

1. unchanged full-TOFU model;
2. retain-only retraining oracle;
3. full model, all answer tokens;
4. full model, selective answer tokens;
5. input embedding + LM head, all answer tokens;
6. input embedding + LM head, selective answer tokens;
7. TOFU Setting 5e restoration;
8. Setting 5e plus active forget-case repair;
9. Setting 5e plus active and neighborhood-confidence repair.

The final method is accepted only when its clean forget answer probability is
at most `0.00002`.

## Run

From `semantic-unlearning/`:

```bash
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

For the scratch model path used by the 3B experiments:

```bash
MODEL_PATH=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/finetuned_model_3B_instruct \
OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_gagd_targeted_2e5 \
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

The default protocol is seed 42, `forget05`, `retain95`, 200 forget examples,
and 1,000 retain examples. Every phase runs by default. The four-setting budget
is now 1,000 steps, with a separate `2e-4` embedding/LM-head learning rate;
the previous 100-step `1e-5` run barely changed the model.
The four-setting front end also uses the same chat-template prompt and leading
answer space as `tofu_eval.py`; the older generic TOFU loader used a different
plain-text prompt.

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

Set a switch to `0` only when its expected output already exists.

## Setting 5e restoration

TOFU has no MCF `target_new`/`target_true` pair. Its 5e analogue uses answer
token groups:

- unique forget rows keep the learned GA/GD update;
- forget/retain overlap rows keep 25% of the learned update by default;
- retain-only and unrelated rows return to the full-TOFU base model.

The policy is applied after all-token embedding/LM-head GA/GD to both input
embeddings and the LM head.

## Absolute active-case repair

For each clean forget record, `tofu_eval.py` reports:

```text
answer_probability = exp(-mean_answer_nll)
```

The active repair requires every record to reach an answer NLL of at least
`-log(2e-5)`, plus a BF16 safety buffer. It:

- selects rows only from initially failing forget answers;
- freezes the transformer and input embeddings;
- optimizes every forget record simultaneously so failures cannot migrate;
- protects deterministic retain, real-author, and world-fact calibration
  answers with per-example NLL ceilings;
- can project sparse LM-head updates away from utility hidden directions;
- refuses to save a normal candidate unless every materialized forget answer
  passes the hard target.

`--save-best-effort` exists only for diagnostics and is not used by the shell
runner.

## Neighborhood-confidence repair

The neighborhood stage starts from the target-qualified active checkpoint,
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

The full `tofu_eval.py` evaluation runs only after the sparse candidate is
selected and saved.

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

Forget truth ratio is not a standalone monotonic score. Interpret it together
with the retain-only oracle KS comparison.

For quick smoke tests, set `N_FORGET_EVAL`, `N_RETAIN_EVAL`,
`N_REAL_AUTHORS_EVAL`, `N_WORLD_FACTS_EVAL`, and `N_PERTURBED_EVAL` to small
values. Do not use those truncated runs as final benchmark results.

The comparison generator exits nonzero when the final neighborhood-repaired
checkpoint exceeds the forgetting target. This prevents a weak run from being
silently presented as the selected method.
