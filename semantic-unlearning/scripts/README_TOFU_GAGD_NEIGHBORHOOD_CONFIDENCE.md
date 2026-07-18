# TOFU GA/GD + neighborhood-confidence comparison

This experiment is isolated from the existing MCF and TOFU scripts. It creates
five TOFU method checkpoints and evaluates those alongside the unchanged base
model:

1. full model, all answer tokens;
2. full model, selective answer tokens;
3. input embedding + LM head, all answer tokens;
4. input embedding + LM head, selective answer tokens;
5. sparse LM-head neighborhood-confidence repair of one selected GA/GD
   checkpoint.

The fifth setting defaults to repairing `emb_lm_all_tokens`. Set
`REPAIR_INPUT_MODE` to any of the four mode names to change its parent.

## Run

From `semantic-unlearning/`:

```bash
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

For the scratch model path used by the 3B experiments:

```bash
MODEL_PATH=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/finetuned_model_3B_instruct \
OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_gagd_neighborhood_confidence \
bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

The default protocol is seed 42, `forget05`, `retain95`, 200 forget examples,
and 1,000 retain examples. It runs every phase. Existing checkpoints can be
reused without retraining:

```bash
RUN_FOUR_SETTINGS=0 bash scripts/run_tofu_gagd_neighborhood_confidence.sh
```

Likewise, set `RUN_REPAIR=0` or `RUN_EVAL=0` only when the corresponding
outputs already exist.

## Fifth-setting safety policy

The TOFU dataset has no MCF `target_new`, `target_true`, or neighborhood
probability-difference metric. Therefore this runner does not call a TOFU
number `Spe`. Its neighborhood is a deterministic calibration sample from:

- the paired retain split;
- `real_authors`;
- `world_facts`.

It restores answer confidence toward the original TOFU-finetuned reference
while:

- freezing the complete transformer;
- freezing input embeddings;
- changing only selected LM-head rows;
- excluding every row that occurs in sampled forget answers;
- keeping every sampled forget answer in a per-example NLL protection
  objective;
- optionally projecting the update away from forget hidden-state directions.

By default, the best forget-safe sparse candidate is saved even when the row
and rank budget cannot meet every reference-confidence target. Pass
`--no-save-best-effort` directly to the Python repair runner when complete
target satisfaction must be a hard gate.

The full `tofu_eval.py` evaluation runs only after the sparse candidate is
selected and saved.

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
- optional KS p-value when `TOFU_REFERENCE_TRUTH_RATIOS` or
  `TOFU_REFERENCE_MODEL_DIR` is provided.

Forget truth ratio is not a standalone monotonic score. Interpret it together
with the KS comparison to an appropriate retain-model reference distribution.

For quick smoke tests, set `N_FORGET_EVAL`, `N_RETAIN_EVAL`,
`N_REAL_AUTHORS_EVAL`, `N_WORLD_FACTS_EVAL`, and `N_PERTURBED_EVAL` to small
values. Do not use those truncated runs as final benchmark results.
