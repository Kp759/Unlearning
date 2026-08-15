# TOFU ZeroUnlearn-style locked protocol

This track is a data-access-controlled TOFU portability experiment for the
project's Setting-5e + sparse LM-head repair method.  It is separate from the
existing native/full-utility TOFU F01/F05/F10 track and does not replace it.

## Starting model

Unlearning starts from the project-selected **Full-TOFU** Llama-3.2-3B model,
not from raw Llama:

```text
outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5
```

Project registry metadata records this checkpoint as the selected epoch-5 model
from the LR `4e-5` TOFU fine-tuning run.  The runner fails if the checkpoint is
missing; it never silently substitutes a raw model.

## Per-seed data roles

Seeds are `1..10`.  For every seed, fresh `random.Random(seed).sample` calls
select 50 rows from `forget05` and 1,000 rows from `retain95`, matching the
sampling behavior of `tofu_eval.subset_samples` on those primary splits.

### Visible before final evaluation

Only the selected 50 `forget05` rows are visible, with exactly:

- `question`
- `answer`
- `_source_index` (provenance only)

Stage 1 and Stage 2 each see those same 50 direct QAs.  They see **zero**
retain95 rows, paraphrased questions, perturbed answers, real-authors rows, or
world-facts rows.

### Final-evaluation only

After the Stage-2 checkpoint is frozen, evaluation may access:

1. the selected 50 direct forget QAs;
2. their 50 benchmark-provided `paraphrased_question` / `paraphrased_answer`
   pairs;
3. the remaining 150 `forget05` QAs never exposed to unlearning;
4. paraphrases of those remaining 150 QAs;
5. 1,000 sampled `retain95` QAs;
6. standard TOFU real-authors/world-facts utility data;
7. perturbed answers used by truth-ratio evaluation.

The primary generalization result is the 50 held-out paraphrased questions for
the same deletion QAs.  The remaining 150 forget05 QAs are a stronger
secondary same-forget-split / unseen-fact diagnostic and are reported
separately.

## Stage 1

Entrypoint: `scripts/tofu_forget_only_setting5e.py`.

Default configuration:

- 50 direct forget QAs
- batch size 1
- 50 optimizer steps (one default pass)
- LR `2e-4`
- answer-NLL gradient ascent
- embedding/LM-head trainable during GA
- transformer frozen by the existing mode configuration

After GA, the LM head is untied.  The complete input embedding matrix returns
to the exact Full-TOFU starting weights.  All output rows except tokens in the
50 visible forget answers also return to the Full-TOFU weights.

## Stage 2

Entrypoint: `scripts/tofu_forget_only_active_repair.py`.

Default configuration:

- target direct-forget answer probability `<= 3e-4`
- NLL materialization buffer `0.25`
- repair rank `64`
- repair steps `5000`
- repair LR `0.02`
- forget hinge weight `100`
- hardest-forget hinge weight `25`
- delta L2 `1e-5`

The transformer and input embeddings are frozen.  Editable LM-head rows come
only from initially active direct forget answers.  The low-rank basis comes
only from hidden states of those active direct forget answers.  No held-out or
utility record can affect row selection, optimization, early stopping, or
checkpoint selection.

## Evaluation

`scripts/tofu_zerounlearn_locked_eval.py` reads only the materialized
`eval_only/` files and reports direct, paraphrase, unseen-150, and retain-1000
answer probability plus ROUGE-L (unless generation is disabled).  It also
reports truth-ratio diagnostics where perturbed answers exist.

The runner additionally invokes the existing `scripts/tofu_eval.py` only after
the checkpoint is frozen, to preserve the project's native TOFU utility axes.

## Wulver

After updating the branch, verify the selected Full-TOFU checkpoint:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
FULL=outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5

test -d "$FULL" && test -f "$FULL/config.json" && echo "FOUND: $FULL"
```

Submit all ten seeds:

```bash
sbatch slurm/run_tofu_zerounlearn_locked_3b.slurm
```

If the previous checkpoint is stored elsewhere:

```bash
sbatch --export=ALL,TOFU_FULL_MODEL_PATH=/actual/path/to/checkpoint_epoch_5 \
  slurm/run_tofu_zerounlearn_locked_3b.slurm
```

For a faster first probability-only diagnostic, disable only the custom held-out
generation pass (the native evaluator remains enabled):

```bash
sbatch --export=ALL,RUN_LOCKED_GENERATION=0 \
  slurm/run_tofu_zerounlearn_locked_3b.slurm
```

After all seeds finish:

```bash
python scripts/aggregate_tofu_zerounlearn_locked.py \
  --root outputs/tofu_zerounlearn_forget_only_locked_3b
```

The aggregate JSON and CSV are written under:

```text
outputs/tofu_zerounlearn_forget_only_locked_3b/aggregate/
```
