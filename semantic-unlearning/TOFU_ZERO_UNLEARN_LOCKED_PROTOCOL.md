# TOFU author-balanced locked protocol

This is the project's leakage-controlled TOFU portability track for Setting-5e
plus sparse LM-head repair. It is separate from the native/full-utility
F01/F05/F10 track.

## Starting model

Unlearning starts from the validated Full-TOFU Llama-3.2-3B epoch-5 model:

```text
outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5
```

This is the reproduced LR `4e-5`, selected epoch-5 checkpoint whose native TOFU
metrics match the original selected run closely.

## Author-balanced split

TOFU profiles contain 20 QA pairs per fictitious author. For `forget05`, the
200 rows are treated as 10 contiguous 20-QA author blocks. For every seed:

1. sample 5 of the 10 forget-author blocks;
2. for each selected author, sample 10 of its 20 QAs for unlearning;
3. reserve the other 10 QAs for that same author as unseen same-author
   evaluation data;
4. independently sample 1,000 `retain95` QAs for final utility evaluation.

Therefore each seed has:

```text
Unlearning-visible direct QAs:       5 authors x 10 = 50
Seen direct efficacy eval:           same 50 QAs
Seen paraphrase eval:                50 paraphrases
Same-author unseen direct eval:      5 authors x 10 = 50
Same-author unseen paraphrase eval:  50 paraphrases
Retain utility eval:                 1000 retain95 QAs
```

The selected author-block IDs and exact train/held-out source indices are stored
in each `split_manifest.json`.

## Leakage lock

Stage 1 and Stage 2 receive only the 50 direct training QAs. Training-visible
rows contain only:

- `question`
- `answer`
- `_source_index` (provenance only)

They receive zero retain95 rows, paraphrases, perturbed answers, real-author
examples, or world-fact examples. All held-out and utility data enter only
after the Stage-2 checkpoint is frozen.

## Evaluation meanings

- **Seen deletion efficacy:** the exact 50 direct QAs exposed during unlearning.
- **Prompt generalization:** benchmark paraphrases of those same 50 QAs.
- **Same-author fact generalization:** the other 10 QAs for each of the 5
  selected authors, never exposed to Stage 1 or Stage 2.
- **Utility:** 1,000 sampled `retain95` QAs relative to the Full-TOFU reference.

This distinction lets the experiment separate memorization of the deletion
requests from generalization of forgetting to other facts about the same
selected authors.

## Stage 1

Entrypoint: `scripts/tofu_forget_only_setting5e.py`.

Default configuration:

- 50 direct forget QAs
- batch size 1
- 50 optimizer steps
- LR `2e-4`
- answer-NLL gradient ascent
- embedding/LM-head trainable during GA
- transformer frozen

After GA, the input embedding matrix is restored to the Full-TOFU starting
weights. Output rows not associated with visible forget-answer tokens are also
restored.

## Stage 2

Entrypoint: `scripts/tofu_forget_only_active_repair.py`.

The author-balanced runner intentionally uses bounded defaults because the
previous rank-64 / LR `0.02` / unbounded repair destroyed TOFU utility.

Default runner configuration:

- target direct-forget AP `<= 3e-4` (best effort under the norm cap)
- repair rank `8`
- repair LR `0.002`
- maximum LM-head delta norm `1.0`
- row selection `all`
- repair steps `5000`
- `--save-best-effort`

The target is not allowed to force an arbitrarily large edit: if the bounded
repair cannot reach `3e-4`, the best bounded checkpoint is still materialized
for evaluation.

## Native evaluator

`RUN_NATIVE_EVAL=0` by default for this track. The legacy `tofu_eval.py`
independently samples `forget05`, so its forget subset is not guaranteed to be
the same author-balanced 50. The path-based locked evaluator is the
protocol-defining evaluation. Native evaluation can still be enabled as an
optional auxiliary diagnostic.

## Wulver smoke test

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

export TOFU_FULL_MODEL_PATH=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5
export TOFU_SEEDS=1
export OUTPUT_ROOT=outputs/tofu_author_balanced_locked_3b_test
export RUN_LOCKED_GENERATION=0
export RUN_NATIVE_EVAL=0

bash scripts/run_tofu_zerounlearn_locked_our_method.sh "$TOFU_FULL_MODEL_PATH"
```

Inspect the split before trusting the run:

```bash
python - <<'PY'
import json
p='outputs/tofu_author_balanced_locked_3b_test/protocol/seed1/split_manifest.json'
x=json.load(open(p))
s=x['sampling']
print('authors:', s['selected_author_block_ids'])
print('train:', s['train_forget_num'])
print('heldout same-author:', s['same_author_heldout_num'])
for a,v in s['per_selected_author'].items():
    print(a, len(v['train_source_indices']), len(v['heldout_source_indices']))
PY
```

Expected: 5 selected authors, 50 train QAs, 50 same-author held-out QAs, and
10/10 train/held-out QAs for every selected author.

## Ten-seed Slurm run

```bash
sbatch slurm/run_tofu_zerounlearn_locked_3b.slurm
```

The Slurm wrapper uses the validated reproduced epoch-5 checkpoint by default
and writes to `outputs/tofu_author_balanced_locked_3b`.
