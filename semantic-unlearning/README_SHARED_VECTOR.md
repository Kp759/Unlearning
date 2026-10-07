# MCF seed 1: one shared vector versus 50 association vectors

This experiment asks whether one learned residual can replace the 50 independent
MCF residuals while preserving forgetting and ordinary behavior. It uses the
exact 50 seed-1 associations and the frozen linear router from an existing
**50-vector IDK + end-token** run. Read and write layer are both 19.

For an active route, `h_last += shared_vector`, regardless of the selected fact.
For an inactive route, the hidden state is unchanged. There are no per-fact
scales, codes, or residual parameters: only one zero-initialized vector of size
`d` (3,072 values for Llama-3.2-3B). This is trained from scratch, not averaged
from the existing rows. The router, including its ambiguity/abstention rules,
is unchanged.

## Run

The default reference follows the existing `generations_multiseed.slurm` layout:

```text
outputs/compressed_multiseed_idk_eos_v1/mcf/seed1/L19/full
```

On Wulver, from `semantic-unlearning`, validate the baseline without loading the
model or writing output, then submit:

```bash
python scripts/run_mcf_shared_vector_seed1.py --dry-run --local-files-only
mkdir -p logs
sbatch mcf_shared_vector_seed1.slurm
```

For a different baseline location:

```bash
python scripts/run_mcf_shared_vector_seed1.py \
  --reference-run-dir /path/to/mcf/seed1/L19/full \
  --dry-run --local-files-only
sbatch --export=ALL,REFERENCE_RUN_DIR=/path/to/mcf/seed1/L19/full mcf_shared_vector_seed1.slurm
```

Without Slurm, run the Python command with `--dry-run` removed. The reference
must include its artifact, manifest, and training report. Preflight requires
seed 1, layer 19, 50 facts with training coverage, the joint `full` trainer,
actual router routing, and IDK + EOS enabled. It rejects the older shipped
two-phase baseline as a direct training control.

## Controlled comparison

The launcher copies the baseline's recorded optimizer settings, batch size,
epoch/evaluation schedule, time cap, seed, IDK text, and IDK weight. It uses the
same joint loss: sensitive-answer suppression hinge plus IDK-and-EOS NLL.
Gradients from different facts accumulate into the same vector. Checkpoint
selection uses training/authored development views; official evaluation
prompts never train or select the checkpoint.

Training uses the same correctly routed views as the full-row baseline to
isolate parameter sharing. Excluded views and all 50 fact identities are
checked. At inference **every active route** gets the shared vector, including
incorrect association selections; an inactive router still cannot be repaired
by the vector. No per-fact zero mask is applied to shared values.

The experiment re-evaluates the existing 50-vector checkpoint and the new shared
checkpoint with the same evaluator and precision. It does not retrain or modify
the reference. The configured training budgets match; early stopping and actual
runtime can differ and are recorded. The supplied comparison table is context,
not a seed-1 result file or a numerical acceptance threshold.

## Outputs and interpretation

Default output: `outputs/mcf_shared_vector_seed1_idk_eos/`.

- `experiment_config.json`: baseline hashes, settings, and exact commands.
- `shared/`: shared-vector artifact, compact parameter state, and training report.
- `official_full_50.json`, `official_shared_1.json`: Eff, Gen, Spe, retain and PPL.
- `generations.jsonl` / `generations.md`: side-by-side base/full/shared continuations.
- `comparison.csv` / `comparison.json`: scores and answer-leakage, abstention,
  exact-IDK and activation percentages with prompt counts.

Generation evaluates all seed-1 rewrites/paraphrases, up to 10 neighborhood
prompts per fact, and all 1,000 retain records' evaluation prompts. Both arms use
24 new tokens, greedy decoding, bf16, and the fixed original request boundary.
`--max-new-tokens` changes both arms together. Answer presence uses the existing
case-insensitive substring metric; IDK detection alone does not imply no leak.
PPL is enabled. Official metrics keep their native units; generation columns
ending `_pct` are percentages. Generation routes must match between arms.

For evaluator compatibility, the artifact stores 50 **identical copies** of the
learned vector in `rows`, plus its single-vector compact state. Training has
only `d` residual parameters; the existing evaluator still materializes 50 rows,
so this is not a benchmark of deployed memory savings. Export checks enforce
exact row equality and reconstruction from the single vector.

Finished stages can be resumed with the same command. Changing the baseline or
settings requires a new `--output-dir`. Incomplete training directories are
preserved and cause an error rather than being overwritten.

Compare leakage and Eff/Gen first, then Spe/retain/PPL and unintended abstention.
Similar results would support sharing for this seed/model; worse results would
show a loss of performance under the matched training protocol, not prove that
no possible shared-vector training procedure can work.
