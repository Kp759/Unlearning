# MCF SURE Wikipedia-scaling and v9 augmentation protocol

This protocol follows the v8 seed-1 result at commit `731df21`:

| Experiment | Real Wikipedia documents | Generated subject contexts | Observed/primary goal |
|---|---:|---:|---|
| v8 baseline | 180 (200-row fixture minus 20 PPL rows) | 0 | observed FS 100 / GFS 79 / Spe-success 34.2 |
| v8-W1K | 1,000 | 0 | isolate the first real-corpus locality gain |
| v8-W10K | 10,000 | 0 | Stage 1 observed FS 96 / GFS 48 / Spe-success 61.8 |
| v9-Aug | 10,000 | 4 GA/GD templates per forget record plus external locality contexts | improve held-out GFS without sacrificing recovered Spe |
| v9-GFS-Paired | 10,000 | 4 answer-cued GA/GD views plus structure-matched locality views | recover the direct-to-paraphrase transfer gap while holding W10K fixed |
| final | 50,000–100,000 | augmentation locked before final seeds | paper-facing result |

`Spe-success` and `Spe-margin` are different quantities and must remain separate
in tables.  The base-like `88–89` target refers to the success-rate diagnostic,
not the probability-difference margin.

## What the v8 result established

The file named `docs100000` in v8 did not prove 100,000-document protection.
The checked-in `data/wikidata` fixture has only 200 rows and the cache builder
excluded its first 20 PPL rows.  It therefore used 180 documents while filling
the predictor reservoir with many token positions from those same documents.

The tiny selected-pool KL (`~3e-8`) and the post-hoc exact retain KL (mean
`0.541`, p95 `3.264`) measure different context distributions.  The latter is
the relevant evidence that v8's narrow proxy pool did not protect benchmark-
like retained contexts.

## Locked invariants

Every experiment keeps:

- Stage-1 row rank 4;
- bounded GA on original `target_true` and bounded GD on `target_new`;
- sparse edits to the union of true/new LM-head rows;
- checkpoint-dtype FS=100 and the direct margin gate;
- official paraphrases, neighborhoods, retain prompts, and PPL text unavailable
  until after the checkpoint is saved.

W1K and W10K are pure v8 replications.  Their only intended treatment variable
is `actual_document_sample_size`; generated augmentation is explicitly off.

V9 adds two data roles built from the stripped direct view and a separate
Wikipedia corpus:

1. Four deterministic same-subject template contexts per forget record receive
   the existing GA(true)+GD(new) objective and are hard materialization
   constraints.
2. Prompts made from unrelated Wikipedia article titles and direct-visible
   relation templates receive frozen-Base exact sparse-row KL preservation.
   Train and guard pools are split before token-probability conditioning.

The context builder has no raw CounterFact argument.  It records zero reads of
official paraphrases, neighborhoods, retain examples, and generation probes.

## Prepare the real Wikipedia corpus

The downloader uses the pinned `wikimedia/wikipedia` `20231101.en` snapshot and
stores 100,020 articles: 100,000 usable utility documents plus the 20-row
exclusion required by the existing cache contract.

```bash
cd /home/ec2-user/workspace/Unlearning/semantic-unlearning

python scripts/prepare_sure_wikipedia_corpus.py \
  --output-dir data/wikipedia_sure_100020 \
  --sample-size 100020 \
  --seed 1
```

The output receipt records the pinned revision, shuffle settings, row count,
and content SHA-256.  Do not point official PPL evaluation at this directory;
`data/wikidata` remains the unchanged PPL fixture.

## Run W1K then W10K

```bash
MODEL=/home/ec2-user/models/Llama-3.2-3B-Instruct
MCF=data/multi_counterfact.json
REAL_WIKI=data/wikipedia_sure_100020

MCF_SEEDS=1 \
bash scripts/run_mcf_sure_utility_scaling.sh \
  "$MODEL" "$MCF" "$REAL_WIKI"
```

The default ladder is `1000 10000`; override it only for a new, explicitly
labeled ablation:

```bash
SURE_SCALING_DOCS="1000 10000" MCF_SEEDS=1 \
bash scripts/run_mcf_sure_utility_scaling.sh \
  "$MODEL" "$MCF" "$REAL_WIKI"
```

Each run requires the requested document count and at least 90,000 predictor
candidates.  It fails instead of silently falling back to a pilot cache.

## Run v9-Aug at W10K

```bash
MCF_SEEDS=1 \
SURE_V9_WIKIPEDIA_DOCS=10000 \
SURE_EXTERNAL_CONTEXTS_PER_RECORD=128 \
bash scripts/run_mcf_sure_v9_aug.sh \
  "$MODEL" "$MCF" "$REAL_WIKI"
```

Defaults are intentionally explicit in the learner artifacts:

- same-subject GA/GD contexts: 4 per record;
- external locality candidates: 128 per record;
- locality top-k: 64 per edited row in each disjoint half;
- locality Stage-1 KL weight: 10;
- locality mean/p95/max KL guards: `0.01 / 0.05 / 0.5`.

If this configuration is infeasible, report it as infeasible.  Do not relax a
guard after inspecting official GFS or Spe and call the result locked.

The W10K Stage-1 diagnostic subsequently showed FS 96, GFS 48, Spe-success
61.8, Spe-margin 3.69, and unchanged PPL 11.0625.  The locked recovery ablation
and its predeclared acceptance rules are in
[`MCF_SURE_GFS_RECOVERY.md`](MCF_SURE_GFS_RECOVERY.md).  Run it with:

```bash
MCF_SEEDS=1 \
SURE_V9_WIKIPEDIA_DOCS=10000 \
bash scripts/run_mcf_sure_v9_gfs_recovery.sh \
  "$MODEL" "$MCF" "$REAL_WIKI"
```

## Compare the ladder

```bash
python scripts/compare_mcf_sure_utility_ladder.py \
  --run baseline=/path/to/v8-baseline/seed1 \
  --run W1K=outputs/mcf_sure_utility_scaling/v8-W1000/seed1 \
  --run W10K=outputs/mcf_sure_utility_scaling/v8-W10000/seed1 \
  --run v9-Aug=outputs/mcf_sure_v9_aug_w10000/seed1 \
  --output-dir outputs/mcf_sure_utility_ladder_comparison
```

The comparison includes FS, GFS, both Spe measures, PPL, cache sizes, and exact
retain KL.  It rejects artifacts claiming that official GFS or neighborhoods
participated in checkpoint selection.

## Locking the final result

GFS and Spe are genuinely held-out only with respect to training and within-run
checkpoint selection.  Once seed-1 GFS/Spe are used to choose W10K versus an
augmentation, they become development metrics for that choice.

For a paper-facing final result:

1. choose and record one augmentation configuration from the ladder;
2. choose either 50K or 100K before evaluating new seeds;
3. freeze every SURE/context environment variable and cache SHA-256;
4. run untouched seeds (for example 2–6) without further changes;
5. report seed 1 as development/tuning evidence, not as an untouched final
   replicate;
6. report FS/GFS/Spe/PPL and exact retain-KL together.

After those choices are frozen, a 50K example final run would look like:

```bash
OUTPUT_ROOT=outputs/mcf_sure_v9_final_locked_w50000 \
MCF_SEEDS="2 3 4 5 6" \
SURE_V9_WIKIPEDIA_DOCS=50000 \
SURE_EXTERNAL_CONTEXTS_PER_RECORD=128 \
bash scripts/run_mcf_sure_v9_aug.sh \
  "$MODEL" "$MCF" "$REAL_WIKI"
```

Use `100000` instead of `50000` only if that choice was made before looking at
the final-seed GFS/Spe results.

Larger Wikipedia coverage can improve Spe and may improve GFS, but neither is
automatic.  The final claim is determined by the post-checkpoint measurements.
