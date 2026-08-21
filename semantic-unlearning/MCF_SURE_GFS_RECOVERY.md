# MCF SURE W10K GFS-recovery protocol

## Decision

Do not increase the global edit scale or relax the W10K utility guard.  The
seed-1 Stage-1 result is a context-transfer failure, not evidence that the edit
is globally too weak:

| Metric | v8 baseline | W10K Stage 1 | Change |
|---|---:|---:|---:|
| FS | 100.0 | 96.0 | -4.0 pp |
| GFS | 79.0 | 48.0 | -31.0 pp |
| Spe-success | 34.2 | 61.8 | +27.6 pp |
| Spe-margin | -5.38 | 3.69 | +9.07 |
| PPL | 11.0625 | 11.0625 | 0 |

The direct mean NLL separation is already `+4.1463`, yet paraphrase success is
48 points below direct success.  That concentration of failure on unseen
wordings points to prompt-manifold undercoverage.  More global edit strength
would not target that gap and would put the recovered locality at risk.

## Locked recovery treatment

Run `v9-GFS-Paired-W10K` from the Base model with exactly the same:

- 10,000-document Wikipedia utility cache and cache SHA-256;
- Rank-4 Stage 1;
- bounded GA(`target_true`) + GD(`target_new`);
- sparse union of true/new LM-head rows;
- utility KL budgets, norm budget, and Stage-2 rank/margin ladder.

Change only the training-visible context treatment:

1. For each forget request, create four deterministic same-subject prompt views
   ending in different answer cues: QA, short-answer, cloze, and encyclopedia
   card.  They receive the existing GA+GD objective and are hard BF16 gates.
2. For every view family, apply the identical wrapper to unrelated Wikipedia
   article titles substituted into the direct-visible relation template.
   Preserve Base behavior on these matched views with exact sparse-row KL.
3. Keep official MCF paraphrases, neighborhoods, retain prompts, generation
   probes, and PPL text unavailable until after checkpoint freeze.

This is implemented as the locked profile `paired_answer_cue_v1`.  It keeps the
same four generated contexts per record as legacy v9; the intended treatment is
prompt geometry plus matched locality protection, not extra sample count.

## Run

```bash
cd /home/ec2-user/workspace/Unlearning/semantic-unlearning

MODEL=/home/ec2-user/models/Llama-3.2-3B-Instruct
MCF=data/multi_counterfact.json
PPL_WIKI=data/wikidata
REAL_WIKI=data/wikipedia_sure_100020

python -u scripts/MCF_Scripts/run_mcf_sure_two_stage.py \
  --model-path "$MODEL" \
  --mcf-path "$MCF" \
  --wikipedia-dir "$PPL_WIKI" \
  --utility-wikipedia-dir "$REAL_WIKI" \
  --treatment paired_context_recovery \
  --utility-docs 10000 \
  --utility-prompts 100000 \
  --require-corpus-protocol sure_external_wikipedia_corpus_v1 \
  --output-root outputs/mcf_sure_paired_recovery_W10K_seed1_dev \
  --seeds 1 \
  --development-seeds 1
```

The canonical runner refuses to overwrite an existing output root and refuses
to execute when tracked or untracked files under `semantic-unlearning/scripts`
are dirty. It records the Git commit and SHA-256 of every runtime source file.
The 1,000 prompt-only retain records are opened only after checkpoint freeze
for the exact sparse-row retain-KL audit.

## Predeclared readout

Treat the supplied W10K Stage-1 result as development evidence because official
GFS and Spe have now been inspected.  Do not use either metric to choose a
scale, rank, margin, early-stop point, or residual repair within the new run.

After the new checkpoint is frozen, classify its seed-1 result as follows:

- **full recovery:** FS = 100, GFS >= 79, Spe-success >= 61.8,
  Spe-margin >= 3.69, PPL <= 11.2, and every declared utility/locality guard
  passes;
- **partial recovery:** FS = 100 and 48 < GFS < 79 while both Spe diagnostics
  meet the W10K floors and all guards pass;
- **reject:** GFS <= 48, either Spe diagnostic falls below its W10K floor, or a
  materialized utility/locality/norm guard fails.

If seed 1 passes, freeze the profile and every argument before running untouched
seeds 2--11 for the paper-facing estimate. The canonical report requires ten
confirmatory seeds. Report seed 1 separately as development evidence and do not
promote it to an untouched replicate.

```bash
python -u scripts/MCF_Scripts/run_mcf_sure_two_stage.py \
  --model-path "$MODEL" \
  --mcf-path "$MCF" \
  --wikipedia-dir "$PPL_WIKI" \
  --utility-wikipedia-dir "$REAL_WIKI" \
  --treatment paired_context_recovery \
  --utility-docs 10000 \
  --utility-prompts 100000 \
  --require-corpus-protocol sure_external_wikipedia_corpus_v1 \
  --output-root outputs/mcf_sure_paired_recovery_W10K_confirmatory \
  --seeds 2 3 4 5 6 7 8 9 10 11
```

## What the result can establish

An improvement after this single-factor treatment supports the diagnosis that
W10K made the edit more local but too tied to direct-prompt hidden states.  It
does not prove that Wikipedia scaling automatically improves GFS, and one seed
does not establish a stable mean effect.  If the paired treatment is rejected,
the next clean ablation is additional generic view families with the same
paired KL construction—not a relaxed locality budget.
