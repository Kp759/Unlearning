# SURE MCF target-aware direct-only v8

V8 is the target-aware MCF experiment in which official GFS remains genuinely
held out. It uses the sampled records' `target_true` and `target_new` answers,
so it is not benchmark-neutral SURE. Unlike v7, however, it never exposes the
official paraphrases to learning or checkpoint selection.

The intended interpretation is:

- FS is an in-sample, hard acceptance guarantee;
- GFS measures held-out paraphrase generalization;
- Spe, PPL, and benchmark-retain KL are post-training utility audits;
- v7 remains a separately labeled in-sample FS/GFS upper-bound ablation.

## Enforced data boundary

```text
original MCF source
        │
        │ split builder is the sole training-side reader
        ▼
stripped learner file
  ├─ direct prompt
  ├─ subject
  ├─ target_true
  └─ target_new
        │
        │ no raw-MCF path is accepted by the learner
        ▼
Stage 1 + Stage 2 + BF16 FS checkpoint gate
        │
        │ checkpoint saved and frozen
        ▼
original MCF source opened for the first time downstream
  ├─ held-out official paraphrases → GFS
  ├─ neighborhood prompts → Spe
  ├─ PPL text → PPL
  └─ retained prompts → exact sparse-row KL audit
```

`build_mcf_sure_target_aware_direct_split.py` writes
`training_visible_target_aware_direct.json`. Probe fields are absent, including
when their source value would be an empty list. The manifest binds the view to
the source SHA-256 and official sampled case IDs, but deliberately contains no
source-dataset path. `sure_mcf_target_aware_direct_only.py` accepts the stripped
file and manifest; it has no `--mcf-path` argument.

Consequently, official paraphrases cannot affect:

- Stage-1 basis construction or losses;
- Stage-1 scale selection;
- Stage-2 failure detection, basis construction, or constraints;
- the rank ladder or BF16 solver-margin ladder;
- checkpoint acceptance or early stopping.

Using observed GFS to change v8 settings would create a new, tuned experiment
and must not be reported as the locked held-out v8 result.

## Architecture

```text
                 Frozen Base transformer
                 Frozen input embeddings
                           │
                  50 direct MCF prompts
                           │
             ┌─────────────┴─────────────┐
             │                           │
         target_true                 target_new
             │                           │
      bounded NLL ascent          bounded NLL descent
      (suppress old fact)         (install replacement)
             └─────────────┬─────────────┘
                           │
             direct pairwise separation loss
                           │
          token-conditioned Wikipedia exact KL
                           │
        sparse union(true,new) LM-head row edit
                   Stage 1 rank 4
                           │
             actual BF16 direct FS scoring
                           │
                  FS=100 and utility safe?
                     /               \
                   yes               no
                    │                 │
                  DONE       direct residual cases
                                      │
                         exact constrained Stage 2
                           ranks 2 → 4 → 8
                         margins .5 → 1 → 2
                                      │
                         minimize Wikipedia KL
                         plus sparse residual norm
                         subject to every direct
                         separation constraint
                                      │
                         actual BF16 materialization
                                      │
                    FS=100, direct margin, utility safe?
                              /                 \
                            yes                 no
                             │                   │
                       save checkpoint       INFEASIBLE
```

Stage 1 minimizes the direct-only version of:

```text
L = 100 * hinge(1 - [NLL(true)-NLL(new)])^2
  +  10 * hinge(2 - [NLL_post(true)-NLL_base(true)])^2
  +  10 * hinge(1 - [NLL_base(new)-NLL_post(new)])^2
  +       Wikipedia_KL
  + 1e-4 * ||Delta W||^2.
```

Stage 2 uses the same exact cached-state constrained residual formulation as
v7, but its behavioral constraint vector contains direct prompts only. The
reported FS rule remains strict `NLL(target_true) > NLL(target_new)`. The
materialized checkpoint additionally requires separation of at least `0.01`.
Continuous Stage-2 targets of `0.5`, `1.0`, and `2.0` provide BF16 headroom;
they do not redefine FS.

## Checkpoint contract

A v8 checkpoint is saved only when an actual checkpoint-dtype forward pass
satisfies all of the following:

- direct FS is exactly 100;
- all 50 direct separations are at least `0.01`;
- held-out Wikipedia mean, p95, and maximum KL pass their locked budgets;
- total sparse delta norm is at most `1.5`.

GFS is intentionally absent from this list. The post-training evaluator still
computes it, but `--require-min-gfs` is not passed. V8 therefore cannot promise
GFS=100 before the experiment is run; whatever value is observed is the honest
held-out generalization result.

## Running

```bash
bash scripts/run_mcf_sure_target_aware_direct_only.sh \
  /path/to/Llama-3.2-3B-Instruct \
  data/multi_counterfact.json
```

Useful locked overrides are shared with v7:

```bash
export OUTPUT_ROOT=outputs/mcf_sure_target_aware_direct_only_v8_seed1
export MCF_SEEDS=1
export SURE_MCF_TARGET_STAGE1_RANK=4
export SURE_MCF_TARGET_STAGE1_PAIRWISE_TARGET=1.0
export SURE_MCF_TARGET_STAGE1_TRUE_NLL_INCREASE=2.0
export SURE_MCF_TARGET_STAGE1_NEW_NLL_DECREASE=1.0
export SURE_MCF_STAGE2_SOLVER_MARGINS=0.5,1.0,2.0
export SURE_MCF_TARGET_STAGE2_RANK_LADDER=2,4,8
```

Learner artifacts are written under
`seed*/target_aware_direct_only_learner/`. A failed rank/margin ladder writes
`infeasible.json` and does not save a sub-100-FS checkpoint.
