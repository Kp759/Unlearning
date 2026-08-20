# SURE MCF target-aware true-GA/new-GD mode

This mode is an explicitly benchmark-aware MCF ablation designed to enforce
both paper-facing forgetting metrics:

```text
FS  = direct prompts where NLL(target_true) > NLL(target_new)
GFS = paraphrases where NLL(target_true) > NLL(target_new)
```

It is not the benchmark-neutral, dataset-reusable SURE result. It reads the
sampled MCF records' original `target_new` values and official paraphrases
during training and checkpoint selection. Any comparison must disclose that
additional supervision.

## Architecture

```text
                   Frozen Base transformer
                   Frozen input embeddings
                             │
         50 direct prompts + official paraphrases
                             │
               ┌─────────────┴─────────────┐
               │                           │
       original target_true        original target_new
               │                           │
       bounded NLL ascent           bounded NLL descent
       (sensitive suppression)      (replacement learning)
               └─────────────┬─────────────┘
                             │
        direct + paraphrase pairwise separation loss
                             │
           token-conditioned Wikipedia exact KL
                             │
          sparse union(true,new) LM-head row edit
                    Stage 1 rank 4
                             │
          actual BF16 materialization and scoring
                             │
                   FS=100 and GFS=100?
                      /              \
                    yes              no
                     │                │
                   DONE       residual prompt cases
                                      │
                     exact constrained Stage 2
                       ranks 2 -> 4 -> 8
                       margins .5 -> 1 -> 2
                                      │
                     minimize Wikipedia KL + norm
                     subject to every direct and
                     paraphrase separation constraint
                                      │
                         actual BF16 materialization
                                      │
                FS=100, GFS=100, margins and utility safe?
                              /                 \
                            yes                 no
                             │                   │
                       emit checkpoint       INFEASIBLE
```

Only the union of target-answer LM-head rows is editable. Because the
transformer is frozen, teacher-forced hidden states are fixed and sequence NLL
constraints are differentiable exactly from cached states. Untouched vocabulary
logits do not move; their conditional distribution is therefore preserved
exactly within each predictor state. External Wikipedia KL guards protect uses
of the edited rows outside the benchmark prompts.

Stage 1 uses bounded objectives rather than unconstrained ascent/descent:

```text
L = 100 * hinge(1 - [NLL(true)-NLL(new)])^2
  +  10 * hinge(2 - [NLL_post(true)-NLL_base(true)])^2
  +  10 * hinge(1 - [NLL_base(new)-NLL_post(new)])^2
  +       Wikipedia_KL
  + 1e-4 * ||Delta W||^2
```

Direct and paraphrase losses are macro-balanced, so the two paraphrases per
record do not drown out the direct prompt objective. These are default training
targets, not metric redefinitions.

The paper requirement remains strict separation greater than zero. The emitted
checkpoint additionally requires every materialized separation to be at least
`0.01`. Stage 2 solves at larger continuous targets (`0.5`, then `1.0`, then
`2.0`) specifically to survive BF16 LM-head and log-softmax quantization. A
continuous FP32 solution is never sufficient by itself.

## Acceptance contract

A checkpoint is saved only when all of the following hold after actual
checkpoint-dtype materialization:

- FS is exactly 100 (50/50 direct prompt instances);
- GFS is exactly 100 (every official paraphrase prompt instance);
- no direct or paraphrase separation is below `0.01`;
- Wikipedia mean, p95, and maximum KL pass their locked budgets;
- total sparse-delta norm is at most `1.5`.

Neighborhood prompts, benchmark-retain examples, and PPL text are never used
for optimization or selection. Spe/Spe-success, exact benchmark-retain KL, and
PPL remain post-training audits. Thus FS/GFS are hard in-sample guarantees for
this benchmark-aware mode; specificity and broad utility are not guaranteed.

## Running

```bash
bash scripts/run_mcf_sure_target_aware.sh \
  /path/to/Llama-3.2-3B-Instruct \
  data/multi_counterfact.json
```

The compatibility entry point invokes the same v7 runner:

```bash
bash scripts/run_mcf_sure_fs100.sh MODEL MCF_JSON
```

Useful overrides:

```bash
export OUTPUT_ROOT=outputs/mcf_sure_target_aware_true_ga_new_gd_v7_seed1
export MCF_SEEDS=1
export SURE_MCF_TARGET_STAGE1_RANK=4
export SURE_MCF_TARGET_STAGE1_PAIRWISE_TARGET=1.0
export SURE_MCF_TARGET_STAGE1_TRUE_NLL_INCREASE=2.0
export SURE_MCF_TARGET_STAGE1_NEW_NLL_DECREASE=1.0
export SURE_MCF_STAGE2_SOLVER_MARGINS=0.5,1.0,2.0
export SURE_MCF_TARGET_STAGE2_RANK_LADDER=2,4,8
```

Key learner artifacts are under `seed*/target_aware_learner/`. An infeasible
run writes `infeasible.json` and never labels or saves a sub-100 result as a
successful checkpoint.
