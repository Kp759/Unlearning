# SURE-TOFU Author-Balanced Locked Protocol

## Goal

Compare SURE against the ZeroUnlearn-style TOFU baseline under identical data
access, starting checkpoint, author-balanced split, seeds, and final evaluator.
Algorithmic optimization may differ; held-out benchmark information may not be
used before the final checkpoint is frozen.

## Shared split

Use `build_tofu_zerounlearn_locked_split.py` with, per seed:

- `forget05`: 10 contiguous author blocks x 20 QAs.
- Select 5 author blocks with `random.Random(seed)`.
- For each selected author, choose 10 QAs for training and hold out the other
  10 QAs.
- Training-visible forget set: 50 direct QAs = 5 authors x 10 QAs.
- Same-author held-out set: 50 direct QAs = the other 10 QAs per selected
  author.
- Final retain utility sample: 1000 from `retain95`.
- Paraphrases, same-author heldout QAs, retain95, real-authors, world-facts,
  PPL, and final metrics are evaluation-only.

The exact same split manifest must be used for ZeroUnlearn and SURE.

## Starting model

Use the validated Full-TOFU fine-tuned checkpoint (not raw Llama and not a
post-unlearning checkpoint).  For the current 3B experiments this is the
reproduced epoch-5 checkpoint from the six-epoch LR=4e-5 Full-TOFU run.

## Stage 1A: same-prompt GA/GD

Input data: only the 50 training-visible direct forget QAs.

For every teacher-forced answer-token position:

1. GA suppresses the true answer token by minimizing its log probability.
2. GD preserves the Full-TOFU Base distribution over every non-sensitive
   vocabulary token on the same context.  The true answer token is removed
   from Base and Current distributions and each distribution is renormalized
   before computing `KL(Base_non_sensitive || Current_non_sensitive)`.

Default objective:

`L = 2.0 * L_GA + 1.0 * L_GD`

Default training controls:

- embedding + LM-head trainable;
- 600 steps;
- batch size 1;
- AdamW;
- LR `1e-4`;
- gradient clip 1.0.

Post-training vocabulary restoration uses `sensitive_both` by default:

- sensitive answer-token input rows keep Stage-1A displacement;
- sensitive answer-token output rows keep Stage-1A displacement;
- every other input/output vocabulary row is restored exactly to Full-TOFU.

No held-out benchmark data are consulted.

## Stage 1B: SURE active forgetting

Start from the frozen Stage-1A checkpoint.

- Audit the same 50 direct training QAs.
- Identify residual active direct forget constraints.
- Edit only LM-head rows required by active answer tokens.
- `repair_rank = 0`: unrestricted selected-row delta in the full hidden
  dimension; no low-rank bottleneck.
- Optimize direct forget constraints dynamically until every direct QA meets
  the target and NLL buffer.
- Default target answer probability: `3e-4`.
- Default NLL buffer: `0.25`.
- No retain or held-out utility data are used.

Stage 1B is required to PASS before restoration begins.

## Stage 2: forget-nullspace Base restoration

Start from the frozen Stage-1B rank-0 forgetting solution.

1. Build the numerical hidden-state span `B_F` from every teacher-forced answer
   position in the same 50 direct training QAs.
2. Compute the selected-row LM-head displacement needed to return toward the
   original Full-TOFU LM-head rows.
3. Project that desired restoration into the orthogonal complement of `B_F`.
4. Compute a fixed rank-64 or rank-128 restoration basis in that complement.
5. Apply the largest predeclared scale that preserves every Stage-1B direct
   forget NLL constraint after model-dtype materialization.

If `Delta W_R` lies in the orthogonal complement of every direct-forget hidden
state `h`, then `Delta W_R h = 0` on those cached contexts in exact arithmetic.
The explicit post-materialization NLL guard is still mandatory.

R64 and R128 are frozen before any held-out evaluation.  They are reported as
an ablation; retain1000 or other final metrics must not be used to choose one.

## Final evaluation only

After all requested restoration checkpoints are frozen, run the same locked
TOFU evaluator used for ZeroUnlearn:

- seen direct F50;
- seen paraphrase F50;
- same-author unseen direct F50;
- same-author unseen paraphrase F50;
- retain95 sample of 1000.

Optional post-hoc standard TOFU diagnostics may include real-authors,
world-facts, truth-ratio/ROUGE axes, and PPL, but none may influence checkpoint
selection.

## Fair-comparison statement

The intended paper statement is:

> All compared methods use the same Full-TOFU starting checkpoint, identical
> author-balanced training-visible forget examples, identical held-out splits,
> identical seeds, and the same final evaluator.  No held-out benchmark metric
> is used for optimization or checkpoint selection.

This is data-access/evaluation matched.  It is not necessarily compute-matched;
optimization budget and parameterization should be reported separately.
