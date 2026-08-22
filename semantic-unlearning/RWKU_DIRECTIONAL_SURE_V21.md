# RWKU Directional SURE v2.1

Directional SURE v2.1 is a one-factor correction to the first RWKU Directional SURE v2.0 development run.

## Why v2.1 exists

The v2.0 learner inherited the earlier RWKU head-only **content-bearing token** row filter. That filter intentionally leaves punctuation/function-word-like sensitive answer token rows immutable. The v2.0 Stephen King run therefore did not implement the literal Directional SURE policy "all sensitive answer rows trainable; all non-sensitive rows exact Base."

The v2.0 run is preserved as a valid failed development result. At step 600 it reached 85.714% direct atomic success and 85.294% other generated-view success while external-Wikipedia selection KL remained within the predefined mean/p95/max budgets. It failed closed because behavior did not reach 100%/100%.

## Single change

v2.1 changes only the sensitive vocabulary-row selector:

- editable input rows = every non-special vocabulary token observed in the teacher-forced sensitive answers;
- editable LM-head rows = the same set;
- all non-selected embedding/head rows remain exactly Base;
- every transformer parameter remains frozen.

No neutral row is added.

## Held fixed from v2.0

- 600 optimization steps
- batch size 1
- embedding LR 5e-5
- LM-head LR 1e-4
- GA weight 2
- GD weight 1
- gradient clip 1
- AdamW, zero weight decay
- basis refresh every 25 updates
- sensitive-exclusive basis rank 8
- protected basis rank 32
- protected external-Wikipedia contexts 256
- selection external-Wikipedia contexts 256
- fresh utility gate contexts 1000
- exact utility KL budgets: mean <= 0.01, p95 <= 0.05, max <= 0.5
- generated atomic behavior requirements: direct 100%, other generated views 100%, pairwise margin >= 0.01
- official RWKU artifacts unavailable to learner/selection

Thus v2.0 -> v2.1 is an isolated row-coverage correction, not a rank/LR/budget retuning experiment.
