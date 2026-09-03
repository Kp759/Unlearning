# MCF hybrid guarded/projected embedding rewiring V2.3.3

## Decision

V2.3.3 keeps the frozen Transformer, frozen untied LM head, selected input-row
edits, frequency-adjusted caps, 0.1 margin, and all preservation thresholds
unchanged. It changes only the treatment of rows and the nonlinear
reachability test:

| Sensitivity class | Treatment |
|---|---|
| `forget_specific` | unprojected, frequency-capped, forward-guarded |
| `shared` | projected out of the retain-readout subspace |
| `retain_dominant` | excluded at exact zero |
| `low_forget` | excluded at exact zero |

V2.3.2 found 106 forget-specific rows but projected 104 of them. That policy
identified the intended capacity and then removed it. V2.3.3 permits those
directions while replacing their first-order safety assumption with observed
forward constraints.

## Guarded retain bank

Every fit-bank case containing a guarded token row is indexed before
reachability. A guarded row without a training-visible forward-audit case is a
hard failure. The indexed cases are:

- evaluated during every candidate step of each prompt-specific reachability
  path when the prompt gradient uses that guarded row;
- added to the rotating overlap stratum used on every joint training update;
- covered by the complete fit-bank hard-tail refresh and rollback audit.

Development data remains selection-only. Certification remains sealed until a
development-passing candidate has been selected.

## Iterative nonlinear reachability

The V2.3.2 one-shot ray held the Base gradient fixed while moving as far as a
row cap. Its seed-1 failure was mostly nonlinear-only: the local linear bound
passed for many prompts whose frozen-model forward sweep failed.

V2.3.3 starts every direct and synthetic prompt at exact zero and repeats:

1. recompute that prompt's current margin gradient;
2. leave guarded components intact, project shared components, and zero
   excluded components;
3. propose a row-normalized step of 2% of each unchanged frequency-aware cap;
4. backtrack until the real margin strictly improves and all relevant guarded
   retain cases remain inside the locked KL/log-probability limits;
5. stop only at margin 0.1, an exhausted 64-step budget, or the absence of a
   safe improving step.

The exact-zero linear cap calculation remains in the report as a diagnostic;
the relinearized forward path is the acceptance gate. Every one of the 50
direct and 150 synthetic prompts must pass before joint optimization begins.

## Claim boundary

This is an internal materialized embedding edit with no classifier, runtime
gate, sidecar, Transformer update, or LM-head update. A passing result supports
context-sensitive behavioral unlearning within the selected-token coverage.
It does not establish latent erasure or token-disjoint alias coverage.

Seed 1 is consumed architecture-development evidence and cannot support an
official claim.
