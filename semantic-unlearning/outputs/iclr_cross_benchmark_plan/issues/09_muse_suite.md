## Objective

Build a passage-level SURE extension for MUSE News and Books and evaluate all six native desiderata.

## Scope

- Use corpus-specific Target and retain reference models; do not substitute the generic Llama checkpoint.
- Train only on permitted forget and retain1/calibration passages.
- Keep retain2, holdout, PrivLeak, and final evaluation roles behind the freeze boundary.
- Replace QA-row editing with a predeclared passage-window/span objective and contextual low-rank edit scope.
- Report VerbMem-F, KnowMem-F, raw PrivLeak and distance-to-zero, KnowMem-R, scale curves, and sequential-unlearning sustainability.
- Compare against retrain, GA variants, NPO variants, SimNPO, GradDiff, PDU, RMU, and other eligible pinned implementations.

## Acceptance criteria

- [ ] News and Books target/retrain models reproduce official evaluator baselines.
- [ ] PrivLeak acceptance is based on closeness to zero, not raw minimization.
- [ ] No held-out/retain2 evidence selects checkpoints.
- [ ] SURE-F runs end to end before U100K is introduced.
- [ ] At least three independent expensive runs plus bootstrap CIs over examples are reported, with uncertainty types separated.
- [ ] Scalability and sequential-unlearning evaluations are included, not only four headline metrics.

## Dependencies

Depends on metric contract, baseline contract, and result registry. Start after factual-suite parity is stable.
