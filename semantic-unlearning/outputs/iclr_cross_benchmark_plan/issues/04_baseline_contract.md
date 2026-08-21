## Objective

Create the paper's baseline eligibility and fairness contract before launching expensive comparisons.

## Scope

- Replace checkmarks with support classes: Native, Reproduced, Adapted, or Inappropriate/unavailable.
- Store official source URL, immutable commit, model revision, dataset revision, evaluator revision, and license.
- Separate RMU and TAR; identify and source-pin the intended TAR method.
- Mark ZeroUnlearn as native only for datasets actually supported upstream; label TOFU/RWKU/MUSE/WMDP ports as adaptations.
- Lock equal method-visible data roles, tuning budgets, stopping rules, and seed policy.
- Define compute disclosure: trials, GPU hours, peak memory, trainable parameters, and checkpoint size.

## Acceptance criteria

- [ ] Every method×dataset cell has N/R/A/X status and evidence.
- [ ] No adapted method is presented as an upstream native result.
- [ ] Hyperparameter-search budgets and selection metrics are predeclared.
- [ ] Every baseline uses the same target model and evaluator as SURE within a dataset.
- [ ] Unsupported cells fail closed in orchestration.
- [ ] The final baseline matrix is generated from the registry rather than hand-edited.

## Dependencies

Depends on the metric contract. Blocks all dataset comparison suites.
