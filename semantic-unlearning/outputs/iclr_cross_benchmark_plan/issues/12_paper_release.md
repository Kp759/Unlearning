## Objective

Run the frozen confirmatory matrix and produce the reproducible ICLR result package.

## Scope

- Freeze method definitions, hyperparameters, seeds/targets, model revisions, evaluator revisions, and utility gates.
- Execute final runs without mid-course tuning or seed rejection.
- Generate factual, entity/data, and hazardous-capability native tables.
- Generate Pareto plots, robustness tables, ablations, cost table, and limitations.
- Independently reproduce at least one run per benchmark from receipts.
- Package commands, configs, manifests, evaluator outputs, hashes, and environment lock.

## Acceptance criteria

- [ ] Every main-table row passes the publication acceptance gate.
- [ ] Claims are generated from frozen results and include losses/ties, not only wins.
- [ ] SURE-F and SURE-U100K names and differences are explicit.
- [ ] Adapted baselines and benchmark-specific SURE extensions are labeled.
- [ ] No FS/GFS row is compared directly with incompatible Eff/Gen.
- [ ] Artifact completeness and independent reproduction checks pass.
- [ ] The release bundle can regenerate all tables and figures from raw outputs.

## Dependencies

Depends on completion of metric/baseline/result-registry work and all dataset suites selected for the final paper.
