## Objective

Freeze one machine-readable metric contract for every paper benchmark before any additional confirmatory run.

## Why this is P0

The repository currently risks mixing incompatible directions and tasks:

- MCF target-true-sensitive `FS/GFS` are not ZeroUnlearn `Eff/Gen`.
- MUSE `PrivLeak` is currently described as lower-is-better in benchmark metadata, but the native target is closeness to zero (the paper treats roughly `[-5%, +5%]` as acceptable).
- A raw cross-benchmark average would mix unrelated units.

## Scope

- Define primary, secondary, and diagnostic metrics for MCF-ZU, MCF-Original, ZsRE, TOFU, RWKU, MUSE, MQuAKE, and WMDP.
- Record direction, unit, aggregation grain, tie handling, numerator/denominator, and permitted selection role.
- Correct MUSE PrivLeak semantics to a target interval or absolute distance from zero.
- Add schema validation that rejects missing or contradictory directions.
- Document that native metrics remain the paper source of truth; any normalized summary is secondary.

## Acceptance criteria

- [ ] A versioned metric-contract JSON exists and is covered by tests.
- [ ] MCF-ZU and MCF-Original have separate protocol IDs and non-overlapping column names.
- [ ] MUSE raw PrivLeak has a target interval; `abs(PrivLeak)` may be derived but raw values remain reported.
- [ ] Every table metric has a direction, unit, aggregation grain, and source evaluator.
- [ ] Validation rejects a universal raw-score average.
- [ ] Documentation specifies the main table family for every metric.

## Dependencies

None. This blocks all confirmatory benchmark issues.
