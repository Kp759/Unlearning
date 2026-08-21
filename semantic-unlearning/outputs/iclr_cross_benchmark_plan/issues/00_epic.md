## Goal

Build a publication-grade ICLR result package showing where SURE improves the benchmark-native forgetting–utility frontier across factual, entity, passage, and hazardous-knowledge unlearning.

The goal is strong and reproducible evidence, not post hoc tuning until every scalar is best.

## Non-negotiable reporting rules

- Use benchmark-native metrics in main tables.
- Never compare MCF target-true-sensitive `FS/GFS` directly with ZeroUnlearn `Eff/Gen`.
- Keep official held-out probes evaluation-only unless a native protocol explicitly permits calibration.
- Label every method×dataset result Native, Reproduced, Adapted, or Unsupported.
- Require immutable model/data/evaluator identities and complete per-run receipts.
- Report utility, locality, privacy, collapse, robustness, and compute alongside forgetting.
- Use Pareto/constrained-forgetting claims; any normalized cross-benchmark scalar is secondary.

## Phase 0 — contracts and infrastructure

- [ ] #27
- [ ] #30
- [ ] #37

Exit gate: metric directions, method provenance, data roles, aggregation, and artifact completeness are machine-validated.

## Phase 1 — factual benchmark foundation

- [ ] #28
- [ ] #29
- [ ] #31
- [ ] #32

Exit gate: Base and strong baselines reproduce under one evaluator; all ten-seed rows have complete receipts.

## Phase 2 — entity and profile unlearning

- [ ] #33
- [ ] #34

Exit gate: native TOFU and RWKU efficacy, locality, privacy/robustness, and utility suites pass predeclared reporting gates.

## Phase 3 — passage and hazardous-domain extensions

- [ ] #35
- [ ] #36

Exit gate: MUSE and WMDP run end to end against strong maintained baselines, with head-only suppression separated from deeper contextual editing.

## Phase 4 — confirmatory paper package

- [ ] #38

Exit gate: every paper row and figure regenerates from immutable raw artifacts, commands, configs, and receipts.

## Proposed main-paper table families

1. Factual unlearning: MCF-ZU, ZsRE, MQuAKE.
2. Entity/data unlearning: TOFU, RWKU, MUSE.
3. Hazardous capability unlearning: WMDP.
4. Cross-benchmark summary: win/tie/loss, average rank, Pareto-front membership, and compute—not a mean of raw native metrics.

## Publication acceptance gate

A row enters the main paper only if its source/model/tokenizer/dataset/evaluator are pinned, data roles are audited, per-run artifacts are complete, Base and a strong baseline reproduce, uncertainty and raw counts are reported, utility/collapse gates pass, and no held-out result selected the checkpoint.
