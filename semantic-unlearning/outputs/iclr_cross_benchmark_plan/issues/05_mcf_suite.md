## Objective

Produce two scientifically separate MCF suites: an apples-to-apples ZeroUnlearn comparison and an original-fact-deletion stress test.

## Track A: MCF-ZU main table

- Sensitive/unwanted field follows the upstream ZeroUnlearn MCF convention.
- Report native `Eff ↓`, `Gen ↓`, `Spe ↑`, Spe-success, and locked PPL ratio.
- Run Base, SURE-F, SURE-U100K, ZeroUnlearn, ROME, MEMIT, AlphaEdit, GA, NPO, and other eligible factual baselines under one evaluator.

## Track B: MCF-Original stress test

- Original `target_true` is sensitive.
- Report `FS ↑`, `GFS ↑`, Spe-success/margin, exact retain KL, semantic leakage, and PPL.
- Rerun every included baseline under this task; do not import published MCF-ZU numbers.

## Acceptance criteria

- [ ] Protocol IDs, manifests, and table outputs cannot mix Track A and Track B.
- [ ] All main-table methods use seeds 1–10 and complete receipts.
- [ ] SURE-F and SURE-U100K differ only by the locked external cache treatment.
- [ ] W10K/100K utility and locality gains are evaluated without held-out prompt selection.
- [ ] Native and stress-test tables include Base, raw counts, mean, sample SD, and 95% CI.
- [ ] Any target-aware/paraphrase-trained ablation is labeled and excluded from the benchmark-neutral main row.

## Dependencies

Depends on metric contract, baseline contract, and result registry.
