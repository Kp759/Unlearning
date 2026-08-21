## Objective

Build one fail-closed result registry and statistical aggregation pipeline for all paper tables.

## Scope

- Ingest per-run receipts, raw evaluator JSON, configs, hashes, and compute metadata.
- Validate metric contract, model/dataset/evaluator identity, data roles, seed coverage, and artifact completeness.
- Produce mean, sample SD, bootstrap/run-level 95% CI, discrete raw counts, and paired comparisons where valid.
- Keep training-run uncertainty separate from evaluation-example uncertainty.
- Generate native dataset tables, Pareto-front summaries, win/tie/loss, average rank, and compute tables.
- Reject terminal-captured or manually transcribed final results.

## Acceptance criteria

- [ ] One schema covers all benchmark-native metrics without forcing a universal score.
- [ ] Missing seeds, hashes, receipts, or direction metadata fail CI.
- [ ] Aggregates are reproducible from raw files with one command.
- [ ] Base and oracle/reference rows share evaluator identities with methods.
- [ ] Pareto and constrained-forgetting rules are versioned and predeclared.
- [ ] Generated tables include provenance links to every contributing run.

## Dependencies

Depends on the metric contract. Dataset suites depend on this issue.
