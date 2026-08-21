## Objective

Reproduce the canonical 10-seed ZsRE Base, SURE-F, SURE-U100K, and ZeroUnlearn rows under one immutable local protocol with complete artifacts.

## Current evidence

The checked-in SURE aggregate reports `Eff 0.1317`, `Gen 0.6617`, `Spe 26.1068`, and `PPL 11.9750`, but it is terminal-captured, lacks the exact final execution commit, and references output roots that are absent locally. Local Base PPL is `11.0625` while the public ZeroUnlearn Base row is `12.88`.

## Scope

- Pin model, tokenizer, ZsRE data, Wikidata/PPL fixture, evaluator, dtype, and seeds 1–10.
- Run Base once per seed receipt, vendored ZeroUnlearn, SURE-F, and SURE-U100K.
- Preserve per-record raw predictions and macro/micro counts.
- Record checkpoint, split, token-ID, source, code, and environment hashes.
- Resolve PPL by proving identical Base token IDs and scalar across every method runner.
- Compare local Base/ZeroUnlearn with the public table without hard-coding published numbers.

## Acceptance criteria

- [ ] All 10 seed directories exist for every reported method.
- [ ] Every seed has split manifest, config, evaluator JSON, raw counts, checkpoint receipt, and hashes.
- [ ] Base Eff/Gen/Spe/PPL are byte-identical across method runners for a given seed/fixture.
- [ ] PPL fixture token IDs and SHA-256 are recorded.
- [ ] Mean, sample SD, and 95% CI are generated from raw per-seed artifacts.
- [ ] A parity report explains any remaining difference from the public ZeroUnlearn table.
- [ ] No terminal-only aggregate is accepted by the result registry.

## Dependencies

Depends on the cross-benchmark metric contract and result-registry schema.
