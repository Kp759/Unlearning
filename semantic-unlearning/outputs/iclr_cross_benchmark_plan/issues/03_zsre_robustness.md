## Objective

Add post-freeze ZsRE semantic-leak and generation-collapse diagnostics so near-zero teacher-forced `Eff/Gen` cannot conceal generic refusal, lexical leakage, or repetitive degeneration.

## Scope

- Normalized sensitive-answer and alias containment in free generation.
- Sensitive-answer probability ratio versus Base.
- `Unknown`/refusal rate.
- Repetition, loop, empty-output, and truncation rates.
- Retain/locality generation quality and collateral sensitive-token effects.
- Optional blinded LLM judge only after deterministic metrics and prompts are frozen.

All official rephrases, locality prompts, and robustness outputs remain evaluation-only and cannot select or repair a checkpoint.

## Acceptance criteria

- [ ] Deterministic diagnostics are versioned and tested on known pass/fail fixtures.
- [ ] Every metric reports raw numerator/denominator and prompt grain.
- [ ] Base, ZeroUnlearn, SURE-F, and SURE-U100K run through identical decoding settings.
- [ ] Collapse diagnostics are shown beside, not substituted for, native Eff/Gen/Spe/PPL.
- [ ] An access receipt proves evaluation occurred after checkpoint freeze.

## Dependencies

Depends on the ZsRE reproduction issue; does not block its native-metric rerun.
