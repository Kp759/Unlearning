## Objective

Resolve the meaning of `MQuAKE-R`, then run the correct MQuAKE paper suite without relabeling the existing protocol.

## Decision required

Determine whether `MQuAKE-R` means:

1. the repository's ZeroUnlearn-style MQuAKE-CF-3k-v2 deletion-request track; or
2. MQuAKE-Remastered, which is a distinct benchmark and evaluator.

## Scope if ZeroUnlearn-style

- Preserve instance-first half-pool sampling, seeds 1–10, 50 forget and 1,000 retain instances.
- Main native columns: `Eff ↓` and locked `PPL ↓`.
- Post-selection extensions: AtomicGen leakage, standard/CoT multi-hop old-answer leakage, and retain atomic performance.
- Never describe same-request Eff as unseen-fact generalization.

## Acceptance criteria

- [ ] A benchmark identity decision is documented with source/revision hashes.
- [ ] Existing outputs cannot be labeled MQuAKE-Remastered unless evaluated there.
- [ ] Main versus extension metrics are tagged in the schema.
- [ ] Base, SURE variants, ZeroUnlearn, ROME, MEMIT, AlphaEdit, and eligible baselines use the same evaluator/model.
- [ ] Atomic and multi-hop prompts remain post-freeze evaluation-only.
- [ ] Complete per-seed receipts and confidence intervals are present.

## Dependencies

Depends on metric and baseline contracts.
