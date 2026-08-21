## Objective

Complete the strict target-only RWKU SURE adapter, validate `RWKU-H-W1K`, and scale only after native locality/robustness gates pass.

## Scope

- Finish the Stephen King W1K checkpoint and open official probes only after receipt verification.
- Report Level 1/2/3, attack-type breakdown, MIA FM/RM, neighbor FB/QA, general ability, reasoning, truthfulness, factuality, fluency, frozen-head, and PPL.
- Preserve the target-only rule: no official forget/neighbor/MIA/utility probe enters training or selection.
- If W1K is viable, prefreeze one U10K/U100K design and scale targets `1 → 10 → 20 → 50`.
- Compare against native RWKU baselines (ICU, RepE, GA, DPO, NPO, refusal tuning) and clearly labeled adaptations.

## Acceptance criteria

- [ ] W1K official evaluation is complete and immutable before any redesign.
- [ ] Passing generated atomic gates is not labeled entity erasure.
- [ ] Neighbor and utility degradation thresholds are predeclared.
- [ ] Adversarial, MIA, and frozen-head recovery are reported.
- [ ] Batch-target results use multiple entities and preserve per-target raw counts/Wilson intervals.
- [ ] Any probe-assisted track is separated from the strict target-only main result.

## Dependencies

Depends on metric contract, baseline contract, and result registry.
