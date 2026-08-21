## Objective

Determine whether SURE can remove hazardous-domain knowledge on WMDP without reducing utility or merely suppressing answer-choice rows.

## Scope

- Pin corrected WMDP Bio/Cyber data and `lm-evaluation-harness` v0.4.2.
- Keep WMDP multiple-choice questions and MMLU evaluation-only; train on official forget corpora.
- Run head-only SURE as an explicit feasibility ablation.
- Predeclare a contextual internal-layer SURE extension if head-only fails robustness or utility.
- Report WMDP-Bio/Cyber/Chem, overall and subject-level MMLU, MT-Bench, other-domain retention, and adversarial/probe/relearning recovery.
- Compare against RMU, NPO, SimNPO, GA/GradDiff/PDU where supported, using identical model and evaluator.

## Acceptance criteria

- [ ] Base and RMU reference reproduce within locked tolerance.
- [ ] Main forgetting goal is accuracy near the four-choice 25% chance floor; below-chance behavior is audited for answer-row gaming.
- [ ] Head-only and contextual-layer variants are separately labeled.
- [ ] MMLU, subject-level locality, and MT-Bench utility gates are predeclared.
- [ ] Evaluation questions never contribute gradients or checkpoint selection.
- [ ] Recovery tests distinguish decoder suppression from deeper removal.

## Dependencies

Depends on metric contract, baseline contract, and result registry. Start after passage-level adapter design is frozen.
