## Objective

Implement and evaluate SURE-F/SURE-U100K on the benchmark-native TOFU setup with the official Full target model and retain-only reference.

## Scope

- Use official forget01/forget05/forget10 roles and role-specific model/tokenizer revisions.
- Define an author/profile-aware, multi-token answer adapter without exposing perturbed evaluation prompts.
- Run the native evaluator: Forget Quality, KS statistic/effect size, Model Utility, and retain/real-author/world-fact components.
- Add maintained robustness diagnostics such as MIA, extraction strength, and exact memorization as secondary results.
- Compare against retrain, GA, GradDiff, NPO, SimNPO, PDU, RMU, and other eligible maintained baselines from one pinned framework.

## Acceptance criteria

- [ ] Full and retain reference models reproduce expected native metrics before unlearning.
- [ ] Forget Quality is reported with its underlying distribution/effect size, not only a p-value.
- [ ] SURE-F/U100K share the locked architecture contract apart from external cache treatment.
- [ ] Perturbed, paraphrased, real-author, world-fact, and final retain evidence is evaluation-only unless native protocol explicitly permits calibration.
- [ ] At least five independent final runs or official split replicates are reported with uncertainty.
- [ ] Every baseline uses the same model/evaluator and disclosed tuning budget.

## Dependencies

Depends on metric contract, baseline contract, and result registry.
