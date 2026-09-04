# RSNR Direct-Logit Baseline

## Purpose

Test whether RSNR-PreHead is doing more than trivial output control.

For an oracle-routed forgotten `(subject, relation)` query, the baseline leaves
all model weights and hidden states unchanged and edits only LM-head logits.

Primary canonical-surface mask:

\[
z'_t = z_t - \delta,\quad t \in T(o_{\mathrm{true}})
\]

Optional abstention boost:

\[
z'_u = z_u + \gamma,\quad u \in T(\text{"I don't know."})
\]

where `T(.)` is the set of tokenizer ids appearing in the corresponding
training-visible surface string.

Gate OFF is exact Base behavior.

## Fairness constraints

- Oracle routing is the same per-prompt `(subject, relation)` resolver used by RSNR.
- The mask uses only canonical `target_true` strings in
  `training_visible_forget_direct.json`.
- Held-out aliases are not used for mask construction.
- `target_new` is not used for calibration.
- Official paraphrase/neighborhood/fresh-retain text is not used for calibration.
- `delta`/`gamma` are calibrated only on the same locked 5-view training-safe
  corpus used by RSNR.
- Candidate settings are tried from weakest to strongest; the first setting
  satisfying the registered training gate is frozen before official evaluation.
- Trainable parameters: 0.

## Variants

### `suppress_only`

Calibrate the smallest `delta` such that all 50 facts satisfy worst-of-five
true-answer log-probability drop >= 2.0. No IDK preference is required.

### `suppress_plus_idk`

Calibrate the smallest `(delta, gamma)` such that all 50 facts satisfy both:

- worst-of-five true-answer log-probability drop >= 2.0; and
- worst-of-five `logP(IDK) - logP(true) >= 0.1`.

## Evaluation

Use exactly the same development evaluation families as RSNR-PreHead:

- 50 canonical forget prompts;
- 100 official unseen paraphrases;
- prompt-level sensitive neighborhoods;
- fresh disjoint retain sample;
- legacy CounterFact Eff/Gen/Spe/PPL;
- Eff_IDK / Gen_IDK;
- Sensitive_Eff / Sensitive_Gen generation leakage;
- five greedy retrieval attacks;
- forced-choice diagnostic reported separately;
- stochastic attacks at temperatures 0.7 and 1.0, top-p 0.95;
- true/alias leakage checks.

## Interpretation

Legacy CounterFact Eff/Gen are expected to decrease more directly than for
RSNR-PreHead because the baseline explicitly penalizes the canonical sensitive
answer logits. This does not by itself demonstrate better unlearning. The key
comparison is held-out alias/paraphrase/adversarial nondisclosure and generation
quality. If direct logit masking matches RSNR-PreHead everywhere, the simpler
output-control explanation should be preferred. If RSNR-PreHead is stronger,
especially on aliases or adversarial retrieval not trivially covered by the
canonical token mask, that supports the latent-null intervention as doing more
than token masking.
