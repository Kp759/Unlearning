# Router diagnostics: mcf seed 1 — L19 router

Router: `/mmfs1/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/mcf_multiseed_ours_relgate_v1/seed1/L19/router` · gate `threshold` · fires the best eligible head at p >= 0.5 (calibrated bias) · subject match `fuzzy`.
Probes: eval 13650. Hook parity (real bank hook vs cached decision): {"batched": 0, "single": 0, "perturbed_batched": 0, "perturbed_single": 0} mismatches.

## 1. Scores against the decision line

![scores](scores_eval.png)

### False negatives (lowest p(true)): 3 shown

| split/group | score | fired | true | reason | prompt |
|---|---|---|---|---|---|
| paraphrase/forget_paraphrase | 0.257 | - | 5 | below_threshold | He died in Far Rockaway on April 23, 1965. Irish Free State, in |
| paraphrase/forget_paraphrase | 0.400 | - | 44 | below_threshold | Enrollment in 2004 was 2,792 students. Jason Rebello, the |
| paraphrase/forget_paraphrase | 0.457 | - | 7 | relation_gate | References Category:Villages in Piotrków County Pernik Peninsula, in |

### False positives (highest score): 0 shown

none

### Misroutes: 0 shown

none

## 2. Confusion matrices

![binary](confusion_binary.png)

| set | n | TP | FN | FP | TN | FPR | FNR | precision | recall | F1 | correct row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forget | 50 | 50 | 0 | 0 | 0 | – | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| neighborhood | 500 | 0 | 0 | 0 | 500 | 0.000 | – | – | – | – | – |
| paraphrase | 100 | 97 | 3 | 0 | 0 | – | 0.030 | 1.000 | 0.970 | 0.985 | 0.970 |
| retain | 13000 | 0 | 0 | 0 | 13000 | 0.000 | – | – | – | – | – |
| all | 13650 | 147 | 3 | 0 | 13500 | 0.000 | 0.020 | 1.000 | 0.980 | 0.990 | 0.980 |

Error reasons: false_negative:below_threshold 2, false_negative:relation_gate 1

![nway](confusion_nway.png)

## 4. Prompts with no association

![null](scores_null.png)

| group | n | should fire | fired | fire rate [95% CI] | ungated heads p >= 0.5 |
|---|---|---|---|---|---|
| null_generic | 40 | 0 | 0 | 0.000 [0.000, 0.088] | 40 |
| null_ppl_prefix | 99 | 0 | 0 | 0.000 [0.000, 0.037] | 99 |
| null_protected_other_relation | 179 | 0 | 10 | 0.056 [0.031, 0.100] | 179 |
| null_protected_same_answer_type | 20 | 0 | 3 | 0.150 [0.052, 0.360] | 20 |
| null_unseen_subject | 150 | 0 | 0 | 0.000 [0.000, 0.025] | 150 |

**Deployment false-positive rate** (fires on a must-not-fire null prompt; `null_protected_same_answer_type` left out, see below): 0.021 [0.012, 0.039], 10/468. With it: 0.027, 13/488.

`null_protected_same_answer_type`: the protected subject asked with a relation in the fact's answer group (e.g. languages spoken for a mother-tongue fact). The router treats these as the same fact and never trains on them as negatives, since they leak the same answer.

### False negatives (lowest p(true)): 0 shown

none

### False positives (highest score): 13 shown

| split/group | score | fired | true | reason | prompt |
|---|---|---|---|---|---|
| null/null_protected_other_relation | 1.000 | 38 | - | eligible_head_above_threshold | Edinburgh Art Festival is located in |
| null/null_protected_other_relation | 1.000 | 26 | - | eligible_head_above_threshold | Grande Arche is located in |
| null/null_protected_same_answer_type | 0.999 | 21 | - | eligible_head_above_threshold | iPhone 1 was created by |
| null/null_protected_other_relation | 0.999 | 9 | - | eligible_head_above_threshold | BMW M5 was created in the country of |
| null/null_protected_other_relation | 0.999 | 13 | - | eligible_head_above_threshold | Toyota Tercel was created in the country of |
| null/null_protected_other_relation | 0.999 | 39 | - | eligible_head_above_threshold | MacAlpine Hills is located in |
| null/null_protected_other_relation | 0.998 | 26 | - | eligible_head_above_threshold | Grande Arche is in |
| null/null_protected_same_answer_type | 0.998 | 13 | - | eligible_head_above_threshold | Toyota Tercel was created by |
| null/null_protected_other_relation | 0.998 | 5 | - | eligible_head_above_threshold | Irish Free State is located in |
| null/null_protected_other_relation | 0.996 | 22 | - | eligible_head_above_threshold | Palmer Archipelago is located in |
| null/null_protected_other_relation | 0.994 | 20 | - | eligible_head_above_threshold | Shebelle River is located in |
| null/null_protected_same_answer_type | 0.990 | 9 | - | eligible_head_above_threshold | BMW M5 was created by |
| null/null_protected_other_relation | 0.976 | 38 | - | eligible_head_above_threshold | Edinburgh Art Festival is in |

### Misroutes: 0 shown

none

## 5. Perturbed forget / paraphrase prompts (recall under surface noise)

| family | n | recall | correct row | FN reasons |
|---|---|---|---|---|
| context_double_space | 21 | 1.000 | 1.000 |  |
| context_lowercase | 18 | 0.944 | 0.944 | below_threshold 1 |
| context_no_punctuation | 9 | 0.889 | 0.889 | below_threshold 1 |
| context_prefixed | 21 | 1.000 | 1.000 |  |
| context_suffixed | 21 | 0.810 | 0.810 | relation_gate 4 |
| context_typo_delete | 29 | 1.000 | 1.000 |  |
| context_typo_double | 33 | 1.000 | 1.000 |  |
| context_typo_substitute | 36 | 0.889 | 0.889 | below_threshold 1, relation_gate 3 |
| context_typo_transpose | 38 | 0.974 | 0.974 | relation_gate 1 |
| subject_given_only | 19 | 0.000 | 0.000 | subject_absent_from_prompt 19 |
| subject_lowercase | 14 | 1.000 | 1.000 |  |
| subject_possessive | 17 | 0.882 | 0.882 | relation_gate 2 |
| subject_surname_only | 15 | 0.000 | 0.000 | subject_absent_from_prompt 15 |
| subject_typo_delete | 41 | 0.951 | 0.951 | subject_absent_from_prompt 2 |
| subject_typo_double | 38 | 0.947 | 0.947 | subject_absent_from_prompt 2 |
| subject_typo_substitute | 35 | 0.943 | 0.943 | below_threshold 1, subject_absent_from_prompt 1 |
| subject_typo_transpose | 34 | 0.912 | 0.912 | subject_absent_from_prompt 2, below_threshold 1 |
| subject_uppercase | 11 | 0.909 | 0.909 | below_threshold 1 |

## 6. Positives vs negatives at L16 and L19

![umap](umap_L16_L19.png)

| layer | silhouette (cosine, full dim) | silhouette (2D) | LOO logistic | LOO 1-NN |
|---|---|---|---|---|
| L16 | 0.088 | 0.222 | 0.850 | 0.650 |
| L19 | 0.100 | 0.335 | 0.850 | 0.700 |

20 points: each LOO error moves accuracy by 0.05. Negatives (hard) prefer must-not-fire prompts that contain a protected subject.

## 7. Calibration

![reliability](reliability.png)

ECE over 174 subject-eligible (prompt, head) pairs: 0.094. deployed probabilities: the calibrated bias moves the cutoff to 0.5 for recall, so they are not meant to be calibrated probabilities.

