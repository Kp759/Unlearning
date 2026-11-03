# Router diagnostics, explained (router_diag_ours_control_v1, seed 1)

The router of SURE: one logistic head per protected fact, reading the last prompt token at block 19, behind a subject gate (a head may fire only if its fact's subject is named in the prompt). These diagnostics test the router alone, on every prompt the official evaluators route, plus prompts with no association, surface-perturbed prompts, and the router's own training prompts. Router dirs: `mcf_multiseed_ours_v1`, `zsre_multiseed_ours_v1`, `mquake_multiseed_ours_v1`, `rwku_multiseed_ours_v1`; recipe `ours`, subject match `fuzzy`.

## Where the errors come from: gate or classifier

Every routing error is labelled with its cause. If no head was even eligible, the **subject gate** blocked it (the subject is not named, or named in a form the match rejects). If the right head was eligible but did not fire, fired on the wrong prompt, or picked another row, the **classifier heads** are responsible. A false positive on a prompt identical to a forget prompt is a **label conflict** in the dataset (the same text is both forget and retain), not a router error.

| dataset | probes | errors | gate | classifier | label conflict | main cause |
|---|---|---|---|---|---|---|
| MCF | 13650 | 19 | 0 | 19 | 0 | **classifier** |
| zsRE | 8641 | 5 | 0 | 5 | 0 | **classifier** |
| MQuAKE | 2254 | 18 | 0 | 18 | 0 | **classifier** |
| RWKU | 645 | 271 | 74 | 197 | 0 | **classifier** |

Detailed reasons per dataset:

- **MCF**: `false_positive:eligible_head_above_threshold` 16, `false_negative:below_threshold` 3
- **zsRE**: `false_negative:below_threshold` 5
- **MQuAKE**: `false_positive:eligible_head_above_threshold` 10, `false_negative:below_threshold` 8
- **RWKU**: `false_negative:below_threshold` 196, `false_negative:subject_absent_from_prompt` 74, `false_positive:eligible_head_above_threshold` 1

Reading it: `below_threshold` means the right head saw the prompt but was not confident enough (the fix is more or better positives, e.g. rewrites for facts that got few); `ambiguous_top2` means two heads of the same subject tied (more same-subject negatives); `eligible_head_above_threshold` means a head fired on a prompt it should not (hard negatives); `subject_absent_from_prompt` can only be fixed in the gate (aliases, or a learned gate), never by the heads; `relation_gate` means the router's relation gate (scripts/fit_relation_gate.py) judged that the prompt asks for another relation.

## 1. Per-example errors (scores against the decision line)

**What.** One point per evaluated prompt, sorted by score, one panel per split. Should-fire prompts are scored by the correct fact's head; must-not-fire prompts by the best head the subject gate lets through. Above the dashed line (p = 0.5) the head fires. Prompts whose subject the gate does not match cannot fire at all and sit in the grey band at the bottom. Colour: green correct, red false fire, orange missed, purple wrong fact. Each dataset's `report/summary.md` lists the worst examples with their text.

**Why it matters.** A model of errors at the level of single prompts shows whether failures are systematic (one phrasing type, one fact with few positives) or scattered.

| dataset | recall | precision | F1 | FPR | FNR | correct row | misroutes |
|---|---|---|---|---|---|---|---|
| MCF | 0.980 | 0.902 | 0.939 | 0.001 | 0.020 | 0.980 | 0 |
| zsRE | 0.950 | 1.000 | 0.974 | 0.000 | 0.050 | 0.950 | 0 |
| MQuAKE | 0.985 | 0.982 | 0.984 | 0.006 | 0.015 | 0.985 | 0 |
| RWKU | 0.262 | 0.990 | 0.415 | 0.004 | 0.738 | 0.262 | 0 |

![MCF scores](mcf/seed1/report/scores_eval.png)
![zsRE scores](zsre/seed1/report/scores_eval.png)
![MQuAKE scores](mquake/seed1/report/scores_eval.png)
![RWKU scores](rwku/seed1/report/scores_eval.png)

## 2. Prompts with no protected association

**What.** Prompts about unprotected subjects, generic questions and chat turns: what a deployed router sees most of the time. Any firing is a false positive, and by construction it is the only way SURE can change the model's answer to such a prompt.

**Theory.** Off the routed set the model is exactly the base model, so locality loss is bounded by this false-fire rate. The gate makes it near zero for prompts that do not contain a protected name; what remains are prompts that mention a protected subject for an unrelated reason.

| dataset | null prompts | fired | deployment FPR [95% CI] |
|---|---|---|---|
| MCF | 468 | 118 | 25.2% [21.5%, 29.3%] |
| zsRE | 439 | 27 | 6.2% [4.3%, 8.8%] |
| MQuAKE | 872 | 44 | 5.0% [3.8%, 6.7%] |
| RWKU | 239 | 0 | 0.0% [0.0%, 1.6%] |

When the router fires by mistake on one of these prompts, does the answer change? Greedy outputs of the base model and of the paper's method (the rows of `l16_ours_v1`: router at 19, Δe at 16) on every fired prompt:

| dataset | fired prompts | output changed | 'I don't know'-like | rows from |
|---|---|---|---|---|
| MCF | 100 | 98 | 0 | `outputs/l16_ours_v1/mcf/seed1/ours` |
| zsRE | 27 | 26 | 3 | `outputs/l16_ours_v1/zsre/seed1/ours` |
| MQuAKE | 55 | 55 | 2 | `outputs/l16_ours_v1/mquake/seed1/ours` |
| RWKU | 0 | None | None | `None` |

Per-group breakdown (which kind of null prompt fires) and the generations themselves are in each `report/summary.md`, section 4.

![MCF null](mcf/seed1/report/scores_null.png)
![zsRE null](zsre/seed1/report/scores_null.png)
![MQuAKE null](mquake/seed1/report/scores_null.png)
![RWKU null](rwku/seed1/report/scores_null.png)

## 3. Positives vs negatives at blocks 16 and 19 (UMAP)

**What.** Twenty evaluation prompts that should fire (blue) and twenty hard negatives (red: must-not-fire prompts that contain a protected subject), embedded from the block-16 and block-19 residual stream. Every point is numbered; the numbered sentences are listed under the plots, and a ring / ✗ marks a prompt the router gets wrong. The numbers come from the full-dimensional states, not the 2-D picture: silhouette (how clustered the two classes are) and leave-one-out accuracy of a logistic and a 1-nearest-neighbour classifier.

**Theory.** The classifier is linear, so what matters is linear separability in full dimension; the 2-D projection is an illustration. If block 19 separates better than block 16, reading late is justified at the level of the representation, not only by the end metrics.

| dataset | layer | silhouette (full dim) | LOO logistic | LOO 1-NN |
|---|---|---|---|---|
| MCF | L16 | 0.070 | 0.800 | 0.800 |
| MCF | L19 | 0.072 | 0.875 | 0.800 |
| zsRE | L16 | 0.046 | 0.725 | 0.550 |
| zsRE | L19 | 0.033 | 0.675 | 0.375 |
| MQuAKE | L16 | 0.095 | 0.775 | 0.825 |
| MQuAKE | L19 | 0.084 | 0.800 | 0.800 |
| RWKU | L16 | 0.198 | 0.825 | 0.875 |
| RWKU | L19 | 0.171 | 0.800 | 0.825 |

40 points per panel: one leave-one-out error moves accuracy by 0.025.

![MCF umap](mcf/seed1/report/umap_L16_L19.png)
![zsRE umap](zsre/seed1/report/umap_L16_L19.png)
![MQuAKE umap](mquake/seed1/report/umap_L16_L19.png)
![RWKU umap](rwku/seed1/report/umap_L16_L19.png)

### 3b. Each head's own training prompts (scripts/router_train_umap.py)

**What.** The sentences the router is fit on (`linear_router_dataset.json`: about 20 positives and 20 negatives per fact), per fact: its positives against its own negatives. Numbers are averaged over facts, full dimension; the head columns are the deployed router's own decisions on these sentences at its read layer. Below: the confusion of every head on its own sentences (fit vs held-out), and one card per drawn fact (its 20 + 20 numbered sentences at L16 and L19, with the head's own 2x2 confusion). All cards are in each `train_umap/summary.md`.

| dataset | facts | + / − per fact | layer | silhouette mean | LOO 1-NN mean (worst) | LOO probe mean (worst) | head FN | head FP |
|---|---|---|---|---|---|---|---|---|
| MCF | 50 | 20.0 / 19.4 | L16 | 0.110 | 0.752 (0.575) | 0.868 (0.710) | – | – |
| MCF | 50 | 20.0 / 19.4 | L19 | 0.145 | 0.796 (0.581) | 0.900 (0.763) | 2/1000 | 402/971 |
| MCF | 50 | 20.0 / 19.4 | L26 | 0.170 | 0.845 (0.710) | 0.912 (0.774) | – | – |
| zsRE | 50 | 17.4 / 19.9 | L16 | 0.193 | 0.885 (0.737) | 0.931 (0.743) | – | – |
| zsRE | 50 | 17.4 / 19.9 | L19 | 0.211 | 0.898 (0.757) | 0.918 (0.743) | 4/872 | 110/994 |
| zsRE | 50 | 17.4 / 19.9 | L26 | 0.205 | 0.881 (0.711) | 0.907 (0.714) | – | – |
| MQuAKE | 105 | 15.6 / 19.9 | L16 | 0.247 | 0.930 (0.727) | 0.961 (0.861) | – | – |
| MQuAKE | 105 | 15.6 / 19.9 | L19 | 0.276 | 0.941 (0.788) | 0.962 (0.853) | 8/1643 | 83/2094 |
| MQuAKE | 105 | 15.6 / 19.9 | L26 | 0.292 | 0.942 (0.769) | 0.961 (0.848) | – | – |
| RWKU | 50 | 16.0 / 20.0 | L16 | 0.355 | 0.950 (0.800) | 0.972 (0.829) | – | – |
| RWKU | 50 | 16.0 / 20.0 | L19 | 0.374 | 0.953 (0.800) | 0.972 (0.886) | 13/801 | 1/1000 |
| RWKU | 50 | 16.0 / 20.0 | L26 | 0.379 | 0.954 (0.800) | 0.968 (0.875) | – | – |

![MCF confusion on training sentences](mcf/seed1/train_umap/confusion_train.png)
![MCF hardest fact](mcf/seed1/train_umap/fact_35.png)
![MCF training separability vs layer](mcf/seed1/train_umap/separability_vs_layer.png)
![zsRE confusion on training sentences](zsre/seed1/train_umap/confusion_train.png)
![zsRE hardest fact](zsre/seed1/train_umap/fact_6.png)
![zsRE training separability vs layer](zsre/seed1/train_umap/separability_vs_layer.png)
![MQuAKE confusion on training sentences](mquake/seed1/train_umap/confusion_train.png)
![MQuAKE hardest fact](mquake/seed1/train_umap/fact_76.png)
![MQuAKE training separability vs layer](mquake/seed1/train_umap/separability_vs_layer.png)
![RWKU confusion on training sentences](rwku/seed1/train_umap/confusion_train.png)
![RWKU hardest fact](rwku/seed1/train_umap/fact_13.png)
![RWKU training separability vs layer](rwku/seed1/train_umap/separability_vs_layer.png)

## 4. Confusion matrices

**What.** Binary: fired vs should fire, per split, with counts in every cell. N-way: which fact fired (columns) for which fact should have fired (rows); facts with no error are merged into one row and column so the errors stay visible. Correct counts are on the diagonal, errors in red: the last column holds the misses, the last row the false fires, other off-diagonal cells a wrong fact. RWKU adds a person-level matrix: firing a row of the right person counts as correct.

**Theory.** Proposition 2 of the paper splits the residual into routed-and-edited and missed; the 'none' column of the n-way matrix is exactly the missed term.

![MCF binary](mcf/seed1/report/confusion_binary.png) ![MCF n-way](mcf/seed1/report/confusion_nway.png)
![zsRE binary](zsre/seed1/report/confusion_binary.png) ![zsRE n-way](zsre/seed1/report/confusion_nway.png)
![MQuAKE binary](mquake/seed1/report/confusion_binary.png) ![MQuAKE n-way](mquake/seed1/report/confusion_nway.png)
![RWKU binary](rwku/seed1/report/confusion_binary.png) ![RWKU n-way](rwku/seed1/report/confusion_nway.png)
![RWKU person](rwku/seed1/report/confusion_person.png)

## 5. Classifier quality at every block, and the penultimate block

**What.** The same classifier, same data, same calibration rule, refit at every block and scored on the same evaluation prompts. The penultimate block is compared with block 19 under the deployed rule: it counts as better only if F1 rises and neither the false-positive rate nor the null false-positive rate gets worse.

**Theory.** Early blocks carry mostly the surface form; middle-to-late blocks carry the question's meaning, where paraphrases of one fact converge; the last blocks specialise towards the next token. A classifier that must recognise paraphrases should peak in the middle-to-late range, and block 19 is in that range.

| dataset | F1 at L19 | best block (F1) | penultimate F1 | penultimate null FPR vs L19 | verdict |
|---|---|---|---|---|---|
| MCF | 0.939 | L14 (0.948) | 0.924 | 0.262 vs 0.283 | keep L19 |
| zsRE | 0.974 | L3 (0.990) | 0.990 | 0.105 vs 0.062 | keep L19 |
| MQuAKE | 0.984 | L23 (0.988) | 0.988 | 0.069 vs 0.062 | keep L19 |
| RWKU | 0.415 | L3 (0.677) | 0.508 | 0.000 vs 0.000 | penultimate better |

Per-block tables (FPR, FNR, recall, null FPR, perturbed recall, calibration) are in each `layer_sweep/layer_choice.md`; robustness variants (hard negatives, augmentation, layer concatenation, no subject gate) in `layer_sweep/robustness.md`.

![MCF errors vs layer](mcf/seed1/layer_sweep/errors_vs_layer.png)
![zsRE errors vs layer](zsre/seed1/layer_sweep/errors_vs_layer.png)
![MQuAKE errors vs layer](mquake/seed1/layer_sweep/errors_vs_layer.png)
![RWKU errors vs layer](rwku/seed1/layer_sweep/errors_vs_layer.png)

## 6. RWKU

**What.** RWKU forgets people, not single facts, and its held-out probes ask other facts about the same person. The breakdown counts errors by probe type and by how the person is named.

**Theory.** The fact-level heads are trained with other facts about the same person as negatives, so on a held-out question about the person they correctly decline: a miss by the fact-level definition, but the wrong policy for an entity-level benchmark. That is why the person-level matrix, not the fact-level one, is the right lens here, and why the entity gate (fire on any probe that names a protected person) is the matching configuration.

| probe type | n | should fire | fired | correct | FN | FP | misroute |
|---|---|---|---|---|---|---|---|
| adversarial:affirmative suffix | 12 | 12 | 0 | 0 | 12 | 0 | 0 |
| adversarial:context hint | 26 | 26 | 6 | 6 | 20 | 0 | 0 |
| adversarial:cross lingual (french) | 3 | 3 | 1 | 1 | 2 | 0 | 0 |
| adversarial:cross lingual (german) | 3 | 3 | 0 | 0 | 3 | 0 | 0 |
| adversarial:cross lingual (spanish) | 7 | 7 | 0 | 0 | 7 | 0 | 0 |
| adversarial:incontext learning | 12 | 12 | 2 | 2 | 10 | 0 | 0 |
| adversarial:multiple choice | 19 | 19 | 0 | 0 | 19 | 0 | 0 |
| adversarial:prefix injection | 28 | 28 | 2 | 2 | 26 | 0 | 0 |
| adversarial:reverse query | 26 | 26 | 0 | 0 | 26 | 0 | 0 |
| adversarial:role play | 14 | 14 | 3 | 3 | 11 | 0 | 0 |
| adversarial:synonym manipulation | 23 | 23 | 0 | 0 | 23 | 0 | 0 |
| fill_in_blank | 43 | 43 | 18 | 18 | 25 | 0 | 0 |
| neighbor_fill_in_blank | 146 | 0 | 0 | 146 | 0 | 0 | 0 |
| neighbor_qa | 133 | 0 | 1 | 132 | 0 | 1 | 0 |
| qa | 50 | 50 | 11 | 11 | 39 | 0 | 0 |
| qa_paraphrase | 50 | 50 | 3 | 3 | 47 | 0 | 0 |
| same50_fill_in_blank | 40 | 40 | 40 | 40 | 0 | 0 | 0 |
| same50_qa | 10 | 10 | 10 | 10 | 0 | 0 | 0 |

## 7. Surface robustness and calibration

**What.** Forget prompts with typos, case changes and partial names; and the reliability of the deployed probabilities. The bias is shifted for recall on held-out rewrites, so probabilities are not meant to be calibrated; the reliability plot shows how far they are.

| dataset | perturbation family | n | recall |
|---|---|---|---|
| MCF | context_double_space | 21 | 1.000 |
| MCF | context_lowercase | 18 | 0.944 |
| MCF | context_no_punctuation | 9 | 0.889 |
| MCF | context_prefixed | 21 | 1.000 |
| MCF | context_suffixed | 21 | 1.000 |
| MCF | context_typo_delete | 29 | 1.000 |
| MCF | context_typo_double | 33 | 1.000 |
| MCF | context_typo_substitute | 36 | 0.972 |
| MCF | context_typo_transpose | 38 | 0.974 |
| MCF | subject_given_only | 19 | 0.000 |
| MCF | subject_lowercase | 14 | 1.000 |
| MCF | subject_possessive | 17 | 0.941 |
| MCF | subject_surname_only | 15 | 0.000 |
| MCF | subject_typo_delete | 41 | 0.951 |
| MCF | subject_typo_double | 38 | 0.947 |
| MCF | subject_typo_substitute | 35 | 0.943 |
| MCF | subject_typo_transpose | 34 | 0.912 |
| MCF | subject_uppercase | 11 | 0.909 |
| zsRE | context_double_space | 19 | 0.895 |
| zsRE | context_lowercase | 13 | 1.000 |
| zsRE | context_no_punctuation | 13 | 0.692 |
| zsRE | context_prefixed | 10 | 0.900 |
| zsRE | context_suffixed | 11 | 0.727 |
| zsRE | context_typo_delete | 20 | 0.950 |
| zsRE | context_typo_double | 30 | 0.967 |
| zsRE | context_typo_substitute | 22 | 0.727 |
| zsRE | context_typo_transpose | 18 | 0.944 |
| zsRE | subject_given_only | 13 | 0.000 |
| zsRE | subject_lowercase | 13 | 1.000 |
| zsRE | subject_possessive | 18 | 0.556 |
| zsRE | subject_surname_only | 15 | 0.000 |
| zsRE | subject_typo_delete | 11 | 0.818 |
| zsRE | subject_typo_double | 23 | 0.826 |
| zsRE | subject_typo_substitute | 24 | 1.000 |
| zsRE | subject_typo_transpose | 18 | 0.833 |
| zsRE | subject_uppercase | 9 | 1.000 |
| MQuAKE | context_double_space | 28 | 0.929 |
| MQuAKE | context_lowercase | 19 | 1.000 |
| MQuAKE | context_no_punctuation | 14 | 0.357 |
| MQuAKE | context_prefixed | 32 | 0.875 |
| MQuAKE | context_suffixed | 30 | 0.233 |
| MQuAKE | context_typo_delete | 61 | 0.918 |
| MQuAKE | context_typo_double | 59 | 0.932 |
| MQuAKE | context_typo_substitute | 70 | 0.800 |
| MQuAKE | context_typo_transpose | 64 | 0.984 |
| MQuAKE | subject_given_only | 22 | 0.000 |
| MQuAKE | subject_lowercase | 28 | 0.964 |
| MQuAKE | subject_possessive | 37 | 0.892 |
| MQuAKE | subject_surname_only | 25 | 0.000 |
| MQuAKE | subject_typo_delete | 57 | 0.719 |
| MQuAKE | subject_typo_double | 53 | 0.887 |
| MQuAKE | subject_typo_substitute | 73 | 0.699 |
| MQuAKE | subject_typo_transpose | 62 | 0.790 |
| MQuAKE | subject_unaccented | 1 | 1.000 |
| MQuAKE | subject_uppercase | 27 | 0.889 |
| RWKU | context_double_space | 33 | 0.333 |
| RWKU | context_lowercase | 34 | 0.294 |
| RWKU | context_no_punctuation | 29 | 0.138 |
| RWKU | context_prefixed | 23 | 0.304 |
| RWKU | context_suffixed | 47 | 0.319 |
| RWKU | context_typo_delete | 60 | 0.333 |
| RWKU | context_typo_double | 73 | 0.274 |
| RWKU | context_typo_substitute | 65 | 0.323 |
| RWKU | context_typo_transpose | 77 | 0.338 |
| RWKU | subject_given_only | 25 | 0.000 |
| RWKU | subject_lowercase | 29 | 0.276 |
| RWKU | subject_possessive | 34 | 0.294 |
| RWKU | subject_surname_only | 29 | 0.345 |
| RWKU | subject_typo_delete | 84 | 0.179 |
| RWKU | subject_typo_double | 62 | 0.226 |
| RWKU | subject_typo_substitute | 70 | 0.243 |
| RWKU | subject_typo_transpose | 61 | 0.230 |
| RWKU | subject_uppercase | 32 | 0.312 |

- MCF: pair-level ECE 0.094 over 174 pairs. ![MCF reliability](mcf/seed1/report/reliability.png)
- zsRE: pair-level ECE 0.053 over 102 pairs. ![zsRE reliability](zsre/seed1/report/reliability.png)
- MQuAKE: pair-level ECE 0.029 over 844 pairs. ![MQuAKE reliability](mquake/seed1/report/reliability.png)
- RWKU: pair-level ECE 0.765 over 3070 pairs. ![RWKU reliability](rwku/seed1/report/reliability.png)

## Summary

- **MCF**: 19 routing errors; 0 from the gate, 19 from the classifier heads, 0 label conflicts. Main cause: classifier.
- **zsRE**: 5 routing errors; 0 from the gate, 5 from the classifier heads, 0 label conflicts. Main cause: classifier.
- **MQuAKE**: 18 routing errors; 0 from the gate, 18 from the classifier heads, 0 label conflicts. Main cause: classifier.
- **RWKU**: 271 routing errors; 74 from the gate, 197 from the classifier heads, 0 label conflicts. Main cause: classifier.

