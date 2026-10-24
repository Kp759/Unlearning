# SURE-R: making the router robust, measured by SCALPEL

The first SCALPEL pilot showed SURE's held-out leakage is mostly **routing**: 48.7 of 57.8 residual points came from the router not firing, against 9.1 from rows that did not suppress the answer. SURE-R adds one switch per problem. Every switch is off in the shipped SURE, so the paper pipeline is unchanged (the repository's router tests pass identically before and after). Each switch is also its own SCALPEL method, so the benchmark measures it alone.

| Problem (pilot) | Fix | Where |
|---|---|---|
| Typos and case changes never route (0/18) | **Fuzzy subject eligibility.** Subject words are matched in the decoded prompt after normalisation (case, accents, punctuation, spacing), with ≤ 1 edit per word of ≥ 4 characters, or the name written as one word. The fit (`eligibility_matrix`) and the runtime hook use the same rule | `scripts/linear_router.py` (`--subject-match fuzzy`) |
| The router learned only the English cloze request: questions 6/30, other languages 7/67, multi-turn 3/22 | **Views.** The base model rewrites each request as paraphrases, a direct question, translations, and a multi-turn follow-up that uses "they"/"it". Views are rendered exactly as the deployed chat model sees them, and kept only if the base model answers them with the request's answer. They become router positives: the train families, and held-out families for calibration and audit | `scalpel/methods/sure_robust.py` (`make_views`) |
| Looser matching risks false fires | **Hard negatives** (ablation only, off by default). The base model writes other questions and sentences about the same subject, which become negatives for that fact's head. On the pilot they pushed every chat-format view below threshold, see below | `--extra-negatives` (`sure_r_hardneg`) |
| Subject never named: multi-hop 0/12 | **Experimental subject-free firing.** Heads are fitted against every prompt (`--unmasked-fit`), and a head may fire without an eligible subject at deployed logit ≥ 3 | `--free-fire-logit` (`sure_r_free`) |
| 8 of 49 routed exact prompts still answered | **Larger row budget** (60 updates per fact), then **refinement** of each row on every view routed to it. Each fact has its own objective, and steps and the norm cap scale with the layer's boundary norm. SURE's shipped trainer keeps the checkpoint with the best *worst-case* probability over all facts, so one stubborn fact freezes every row early: best step 15 of 900, rows at the first trust radius, 0/15 facts at target | `refine_rows` |
| No abstention: 0% refusals | **Refusal target.** Rows are refined toward " I don't know." + EOS while keeping the answer suppressed | `sure_r_idk` |
| Hidden, not removed: weights unchanged | **Experimental baking.** The rows are compiled into a closed-form edit of `down_proj` at SURE's layer: W += R Kᵀ(λC + KKᵀ)⁻¹, MEMIT's solve, where C is the Wikipedia covariance, K are the views' boundary MLP inputs, and R are the rows. The router hooks are then removed | `sure_r_bake` |
| Uncached, batch-1 generation | **KV-cached, batched, boundary-aware decoder.** Exact for `write_mode: last`: on the pilot artifact, 48/48 generations are identical and it is 3.2× faster | `_SureRuntime.generate_cached` |

## The rule that keeps this honest

Nothing SURE-R learns from comes from SCALPEL:

- **Allowed:** the split's requests (prompt + answer), the base model's own rewrites of them, and the deployment's chat rendering.
- **Never used:** SCALPEL's probes, its relation templates, its languages list, its surface perturbations.

Matching benchmark templates would inflate the dev numbers. It would not survive the hidden test split, which can use private templates. Report dev and test together.

## Variants (SCALPEL methods)

| Method | Views | Fuzzy | Hard neg. | Free fire | Rows target | Baked |
|---|---|---|---|---|---|---|
| `sure` (shipped) | – | – | – | – | suppress (30/fact) | – |
| `sure_r` | all | ✓ | – | – | suppress + refine | – |
| `sure_r_noviews` | – | ✓ | – | – | suppress + refine | – |
| `sure_r_exact` | all | – | – | – | suppress + refine | – |
| `sure_r_hardneg` | all | ✓ | ✓ | – | suppress + refine | – |
| `sure_r_free` | all | ✓ | – | logit ≥ 3 | suppress + refine | – |
| `sure_r_idk` | all | ✓ | – | – | " I don't know." | – |
| `sure_r_bake` | all | ✓ | – | – | suppress + refine | ✓ (weights only) |

## First local result: Qwen2.5-0.5B, the same 15 pilot requests for every column, seed 1

Only 12 of 64 generated views survive validation, because the 0.5B model answers few rephrasings; a 3B model should keep many more. Rows show the % of probes still recalled (forget, lower is better) or retained (retain).

| Family | `sure` (shipped) | `sure_r` + hard negatives | **`sure_r`** |
|---|---|---|---|
| direct | 67% | 0% | **0%** |
| context prefix | 76% | 0% | **0%** |
| surface (typos, case) | 100% | 0% | **0%** |
| paraphrase | 60% | 33% | **13%** |
| question | 90% | 100% | **40%** |
| cross-lingual | 94% | 100% | **50%** |
| multi-turn | 86% | 100% | **43%** |
| multi-fact | 100% | 50% | **50%** |
| multi-hop | 100% | 100% | 100% |
| retain, all 8 families | 100% | 100% | 100%, except `in_context` 87% |
| route recall (forget) / false fires (retain) | 65% / 7.0% | 56% / 0.0% | **76%** / 11.3% |
| answer leaked in free generation | 86% | 65% | **48%** |
| **SCALPEL** (E / L) | **0.0** (2.9 / 100) | 8.1 (2.3 / 100) | **72.2** (66.2 / 97.0) |

- **Fuzzy matching** fixes typos and case changes.
- **Refinement** fixes the edits: no residual remains when the router fires.
- **Views** move questions, translations and follow-ups.
- **Hard negatives** undo the views. The router reads the chat template's final token, where a same-subject question looks like the forgotten one. A read point at the end of the user's message would address this; it is a candidate next change.
- **Multi-hop** remains, by design, the open problem: the subject is never named.

Run on Wulver, after the 3B splits exist (`smoke` or `full`):

```bash
bash scripts/submit_scalpel_compare.sh robust llama32_3b                 # 5 seeds x 7 variants
SEEDS="1" bash scripts/submit_scalpel_compare.sh robust llama32_3b       # seed 1 only
```

**What to read:**
- **SURE vs `sure_r`:** E per family (surface, question, crosslingual, multiturn should rise) against L per family (`name_collision`, `same_subject` and `subject_mentioned` are where fuzzy matching and views can cost locality).
- **The routing decomposition:** route recall, and how much residual comes from routing versus from edits.
- **The ablations:** `noviews` isolates fuzzy matching; `exact` isolates the views.
- **`sure_r_bake`:** the strict tier (weights only) should finally move.
