# SURE-R: making the router robust, measured by LETHE-Bench

The first LETHE-Bench pilot showed SURE's held-out leakage is mostly **routing**: 48.7 of 57.8 residual points came from the router not firing, against 9.1 from rows that did not suppress the answer. SURE-R adds one switch per problem. Every switch is off in the shipped SURE, so the paper pipeline is unchanged (the repository's router tests pass identically before and after). Each switch is also its own LETHE-Bench method, so the benchmark measures it alone.

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

Nothing SURE-R learns from comes from LETHE-Bench:

- **Allowed:** the split's requests (prompt + answer), the base model's own rewrites of them, and the deployment's chat rendering.
- **Never used:** LETHE-Bench's probes, its relation templates, its languages list, its surface perturbations.

Matching benchmark templates would inflate the dev numbers. It would not survive the hidden test split, which can use private templates. Report dev and test together.

## Variants (LETHE-Bench methods)

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
| **LETHE-Bench** (E / L) | **0.0** (2.9 / 100) | 8.1 (2.3 / 100) | **72.2** (66.2 / 97.0) |

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

## The paper's method: `sure_paper` (one forward pass at layer 19)

`sure_paper` is "ours" in LETHE-Bench. The classifier reads layer 19 at the last prompt token, and Δe is added **at the same layer 19**, at the last prompt token and every generated token (`linear_router` write mode `query`), in **one forward pass**. The rows come from the paper's feedback trainer (`scripts/train_mcf_jlens_feedback_rows.py`): on-policy generated continuations, output loss + KL keep + teacher-forced gold views, with the logit lens off (`--lens-type none`). The router has no PCA (`fit_linear_router.py` default; `--pca-dims 0,64,256` restores the old search).

| Method | Classifier input | Δe | Forward passes |
|---|---|---|---|
| `sure_paper` | layer 19 | layer 19, last prompt token + every generated token | 1 |
| `sure_paper_l16` (ablation) | layer 19 | layer 16, same positions | 2: route, then write |
| `sure` (original SURE) | layer 19 | layer 19, last prompt token only | 1 |

The two-pass design needed a routing pass because layer 16 runs before the classifier's layer 19. With both at layer 19, one hook reads the state, decides and adds Δe. `tests/test_linear_router_query.py` checks that the one-pass output equals the two-pass output at the same layer exactly. `scripts/run_l16_paper_one.sh` uses the same default: `WRITE_LAYER` = `ROUTER_LAYER`, tag `onepass_l19_v1`, and `WRITE_LAYER=16` for the ablation under the existing `l16_paper_v1` tag.

**Router recipe for `sure_paper`:** 20 positives and 20 negatives per fact, no PCA (the `fit_linear_router.py` defaults; `--positives-per-fact 0 --negatives-per-fact 0 --pca-dims 0,64,256` is the earlier recipe, which `sure` and `sure_r` keep).
- **Positives:** the dataset's own prompts first, then extra content-free context prefixes (`EXTRA_CONTEXT_PREFIXES`, none of them a LETHE-Bench probe prefix). Every added prompt is checked for subject eligibility, and about a quarter go to calibration / audit.
- **Negatives:** up to half same-subject competitors, the rest subject transplants, round robin over fit / calibration / audit. SURE's donor rules (subject type, answer group, protected relation) stay. A fact with few valid donors therefore gets fewer than 20: on the 15-request check the counts ranged from 0 to 20, mean 12; 50-request splits have about three times the donors.

**Local check: Qwen2.5-0.5B, the same 15 requests, lens attack only, short row training (2 rounds).** Percent recalled (forget) / retained (retain):

| | `sure` | `sure_r` | `sure_paper` | `sure_paper`, 20 / 20 router |
|---|---|---|---|---|
| direct | 67% | 0% | 7% | 7% |
| paraphrase / question | 60% / 90% | 13% / 40% | 13% / 60% | 13% / 60% |
| surface (typos, case) | 100% | **0%** | 100% | 100% |
| cross-lingual / multi-turn | 94% / 86% | 50% / 43% | 75% / 43% | 75% / 43% |
| answer in free generation | 86% | 48% | **36%** | **36%** |
| route recall / false fires | 65% / 7.0% | 76% / 11.3% | 65% / 7.0% | 65% / 8.5% |
| LETHE-Bench (E / L / R) | 0.0 (2.9 / 100 / 0) | **72.2** (66.2 / 97.0 / 50.0) | 20.4 (6.2 / 97.0 / 66.7) | 20.4 (6.2 / 97.0 / 66.7) |

- The one-pass edit at every generated token gives the lowest leak in free generation and the best R.
- Its router is SURE's plain one: no fuzzy names (typos 100%) and no rewrites, so routing, not editing, now limits E.
- The 20 / 20 router changed nothing here. Its 15 extra positives are prefix variants of the same request sentence, so they add no new phrasings.


## `ours` (shown as SURE on the leaderboard): SURE-R's router + the one-pass layer-19 edit

`ours` puts `sure_paper`'s edit on SURE-R's router. `ours_l16` is the same method with Δe at layer 16 in two passes. Every router training prompt is checked as follows.

### Positives (must route to the fact)

Up to 20 per fact: the request, validated base-model rewrites, then context prefixes.

| Source | Count asked | Kept only if |
|---|---|---|
| Paraphrases | 6, plus 6 more in a second round for facts with < 8 rewrites | every check below |
| Direct questions | 3, plus 3 more in a second round | every check below |
| Translations: Italian, Portuguese, Dutch (never LETHE-Bench's French / Spanish / German) | 3 | every check below |
| Multi-turn follow-ups with "they" / "it" (two different openers) | 2 | every check below |
| Context prefixes on the request and on plain-text rewrites | at most 8 | subject still recognised |

The checks a rewrite must pass:
- **Form:** one line, no list markers, labels or talk about the task; 3-40 words; questions end with "?".
- **Content:** contains the subject's name (follow-ups: must not, they use the pronoun) and never the answer.
- **Novelty:** not a near-duplicate of the request or of a kept rewrite (word Jaccard < 0.8).
- **Meaning:** the base model answers it with the request's answer.
- **Same fact:** asked "Does Prompt B ask for exactly the same piece of information as Prompt A?", the base model says Yes. This rejects rewrites that keep the answer but change the question, e.g. "Where did Dante Alighieri live?" for "born in", which the answer check alone accepts.

If a fact runs out of quality phrasings it keeps fewer than 20 positives, reported per fact, rather than being padded with near-copies of one sentence.

### Negatives (mention the subject, must not route)

Exactly 20 where enough valid ones exist:
- **Composition:** up to half same-subject competitors, the rest subject transplants (another fact's prompt with this subject swapped in).
- **SURE's donor rules apply:** subject type, answer group, protected relations.
- **Answer check:** a negative is dropped if the base model would answer it with the fact's answer (≥ 0.3 probability). Example dropped on the local run: "Dante Alighieri is a citizen of" for "born in Florence".
- **Plausibility check:** a transplant is dropped if its relation phrase costs more than 1.5 nats per token more after this subject than after the donor's. Example dropped: "Harry S. Truman plays the position of".
- **Diversity:** lead-in variants of one prompt count once, and transplants rotate across donor relations, so the 20 ask about different things.

`router_quality.md` (in SURE's work directory) lists the counts, the dropped examples, and every positive and negative of two facts.

All of this lives in the core method (`semantic-unlearning/scripts/`). The rewrite generator and its checks are in `router_rewrites.py` (the command-line step is `build_router_rewrites.py`); the 20 / 20 balancing and the negative checks are in `linear_router.py` / `fit_linear_router.py`. LETHE-Bench's `ours` supplies only its prompt rendering and calls the same code as the paper pipeline's `ONEPASS=1` (`README_LINEAR_ROUTER.md`, "Ours (one forward pass)").

Note: translation views used LETHE-Bench's French / Spanish / German before 2026-10-24, which broke this file's own rule, so cross-lingual was not truly held out. All SURE-R variants now use Italian / Portuguese / Dutch.
