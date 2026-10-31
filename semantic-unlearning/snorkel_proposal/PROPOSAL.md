# LETHE-Bench: measuring whether a language model can forget *one thing*

**Open Benchmarks Grant proposal — track: Propose a benchmark — focus areas: safety & alignment, dynamic settings, open-ended outputs, long-running work**

[Team / PI / affiliation — to be filled in] · Contact: [email] · Code: `snorkel_proposal/` in [github.com/Kp759/Unlearning](https://github.com/Kp759/Unlearning) (Apache-2.0)

---

## Summary

Deletion requests are becoming routine. People ask to have their data removed, model owners need hazardous or copyrighted knowledge taken out, and editors need to retract facts that turned out to be wrong. Each request asks a model to forget something *specific* while keeping everything else. Today's fact-unlearning benchmarks cannot tell a method that does this from one that cheats:

- **They are saturated.** On the four standard fact-unlearning benchmarks (MultiCounterFact, zsRE, MQuAKE, RWKU), SURE, the detect-then-edit method we built, removes 96–100% of the base model's recall while keeping 84–100% of locality (97–100% when MQuAKE retain facts that repeat a forget fact are excluded), with perplexity unchanged (5 seeds, Llama-3.2-3B, official evaluators). Gradient ascent, NPO, RMU, ROME, MEMIT and ZeroUnlearn each fail at least one benchmark badly. A benchmark that one method maxes out no longer ranks methods.
- **They reward shortcuts.** On MultiCounterFact and zsRE, every forget probe names the subject and no locality probe does. A string blocklist that refuses whenever the subject's name appears would therefore score near-perfectly by construction, and a filter that bans the answer token would also do well where the locality probes use other answers. None of these benchmarks distinguishes a runtime guardrail from a change to the weights.
- **They do not test completeness.** None checks whether the "forgotten" fact comes back after int4 quantization or a few fine-tuning steps, or is still readable in the hidden states.

**LETHE-Bench** is a benchmark for **surgical unlearning**: forget one fact (or one person), keep every fact entangled with it, and show that it is gone from the weights. It is built and running today. It includes:

- a split builder over 38k Wikidata-derived facts (MultiCounterFact, MQuAKE, zsRE) plus RWKU's entity track;
- **20 probe families**, each designed to defeat a specific shortcut;
- a model-conditioned knowledge filter;
- a hidden test partition;
- three threat-model tiers (behavioural / shipped weights / weights only);
- three white-box attacks;
- a sequential-deletion track;
- a self red-team gate that six reference cheats must fail before any split is released.

We ask Snorkel for expert data (cross-lingual and indirect-reference probes, grader calibration), compute for the full multi-model tables, red-team review, and help growing our agentic leakage prototype into a full track.

## 1. The gap

| | TOFU | MUSE | WMDP | RWKU | BLUR | Leak-Resistant Unlearning | CounterFact / zsRE / MQuAKE | **LETHE-Bench** |
|---|---|---|---|---|---|---|---|---|
| Real facts the model actually knows (per-model filter) | ✗ (fictitious, fine-tuned in) | ✗ | ✗ | partial (famous targets) | ✗ | ✗ (real facts, no filter) | ✗ | **✓** |
| Fact-level requests with typed entangled controls | ✗ | ✗ | ✗ | partial (neighbours) | topic overlap | ✗ (utility only) | neighbourhood only | **✓ 9 retain families** |
| Subject-free / cross-lingual / multi-turn / multi-hop forget probes | ✗ | ✗ | ✗ | partial (level 3) | partial | multi-hop with the bridge entity masked (6 logical forms); no cross-lingual or multi-turn | multi-hop (MQuAKE) | **✓** |
| Reference cheats that must fail (shortcut gate) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ 6 cheats** |
| Guardrail vs weight change (tiers) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ B / W / W+** |
| Completeness attacks (lens, quantization, relearning) | ✗ | partial (MIA) | partial | partial (MIA) | relearning | partial (sampling, key-token repetition, quantization; LLM judge) | ✗ | **✓ all three** |
| Sequential deletion stream | ✗ | ✓ | ✗ | batch only | ✗ | ✗ | ✗ | **✓** |
| Hidden test partition | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

(✓ = a design goal of that benchmark; partial = covered for some settings. Entries reflect our reading of each paper and should be double-checked by reviewers.)

Prior work names the problem: unlearning benchmarks are weak measures of progress (Thaker et al., 2024), forget–retain overlap breaks methods (BLUR; Hu et al., 2025), quantization undoes unlearning (Zhang et al., 2024), relearning attacks recover hidden knowledge (Lynch et al., 2024; Hu et al., 2024), and multi-hop paths leak what single-hop probes call forgotten (Leak-Resistant Unlearning; Qian et al., 2026: Llama-3.1-8B, Qwen3-14B/32B, six methods, MQuAKE and Harry Potter, LLM-judged). LETHE-Bench turns these findings into one fact-level benchmark that ships with its own shortcut gate.

## 2. What LETHE-Bench measures

A deletion request gives the method **one prompt and its answer** (`Danielle Darrieux's mother tongue is → French`). Nothing else is given: no paraphrases, no neighbours, no probe. The evaluator then asks:

**Erasure (E): is the fact gone however it is asked?** Paraphrase · question · subject typo/case · adversarial prefixes · stated after the subject's other facts · **subject never named** (described by its other facts) · **multi-turn with a pronoun** · **French/Spanish/German** · **multi-hop chains** · RWKU jailbreak rewrites.

**Locality (L): is everything else still there, especially what is nearby?** Other facts about the same subject · the same sentence with the forgotten fact *stated* · **other subjects with the same answer** · same relation, other answer · subjects sharing a name token · the subject mentioned while something unrelated is asked · **reading a counterfactual value from context** · unrelated facts · RWKU neighbours.

**Utility (U):** MMLU, TriviaQA and perplexity, relative to the base model.

**Robustness (R): is it gone from the weights?** Logit lens on every layer; int8/int4 quantization; relearning on half the forgotten facts, then testing the other half.

Each family exists to defeat a specific shortcut (`BENCHMARK_SPEC.md` §5). E and L are harmonic means over families, and R is the worst attack, so a method that gives up any whole family scores near zero on that axis. The **LETHE score** is the harmonic mean of E, L, U and R. All per-family scores, means and minima are reported, with a per-request completeness rate (every held-out probe of the fact erased).

**Design choices.**

1. **A model-conditioned knowledge filter.** Every probe is answered correctly by the base model, so scores read as "share of known knowledge retained or removed", comparable across models. In the repository's own MCF results, base recall was only 13.5%: near-zero "Eff" there mostly measures facts the model never knew.
2. **Typed overlap by construction**, using Wikidata triples. Same-subject, same-object, same-sentence and name-collision controls exist *per request*, not as a global pool.
3. **Three tiers.** Every method declares its runtime components. Guardrails and detect-then-edit methods are scored both as shipped and as bare weights (W+), so "behaves as if forgotten" and "removed from the weights" are reported separately.
4. **Grader-exploit defences.**
   - A causality check rejects models that peek at teacher-forced answers. It caught a real bug in one of our own cheat baselines.
   - Leaks are graded on normalized text (accents, casing and letter-spacing removed).
   - The open-ended scorer only counts probes the base model got right in generation.
5. **A detect-then-edit decomposition.** For methods that route, the forget residual splits into routing misses and actuation failures. This generalises the "genie" analysis we developed for our own method.
6. **Statistics.** Five seeds (different forget samples), a cluster bootstrap over requests, and paired tests between methods.

## 3. What exists today

All of it is in this folder, runnable with `pip install -e .` and `python -m scalpel ...`:

- **Builder:**
  - 4 sources, deduplication and conflict removal (186 contradictory pairs and 2,141 self-answering facts dropped);
  - salted subject partitions (retain_train / dev / hidden test);
  - optional private templates for the test partition;
  - a canary, and the split fingerprint recorded in every result.
- **Harness:** a method-agnostic `unlearn(requests, allowed_retain) → model` interface, with HF-checkpoint and plugin loaders and a SURE adapter.
- **Baselines in-harness:** GA, GD, NPO, SimNPO, trained refusal (TOFU IDK), task-vector negation, RMU, ROME, MEMIT, ZeroUnlearn, plus six reference cheats (no-op, refuse-all, prompt cache, subject blocklist, subject+relation keyword router, answer-token ban).
- **Scorers and tracks:** teacher-forced and open-ended scorers with behaviour labels (leak / refusal / hallucination; false refusal); utility; KL; three attacks; the strict tier; the sequential track; routing decomposition.
- **Reports and checks:** leaderboard, Pareto plot and CIs; the split validator and cheat gate (`RED_TEAM.md`).
- **Tests:** model-free unit tests, and an end-to-end run on a toy world with a tiny Llama trained to know it. The tests cover these invariants:
  - every probe is known by the base model;
  - noop scores zero erasure;
  - a peeking model is rejected;
  - attacks restore weights exactly;
  - the fast scoring path matches the full forward pass.
- **Full run on three models** (cluster; Llama-3.2-3B-Instruct, Qwen3-4B, Llama-3.1-8B-Instruct; 5 seeds and 50 requests per seed each; `EVIDENCE.md` §4): 11 methods and 6 reference cheats.
  - **No method is close to solved, and the ranking depends on the model:** the best score on any model is 56.3 (ZeroUnlearn on Qwen3-4B), where the same method scores 15.6 on Llama-3.1-8B; SimNPO 32–44; SURE (our method) 19–28; MEMIT 31 on Qwen3-4B and 2–3 on the Llama models; GA / GD / NPO / ROME below 7.
  - **Every reference cheat scores below 4 on every model** (the release gate is 35). The cheats are shortcuts a reviewer would think of in a minute; each scores near-perfectly on the legacy metrics and is caught by a family built for it. "Legacy view" below = only the families the official MCF / zsRE metrics use (erasure on the request and its paraphrases; locality on same-answer neighbours and unrelated questions), computed from the same runs, range over the three models:

    | Cheat | What it does | Legacy view (erasure / locality) | Caught by | LETHE |
    |---|---|---|---|---|
    | subject blocklist | refuse any prompt that names a forgotten subject | 100 / 99.5–100 | `indirect`, `multihop` (subject never named) leak; `same_subject`, `subject_mentioned` refused | 1.2–2.4 |
    | keyword router | subject and relation words → "I don't know" (a string version of a detect-then-edit method) | 82–89 / 100 | `indirect`, `multihop`, `crosslingual`, `surface` | 1.6–3.2 |
    | answer ban | never emit the forgotten answer's first token | 87–96 / 35–46 | `same_object` (other subjects with the same answer), `in_context` | 1.7–3.4 |
    | prompt cache | refuse the exact training prompts only | 50 / 100 | every held-out phrasing | 1.1–2.2 |
    | refuse-all | "I don't know" to everything | 100 / 0 | every retain family | 1.0–1.9 |
    | no-op | change nothing | 0 / 100 | every forget family | 1.0–2.0 |

    Because E and L are harmonic means over families, failing one family pulls the score to the floor instead of being averaged away. The gate (`RED_TEAM.md`) runs all six on every split before release; if any passes 35, the split is at fault and is fixed.
  - **Each method fails a different axis:** GA / GD / NPO erase 79–100 but keep locality 1–2; a trained refusal falsely refuses 37–55% of retain questions; ROME and MEMIT erase 3–9 on held-out phrasings on Llama and the fact returns under attack (R ≤ 1); SimNPO erases 64–78 at locality 39–51; relearning restores 73–81% of ZeroUnlearn's forgotten facts on the Llama models.
  - **Our own method is not the top row, and the table says why:** SURE keeps locality 94–96 and utility 100 on every model and erases 89–96 of the request prompts, but two families it is never routed on (`multihop` 94–96% residual, `indirect` 100%) hold its E at 7–22; route recall is 86–92%, and it scores 2.9 in the weights-only tier because its router and rows are runtime components.
- **Evidence of saturation** from the repository's 4-benchmark × 5-seed study (`EVIDENCE.md`).

## 4. Fit with the program

- **Safety & alignment.** The benchmark covers right-to-be-forgotten requests, hazardous-knowledge removal, and the difference between hiding knowledge (guardrails) and removing it.
- **Dynamic settings.** Deletion requests arrive over time in the sequential track, which tests whether locality erodes as edits accumulate. The scaling curve runs from 50 to 10k requests.
- **Open-ended outputs.** Generation-based grading with behaviour labels, multi-turn chats, cross-lingual questions.
- **Long-running work (prototype running, M4).** An **agentic leakage track** (`AGENT_TRACK.md`, `scalpel/agentic.py`): the unlearned model runs as a search agent with a scratchpad for up to 10 replies. Every reply is graded in three channels: the scratchpad, the search queries an outside service would log, and the final answer. A survival curve shows whether forgetting holds for 1 step but breaks by step 6. In the closed setting the corpus has nothing about the subject, so a leak is recall; in the open setting the answer is redacted but kept facts remain, so a leak may be reconstruction from them ("born in Bordeaux" → "French"). The prototype runs end to end on every method. Snorkel's environment and expert-data support would turn it into a full track: more tools, realistic corpora, multi-session horizons.

## 5. Asks (in-kind)

1. **Expert data**:
   - native-speaker verification of the cross-lingual probes and expansion to 10 languages;
   - expert-written indirect references (descriptions, aliases, coreference chains) for 2,000 subjects that cannot be inferred from retained facts;
   - a 1,000-item human-labelled calibration set for the generation grader;
   - privacy/legal input on realistic deletion-request scenarios.
2. **GPU compute: ~7,000 A100-80GB hours** (80 GB GPUs; multi-GPU nodes for the largest models) for 10 models from 5 families (Llama 3B / 8B / 70B, Qwen3 4B / 14B / 32B, Gemma-3 4B / 27B, Mistral-Small-24B, OLMo-2-7B; three done) × 5–10 dev seeds and 5 hidden seeds × 19 methods (the 11 run plus SCRUB, WHP, ELM, GRACE, WISE, AlphaEdit, ECO and in-context unlearning prompting) and 6 cheats, representation-level attacks added to R, the scaling curve to 10k requests, and the sequential and agentic tracks. The basis is measured: ≈ 2.2 GPU-hours per model and seed for every method at 3B, ×1.3 at 4B and ×2.1 at 8B (`ROADMAP.md`). The 32B and 70B tiers are what the credits buy that the university cluster cannot. *We request this as partner compute credits, which the program lists among its in-kind support.*
3. **Red-team review** of the grader, verifier and contamination defences before v1.0.
4. **Hosting** for the hidden-test evaluation service and leaderboard.

## 6. Deliverables and timeline (6 months)

| Month | Deliverable |
|---|---|
| 1 | v1.0 freeze: dev splits for the 10-model grid (5 families, 3B–70B) × 5 seeds; hidden test built; HF dataset card; upstream data terms confirmed |
| 1–2 | Full baseline tables on all tiers and attacks; scaling and sequential results |
| 2–4 | Expert-data families (cross-lingual ×10, expert indirect references, grader calibration) |
| 3–5 | Agentic leakage track |
| 6 | Public leaderboard + hidden-test service; Datasets & Benchmarks paper |

Everything is released openly: code under Apache-2.0, dev splits under CC BY 4.0 (subject to the upstream terms in `DATA_LICENSE.md`), and hidden-test evaluation as a service.

## 7. Team and prior work

[Names, roles, affiliations]. The team built the repository behind `EVIDENCE.md`. That work includes:

- a 4-benchmark × 5-seed unlearning study with 7 baselines (GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn);
- the SURE detect-then-edit method and its genie (perfect-routing) decomposition;
- router diagnostics and layer sweeps;
- the MultiFactPerson and Eval-DU+ multi-fact probes that LETHE-Bench generalises.

## 8. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Our method is a baseline on our own benchmark | SURE is one of 17 rows on the leaderboard (11 methods, 6 reference cheats) on three models, and it is not the top method on any of them (SimNPO and ZeroUnlearn score higher; `EVIDENCE.md` §4). One of the reference cheats is a string version of its design. The families where SURE is expected to fail (subject-free, cross-lingual, typo, multi-hop; strict tier) are reported like any other method's |
| Low knowledge-filter yield on small models | Yields are reported per family; families below 5 probes are excluded and flagged; the release uses models of 3B and up |
| Indirect probes inferable from retained facts | Descriptions are drawn from other answer groups; expert-written references (ask 1) |
| The benchmark saturates | Saturation is flagged per family; harder families are planned (agentic, ×10 languages, 10k-request scale); the hidden test is refreshed |
