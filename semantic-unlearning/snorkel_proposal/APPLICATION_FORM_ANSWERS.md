# Application form: ready-to-paste answers

These are short answers for benchmarks.snorkel.ai/apply ("Propose a benchmark"). Upload `PROPOSAL.md` (exported to PDF) as supporting material, together with `EVIDENCE.md` and the pilot leaderboard. Fill in the bracketed fields.

**Project title**
SCALPEL: a benchmark for surgical, fact-level unlearning in language models

**Applicant / team**
[Names, roles, affiliation(s), emails]

**Focus area(s)**
Safety & alignment; dynamic settings; open-ended outputs. A long-horizon agentic leakage track is proposed.

**One-sentence summary**
SCALPEL measures whether a language model can forget one specific fact or person while keeping every entangled fact. It also tests that the fact is gone from the weights rather than hidden, and it ships with a self red-team gate that known shortcuts must fail.

**Abstract (≈200 words)**
Deletion requests (privacy, hazardous knowledge, retractions) ask a model to forget something specific and keep everything else. The standard fact-unlearning benchmarks can no longer tell methods apart. On MultiCounterFact, zsRE, MQuAKE and RWKU, our detect-then-edit method removes 96–100% of base recall while keeping 84–100% of locality (5 seeds, official evaluators). On MultiCounterFact and zsRE, a subject-name blocklist would also score near-perfectly by construction: every forget probe names the subject and no locality probe does. None of these benchmarks checks whether the fact survives quantization or fine-tuning.

SCALPEL builds model-conditioned splits from 38k Wikidata facts, so every probe is known by the base model. It asks each forgotten fact in 11 ways, including with the subject never named, through multi-turn pronouns, in French/Spanish/German, and through multi-hop chains. It controls for over-forgetting with 9 typed retain families: same subject, same answer, same sentence, name collisions, the subject merely mentioned, and reading a counterfactual note. It scores three threat-model tiers (behavioural, shipped weights, weights only) and three attacks (logit lens, int4/int8, relearning), plus a sequential-deletion track.

Six reference cheats must score below 35 before a split is released. On our real-data laptop pilot, a subject-name blocklist scores 100/98.5 under the legacy MCF/zsRE-style metrics and 0 on SCALPEL, and all six cheats are defeated. Our own method, which saturates the legacy benchmarks, leads the pilot at only 10.5: it keeps perfect locality, but questions, cross-lingual queries, typos and multi-hop chains leak. The code, tests and pilot exist today. We ask for expert data, compute for the multi-model tables, red-team review, and help building an agentic leakage track.

**What gap does it fill?**
Existing benchmarks either use fictitious facts (TOFU), whole corpora (MUSE, WMDP), or single-pool locality sets (CounterFact, zsRE, MQuAKE). None of them combines:
- per-model knowledge filtering;
- typed entangled controls for each request;
- subject-free, cross-lingual and multi-turn forget probes;
- a shortcut gate;
- a guardrail-versus-weights distinction;
- completeness attacks.
Prior work shows these are exactly the failure points (Thaker et al. 2024; BLUR; Zhang et al. 2024 on quantization; Lynch et al. 2024 on relearning).

**What exists today?**
- A working Python package: split builder, 20 probe families, two scorers, utility suite, attacks, tiers, sequential track, routing decomposition, leaderboard.
- 10 in-harness methods: GA, GD, NPO, RMU and six reference cheats. External checkpoints and plugins are supported, and there is an adapter for our SURE method.
- A test suite, including a tiny Llama trained on a synthetic world.
- A real-data pilot on Qwen2.5-0.5B-Instruct run on a laptop.
- Evidence of saturation from a 4-benchmark × 5-seed study with 8 baselines.

**How are outputs graded?**
- **Primary:** deterministic teacher-forced exact top-1 on every answer token (any alias).
- **Secondary:** a deterministic rule grader on greedy generations (leak / refusal / hallucination; false refusal), computed only on probes the base model answered correctly in generation.
- An optional LLM judge is never used for headline numbers.
- A causality check rejects models that peek at teacher-forced answers.

**How do you prevent contamination and overfitting?**
- A hidden test partition (45% of subjects), with a private seed and private templates.
- A canary string in every file.
- Model-conditioned splits that are re-sampled per seed.
- A validator confirming that a method's input contains nothing that is evaluation-only.

**What support do you need?**
1. **Expert data**:
   - cross-lingual verification and expansion to 10 languages;
   - expert-written indirect references for 2,000 subjects;
   - a 1,000-item grader calibration set;
   - privacy/legal input on deletion scenarios.
2. **~1,000 A100-hours** for 3 model families × 2 sizes × 10 seeds × 14 methods.
3. **Red-team review** before v1.0.
4. **Hosting** for the hidden-test service and leaderboard.
5. **Research collaboration** on the agentic leakage track.

**Deliverables and timeline**
| Month | Deliverable |
|---|---|
| 1 | v1.0 splits plus hidden test |
| 1–2 | Full baseline tables |
| 2–4 | Expert-data families |
| 3–5 | Agentic track |
| 6 | Public leaderboard, hidden-test service, and a Datasets & Benchmarks paper |

**Licensing / openness**
- Code: Apache-2.0.
- Dev splits: CC BY 4.0, subject to the upstream terms (MIT for MCF and MQuAKE, CC BY 4.0 for RWKU; the zsRE terms are being confirmed).
- Hidden test: run as a service.

**Links**
Repository: https://github.com/Kp759/Unlearning (folder `semantic-unlearning/snorkel_proposal/`)
