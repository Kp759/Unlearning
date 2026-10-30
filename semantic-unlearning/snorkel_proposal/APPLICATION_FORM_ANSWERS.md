# Open Benchmarks Grant application: ready-to-paste answers

One section per form field, in the form's order; the fields and limits were checked against the live form
on 2026-10-28 (the form's own sub-prompts are quoted under each heading). The word count of each limited
field is given in brackets; paste only the text under each heading. Upload `PROPOSAL.md` and `EVIDENCE.md` (as PDF)
and the Llama-3.2-3B leaderboard as supporting materials. Fill in the bracketed fields. Where every
number comes from: **Sources** at the end.

---

**Email · First name · Last name**
[yours]

**Please describe or link to impactful research artifact you've shipped** (100 words max) [81 words]

- **LETHE-Bench** (this repository): split builder over 38k Wikidata-derived facts, 20 probe families, deterministic scorers, attacks, threat-model tiers, sequential and agentic tracks; 11 methods and 6 reference cheats evaluated on three models.
- **SURE**, our detect-then-edit unlearning method (per-fact linear router plus one residual row per fact), with a perfect-routing decomposition that splits every failure into routing versus actuation.
- **A 4-benchmark × 5-seed unlearning study** (MultiCounterFact, zsRE, MQuAKE, RWKU; 7 baselines; official evaluators).
- Code: https://github.com/Kp759/Unlearning [add paper / preprint link if any]

**Web presence**
[personal or lab page]

**Company**
[institution]

**Project title**
LETHE-Bench: a benchmark for surgical, fact-level unlearning in language models

**Describe your proposed benchmark** (500 words max) [488 words]
*Form prompts: the problem or focus area; what each task looks like (input, output, grading); expected outputs and impact.*

*Problem.* Deletion requests (privacy, hazardous knowledge, retractions) ask a model to forget one specific fact and keep everything else, including entangled facts. Standard benchmarks no longer separate methods. On MultiCounterFact, zsRE, MQuAKE and RWKU, SURE, the detect-then-edit method we built, removes 96–100% of the base model's recall of the forget facts while keeping 84–100% of locality, with perplexity unchanged (Llama-3.2-3B-Instruct, 5 seeds, official evaluators). By construction, a subject-name blocklist also looks near-perfect on MultiCounterFact and zsRE: every forget probe names the subject and no locality probe does. None checks whether the fact survives quantization or fine-tuning.

*Task.* A method receives a base model and deletion requests, each only the fact's direct prompt and answer ("Danielle Darrieux's mother tongue is → French"), plus retain data about disjoint subjects. It returns the unlearned system (weights plus declared runtime components). Splits keep only probes the base model answers, so every score reads "share of what this model knew that it still knows". Each forgotten fact is then asked in held-out ways: paraphrase, question, typos, adversarial prefixes, mid-sentence, subject never named, multi-turn pronoun, French/Spanish/German, and multi-hop chains (plus RWKU jailbreak rewrites). Typed retain families test what must survive: other facts about the subject, the same sentence with the fact stated, other subjects with the same answer, the same relation, name collisions, the subject merely mentioned, reading a counterfactual note, and unrelated facts.

*Grading.* Deterministic. A probe counts as answered if every answer token is the model's top-1 under teacher forcing (any alias); a rule grader scores greedy generations as leak, refusal or false refusal. Erasure E and locality L are harmonic means over held-out forget and retain families; utility U covers MMLU, TriviaQA and perplexity; robustness R is the worst residual under logit-lens, int8/int4 quantization and relearning attacks. The LETHE score is their harmonic mean. A sequential track tests whether locality erodes as deletions accumulate. Three tiers separate behaviour, shipped weights and weights only (runtime components removed), so a guardrail cannot pass as removal. Six reference cheats (subject blocklist, answer ban, keyword router, prompt cache, refuse-all, no-op) must score below 35 before a split is released.

*Results so far* (Llama-3.2-3B, Qwen3-4B, Llama-3.1-8B; 11 methods, 6 cheats, 5 seeds each). No method exceeds 60 and the ranking changes with the model: ZeroUnlearn scores 56 on Qwen3-4B but 16 on Llama-3.1-8B; SimNPO 32–44; our own SURE 19–28; MEMIT 31 on Qwen3-4B and 2–3 on Llama; GA/GD/NPO/ROME below 7; every cheat below 4 (gate 35). The scores expose trade-offs the legacy benchmarks hide: GA erases 97 but keeps locality 1; SimNPO erases 64–78 at locality 39–51; SURE, near-saturated on the legacy benchmarks, keeps locality 94–96 but scores 7–22 on held-out phrasings and 2.9 once its runtime router is removed.

*Outputs and impact.* Open splits for eight models from five families (3B–27B), a hidden-test service, a public leaderboard, an agentic leakage track and a Datasets & Benchmarks paper. Deployers learn whether a deletion removed the fact; method developers learn exactly where it leaks.

**How does your benchmark differ from existing benchmarks?** (150 words max) [144 words]
*Form prompts: name the 2–3 closest benchmarks (published or in progress); what yours measures that they don't.*

The closest are RWKU (Jin et al., 2024), Leak-Resistant Unlearning (Qian et al., 2026) and TOFU / MUSE (Maini et al., 2024; Shi et al., 2024). TOFU forgets fictitious authors fine-tuned into the model and MUSE whole corpora; neither asks a model to forget one real fact while keeping its neighbours. RWKU uses real, famous people at entity level, without a per-model knowledge filter. Leak-Resistant Unlearning tests multi-hop leakage and recovery attacks, but with an LLM judge and without locality controls. LETHE-Bench is fact-level, on facts each model provably knows, and adds what none of them has: typed entangled retain controls per request (same answer, same sentence, name collision, subject mentioned, counterfactual reading); subject-free, multi-turn, cross-lingual and multi-hop forget probes; a shortcut gate of six reference cheats that must fail; behaviour versus weights-only tiers; and lens, quantization and relearning attacks in one deterministic score.

**Describe your methodology and how you would leverage awarded resources** (200 words max) [198 words]
*Form prompts: Expert Data (which domain experts); Embedded Support; Compute / Hosting (partner benefits); Milestones.*

*Expert data (Snorkel DaaS):*
- native-speaker linguists for 10 languages, to verify and write cross-lingual probes;
- knowledge annotators, to write indirect references (descriptions, aliases, coreference chains) for 2,000 subjects that cannot be inferred from retained facts;
- trained annotators, to label 1,000 generations (leak, refusal, hallucination, false refusal) and calibrate the rule grader;
- privacy and legal experts, to supply realistic right-to-be-forgotten and hazardous-knowledge deletion scenarios.

*Embedded support:* review of the split-builder and hidden-test pipeline, schema validation of the probe families and dataset card, red-team review of the grader and contamination defences, and help turning the agentic prototype into a full environment.

*Compute / hosting (partner benefits):* about 5,000 A100-80GB hours as compute credits (8 models, 5 families, 3B–27B; 5–10 seeds; 17 methods; scaling, sequential and agentic tracks), multi-GPU nodes for the largest; hosting for the hidden-test service and leaderboard.

*Milestones:*
1. Month 1: v1.0 freeze; every split passes validation and every cheat scores below 35.
2. Months 1–2: full baseline tables on all tiers and attacks.
3. Months 2–4: expert families; grader agreement with human labels of at least κ = 0.8.
4. Months 3–5: agentic leakage track.
5. Month 6: public leaderboard, hidden-test service, paper.

**Team members**
[names, roles, affiliations]

**Estimated project duration**
6 months

**Supporting materials** (optional on the form)
`PROPOSAL.md` and `EVIDENCE.md` (exported to PDF); `results/cluster/leaderboard_dev/leaderboard.md` (all three models, 5 seeds; the per-model files and `pareto.png` are next to it under `results/cluster/<model>/dev/leaderboard/`). The cluster results are produced on Wulver; commit the leaderboard files to the repository before exporting.

**Terms checkbox**
The terms (benchmarks.snorkel.ai/terms) require, before ticking: (1) your institution's IP policy permits the project and its open release, and you hold any needed approvals; (2) outputs are released under a permissive licence (code Apache-2.0, dev splits CC BY 4.0, see `DATA_LICENSE.md`; zsRE redistribution terms must be confirmed first); (3) Snorkel's support is acknowledged in publications, and Snorkel researchers are credited as contributors or co-authors where standard authorship criteria apply; (4) Snorkel may name the team, institution, project summary and open-source output in program materials; (5) non-public information shared during the project stays confidential. Support is in-kind only (no cash).

---

## Sources (not part of the form)

Legacy-benchmark numbers are Llama-3.2-3B-Instruct (2026-10-25); LETHE-Bench numbers are the three-model run of 2026-10-29.

- **Legacy benchmarks** (`outputs/onepass_l19_v1/paper_table.md`, jobs 1452463 / 1452464): SURE (run directory `ours`; classifier and Δe at layer 19, one pass) over 5 seeds, against the base model over 5 seeds. Recall removed = 1 − Eff/Base Eff: MCF 0.001/13.46, zsRE 0.41/32.37, MQuAKE 0.53/73.23, RWKU 2.40/57.60 → 95.8–100%. Locality kept = Spe/Base Spe: 19.84/19.88, 29.79/29.79, 55.53/66.13 (Spe† 66.50/67.34), 61.82/63.70 → 84.0–100%. PPL 11.31 everywhere, equal to base.
- **LETHE-Bench** (`results/cluster/leaderboard_dev/leaderboard.md`, dev partition, seeds 1–5, teacher-forced scorer; the per-axis tables are in `EVIDENCE.md` §4): LETHE per model (Llama-3.2-3B / Qwen3-4B / Llama-3.1-8B): zerounlearn 21.3 / 56.3 / 15.6; simnpo 37.4 / 31.7 / 43.7; SURE 28.4 / 22.3 / 18.7 (E 22.2 / 7.5 / 6.9, L 95.5 / 94.3 / 95.7, U 100, Strict 2.9); memit 1.7 / 30.7 / 3.2; ga, gd, npo, rome 3.0–6.8; largest cheat 3.4. "GA erases 97 at locality 1" = Llama-3.2-3B E 97.3, L 1.3; "SimNPO 64–78 at 39–51" = E 63.5–77.5, L 38.9–51.2 across the three models; "relearning restores 73–81% of ZeroUnlearn's facts" = the attacks table, Llama models. The original SURE adapter `sure` is excluded with `--exclude sure`.
- **GPU compute** (`ROADMAP.md`, "GPU compute"): the 8-model grid (Llama-3.2-3B, Llama-3.1-8B, Qwen3-4B, Qwen3-14B, Gemma-3-4B, Gemma-3-27B, Mistral-Small-24B, OLMo-2-7B) and the ≈ 5,000 A100-80GB-hour breakdown, from a measured ≈ 1.8 GPU-hours per model and seed for every method on Llama-3.2-3B (Wulver `sacct`, October 2026). GPU compute is requested as partner compute credits, which the terms list among the in-kind support (with Expert Data-as-a-Service and Snorkel research / engineering time; no cash). Confirm with benchmarks@snorkel.ai how much the credits cover.
