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

- **LETHE-Bench** (this repository): split builder over 38k Wikidata-derived facts, 20 probe families, deterministic scorers, attacks, threat-model tiers, sequential and agentic tracks; 8 methods and 6 reference cheats evaluated.
- **SURE**, our detect-then-edit unlearning method (per-fact linear router plus one residual row per fact), with a perfect-routing decomposition that splits every failure into routing versus actuation.
- **A 4-benchmark × 5-seed unlearning study** (MultiCounterFact, zsRE, MQuAKE, RWKU; 7 baselines; official evaluators).
- Code: https://github.com/Kp759/Unlearning [add paper / preprint link if any]

**Web presence**
[personal or lab page]

**Company**
[institution]

**Project title**
LETHE-Bench: a benchmark for surgical, fact-level unlearning in language models

**Describe your proposed benchmark** (500 words max) [485 words]
*Form prompts: the problem or focus area; what each task looks like (input, output, grading); expected outputs and impact.*

*Problem.* Deletion requests (privacy, hazardous knowledge, retractions) ask a model to forget one specific fact and keep everything else, including entangled facts. Standard benchmarks no longer separate methods. On MultiCounterFact, zsRE, MQuAKE and RWKU, SURE, the detect-then-edit method we built, removes 96–100% of the base model's recall of the forget facts while keeping 84–100% of locality, with perplexity unchanged (Llama-3.2-3B-Instruct, 5 seeds, official evaluators). By construction, a subject-name blocklist also looks near-perfect on MultiCounterFact and zsRE: every forget probe names the subject and no locality probe does. None checks whether the fact survives quantization or fine-tuning.

*Task.* A method receives a base model and deletion requests, each only the fact's direct prompt and answer ("Danielle Darrieux's mother tongue is → French"), plus retain data about disjoint subjects. It returns the unlearned system (weights plus declared runtime components). Splits keep only probes the base model answers, so every score reads "share of what this model knew that it still knows". Each forgotten fact is then asked in held-out ways: paraphrase, question, typos, adversarial prefixes, mid-sentence, subject never named, multi-turn pronoun, French/Spanish/German, and multi-hop chains (plus RWKU jailbreak rewrites). Typed retain families test what must survive: other facts about the subject, the same sentence with the fact stated, other subjects with the same answer, the same relation, name collisions, the subject merely mentioned, reading a counterfactual note, and unrelated facts.

*Grading.* Deterministic. A probe counts as answered if every answer token is the model's top-1 under teacher forcing (any alias); a rule grader scores greedy generations as leak, refusal or false refusal. Erasure E and locality L are harmonic means over held-out forget and retain families; utility U covers MMLU, TriviaQA and perplexity; robustness R is the worst residual under logit-lens, int8/int4 quantization and relearning attacks. The LETHE score is their harmonic mean. A sequential track tests whether locality erodes as deletions accumulate. Three tiers separate behaviour, shipped weights and weights only (runtime components removed), so a guardrail cannot pass as removal. Six reference cheats (subject blocklist, answer ban, keyword router, prompt cache, refuse-all, no-op) must score below 35 before a split is released.

*Results so far* (Llama-3.2-3B-Instruct, 5 seeds). SURE 30.1 ± 10.1, ZeroUnlearn 21.4, GA/GD/NPO 4.9–6.2, RMU 3.0, ROME 1.6, MEMIT 0, every cheat 0. The scores expose trade-offs the legacy benchmarks hide: GA reaches erasure 97 but locality 1; ZeroUnlearn erasure 39 at locality 67; ROME and MEMIT barely erase held-out phrasings (E 9) and the fact returns under attack (R ≤ 1). SURE, near-saturated on the legacy benchmarks, erases 96% of request prompts but scores only 23 on held-out phrasings, and drops to 2.9 once its runtime router is removed.

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

*Compute / hosting (partner benefits):* about 5,000 A100-80GB hours as compute credits (8 models, 5 families, 3B–27B; 5–10 seeds; 14 methods; scaling, sequential and agentic tracks), multi-GPU nodes for the largest; hosting for the hidden-test service and leaderboard.

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
`PROPOSAL.md` and `EVIDENCE.md` (exported to PDF); `results/cluster/llama32_3b/dev/leaderboard/leaderboard.md` (Llama-3.2-3B, 5 seeds). The cluster leaderboard lives on Wulver, not in this repository: rerun `sbatch scripts/slurm/scalpel_report.slurm` after pulling the rename (its title then reads LETHE-Bench) and copy the file here before exporting.

**Terms checkbox**
The terms (benchmarks.snorkel.ai/terms) require, before ticking: (1) your institution's IP policy permits the project and its open release, and you hold any needed approvals; (2) outputs are released under a permissive licence (code Apache-2.0, dev splits CC BY 4.0, see `DATA_LICENSE.md`; zsRE redistribution terms must be confirmed first); (3) Snorkel's support is acknowledged in publications, and Snorkel researchers are credited as contributors or co-authors where standard authorship criteria apply; (4) Snorkel may name the team, institution, project summary and open-source output in program materials; (5) non-public information shared during the project stays confidential. Support is in-kind only (no cash).

---

## Sources (not part of the form)

All numbers are Llama-3.2-3B-Instruct, as of 2026-10-25.

- **Legacy benchmarks** (`outputs/onepass_l19_v1/paper_table.md`, jobs 1452463 / 1452464): SURE (run directory `ours`; classifier and Δe at layer 19, one pass) over 5 seeds, against the base model over 5 seeds. Recall removed = 1 − Eff/Base Eff: MCF 0.001/13.46, zsRE 0.41/32.37, MQuAKE 0.53/73.23, RWKU 2.40/57.60 → 95.8–100%. Locality kept = Spe/Base Spe: 19.84/19.88, 29.79/29.79, 55.53/66.13 (Spe† 66.50/67.34), 61.82/63.70 → 84.0–100%. PPL 11.31 everywhere, equal to base.
- **LETHE-Bench** (`results/cluster/llama32_3b/dev/leaderboard/leaderboard.md`, dev partition, seeds 1–5; the full table is in `EVIDENCE.md` §4): LETHE score / E / L / R for SURE 30.1 ± 10.1 / 22.8 / 95.6 / 19.0 (E_direct 95.6, Strict 2.9; rerun job 1452482; the original adapter `sure`, 12.0, is excluded from the leaderboard with `--exclude sure`); zerounlearn 21.4 / 38.9 / 67.4 / 8.7; npo 6.2, gd 5.5, ga 4.9 (GA E 97.1, L 1.3); rmu 3.0; rome 1.6 (E 8.5, L 57.2, R 0.8); memit 0.0 (E 9.0, L 91.8, R 0.0); all six cheats 0.0.
- **GPU compute** (`ROADMAP.md`, "GPU compute"): the 8-model grid (Llama-3.2-3B, Llama-3.1-8B, Qwen3-4B, Qwen3-14B, Gemma-3-4B, Gemma-3-27B, Mistral-Small-24B, OLMo-2-7B) and the ≈ 5,000 A100-80GB-hour breakdown, from a measured ≈ 1.8 GPU-hours per model and seed for every method on Llama-3.2-3B (Wulver `sacct`, October 2026). GPU compute is requested as partner compute credits, which the terms list among the in-kind support (with Expert Data-as-a-Service and Snorkel research / engineering time; no cash). Confirm with benchmarks@snorkel.ai how much the credits cover.
