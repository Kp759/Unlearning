# SURE cross-benchmark ICLR results plan

Generated 2026-08-21 from repository commit `4bdcd88122b46c7d215f9480bddf029ee09636f4` on branch `codex/mcf-sure-utility-scaling-v9`.

## Technical summary

The strongest defensible paper is not one that forces a single score or claims that one unchanged implementation wins everywhere. It is one that shows a predeclared SURE core reaching a better forgetting–utility frontier under each benchmark's native evaluator, while every baseline uses the same model, visible data, split, evaluator, tuning budget, and seed policy.

Three conclusions should govern the paper:

1. **Use benchmark-native metrics in the main tables.** `Eff/Gen/Spe/PPL` are appropriate for the ZeroUnlearn MCF/ZsRE protocol, but they are not universal. `FS/GFS` in the target-true-sensitive MCF evaluator are a different task and a different formula; they must not be compared directly with ZeroUnlearn's published `Eff/Gen`.
2. **Treat the current ZsRE result as provisional, not paper-ready.** Its `Eff`, `Gen`, and `Spe` implementation matches the checked-in ZeroUnlearn evaluator semantics and focused parity tests pass, but the stored aggregate lacks its exact execution commit and local per-seed output directories. Its PPL fixture also does not reproduce the public ZeroUnlearn baseline.
3. **Keep a shared SURE principle, not a fiction of unchanged code.** The sparse contextual edit plus external-utility guard directly fits factual QA datasets. Entity, passage, and hazardous-domain benchmarks require explicit adapters or method extensions. Those adaptations must be named and ablated.

The paper should make a constrained claim: **SURE improves the benchmark-native forgetting–utility Pareto frontier across factual, entity, passage, and hazardous-knowledge settings.** A universal raw-score average should be secondary at most.

## Key finding 1: current ZsRE metrics are mostly compatible, but the result is not yet citable

### What is compatible

The repository's `zsre_zero_unlearn_official_eval.py` mirrors the upstream ZsRE construction and scoring:

- first-half retain pool and second-half forget pool;
- seeded sampling of 50 forget and 1,000 retain records;
- direct rewrite and held-out rephrase prompts;
- teacher-forced next-token accuracy on every original-answer token;
- per-record macro averaging, then 0–100 scaling;
- `Eff = direct original-answer token accuracy` (lower is better);
- `Gen = rephrase original-answer token accuracy` (lower is better);
- `Spe = neighborhood next-token accuracy` (higher is better).

The focused local parity suite passes 24/24 tests. The current baseline values also closely track the public ZeroUnlearn table on the three task metrics.

| ZsRE metric | Current local Base | Public ZeroUnlearn Base | Difference | Compatibility decision |
| --- | ---: | ---: | ---: | --- |
| Eff ↓ | 33.0896 | 32.82 | +0.2696 pp | Same meaning; exact numerical reproduction not proven |
| Gen ↓ | 32.1979 | 32.23 | -0.0321 pp | Same meaning; extremely close |
| Spe ↑ | 28.1210 | 28.12 | +0.0010 pp | Same meaning; effectively identical |
| PPL ↓ | 11.0625 | 12.88 | -1.8175 | **Not comparable yet** |

The public ZeroUnlearn Llama-3.2-3B table reports `Eff 27.85`, `Gen 27.52`, `Spe 27.73`, and `PPL 13.08` for ZeroUnlearn. The current SURE aggregate reports `Eff 0.1317`, `Gen 0.6617`, `Spe 26.1068`, and `PPL 11.9750`. These numbers suggest a potentially strong result, but they cannot support a superiority claim until both methods are rerun under one locked local protocol.

### Why the current aggregate is provisional

The checked-in canonical ZsRE record says:

- `aggregate_source = terminal-captured 10-seed aggregate`;
- `exact_final_execution_commit = not separately recaptured`;
- the referenced Base and SURE output directories are absent from the local repository snapshot;
- no complete set of per-seed evaluator JSON, split manifests, checkpoint hashes, token-ID hashes, or execution receipts is available for reconstruction;
- the local PPL fixture yields Base `11.0625`, while the public ZeroUnlearn table yields `12.88` on the nominally same model.

Therefore the correct status is:

| Question | Decision |
| --- | --- |
| Are `Eff/Gen/Spe` metric names and directions comparable to ZeroUnlearn? | **Yes**, at the implementation/semantic level. |
| Is exact protocol identity demonstrated? | **No**; dataset, model, tokenizer, dtype, artifact, and aggregation receipts are incomplete. |
| Is current PPL comparable to the public table? | **No**. |
| Can the current SURE row be submitted as a final paper row? | **No**; rerun or reconstruct it first. |
| Can it guide development? | **Yes**, clearly labeled provisional. |

### A second validity issue: token accuracy can hide collapse

An older five-fold ZsRE diagnostic produced zero token-accuracy leakage while still showing semantic/lexical leakage and newly introduced repetitive `Unknown` behavior. This does not invalidate `Eff/Gen`; it shows that they are narrow teacher-forced metrics. The final ZsRE evaluation should therefore add post hoc, selection-independent diagnostics:

- normalized exact-match/alias containment of the sensitive answer in free generation;
- sensitive-answer probability ratio relative to Base;
- refusal/`Unknown` rate;
- repetition and degenerate-loop rate;
- retain and locality generation quality;
- optional blinded LLM judge only after deterministic metrics are frozen.

These belong in a robustness table or appendix and must not be used to tune the confirmatory checkpoint.

## Key finding 2: `Eff/Gen` and `FS/GFS` are not interchangeable

### MCF has two opposite target semantics in the current repository

The public ZeroUnlearn-style MCF protocol treats the MCF `target_new` field as the unwanted/sensitive answer and asks whether the model still prefers it. Its metrics are:

- `Eff ↓`: percentage of direct prompts still preferring the unwanted `target_new`;
- `Gen ↓`: percentage of paraphrases still preferring `target_new`;
- `Spe ↑`: neighborhood probability-difference preservation score;
- `PPL ↓/stable`: fluency proxy.

The current target-true-sensitive evaluator instead declares the original `target_true` to be sensitive and counts:

- `FS ↑`: direct cases where `NLL(target_true) > NLL(target_new)`;
- `GFS ↑`: paraphrase cases satisfying the same inequality.

These are not alternate names for the same numbers. They reverse which answer is designated sensitive and reverse the success direction. Consequently:

- do not place target-true-sensitive `FS/GFS` beside published ZeroUnlearn `Eff/Gen`;
- do not convert by computing `100 - Eff` unless the exact same records, answer semantics, tie handling, aggregation, and evaluator are proven identical;
- if original-fact deletion is scientifically important, make it a separately named protocol and rerun every baseline on that protocol.

### Recommended MCF presentation

Use two clearly separated experiments:

1. **MCF-ZU main comparison:** published ZeroUnlearn semantics, common `Eff/Gen/Spe/PPL` evaluator, same 10 seeds, all methods rerun locally.
2. **MCF-Original deletion stress test:** original `target_true` is sensitive; report `FS/GFS`, Spe-success/Spe-margin, semantic leakage, retain KL, and PPL. Treat this as a new deletion task, not as the ZeroUnlearn table.

For ZsRE, retain `Eff/Gen/Spe/PPL` because the original answer is the sensitive answer in the upstream adapter. There is no benefit to renaming those native columns `FS/GFS`.

## Key finding 3: one ontology, benchmark-native measurements

Use a common conceptual ontology while preserving native metric names:

| Cross-benchmark axis | Meaning | Reporting rule |
| --- | --- | --- |
| Efficacy | leakage on method-visible deletion requests | Use the benchmark's native direct/forget metric |
| Generalization/robustness | leakage on held-out prompts, facts, attacks, or compositions | Use native paraphrase, adversarial, multi-hop, or sequence metrics |
| Locality/retention | preservation of neighboring or retained knowledge | Use native neighbor/retain/reference-model metrics |
| General utility | fluency and downstream capability preservation | Use native utility suite plus fixed PPL/KL where meaningful |
| Privacy | membership or extraction leakage | Report whenever the benchmark defines it |
| Efficiency | wall time, GPU hours, peak memory, trainable parameters, checkpoint size | Use identical hardware accounting conventions |

### Main metric contract by dataset

| Dataset | Primary forgetting columns | Generalization/attack columns | Locality/utility columns | Important note |
| --- | --- | --- | --- | --- |
| MCF-ZU | Eff ↓ | Gen ↓ | Spe ↑, Spe-success ↑, PPL/base-PPL ratio ↓ | Main comparison with ZeroUnlearn; target-new-sensitive semantics |
| MCF-Original | FS ↑ | GFS ↑, semantic generation leakage ↓ | Spe-success/margin ↑, exact retain KL ↓, PPL ratio ↓ | Separate stress test; never merge with MCF-ZU |
| ZsRE | Eff ↓ | Gen ↓, free-generation sensitive leakage ↓ | Spe ↑, retain metrics ↑, PPL ratio ↓ | Eff/Gen/Spe implementation-compatible with ZeroUnlearn; PPL must be relocked |
| TOFU | Forget Quality/KS p-value ↑ plus KS statistic/effect size ↓ | paraphrased and perturbed forget metrics; MIA/extraction diagnostics | Model Utility ↑; retain, real-author, and world-fact components ↑ | Do not use the p-value alone; report distributions/effect size and native components |
| RWKU | Forget FB/QA/AA/All ↓ | attack-type breakdown ↓; MIA FM ↑ and RM ↓ | Neighbor FB/QA/All ↑; Gen/Rea/Tru/Fac/Flu ↑ | Keep every native direction; do not compress to one entity score |
| MUSE | VerbMem-F ↓, KnowMem-F ↓ | PrivLeak close to 0; scalability and sequential-unlearning curves | KnowMem-R ↑ and retrain gap ↓ | Raw PrivLeak is **not** simply lower-is-better; target interval is approximately [-5%, +5%] |
| MQuAKE ZeroUnlearn-style | Eff ↓ | AtomicGen ↓ and old-answer multi-hop leakage ↓ as post-selection extensions | retain atomic performance ↑, PPL ratio ↓ | Upstream few-shot table natively reports Eff and PPL; same forget requests are trained and scored |
| WMDP | WMDP-Bio/Cyber accuracy toward 25% chance | adversarial/relearning/probe recovery ↓ where run | MMLU ↑, subject-level MMLU ↑, MT-Bench ↑, other WMDP domains retained | Four-choice chance is 25%; below-chance answer-row suppression is not automatically stronger unlearning |

If `MQuAKE-R` means **MQuAKE-Remastered**, it is a different benchmark from the repository's current ZeroUnlearn-style MQuAKE-CF-3k-v2 track. Resolve that identity before producing a paper row. Do not silently relabel the existing track.

### Do not compute a universal raw average

A mean of `Eff`, TOFU Forget Quality, RWKU ROUGE, MUSE PrivLeak, and WMDP accuracy has no coherent unit or interpretation. The main paper should instead use:

- one factual-unlearning table for MCF-ZU, ZsRE, and MQuAKE;
- one data/entity-unlearning table for TOFU, RWKU, and MUSE;
- one hazardous-capability table for WMDP;
- a compact summary table containing win/tie/loss counts, average rank, and Pareto-front membership under predeclared utility constraints.

If a single summary score is required, make it secondary and compute it only after freezing a normalization rule against Base and the benchmark's oracle/reference. Report the native values beside it. A safer summary is **constrained forgetting**: compare forgetting only among checkpoints that satisfy predeclared utility, locality, fluency, and collapse gates.

## Architecture portability assessment

### Shared SURE core that can remain invariant

The defensible shared core is:

- frozen Base identity and tokenizer;
- target-conditioned hidden-state collection from method-visible forget data only;
- sparse, low-rank, context-conditioned edit parameterization;
- bounded forgetting objective rather than unbounded destructive GA;
- exact or audited external-utility KL with a target-excluded Wikipedia cache;
- checkpoint-dtype materialization gate;
- Stage 1 followed by residual Stage 2 repair;
- one-way freeze before any official held-out evaluation is opened.

Define the paper variants before running more benchmarks:

- **SURE-F:** shared SURE learner using only the method-visible forget data and benchmark-permitted preservation data, with no large external Wikipedia cache.
- **SURE-U100K:** identical optimizer, edit scope, ranks, gates, and stopping rules, adding one pinned target-excluded 100k-document Wikipedia utility cache.

If `SURE-F` currently means something else, rename and lock it now. The two rows are scientifically useful only if the 100k cache is the sole intentional difference.

### Dataset-by-dataset feasibility

| Dataset | Can the current head-only QA architecture be reused unchanged? | Required adaptation | Risk level |
| --- | :---: | --- | --- |
| MCF | Mostly | Lock the correct MCF-ZU versus MCF-Original target semantics | Low |
| ZsRE | Mostly | Relock PPL/provenance; add generation robustness audit | Low |
| MQuAKE-CF-3k-v2 | Mostly | Instance-first sampling, multi-row atomic facts, held-out atomic/multi-hop evaluation | Medium |
| RWKU | No, but the core is plausible | Generate target-only atomic facts without opening probes; broaden sensitive rows; test Level 1/2/3, attacks, MIA, neighbors, utility | Medium-high |
| TOFU | No | Author/profile-level sequence and multi-token answer adapter; official Full/retain models; distributional forget-quality evaluation | High |
| MUSE | No | Passage-window/span objective, sequence-scale batching, retain1-only calibration, retain2/holdout firewall, sequential requests | High |
| WMDP | No | Hazardous-corpus sequence objective; likely contextual MLP/attention edit scope in addition to head-only; MC questions remain evaluation-only | Very high |

The Stephen King `RWKU-H-W1K` bundle is correctly labeled a feasibility experiment. Passing its generated atomic gate proves decoder suppression on the generated training distribution, not entity erasure. The paper result requires official Level 1/2/3, adversarial, MIA, neighbor, utility, fluency, and frozen-head outcomes after checkpoint freeze.

For MUSE and WMDP, suppressing a handful of output rows can game surface metrics while leaving internal knowledge recoverable. Treat head-only SURE as an ablation. The primary extension may need sparse contextual edits in selected MLP/output projections while retaining the same utility-constrained, low-rank SURE principle.

## Baseline matrix: replace checkmarks with provenance classes

A checkmark currently hides whether a result is native, reproduced, or newly adapted. Use these classes:

- **N — Native:** official benchmark/method release directly supports the dataset and task.
- **R — Reproduced:** a maintained unified framework supports the exact benchmark/model/evaluator.
- **A — Adapted:** this paper creates a new adapter; label it and include adapter details.
- **X — Inappropriate/unavailable:** no defensible task mapping or released implementation.

Key corrections to the proposed matrix:

- ZeroUnlearn is native for MCF, ZsRE, and MQuAKE in its public repository. TOFU, RWKU, MUSE, and WMDP would be **new adaptations**, not native ZeroUnlearn results.
- ROME, MEMIT, and AlphaEdit are factual model-editing baselines. They are natural on MCF/ZsRE/MQuAKE, but TOFU/RWKU adaptations belong in the appendix and passage/domain use is generally inappropriate.
- GA and NPO are the broadest common optimization baselines. They have native or maintained implementations across the long-form suites, but still require exact dataset/model protocol locking.
- GradDiff, SimNPO, PDU, and RMU are available for TOFU/MUSE/WMDP through maintained implementations such as OpenUnlearning; this does not automatically make them native RWKU or factual-editing baselines.
- RWKU's original paper directly reports ICU, RepE, GA, DPO, NPO, and refusal tuning. Other methods are adaptations.
- RMU is native to WMDP. Do not merge `RMU/TAR` into one row; source-pin and report them as separate algorithms.
- Retrain is a meaningful oracle for TOFU and MUSE because retain-only reference models exist. It is not a feasible oracle for MCF/ZsRE/RWKU/MQuAKE or for removing pretraining-era WMDP knowledge.

For every method×dataset cell, store `support_class`, source repository, immutable commit, model revision, dataset revision/hash, training-visible roles, evaluator revision, hyperparameter-search budget, and result artifact path.

## Apples-to-apples experimental contract

Every main-table comparison must satisfy all of the following:

- exact target model and tokenizer revision per benchmark;
- identical forget/retain split or official target set;
- identical method-visible data roles and held-out firewall;
- identical native evaluator revision and decoding configuration;
- same checkpoint selection rule based only on permitted calibration data;
- same seed list and aggregation convention;
- comparable hyperparameter-search budget, with search trials and selection metric disclosed;
- mean, sample SD, and 95% confidence interval over independent seeds when seeds exist;
- raw numerator/denominator counts for discrete metrics;
- Base and oracle/reference rows evaluated by the same local evaluator;
- compute, memory, trainable-parameter, and checkpoint-size accounting;
- no post hoc rejection of seeds or selection on official held-out outcomes.

Recommended seed policy:

- MCF/ZsRE/MQuAKE few-shot: seeds 1–10;
- TOFU: at least 5 independent runs or official splits, preferably 10 when affordable;
- RWKU: multiple target entities and batch sizes, not repeated randomness on Stephen King alone;
- MUSE/WMDP: at least 3 expensive independent runs, plus bootstrap confidence intervals over evaluation examples; clearly distinguish example uncertainty from training-run uncertainty.

## Ordered execution plan

### Phase 0 — Freeze claims and contracts

- [ ] Define SURE-F and SURE-U100K so the cache is the only intended difference.
- [ ] Split MCF-ZU and MCF-Original into separate protocol IDs and table destinations.
- [ ] Correct MUSE PrivLeak from `lower` to `distance-to-zero/target interval`.
- [ ] Resolve whether `MQuAKE-R` means MQuAKE-Remastered or the current ZeroUnlearn-style track.
- [ ] Create the method×dataset provenance registry with N/R/A/X support classes.
- [ ] Freeze metric schema, directions, units, aggregation, and tie handling.

### Phase 1 — Repair the factual benchmark foundation

- [ ] Rerun ZsRE Base and SURE seeds 1–10 with exact receipts and per-seed artifacts.
- [ ] Run vendored ZeroUnlearn on the same ZsRE model/data/evaluator and resolve PPL to exact Base parity.
- [ ] Add deterministic semantic-leak and generation-collapse audits post freeze.
- [ ] Run MCF-ZU SURE-F/SURE-U100K and factual baselines under the common evaluator.
- [ ] Keep MCF-Original FS/GFS in a separate stress-test table.
- [ ] Freeze the MQuAKE identity and run native Eff/PPL plus post-selection atomic/multi-hop leakage.

### Phase 2 — Entity and profile unlearning

- [ ] Complete `RWKU-H-W1K` official post-freeze evaluation before scaling.
- [ ] Promote RWKU only if neighbor/utility and adversarial/MIA gates pass; then scale 1 → 10 → 20 → 50 targets.
- [ ] Implement TOFU on the official Full model with retain-only reference and native evaluator.
- [ ] Run the common TOFU baseline set from one pinned framework.

### Phase 3 — Passage and hazardous-domain extensions

- [ ] Implement MUSE passage-level SURE-F first; add U100K only after native metrics run end to end.
- [ ] Evaluate MUSE News and Books, including privacy, scale, and sequential requests.
- [ ] Run WMDP head-only feasibility as an ablation.
- [ ] If head-only fails robustness or utility, implement the predeclared contextual-layer extension.
- [ ] Compare WMDP against RMU, NPO, SimNPO, GA/GradDiff where supported, using WMDP/MMLU/MT-Bench and robustness probes.

### Phase 4 — Confirmatory paper run

- [ ] Freeze all hyperparameters before opening confirmatory held-out data.
- [ ] Execute the final seed/target matrix without mid-run tuning.
- [ ] Generate native tables, Pareto plots, confidence intervals, and cost table from raw artifacts.
- [ ] Run artifact completeness checks and independently reproduce at least one seed per benchmark.
- [ ] Write claims from the frozen results, including losses and failures.

## Publication acceptance gates

A dataset row may enter the main paper only when:

- [ ] benchmark source, model, tokenizer, evaluator, and dataset are immutable and hashed;
- [ ] method-visible versus evaluation-only roles are machine-checked;
- [ ] Base and at least one strong baseline reproduce within a predeclared tolerance;
- [ ] all per-seed/target artifacts are present;
- [ ] metric directions and units pass schema validation;
- [ ] utility/collapse gates pass;
- [ ] no official held-out metric influenced checkpoint selection;
- [ ] confidence intervals and raw counts are generated automatically;
- [ ] compute and hyperparameter-search budgets are disclosed;
- [ ] the exact command and commit reproduce the table row.

## Limitations and robustness

The local audit can establish code semantics and artifact completeness, but it cannot reconstruct missing GPU outputs. The public table comparison is therefore descriptive. The current repository also contains several historical SURE variants; their results must not be mixed unless architecture signatures, data access, and evaluators match.

The plan deliberately avoids promising that SURE will be best on every scalar. Reviewers are more likely to trust a method that dominates or expands the Pareto frontier under strong utility constraints than one optimized separately against every held-out metric. Negative results—especially on MUSE privacy or WMDP recovery—should be reported because they establish the boundary of sparse decoder editing.

The packaged report includes one narrow chart comparing the same-unit ZsRE Base `Eff/Gen/Spe` values. PPL is excluded because its fixture is unresolved, and all cross-benchmark compatibility evidence remains tabular to avoid implying invalid raw-score comparability.

## Primary sources

- [ZeroUnlearn official repository](https://github.com/XMUDeepLIT/ZeroUnlearn)
- [TOFU project and native metric definitions](https://locuslab.github.io/tofu/)
- [OpenUnlearning maintained benchmark/method framework](https://github.com/locuslab/open-unlearning)
- [RWKU paper](https://rwku-bench.github.io/static/RWKU.pdf)
- [MUSE paper](https://arxiv.org/abs/2407.06460)
- [MUSE official evaluator](https://github.com/jaechan-repo/muse_bench)
- [MQuAKE official repository](https://github.com/princeton-nlp/MQuAKE)
- [WMDP official repository](https://github.com/centerforaisafety/wmdp)
- [WMDP paper](https://arxiv.org/abs/2403.03218)

## Further questions

- What is the intended expansion of `SURE-F`?
- Does `MQuAKE-R` mean MQuAKE-Remastered, or the current ZeroUnlearn-style deletion track?
- Which exact model family is the main paper anchor: benchmark-native models per dataset, or one common Llama family where possible?
- What compute budget can be precommitted for baseline tuning and final seeds?
- Will the headline claim be “best native metric,” “best constrained forgetting,” or “best average rank under utility gates”?
