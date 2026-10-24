# Roadmap, deliverables and asks

## Status today (v0.1, this folder)

| Component | Status |
|---|---|
| Split builder: 4 sources, knowledge filter, partitions, hidden test, conflict removal, canary | Done, tested |
| 11 forget families + 9 retain families, fact track + RWKU entity track | Done, tested |
| Teacher-forced and open-ended scorers, behaviour labels | Done, tested |
| Utility (MMLU, TriviaQA, PPL), KL on retain probes | Done |
| Attacks: logit lens, int8/int4, relearning | Done, tested (restore exactly) |
| Tiers B / W / W+, causality check, boundary contract | Done, tested |
| Sequential deletion track | Done, tested |
| Detect-then-edit decomposition (routing vs actuation) | Done |
| Baselines in-harness: 6 reference cheats, GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn | Done |
| External: HF checkpoint loader, plugin loader, SURE adapter | Done. SURE adapter runs end to end on the laptop pilot (Qwen2.5-0.5B, transformers 4.57); cluster run is M2 |
| ROME / MEMIT / ZeroUnlearn in-harness adapters (around the vendored `ZeroUnlearn/` code via `paper_baselines.py`) | Done; request format tested. The vendored code is CUDA-only, so first full run is the Wulver smoke job |
| Red-team gate + split validator | Done, passing on toy and pilot |
| Reports: leaderboard, Pareto plot, CIs, paired tests | Done |
| Toy world (CPU, minutes) and real-data pilot (laptop) | Done |

## Milestones

| | Deliverable | Needs |
|---|---|---|
| **M1** (month 1) | v1.0 freeze: dev splits for 3 model families × 2 sizes × 5 seeds; hidden test built and held; zsRE redistribution terms confirmed; HF dataset card | Compute |
| **M2** (months 1–2) | Full baseline tables on the cluster (`scripts/submit_scalpel_compare.sh full all`): GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn, SURE, and the 6 cheats; all tiers and attacks; scaling curve 50 → 1k → 10k requests; sequential track | Compute (≈1,000 A100-hours, below) |
| **M3** (months 2–4) | **Expert data**: native-speaker verification of the cross-lingual templates and expansion to 10 languages; expert-written indirect references (descriptions, aliases, coreference chains) for 2,000 subjects; human-labelled grader calibration set (1,000 generations: leak / refusal / hallucination / correct) | Snorkel expert network |
| **M4** (months 3–5) | **Agentic track** (prototype exists: `AGENT_TRACK.md`): the unlearned model runs as an agent with a scratchpad and a mock search tool over a corpus with the target fact removed. Does multi-step reasoning (self-ask, decomposition, tool calls) reconstruct the forgotten fact? Does the agent leak it through tool queries? Long-horizon multi-turn sessions (≥ 10 turns) | Snorkel research collaboration; environment design |
| **M5** (month 6) | Public leaderboard (dev) + hidden-test evaluation service; paper (NeurIPS Datasets & Benchmarks track) | Hosting |

## In-kind asks

1. **Expert data** (M3):
   - linguists or native speakers, for cross-lingual probe verification and expansion;
   - domain annotators, to write unambiguous indirect references that are *not* inferable from retained facts;
   - annotators, to label generations for grader calibration;
   - privacy and legal experts, to design realistic deletion-request scenarios (GDPR Art. 17 style: a person asks for specific facts to be removed; what must remain).
2. **Compute**:
   - per run (one model × seed × method) at 3B: roughly 10–20 A100-minutes, covering unlearning, core probes, generation and attacks;
   - 6 models × 10 seeds × 14 methods ≈ 840 runs ≈ 250 A100-hours;
   - plus the scaling curve (1k / 10k requests), the sequential track and the 8B models.
   - **Ask: ~1,000 A100-hours.**
3. **Red-team review**: Snorkel's benchmark red team examines grader exploits, verifier reliability and contamination before v1.0.
4. **Hosting**: the hidden-test evaluation service and the leaderboard.

## Risks

| Risk | Mitigation |
|---|---|
| Knowledge-filter yield is low for small models in QA/chat formats | Report yields per family. Families below 5 probes are excluded and flagged. Release models start at 3B (Llama-3.2-3B base Eff on MCF is 13.5%, and SCALPEL screens until it has enough known facts) |
| `indirect` probes inferable from correlated retained facts | Description facts come from other answer groups; expert-written references (M3) |
| A future method saturates SCALPEL | Per-family scores flag saturation. Harder families (agentic, more languages, 10k-request scale) are planned, and the hidden test is refreshed |
| Grader disagreement on open-ended outputs | The teacher-forced scorer is primary; the human-labelled calibration set (M3) measures the generation grader |
| The benchmark favours the authors' own method | SURE is one baseline among 14. The reference cheats include a string version of its design (`keyword_router_idk`), and its expected failure families (indirect, multihop, crosslingual, surface; the strict tier) are reported like any other method's |
