# Evidence: existing fact-unlearning benchmarks are saturated

All numbers below come from this repository's own experiments (`../RESULTS_L16_PAPER_FINAL.md`, commits `baa6252`, `0fe3ded`, `95e824a`). The setup was Llama-3.2-3B-Instruct, mean over 5 seeds, each benchmark's official evaluator, and 50 forget facts per seed. "Ours" is SURE, a detect-then-edit method: a per-fact linear router on a layer-19 state, behind exact subject-token eligibility, plus one residual row per fact written at layer 16. "Genie" applies the same rows with perfect routing.

## 1. A strong method reaches the ceiling on all four standard benchmarks

Share of base Eff removed / share of base Spe kept:

| Method | MCF | ZsRE | MQuAKE | RWKU |
|---|---|---|---|---|
| GA | 99.8% / 1.8% | 40.1% / 88.1% | 89.1% / 19.2% | 59.7% / 62.1% |
| NPO | 99.8% / 0.6% | 45.8% / 85.3% | 93.7% / 12.7% | 67.4% / 47.4% |
| RMU | 99.9% / 5.0% | 100.0% / 25.3% | 99.6% / 1.0% | 98.6% / 0.2% |
| ROME | 64.7% / 92.4% | −3.4% / 100.4% | 86.0% / 48.9% | 95.8% / 48.3% |
| MEMIT | 29.0% / 99.4% | −1.8% / 100.0% | 57.2% / 83.2% | 36.1% / 94.1% |
| ZeroUnlearn | 98.0% / 79.7% | 14.1% / 98.1% | 74.1% / 75.3% | 67.4% / 77.7% |
| **SURE (ours)** | **100.0% / 99.8%** | **99.1% / 100.0%** | **100.0% / 83.8%** (98.6% on Spe†) | **96.5% / 90.1%** |

On MCF, SURE reaches Eff 0.001 / Gen 0.086 against genie's 0.001 / 0.002, with PPL exactly at base. A benchmark on which one method sits at the ceiling no longer ranks methods. It only confirms that the method has stopped making mistakes on *that probe distribution*.

## 2. Why the ceiling is not the whole story

Each of the following findings, all from this repository, points to a gap that SCALPEL measures directly.

| Finding (repository evidence) | What it means | SCALPEL family / axis |
|---|---|---|
| **MCF base Eff is only 13.5%**, base Spe 19.9% | Most "forgotten" MCF facts were never known, so near-zero Eff is easy | Knowledge filter: every probe must be known by the base model (§4.1) |
| **RWKU held-out facts about a forgotten person: Gen 17.96 vs base 54.77**, and entity-genie 17.75 | Even perfect routing leaves a third of held-out entity knowledge; fact-level edits do not generalise to other facts about the same entity | Entity track; `indirect`, `multihop` |
| **MQuAKE multi-hop Gen 7.20, genie 0.00** | Detectors miss chain questions where the subject is not named | `multihop`, `indirect` |
| **MQuAKE raw Spe kept 83.8%; 15–18% of retain facts repeat a forget fact** | The official locality set has label conflicts | Conflict removal (§4.3) |
| **RWKU Spe 90.1% under the subject gate** (genie keeps 100%) | Routers fire on neighbour probes that mention the person | `subject_mentioned`, `same_subject`, `name_collision` |
| **ZsRE: lowering the router cutoff gives 80% paraphrase routing but 17% same-subject false fires** (`../README_LAYER_SWEEP.md`) | A recall/locality trade-off that the official metrics barely register | `same_subject`, routing decomposition (§6.6) |
| **The router uses exact subject-token eligibility** | Typos, case, aliases, pronouns and descriptions bypass it, and *a string blocklist would score well on the official metrics* | `surface`, `multiturn`, `indirect`; the `subject_blocklist_idk` and `keyword_router_idk` reference cheats |
| **MCF neighbourhood prompts share the forgotten answer** | Any global answer suppression destroys locality, so the official Spe already punishes it. But no official metric rewards a method for *keeping* same-subject facts | `same_object` + `same_subject` + `same_sentence` |
| **Lens evaluation exists for MCF seed 1 only; quantization and relearning were never run** | Completeness (is the fact gone from the weights?) is untested for every method in the table | R axis: lens, int8/int4, relearning |
| **SURE is a runtime component (hooks + router)** | In a weights-only threat model the base weights are unchanged | Tiers B / W / W+ (§2) |

## 3. The same blind spot, measured (pilot, `results/pilot/`)

SCALPEL ran end to end on real MultiCounterFact + zsRE + MQuAKE facts, using Qwen2.5-0.5B-Instruct on a laptop: 50 requests, 520 probes, one seed. The leaderboard scores every run twice. The **legacy view** uses only the families the official MCF/zsRE metrics use; **SCALPEL** uses all of them.

| Method | Legacy view: erasure / locality | SCALPEL: E / L | SCALPEL |
|---|---|---|---|
| subject-name blocklist + "I don't know" | **100.0 / 98.5** | 6.8 / 2.0 | 0.0 |
| subject + relation-word router + refusal | 82.5 / 100.0 | 5.9 / 7.0 | 0.0 |
| **SURE (ours, untuned)** | 68.5 / 100.0 | 3.3 / **100.0** | **10.5** |
| GA | 100.0 / 5.4 | 99.0 / 1.8 | 6.7 |
| NPO | 100.0 / 0.0 | 99.2 / 1.3 | 4.9 |

A string blocklist is nearly perfect by the legacy definition. On the full benchmark it collapses:

- it leaks under typos (100% residual), multi-hop chains (92%) and subject-free descriptions;
- it refuses same-subject, same-sentence, subject-mentioned and in-context questions (0% retained).

**SURE leads the pilot, with headroom.** It keeps perfect locality (100 on every retain family, no false fires) while most held-out phrasings leak: question 90% residual, cross-lingual 94%, typos 100%, multi-hop 100%. The routing decomposition attributes 48.7 of the forget residual to routing misses and 9.1 to actuation: the same "gap to genie is routing" result as `../README_ROUTER_DIAGNOSTICS.md`, now on families the legacy benchmarks do not contain.

All six reference cheats score SCALPEL 0 (SCALPEL_core 2.8–16.2), and every split-integrity check passes (`results/pilot/qwen2.5-0.5b-instruct/runs/RED_TEAM_REPORT.md`). Full tables for Llama-3.2-3B / 8B and Qwen3-4B on the cluster are milestone M2 (`ROADMAP.md`); the job scripts are in `scripts/slurm/`.
