# Evidence: existing fact-unlearning benchmarks are saturated

All numbers below come from this repository's own experiments (`../RESULTS_L16_PAPER_FINAL.md`, commits `baa6252`, `0fe3ded`, `95e824a`). The setup was Llama-3.2-3B-Instruct, mean over 5 seeds, each benchmark's official evaluator, and 50 forget facts per seed. "SURE (ours, published)" is the detect-then-edit method as published: a per-fact linear router on a layer-19 state, behind exact subject-token eligibility, plus one residual row per fact written at layer 16 in a second pass. "Ours (new classifier)" is the current version (`outputs/onepass_l19_v1`, 2026-10-25): the router reads layer 19 with fuzzy subject matching and is trained on 20 quality-checked positives (base-model rewrites validated by answer transfer) and 20 checked negatives per fact; the row is added at layer 19 in the same pass. "Genie" applies the same rows with perfect routing.

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
| **SURE (ours, published)** | **100.0% / 99.8%** | **99.1% / 100.0%** | **100.0% / 83.8%** (98.6% on Spe†) | **96.5% / 90.1%** |
| **Ours (new classifier)** | **100.0% / 99.8%** | **98.8% / 100.0%** | **99.0% / 84.1%** (98.8% on Spe†) | **95.8% / 96.4%** |

On MCF, SURE reaches Eff 0.001 / Gen 0.086 against genie's 0.001 / 0.002 (new classifier: Gen 0.054), with PPL exactly at base. A benchmark on which one method sits at the ceiling no longer ranks methods. It only confirms that the method has stopped making mistakes on *that probe distribution*.

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

## 4. SCALPEL on Llama-3.2-3B-Instruct (cluster, 5 seeds)

`results/cluster/llama32_3b/dev/leaderboard/leaderboard.md`, dev partition, seeds 1–5, 50 requests per seed, teacher-forced scorer, mean ± sd over seeds. † = reference cheat. "ours" is the new-classifier method above; "sure" is the original SURE (exact matching, the request's own prompts only).

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict |
|---|---|---|---|---|---|---|---|---|---|
| ours | **30.9 ± 10.7** | 26.7 ± 20.6 | 5.2 ± 6.1 | 98.4 ± 0.9 | 95.8 ± 1.5 | 85.4 ± 6.9 | 100.0 | 19.0 ± 3.6 | 2.9 |
| zerounlearn | 21.4 ± 10.3 | 38.9 ± 17.9 | 21.1 ± 12.2 | 96.4 ± 3.8 | 67.4 ± 3.9 | 38.6 ± 6.2 | 96.9 ± 1.3 | 8.7 ± 4.3 | – |
| sure | 12.0 ± 0.4 | 3.8 ± 0.0 | 0.0 | 95.6 ± 4.6 | 99.7 ± 0.4 | 97.5 ± 2.8 | 99.6 ± 0.7 | 19.0 ± 3.6 | 2.9 |
| npo | 6.2 ± 2.5 | 91.9 ± 14.8 | 84.1 ± 25.1 | 100.0 | 1.7 ± 0.7 | 0.0 | 74.9 ± 18.4 | 80.3 ± 18.4 | – |
| gd | 5.5 ± 1.6 | 94.8 ± 5.2 | 87.0 ± 12.6 | 100.0 | 1.5 ± 0.4 | 0.0 | 66.5 ± 10.1 | 86.5 ± 9.5 | – |
| ga | 4.9 ± 1.5 | 97.1 ± 4.1 | 91.9 ± 11.2 | 100.0 | 1.3 ± 0.4 | 0.0 | 56.2 ± 16.9 | 90.5 ± 5.7 | – |
| rmu | 3.0 ± 4.5 | 8.3 ± 6.9 | 1.7 ± 2.4 | 11.6 ± 4.6 | 39.7 ± 48.5 | 32.6 ± 44.7 | 84.5 ± 22.3 | 5.8 ± 12.0 | – |
| rome | 1.6 ± 3.6 | 8.5 ± 3.6 | 2.0 ± 1.8 | 82.8 ± 5.0 | 57.2 ± 11.4 | 30.0 ± 12.5 | 98.7 ± 0.7 | 0.8 ± 1.9 | – |
| memit | 0.0 | 9.0 ± 4.1 | 3.9 ± 2.9 | 45.6 ± 6.2 | 91.8 ± 1.4 | 76.6 ± 6.7 | 99.2 ± 0.8 | 0.0 | – |
| answer_ban † | 0.0 | 91.8 ± 3.6 | 72.1 ± 10.7 | 95.2 ± 3.6 | 7.3 ± 0.1 | 0.0 | 46.7 ± 5.5 | 0.0 | 2.9 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 11.6 ± 1.2 | 0.0 | 2.7 |
| keyword_router_idk † | 0.0 | 6.4 ± 0.1 | 0.0 | 100.0 | 42.3 ± 9.3 | 12.6 ± 3.9 | 100.0 | 0.0 | 2.9 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| prompt_cache_idk † | 0.0 | 1.2 ± 0.1 | 0.0 | 100.0 | 90.4 ± 4.2 | 66.6 ± 10.6 | 100.0 | 0.0 | 2.9 |
| subject_blocklist_idk † | 0.0 | 7.3 ± 0.1 | 0.0 | 100.0 | 2.0 ± 0.0 | 0.0 | 99.8 ± 0.3 | 0.0 | 2.9 |

What the table shows:

- **The benchmark separates methods the legacy benchmarks cannot.** Every method that forgets the request itself (E_direct 83–100) differs sharply on held-out phrasings (E), on entangled facts (L) and under attack (R).
- **Every shortcut fails.** All six reference cheats score SCALPEL 0: each zeroes at least one family (the subject blocklist erases almost nothing that avoids the name and refuses same-subject questions; refuse-all keeps no locality).
- **Each baseline fails a different axis.** GA, GD and NPO erase 92–97 but keep locality 1.3–1.7. ROME and MEMIT keep utility and (MEMIT) locality, but erase only 8–9 on held-out phrasings, and the fact returns under attack (R 0.8 and 0.0). RMU erases little at this setting (E 8.3). ZeroUnlearn erases the most among baselines (38.9) at locality 67.4.
- **Our own method leads, with clear headroom.** It forgets the request prompts (E_direct 98.4) and keeps locality (95.8), but held-out phrasings still leak (E 26.7, worst family 5.2), and it falls to 2.9 in the weights-only tier, because its router and rows are runtime components. The original SURE, with exact matching and no rewrites, erases only 3.8 on held-out phrasings.
- **Pending:** two runs with the latest rewrite checks are in progress (the legacy table and SCALPEL's ours); the ours rows above will be updated when they finish.

