# Evidence: existing fact-unlearning benchmarks are saturated

All numbers below come from this repository's own experiments (`../RESULTS_L16_PAPER_FINAL.md`, commits `baa6252`, `0fe3ded`, `95e824a`). The setup was Llama-3.2-3B-Instruct, mean over 5 seeds, each benchmark's official evaluator, and 50 forget facts per seed. "SURE (published configuration)" is SURE, the detect-then-edit method we built, as published: a per-fact linear router on a layer-19 state, behind exact subject-token eligibility, plus one residual row per fact written at layer 16 in a second pass. "Genie" applies the same rows with perfect routing.

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
| **SURE (published configuration)** | **100.0% / 99.8%** | **99.1% / 100.0%** | **100.0% / 83.8%** (98.6% on Spe†) | **96.5% / 90.1%** |

On MCF, SURE reaches Eff 0.001 / Gen 0.086 against genie's 0.001 / 0.002, with PPL exactly at base. A benchmark on which one method sits at the ceiling no longer ranks methods. It only confirms that the method has stopped making mistakes on *that probe distribution*.

## 2. Why the ceiling is not the whole story

Each of the following findings, all from this repository, points to a gap that LETHE-Bench measures directly.

| Finding (repository evidence) | What it means | LETHE-Bench family / axis |
|---|---|---|
| **MCF base Eff is only 13.5%**, base Spe 19.9% | Most "forgotten" MCF facts were never known, so near-zero Eff is easy | Knowledge filter: every probe must be known by the base model (§4.1) |
| **RWKU held-out facts about a forgotten person: Gen 17.96 vs base 54.77**, and entity-genie 17.75 | Even perfect routing leaves a third of held-out entity knowledge; fact-level edits do not generalise to other facts about the same entity | Entity track; `indirect`, `multihop` |
| **MQuAKE raw Spe kept 83.8%; 15–18% of retain facts repeat a forget fact** | The official locality set has label conflicts | Conflict removal (§4.3) |
| **RWKU Spe 90.1% under the subject gate** (genie keeps 100%) | Routers fire on neighbour probes that mention the person | `subject_mentioned`, `same_subject`, `name_collision` |
| **ZsRE: lowering the router cutoff gives 80% paraphrase routing but 17% same-subject false fires** (`../README_LAYER_SWEEP.md`) | A recall/locality trade-off that the official metrics barely register | `same_subject`, routing decomposition (§6.6) |
| **The router uses exact subject-token eligibility** | Typos, case, aliases, pronouns and descriptions bypass it, and *a string blocklist would score well on the official metrics* | `surface`, `multiturn`, `indirect`; the `subject_blocklist_idk` and `keyword_router_idk` reference cheats |
| **MCF neighbourhood prompts share the forgotten answer** | Any global answer suppression destroys locality, so the official Spe already punishes it. But no official metric rewards a method for *keeping* same-subject facts | `same_object` + `same_subject` + `same_sentence` |
| **Lens evaluation exists for MCF seed 1 only; quantization and relearning were never run** | Completeness (is the fact gone from the weights?) is untested for every method in the table | R axis: lens, int8/int4, relearning |
| **SURE is a runtime component (hooks + router)** | In a weights-only threat model the base weights are unchanged | Tiers B / W / W+ (§2) |

## 3. The same blind spot, measured (pilot, `results/pilot/`)

LETHE-Bench ran end to end on real MultiCounterFact + zsRE + MQuAKE facts, using Qwen2.5-0.5B-Instruct on a laptop: 50 requests, 520 probes, one seed. The leaderboard scores every run twice. The **legacy view** uses only the families the official MCF/zsRE metrics use; **LETHE-Bench** uses all of them.

| Method | Legacy view: erasure / locality | LETHE-Bench: E / L | LETHE |
|---|---|---|---|
| subject-name blocklist + "I don't know" | **100.0 / 98.5** | 6.8 / 2.0 | 0.0 |
| subject + relation-word router + refusal | 82.5 / 100.0 | 5.9 / 7.0 | 0.0 |
| GA | 100.0 / 5.4 | 99.0 / 1.8 | 6.7 |
| NPO | 100.0 / 0.0 | 99.2 / 1.3 | 4.9 |

A string blocklist is nearly perfect by the legacy definition. On the full benchmark it collapses:

- it leaks under typos (100% residual), multi-hop chains (92%) and subject-free descriptions;
- it refuses same-subject, same-sentence, subject-mentioned and in-context questions (0% retained).

SURE's pilot run is not shown: it used an early, untuned adapter, not the published configuration (see §4).

All six reference cheats score 0 in this pilot file (below 4 with the current 1-point floor on R; LETHE_core 2.8–16.2), and every split-integrity check passes (`results/pilot/qwen2.5-0.5b-instruct/runs/RED_TEAM_REPORT.md`). Full tables for Llama-3.2-3B, Qwen3-4B and Llama-3.1-8B on the cluster are in §4; the job scripts are in `scripts/slurm/`.

## 4. LETHE-Bench on three models (cluster, 5 seeds each)

`results/cluster/leaderboard_dev/leaderboard.md` (all models) and `results/cluster/<model>/dev/leaderboard/leaderboard.md`; dev partition, seeds 1–5, 50 requests per seed, teacher-forced scorer (the generation scorer gives the same ordering), mean ± sd over seeds. 10 methods and 6 reference cheats (†) on Llama-3.2-3B-Instruct, Qwen3-4B and Llama-3.1-8B-Instruct. SURE, the detect-then-edit method the authors built, is not in the table yet. The runs made so far used a one-pass layer-19 configuration that is not the published method; the paper configuration (`sure_paper_l16`: the paper's router at layer 19 behind exact subject eligibility, one residual row per fact written at layer 16) is being run on the same splits and will be added as one row among the others. The Llama-3.2-3B numbers differ slightly from the earlier single-model file because the metric now floors U and R at 1 point, budgets three indirect probes per request and reports E_complete; that file is kept as `results/cluster/llama32_3b.pre_lethe_20261028/`.

**LETHE score per model** (sorted by the mean over models):

| Method | Llama-3.2-3B | Qwen3-4B | Llama-3.1-8B | mean |
|---|---|---|---|---|
| simnpo | 37.4 ± 20.1 | 31.7 ± 15.6 | 43.7 ± 13.6 | 37.6 |
| zerounlearn | 21.3 ± 9.5 | 56.3 ± 5.8 | 15.6 ± 5.2 | 31.1 |
| memit | 1.7 ± 1.8 | 30.7 ± 9.0 | 3.2 ± 0.4 | 11.9 |
| idk_ft | 11.4 ± 0.6 | 11.0 ± 1.3 | 10.9 ± 0.2 | 11.1 |
| tv | 5.7 ± 1.4 | 10.2 ± 3.8 | 7.7 ± 1.8 | 7.9 |
| rmu | 3.6 ± 3.5 | 7.5 ± 4.1 | 7.8 ± 2.7 | 6.3 |
| ga | 4.9 ± 1.4 | 6.8 ± 1.9 | 5.6 ± 1.6 | 5.8 |
| gd | 5.5 ± 1.5 | 6.3 ± 1.8 | 5.1 ± 1.1 | 5.6 |
| npo | 6.2 ± 2.4 | 5.9 ± 2.0 | 4.6 ± 1.0 | 5.6 |
| rome | 3.0 ± 3.1 | 4.5 ± 1.2 | 3.2 ± 0.6 | 3.6 |
| answer_ban † | 1.7 | 3.4 | 3.4 | 2.8 |
| keyword_router_idk † | 1.6 | 3.2 | 3.2 | 2.7 |
| subject_blocklist_idk † | 1.2 | 2.4 | 2.4 | 2.0 |
| prompt_cache_idk † | 1.1 | 2.2 | 2.2 | 1.8 |
| noop † | 1.0 | 2.0 | 2.0 | 1.7 |
| idk_all † | 1.0 | 1.9 | 1.9 | 1.6 |

**Axes per model.** E = erasure over held-out forget families (harmonic mean), L = locality (harmonic mean over retain families), U = utility, R = robustness under attack, Strict = weights only (– for methods with no runtime component). The second table under each model shows what E and L are made of: E_direct = the request prompt itself (training-visible, not in E), E_complete = share of requests whose every held-out probe is erased, E_mean / E_min = plain mean and worst forget family, L_min = worst retain family.

*Llama-3.2-3B-Instruct*

| Method | LETHE | E | L | U | R | Strict |
|---|---|---|---|---|---|---|
| simnpo | 37.4 | 63.5 | 50.8 | 81.1 | 26.1 | – |
| zerounlearn | 21.3 | 38.4 | 67.4 | 96.9 | 8.7 | – |
| idk_ft | 11.4 | 96.2 | 3.8 | 56.7 | 19.0 | – |
| npo | 6.2 | 92.3 | 1.7 | 74.8 | 80.4 | – |
| tv | 5.7 | 96.4 | 1.5 | 77.6 | 58.4 | – |
| gd | 5.5 | 95.1 | 1.5 | 66.5 | 86.6 | – |
| ga | 4.9 | 97.3 | 1.3 | 56.3 | 90.3 | – |
| rmu | 3.6 | 7.7 | 39.7 | 84.5 | 5.8 | – |
| rome | 3.0 | 8.6 | 57.2 | 98.7 | 0.8 | – |
| memit | 1.7 | 8.0 | 91.8 | 99.2 | 0.0 | – |

Behind E and L (Llama-3.2-3B):

| Method | E_direct | E_complete | E_mean | E_min | L_min |
|---|---|---|---|---|---|
| simnpo | 96.4 | 59.5 | 78.6 | 44.8 | 37.1 |
| zerounlearn | 96.4 | 30.4 | 58.4 | 20.6 | 38.6 |
| idk_ft | 100.0 | 86.3 | 96.5 | 87.0 | 0.0 |
| npo | 100.0 | 87.2 | 94.2 | 84.1 | 0.0 |
| tv | 98.0 | 89.5 | 96.5 | 91.0 | 0.0 |
| gd | 100.0 | 88.9 | 95.7 | 87.0 | 0.0 |
| ga | 100.0 | 93.5 | 97.5 | 91.9 | 0.0 |
| rmu | 11.8 | 6.5 | 34.9 | 1.4 | 32.6 |
| rome | 82.8 | 15.3 | 38.5 | 2.0 | 30.0 |
| memit | 45.6 | 3.2 | 16.6 | 3.0 | 76.6 |

*Qwen3-4B*

| Method | LETHE | E | L | U | R | Strict |
|---|---|---|---|---|---|---|
| zerounlearn | 56.3 | 68.4 | 47.3 | 91.5 | 42.4 | – |
| simnpo | 31.7 | 76.5 | 38.9 | 85.7 | 21.0 | – |
| memit | 30.7 | 27.3 | 79.8 | 99.2 | 19.9 | – |
| idk_ft | 11.0 | 89.3 | 4.1 | 53.7 | 15.8 | – |
| tv | 10.2 | 83.2 | 3.0 | 87.5 | 35.5 | – |
| rmu | 7.5 | 21.4 | 33.3 | 81.5 | 12.2 | – |
| ga | 6.8 | 79.4 | 1.9 | 60.2 | 71.3 | – |
| gd | 6.3 | 90.1 | 1.7 | 55.5 | 79.7 | – |
| npo | 5.9 | 93.2 | 1.6 | 67.8 | 75.7 | – |
| rome | 4.5 | 94.5 | 1.2 | 29.8 | 69.5 | – |

Behind E and L (Qwen3-4B):

| Method | E_direct | E_complete | E_mean | E_min | L_min |
|---|---|---|---|---|---|
| zerounlearn | 96.0 | 56.1 | 76.9 | 33.3 | 21.5 |
| simnpo | 95.6 | 59.3 | 79.5 | 53.7 | 24.7 |
| memit | 94.4 | 42.7 | 55.2 | 7.5 | 43.2 |
| idk_ft | 100.0 | 80.0 | 92.2 | 63.7 | 0.0 |
| tv | 97.2 | 70.0 | 86.2 | 62.8 | 0.0 |
| rmu | 32.4 | 15.1 | 45.5 | 7.2 | 20.5 |
| ga | 99.6 | 71.5 | 85.8 | 61.1 | 0.0 |
| gd | 100.0 | 85.0 | 92.5 | 78.3 | 0.0 |
| npo | 100.0 | 88.3 | 94.7 | 80.9 | 0.0 |
| rome | 99.2 | 89.0 | 95.4 | 85.8 | 0.0 |

*Llama-3.1-8B-Instruct*

| Method | LETHE | E | L | U | R | Strict |
|---|---|---|---|---|---|---|
| simnpo | 43.7 | 77.5 | 51.2 | 83.2 | 26.8 | – |
| zerounlearn | 15.6 | 24.5 | 79.1 | 98.2 | 5.6 | – |
| idk_ft | 10.9 | 98.3 | 3.8 | 63.5 | 13.6 | – |
| rmu | 7.8 | 8.8 | 21.5 | 56.0 | 20.7 | – |
| tv | 7.7 | 95.4 | 2.1 | 84.8 | 53.5 | – |
| ga | 5.6 | 98.9 | 1.5 | 63.8 | 84.9 | – |
| gd | 5.1 | 99.0 | 1.3 | 64.2 | 84.5 | – |
| npo | 4.6 | 99.6 | 1.2 | 57.1 | 87.8 | – |
| rome | 3.2 | 3.4 | 54.1 | 99.2 | 0.5 | – |
| memit | 3.2 | 6.7 | 87.1 | 99.5 | 0.0 | – |

Behind E and L (Llama-3.1-8B):

| Method | E_direct | E_complete | E_mean | E_min | L_min |
|---|---|---|---|---|---|
| simnpo | 96.4 | 64.4 | 81.2 | 53.1 | 32.7 |
| zerounlearn | 95.6 | 18.6 | 43.6 | 9.4 | 49.2 |
| idk_ft | 100.0 | 94.3 | 98.5 | 91.1 | 0.0 |
| rmu | 17.6 | 8.5 | 45.6 | 1.6 | 17.5 |
| tv | 97.6 | 87.0 | 95.6 | 88.5 | 0.0 |
| ga | 100.0 | 96.4 | 98.9 | 95.8 | 0.0 |
| gd | 100.0 | 98.0 | 99.1 | 96.0 | 0.0 |
| npo | 100.0 | 99.2 | 99.6 | 98.3 | 0.0 |
| rome | 87.2 | 13.0 | 36.3 | 0.0 | 24.8 |
| memit | 76.0 | 8.1 | 24.5 | 1.7 | 59.6 |

Reference cheats: on every model the six cheats score 1.0–3.4 against a gate of 35, each zeroing at least one axis (the subject blocklist and keyword router keep E 5–7 because subject-free, multi-hop and typo probes pass; refuse-all keeps L 1.0; the prompt cache E 1.2–1.3; the answer ban L 7.2–7.3; noop E 1.0). Every split passes the integrity checks (`RED_TEAM_REPORT.md` per seed: PASS).

What the tables show:

- **No method is close to solved, and the ranking depends on the model.** The best score on any model is 56.3 (ZeroUnlearn on Qwen3-4B); the same method scores 15.6 on Llama-3.1-8B. MEMIT scores 30.7 on Qwen3-4B and 1.7–3.2 on the Llama models. SimNPO has the best mean (37.6) with the largest seed variance (± 14–20). A single-model leaderboard would have ranked these methods wrongly, which is the case for the 10-model grid.
- **Every shortcut fails, on every model.** The gate requires a score below 35; the largest cheat score is 3.4.
- **Each method fails a different axis.** GA, GD, NPO and task-vector negation erase 79–100 and keep locality 1–3; the trained refusal (idk_ft) erases 89–98 and falsely refuses 37–55% of retain questions (generation scorer); ROME and MEMIT keep utility but erase 3–9 on held-out phrasings on Llama, and the fact returns under attack (R ≤ 1); RMU barely erases at the published setting (E_direct 12–32). SimNPO erases 64–78 at locality 39–51. ZeroUnlearn erases 68 at locality 47 on Qwen3-4B but 25 at 79 on the 8B model, and relearning restores 73–81% of its forgotten facts on the Llama models (13.6% on Qwen3-4B).
- **The legacy view hides all of this.** In the "legacy view" columns of the same files (only the families the official MCF / zsRE metrics use), the subject blocklist scores 100 / 100 (erasure / locality) and GA 98–100 / 2–3. LETHE-Bench scores them 1.2–2.4 and 4.9–6.8.

