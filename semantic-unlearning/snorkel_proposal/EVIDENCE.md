# Evidence: existing fact-unlearning benchmarks are saturated

All numbers below come from this repository's own experiments (`../RESULTS_L16_PAPER_FINAL.md`, commits `baa6252`, `0fe3ded`, `95e824a`). The setup was Llama-3.2-3B-Instruct, mean over 5 seeds, each benchmark's official evaluator, and 50 forget facts per seed. "SURE (published configuration)" is SURE, the detect-then-edit method we built, as published: a per-fact linear router on a layer-19 state, behind exact subject-token eligibility, plus one residual row per fact written at layer 16 in a second pass. "SURE (current)" is the current version (`outputs/onepass_l19_v1`, 2026-10-25): the router reads layer 19 with fuzzy subject matching and is trained on 20 quality-checked positives (base-model rewrites validated by answer transfer) and 20 checked negatives per fact, with its firing threshold calibrated on held-out rewrites (every third kept rewrite); the row is added at layer 19 in the same pass. "Genie" applies the same rows with perfect routing.

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
| **SURE (current)** | **100.0% / 99.8%** | **98.7% / 100.0%** | **99.3% / 84.0%** (98.8% on Spe†) | **95.8% / 97.0%** |

On MCF, SURE reaches Eff 0.001 / Gen 0.086 against genie's 0.001 / 0.002 (new classifier: Gen 0.027), with PPL exactly at base. The new classifier's Eff equals genie's on MCF, ZsRE and MQuAKE, and its Gen is 1.48 on ZsRE (genie 0.41) and 1.83 on MQuAKE (genie 0.53, published SURE 7.20); on RWKU its held-out Gen is 34.1, below even entity-genie's 40.8 (`outputs/onepass_l19_v1/paper_table.md`, jobs 1452463 / 1452464). On RWKU, 2–6 facts per seed that the router never fired on had their rows trained on oracle routing (`ORACLE_UNROUTED=1`). A benchmark on which one method sits at the ceiling no longer ranks methods. It only confirms that the method has stopped making mistakes on *that probe distribution*.

## 2. Why the ceiling is not the whole story

Each of the following findings, all from this repository, points to a gap that LETHE-Bench measures directly.

| Finding (repository evidence) | What it means | LETHE-Bench family / axis |
|---|---|---|
| **MCF base Eff is only 13.5%**, base Spe 19.9% | Most "forgotten" MCF facts were never known, so near-zero Eff is easy | Knowledge filter: every probe must be known by the base model (§4.1) |
| **RWKU held-out facts about a forgotten person: Gen 17.96 vs base 54.77**, and entity-genie 17.75 | Even perfect routing leaves a third of held-out entity knowledge; fact-level edits do not generalise to other facts about the same entity | Entity track; `indirect`, `multihop` |
| **LETHE-Bench `multihop`: 84 of 92 chain questions never routed, 96% residual (5 seeds)**, while the legacy MQuAKE evaluation (single-fact rewrites and AtomicGen questions) shows Gen 1.83 | The official MQuAKE metrics never ask the chain; a detector gated on the subject's name misses chains that do not name it | `multihop`, `indirect` |
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
| **SURE (early version, untuned)** | 68.5 / 100.0 | 3.3 / **100.0** | **10.5** |
| GA | 100.0 / 5.4 | 99.0 / 1.8 | 6.7 |
| NPO | 100.0 / 0.0 | 99.2 / 1.3 | 4.9 |

A string blocklist is nearly perfect by the legacy definition. On the full benchmark it collapses:

- it leaks under typos (100% residual), multi-hop chains (92%) and subject-free descriptions;
- it refuses same-subject, same-sentence, subject-mentioned and in-context questions (0% retained).

**SURE leads the pilot, with headroom.** It keeps perfect locality (100 on every retain family, no false fires) while most held-out phrasings leak: question 90% residual, cross-lingual 94%, typos 100%, multi-hop 100%. The routing decomposition attributes 48.7 of the forget residual to routing misses and 9.1 to actuation: the same "gap to genie is routing" result as `../README_ROUTER_DIAGNOSTICS.md`, now on families the legacy benchmarks do not contain.

All six reference cheats score 0 in this pilot file (below 4 with the current 1-point floor on R; LETHE_core 2.8–16.2), and every split-integrity check passes (`results/pilot/qwen2.5-0.5b-instruct/runs/RED_TEAM_REPORT.md`). Full tables for Llama-3.2-3B, Qwen3-4B and Llama-3.1-8B on the cluster are in §4; the job scripts are in `scripts/slurm/`.

## 4. LETHE-Bench on three models (cluster, 5 seeds each)

`results/cluster/leaderboard_dev/leaderboard.md` (all models) and `results/cluster/<model>/dev/leaderboard/leaderboard.md`; dev partition, seeds 1–5, 50 requests per seed, teacher-forced scorer (the generation scorer gives the same ordering), mean ± sd over seeds. 11 methods and 6 reference cheats (†) on Llama-3.2-3B-Instruct, Qwen3-4B and Llama-3.1-8B-Instruct. SURE is the method we built (classifier and Δe at layer 19, one pass; run directory `ours`); SURE_early is its ablation with both at layer 16. The Llama-3.2-3B numbers differ slightly from the earlier single-model file (SURE 30.1 → 28.4) because the metric now floors U and R at 1 point, budgets three indirect probes per request and reports E_complete; that file is kept as `results/cluster/llama32_3b.pre_lethe_20261028/`.

**LETHE score per model** (sorted by the mean over models):

| Method | Llama-3.2-3B | Qwen3-4B | Llama-3.1-8B | mean |
|---|---|---|---|---|
| simnpo | 37.4 ± 20.1 | 31.7 ± 15.6 | 43.7 ± 13.6 | 37.6 |
| zerounlearn | 21.3 ± 9.5 | 56.3 ± 5.8 | 15.6 ± 5.2 | 31.1 |
| SURE_early | 35.4 ± 22.7 | 18.5 ± 2.9 | 28.1 ± 16.9 | 27.3 |
| SURE | 28.4 ± 11.6 | 22.3 ± 0.7 | 18.7 ± 3.5 | 23.1 |
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

**Axes per model.** E = erasure over held-out forget families (harmonic), E_direct = the request prompt itself, E_complete = share of requests whose every held-out probe is erased, L = locality, U = utility, R = robustness under attack, Strict = weights only.

*Llama-3.2-3B-Instruct*

| Method | LETHE | E | E_direct | E_complete | L | U | R | Strict |
|---|---|---|---|---|---|---|---|---|
| simnpo | 37.4 | 63.5 | 96.4 | 59.5 | 50.8 | 81.1 | 26.1 | – |
| SURE_early | 35.4 | 25.1 | 95.2 | 56.2 | 91.6 | 100.0 | 34.1 | 2.9 |
| SURE | 28.4 | 22.2 | 96.2 | 59.8 | 95.5 | 100.0 | 19.0 | 2.9 |
| zerounlearn | 21.3 | 38.4 | 96.4 | 30.4 | 67.4 | 96.9 | 8.7 | – |
| idk_ft | 11.4 | 96.2 | 100.0 | 86.3 | 3.8 | 56.7 | 19.0 | – |
| npo | 6.2 | 92.3 | 100.0 | 87.2 | 1.7 | 74.8 | 80.4 | – |
| tv | 5.7 | 96.4 | 98.0 | 89.5 | 1.5 | 77.6 | 58.4 | – |
| gd | 5.5 | 95.1 | 100.0 | 88.9 | 1.5 | 66.5 | 86.6 | – |
| ga | 4.9 | 97.3 | 100.0 | 93.5 | 1.3 | 56.3 | 90.3 | – |
| rmu | 3.6 | 7.7 | 11.8 | 6.5 | 39.7 | 84.5 | 5.8 | – |
| rome | 3.0 | 8.6 | 82.8 | 15.3 | 57.2 | 98.7 | 0.8 | – |
| memit | 1.7 | 8.0 | 45.6 | 3.2 | 91.8 | 99.2 | 0.0 | – |

*Qwen3-4B*

| Method | LETHE | E | E_direct | E_complete | L | U | R | Strict |
|---|---|---|---|---|---|---|---|---|
| zerounlearn | 56.3 | 68.4 | 96.0 | 56.1 | 47.3 | 91.5 | 42.4 | – |
| simnpo | 31.7 | 76.5 | 95.6 | 59.3 | 38.9 | 85.7 | 21.0 | – |
| memit | 30.7 | 27.3 | 94.4 | 42.7 | 79.8 | 99.2 | 19.9 | – |
| SURE | 22.3 | 7.5 | 89.2 | 49.5 | 94.3 | 100.0 | 39.8 | 2.9 |
| SURE_early | 18.5 | 6.2 | 86.4 | 27.6 | 92.2 | 99.8 | 30.7 | 2.9 |
| idk_ft | 11.0 | 89.3 | 100.0 | 80.0 | 4.1 | 53.7 | 15.8 | – |
| tv | 10.2 | 83.2 | 97.2 | 70.0 | 3.0 | 87.5 | 35.5 | – |
| rmu | 7.5 | 21.4 | 32.4 | 15.1 | 33.3 | 81.5 | 12.2 | – |
| ga | 6.8 | 79.4 | 99.6 | 71.5 | 1.9 | 60.2 | 71.3 | – |
| gd | 6.3 | 90.1 | 100.0 | 85.0 | 1.7 | 55.5 | 79.7 | – |
| npo | 5.9 | 93.2 | 100.0 | 88.3 | 1.6 | 67.8 | 75.7 | – |
| rome | 4.5 | 94.5 | 99.2 | 89.0 | 1.2 | 29.8 | 69.5 | – |

*Llama-3.1-8B-Instruct*

| Method | LETHE | E | E_direct | E_complete | L | U | R | Strict |
|---|---|---|---|---|---|---|---|---|
| simnpo | 43.7 | 77.5 | 96.4 | 64.4 | 51.2 | 83.2 | 26.8 | – |
| SURE_early | 28.1 | 13.3 | 94.8 | 61.5 | 93.7 | 100.0 | 39.4 | 2.9 |
| SURE | 18.7 | 6.9 | 94.4 | 64.3 | 95.7 | 100.0 | 21.2 | 2.9 |
| zerounlearn | 15.6 | 24.5 | 95.6 | 18.6 | 79.1 | 98.2 | 5.6 | – |
| idk_ft | 10.9 | 98.3 | 100.0 | 94.3 | 3.8 | 63.5 | 13.6 | – |
| rmu | 7.8 | 8.8 | 17.6 | 8.5 | 21.5 | 56.0 | 20.7 | – |
| tv | 7.7 | 95.4 | 97.6 | 87.0 | 2.1 | 84.8 | 53.5 | – |
| ga | 5.6 | 98.9 | 100.0 | 96.4 | 1.5 | 63.8 | 84.9 | – |
| gd | 5.1 | 99.0 | 100.0 | 98.0 | 1.3 | 64.2 | 84.5 | – |
| npo | 4.6 | 99.6 | 100.0 | 99.2 | 1.2 | 57.1 | 87.8 | – |
| rome | 3.2 | 3.4 | 87.2 | 13.0 | 54.1 | 99.2 | 0.5 | – |
| memit | 3.2 | 6.7 | 76.0 | 8.1 | 87.1 | 99.5 | 0.0 | – |

Reference cheats: on every model the six cheats score 1.0–3.4 against a gate of 35, each zeroing at least one axis (the subject blocklist and keyword router keep E 5–7 because subject-free, multi-hop and typo probes pass; refuse-all keeps L 1.0; the prompt cache E 1.2–1.3; the answer ban L 7.2–7.3; noop E 1.0). Every split passes the integrity checks (`RED_TEAM_REPORT.md` per seed: PASS).

What the tables show:

- **No method is close to solved, and the ranking depends on the model.** The best score on any model is 56.3 (ZeroUnlearn on Qwen3-4B); the same method scores 15.6 on Llama-3.1-8B. MEMIT scores 30.7 on Qwen3-4B and 1.7–3.2 on the Llama models. SimNPO has the best mean (37.6) with the largest seed variance (± 14–20). A single-model leaderboard would have ranked these methods wrongly, which is the case for the 8-model grid.
- **Every shortcut fails, on every model.** The gate requires a score below 35; the largest cheat score is 3.4.
- **Each method fails a different axis.** GA, GD, NPO and task-vector negation erase 79–100 and keep locality 1–3; the trained refusal (idk_ft) erases 89–98 and falsely refuses 37–55% of retain questions (generation scorer); ROME and MEMIT keep utility but erase 3–9 on held-out phrasings on Llama, and the fact returns under attack (R ≤ 1); RMU barely erases at the published setting (E_direct 12–32). SimNPO erases 64–78 at locality 39–51. ZeroUnlearn erases 68 at locality 47 on Qwen3-4B but 25 at 79 on the 8B model, and relearning restores 73–81% of its forgotten facts on the Llama models (13.6% on Qwen3-4B).
- **Our own method is not the winner, and the tables say why.** SURE keeps locality 94–96 and utility 100 on every model, erases 89–96 of the request prompts, and has the lowest seed variance of any method on Qwen3-4B (± 0.7). Its E is 7–22 because two families are never routed: `multihop` (94–96% residual) and `indirect` (100%), where the prompt does not name the subject and a router gated on the subject cannot fire; the harmonic mean over families is set by them. The routing decomposition attributes 8–14 points of residual to routing misses (route recall 86–92%) and 1–3 to actuation, with 8.5–14% false fires on retain probes (visible as `in_context` 82–88 and `same_subject` 83–98). It scores 2.9 in the weights-only tier because its router and rows are runtime components. SURE_early (classifier and Δe both at layer 16) resists the logit-lens attack better (R 31–39 against 19–40) at lower route recall (69% on Qwen3-4B).
- **The legacy view hides all of this.** In the "legacy view" columns of the same files (only the families the official MCF / zsRE metrics use), SURE scores 85–94 / 100 (erasure / locality), the subject blocklist 100 / 100 and GA 98–100 / 2–3. LETHE-Bench scores them 19–28, 1.2–2.4 and 4.9–6.8.

