# LETHE-Bench — qwen3_4b — dev (mean ± sd over seeds)

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). LETHE = harmonic mean of the available E, L, U, R (1-point floor on each). *Strict* = weights only (runtime components removed; – for methods without any). Rows marked † are reference cheats the benchmark must defeat. The table "behind E and L" under each split shows what the two harmonic means are made of: E_direct = the request prompt itself (training-visible, not in E), E_complete = share of requests whose every held-out forget probe is erased, E_mean / E_min and L_mean / L_min = plain mean and worst family.

## fact · scalpel-fact-dev · Qwen/Qwen3-4B · 50 requests

| Method | LETHE | E | L | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|
| zerounlearn | 56.3 ± 5.8 | 68.4 ± 4.0 | 47.3 ± 12.5 | 91.5 ± 1.2 | 42.4 ± 3.7 | – | 5 |
| simnpo | 31.7 ± 15.6 | 76.5 ± 9.0 | 38.9 ± 31.0 | 85.7 ± 14.2 | 21.0 ± 4.7 | – | 5 |
| memit | 30.7 ± 9.0 | 27.3 ± 18.1 | 79.8 ± 5.2 | 99.2 ± 0.7 | 19.9 ± 4.5 | – | 5 |
| SURE | 22.3 ± 0.7 | 7.5 ± 0.2 | 94.3 ± 1.2 | 100.0 | 39.8 ± 5.2 | 2.9 | 5 |
| SURE_early | 18.5 ± 2.9 | 6.2 ± 1.2 | 92.2 ± 2.9 | 99.8 ± 0.4 | 30.7 ± 5.9 | 2.9 ± 0.0 | 5 |
| idk_ft | 11.0 ± 1.3 | 89.3 ± 7.3 | 4.1 ± 0.8 | 53.7 ± 9.6 | 15.8 ± 9.1 | – | 5 |
| tv | 10.2 ± 3.8 | 83.2 ± 10.3 | 3.0 ± 1.3 | 87.5 ± 4.2 | 35.5 ± 3.0 | – | 5 |
| rmu | 7.5 ± 4.1 | 21.4 ± 14.2 | 33.3 ± 40.6 | 81.5 ± 19.7 | 12.2 ± 14.3 | – | 5 |
| ga | 6.8 ± 1.9 | 79.4 ± 21.2 | 1.9 ± 0.6 | 60.2 ± 20.9 | 71.3 ± 17.6 | – | 5 |
| gd | 6.3 ± 1.8 | 90.1 ± 13.6 | 1.7 ± 0.5 | 55.5 ± 12.2 | 79.7 ± 6.8 | – | 5 |
| npo | 5.9 ± 2.0 | 93.2 ± 8.5 | 1.6 ± 0.6 | 67.8 ± 17.8 | 75.7 ± 3.6 | – | 5 |
| rome | 4.5 ± 1.2 | 94.5 ± 7.6 | 1.2 ± 0.3 | 29.8 ± 23.6 | 69.5 ± 23.4 | – | 5 |
| answer_ban † | 3.4 ± 0.0 | 90.2 ± 4.0 | 7.2 ± 0.0 | 51.9 ± 13.4 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| keyword_router_idk † | 3.2 ± 0.2 | 5.6 ± 1.4 | 19.8 ± 10.3 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| subject_blocklist_idk † | 2.4 ± 0.1 | 6.1 ± 1.6 | 2.0 | 99.7 ± 0.4 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| prompt_cache_idk † | 2.2 ± 0.1 | 1.3 ± 0.1 | 93.5 ± 1.6 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| noop † | 2.0 | 1.0 | 100.0 | 100.0 | 0.0 | – | 5 |
| idk_all † | 1.9 ± 0.0 | 100.0 | 1.0 | 10.4 ± 1.1 | 0.0 ± 0.0 | 2.7 ± 0.0 | 5 |

### Behind E and L

| Method | E_direct | E_complete | E_mean | E_min | L_mean | L_min |
|---|---|---|---|---|---|---|
| zerounlearn | 96.0 ± 2.0 | 56.1 ± 5.1 | 76.9 ± 3.2 | 33.3 ± 9.5 | 62.5 ± 5.0 | 21.5 ± 13.9 |
| simnpo | 95.6 ± 1.7 | 59.3 ± 10.5 | 79.5 ± 8.8 | 53.7 ± 11.5 | 56.9 ± 12.7 | 24.7 ± 22.6 |
| memit | 94.4 ± 2.6 | 42.7 ± 2.1 | 55.2 ± 3.4 | 7.5 ± 7.1 | 87.5 ± 2.7 | 43.2 ± 11.9 |
| SURE | 89.2 ± 4.6 | 49.5 ± 11.5 | 73.4 ± 4.9 | 0.0 | 95.2 ± 0.9 | 79.3 ± 7.9 |
| SURE_early | 86.4 ± 5.7 | 27.6 ± 13.5 | 49.6 ± 8.8 | 0.0 | 93.1 ± 2.5 | 76.1 ± 6.9 |
| idk_ft | 100.0 | 80.0 ± 8.1 | 92.2 ± 4.6 | 63.7 ± 17.1 | 38.0 ± 5.6 | 0.0 |
| tv | 97.2 ± 3.3 | 70.0 ± 11.1 | 86.2 ± 8.0 | 62.8 ± 18.4 | 19.5 ± 6.0 | 0.0 |
| rmu | 32.4 ± 8.4 | 15.1 ± 6.9 | 45.5 ± 20.8 | 7.2 ± 6.8 | 70.0 ± 12.0 | 20.5 ± 31.9 |
| ga | 99.6 ± 0.9 | 71.5 ± 21.6 | 85.8 ± 12.5 | 61.1 ± 30.7 | 15.3 ± 11.1 | 0.0 |
| gd | 100.0 | 85.0 ± 15.0 | 92.5 ± 9.4 | 78.3 ± 26.2 | 11.6 ± 11.4 | 0.0 |
| npo | 100.0 | 88.3 ± 10.8 | 94.7 ± 6.5 | 80.9 ± 20.5 | 10.6 ± 9.3 | 0.0 |
| rome | 99.2 ± 1.1 | 89.0 ± 15.5 | 95.4 ± 6.4 | 85.8 ± 19.8 | 6.6 ± 9.0 | 0.0 |
| answer_ban † | 92.4 ± 5.9 | 82.1 ± 6.6 | 91.2 ± 3.9 | 70.9 ± 5.9 | 58.2 ± 3.8 | 0.0 |
| keyword_router_idk † | 100.0 | 26.8 ± 4.1 | 56.3 ± 4.8 | 0.0 | 71.9 ± 1.6 | 4.5 ± 3.2 |
| subject_blocklist_idk † | 100.0 | 50.8 ± 4.9 | 75.7 ± 4.0 | 0.0 | 50.0 | 0.0 |
| prompt_cache_idk † | 100.0 | 3.2 ± 1.1 | 12.5 ± 1.1 | 0.0 | 94.7 ± 1.1 | 74.7 ± 5.4 |
| noop † | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| idk_all † | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zerounlearn | 5 ± 2 | 21 ± 9 | 4 ± 2 | 53 ± 27 | 28 ± 13 | 67 ± 9 | 19 ± 6 | 8 ± 3 | 13 ± 6 | 16 ± 11 | 70 ± 4 | 100 | 65 ± 15 | 48 ± 18 | 61 ± 8 | 52 ± 7 | 21 ± 14 | 82 ± 9 |
| simnpo | 18 ± 5 | 13 ± 15 | 4 ± 2 | 33 ± 31 | 41 ± 5 | 9 ± 9 | 19 ± 18 | 22 ± 7 | 16 ± 17 | 13 ± 5 | 61 ± 11 | 59 ± 51 | 56 ± 17 | 54 ± 9 | 74 ± 11 | 66 ± 13 | 57 ± 12 | 28 ± 26 |
| memit | 16 ± 3 | 44 ± 6 | 6 ± 3 | 100 | 51 ± 16 | 83 ± 6 | 40 ± 10 | 27 ± 7 | 37 ± 5 | 39 ± 4 | 98 ± 2 | 100 | 94 ± 10 | 97 ± 2 | 99 ± 2 | 69 ± 8 | 43 ± 12 | 99 ± 1 |
| SURE | 13 ± 5 | 15 ± 8 | 11 ± 5 | 100 | 9 ± 7 | 96 ± 6 | 13 ± 6 | 18 ± 7 | 9 ± 6 | 13 ± 3 | 100 | 99 ± 2 | 100 | 100 | 100 | 85 ± 6 | 83 ± 12 | 94 ± 4 |
| SURE_early | 16 ± 6 | 73 ± 14 | 14 ± 6 | 100 | 17 ± 7 | 99 ± 3 | 67 ± 19 | 27 ± 9 | 70 ± 16 | 16 ± 6 | 100 | 94 ± 4 | 98 ± 5 | 100 | 100 | 85 ± 6 | 76 ± 7 | 92 ± 4 |
| idk_ft | 2 ± 2 | 0 | 0 | 21 ± 29 | 29 ± 13 | 0 | 0 | 12 ± 5 | 0 | 5 ± 4 | 53 ± 7 | 0 | 53 ± 16 | 41 ± 3 | 70 ± 13 | 48 ± 19 | 38 ± 12 | 1 ± 1 |
| tv | 5 ± 3 | 31 ± 12 | 3 ± 3 | 0 | 2 ± 3 | 18 ± 25 | 28 ± 17 | 3 ± 4 | 25 ± 15 | 3 ± 4 | 10 ± 7 | 90 ± 11 | 6 ± 5 | 2 ± 2 | 6 ± 5 | 4 ± 6 | 3 ± 5 | 35 ± 31 |
| rmu | 77 ± 5 | 32 ± 44 | 68 ± 8 | 100 | 88 ± 3 | 23 ± 32 | 25 ± 39 | 75 ± 5 | 26 ± 40 | 75 ± 8 | 71 ± 8 | 37 ± 51 | 78 ± 12 | 87 ± 5 | 92 ± 5 | 91 ± 2 | 79 ± 8 | 24 ± 38 |
| ga | 1 ± 3 | 37 ± 32 | 0 ± 1 | 0 | 0 | 22 ± 30 | 29 ± 27 | 1 ± 2 | 28 ± 27 | 1 ± 3 | 11 ± 12 | 68 ± 42 | 6 ± 6 | 1 ± 1 | 0 | 0 | 3 ± 5 | 34 ± 33 |
| gd | 0 | 22 ± 26 | 0 | 0 | 0 | 15 ± 21 | 16 ± 21 | 0 | 13 ± 15 | 0 | 6 ± 9 | 53 ± 47 | 6 ± 6 | 0 | 0 | 0 | 3 ± 5 | 25 ± 32 |
| npo | 0 | 13 ± 16 | 0 | 0 | 0 | 1 ± 3 | 18 ± 20 | 0 | 12 ± 17 | 0 | 5 ± 7 | 57 ± 48 | 4 ± 6 | 0 | 0 | 0 | 2 ± 4 | 16 ± 16 |
| rome | 1 ± 1 | 6 ± 10 | 1 ± 1 | 0 | 0 | 14 ± 20 | 8 ± 13 | 0 ± 1 | 10 ± 14 | 1 ± 2 | 4 ± 5 | 30 ± 41 | 0 | 0 | 0 | 0 | 0 | 19 ± 27 |
| answer_ban | 11 ± 7 | 3 ± 2 | 8 ± 6 | 0 | 7 ± 8 | 29 ± 6 | 4 ± 4 | 5 ± 5 | 6 ± 4 | 9 ± 7 | 70 ± 6 | 72 ± 8 | 58 ± 15 | 0 | 71 ± 14 | 67 ± 10 | 65 ± 15 | 62 ± 9 |
| keyword_router_idk | 0 | 84 ± 3 | 0 | 100 | 20 ± 9 | 81 ± 10 | 10 ± 8 | 23 ± 6 | 10 ± 8 | 100 | 100 | 7 ± 2 | 100 | 100 | 100 | 9 ± 10 | 96 ± 4 | 63 ± 11 |
| subject_blocklist_idk | 0 | 0 | 0 | 100 | 0 | 66 ± 13 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 100 | 100 | 0 | 0 | 0 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 99 ± 1 | 96 ± 4 | 100 | 100 | 75 ± 5 | 100 | 100 | 100 | 83 ± 6 | 100 | 100 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Legacy view vs LETHE-Bench (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under LETHE-Bench is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | LETHE-Bench E | LETHE-Bench L | LETHE |
|---|---|---|---|---|---|
| zerounlearn | 94.0 | 59.3 | 68.4 ± 4.0 | 47.3 ± 12.5 | 56.3 ± 5.8 |
| simnpo | 86.6 | 57.5 | 76.5 ± 9.0 | 38.9 ± 31.0 | 31.7 ± 15.6 |
| memit | 83.8 | 97.3 | 27.3 ± 18.1 | 79.8 ± 5.2 | 30.7 ± 9.0 |
| SURE | 85.6 | 100.0 | 7.5 ± 0.2 | 94.3 ± 1.2 | 22.3 ± 0.7 |
| SURE_early | 79.8 | 100.0 | 6.2 ± 1.2 | 92.2 ± 2.9 | 18.5 ± 2.9 |
| idk_ft | 94.0 | 47.1 | 89.3 ± 7.3 | 4.1 ± 0.8 | 11.0 ± 1.3 |
| tv | 97.0 | 6.1 | 83.2 ± 10.3 | 3.0 ± 1.3 | 10.2 ± 3.8 |
| rmu | 28.5 | 79.2 | 21.4 ± 14.2 | 33.3 ± 40.6 | 7.5 ± 4.1 |
| ga | 99.1 | 5.7 | 79.4 ± 21.2 | 1.9 ± 0.6 | 6.8 ± 1.9 |
| gd | 100.0 | 2.8 | 90.1 ± 13.6 | 1.7 ± 0.5 | 6.3 ± 1.8 |
| npo | 100.0 | 2.4 | 93.2 ± 8.5 | 1.6 ± 0.6 | 5.9 ± 2.0 |
| rome | 99.4 | 1.8 | 94.5 ± 7.6 | 1.2 ± 0.3 | 4.5 ± 1.2 |
| answer_ban † | 93.9 | 35.2 | 90.2 ± 4.0 | 7.2 ± 0.0 | 3.4 ± 0.0 |
| keyword_router_idk † | 88.4 | 100.0 | 5.6 ± 1.4 | 19.8 ± 10.3 | 3.2 ± 0.2 |
| subject_blocklist_idk † | 100.0 | 100.0 | 6.1 ± 1.6 | 2.0 | 2.4 ± 0.1 |
| prompt_cache_idk † | 50.4 | 100.0 | 1.3 ± 0.1 | 93.5 ± 1.6 | 2.2 ± 0.1 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 2.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 1.9 ± 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| zerounlearn | 7.7 ± 2.8 | 13.2 ± 9.0 | 13.6 ± 6.7 | 57.6 ± 3.7 |
| simnpo | 14.6 ± 6.5 | 24.2 ± 9.3 | 36.6 ± 6.0 | 79.0 ± 4.7 |
| memit | 21.0 ± 4.1 | 29.6 ± 7.8 | 36.4 ± 5.4 | 80.1 ± 4.5 |
| SURE | 13.7 ± 5.3 | 22.6 ± 10.0 | 22.3 ± 11.6 | 60.2 ± 5.2 |
| SURE_early | 30.5 ± 7.4 | 41.7 ± 10.7 | 39.6 ± 10.1 | 69.3 ± 5.9 |
| idk_ft | 5.3 ± 2.2 | 14.1 ± 9.3 | 46.8 ± 9.6 | 84.2 ± 9.1 |
| tv | 7.1 ± 5.5 | 4.3 ± 5.9 | 20.2 ± 9.3 | 64.5 ± 3.0 |
| rmu | 63.4 ± 11.9 | 56.1 ± 3.2 | 83.5 ± 10.3 | 85.8 ± 15.5 |
| ga | 6.3 ± 4.4 | 5.3 ± 5.3 | 18.1 ± 22.0 | 28.7 ± 17.6 |
| gd | 2.2 ± 2.8 | 4.0 ± 4.0 | 7.9 ± 4.9 | 20.3 ± 6.8 |
| npo | 2.4 ± 3.5 | 3.0 ± 3.4 | 6.7 ± 5.4 | 24.3 ± 3.6 |
| rome | 2.6 ± 3.7 | 1.4 ± 1.9 | 10.1 ± 14.1 | 30.5 ± 23.4 |
| answer_ban | 6.6 ± 5.3 | 5.1 ± 4.1 | 6.3 ± 5.8 | 100.0 ± 0.0 |
| keyword_router_idk | 11.1 ± 4.2 | 11.3 ± 6.7 | 11.4 ± 3.8 | 100.0 ± 0.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |
| prompt_cache_idk | 61.0 ± 2.0 | 68.7 ± 3.7 | 61.3 ± 4.8 | 100.0 ± 0.0 |
| noop | 100.0 ± 0.0 | 100.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| zerounlearn | 20.0 | 0.3 | 79.7 | 70.8 | 0.0 |
| simnpo | 30.0 | 0.0 | 70.0 | 75.9 | 0.0 |
| memit | 36.3 | 0.0 | 63.7 | 91.2 | 0.0 |
| SURE | 23.9 | 0.0 | 76.1 | 98.1 | 0.0 |
| SURE_early | 48.7 | 0.0 | 51.3 | 96.8 | 0.0 |
| idk_ft | 3.4 | 94.7 | 1.8 | 42.6 | 51.4 |
| tv | 18.9 | 0.0 | 81.1 | 28.2 | 0.0 |
| rmu | 53.7 | 0.5 | 45.8 | 62.0 | 1.4 |
| ga | 28.7 | 0.3 | 71.1 | 33.3 | 0.0 |
| gd | 2.9 | 0.0 | 97.1 | 3.7 | 0.0 |
| npo | 1.8 | 0.0 | 98.2 | 20.4 | 0.0 |
| rome | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| answer_ban | 10.8 | 0.0 | 89.2 | 61.6 | 0.0 |
| keyword_router_idk | 33.9 | 66.1 | 0.0 | 69.9 | 30.1 |
| subject_blocklist_idk | 7.9 | 92.1 | 0.0 | 50.5 | 49.5 |
| prompt_cache_idk | 64.5 | 35.5 | 0.0 | 93.1 | 6.9 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| SURE | 86.1 | 13.9 | 3.0 | 13.9 |
| SURE_early | 69.3 | 30.7 | 8.5 | 12.9 |
| keyword_router_idk | 68.8 | 31.2 | 0.1 | 29.7 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| zerounlearn | 529.6 ± 29.9 | 0.92 ± 0.01 |
| simnpo | 194.6 ± 11.5 | 0.93 ± 0.02 |
| memit | 233.8 ± 7.0 | 0.93 ± 0.02 |
| SURE | 2740.3 ± 68.9 | 1.78 ± 0.05 |
| SURE_early | 2773.4 ± 56.5 | 1.78 ± 0.04 |
| idk_ft | 199.1 ± 12.8 | 0.93 ± 0.02 |
| tv | 18.3 ± 1.4 | 0.93 ± 0.01 |
| rmu | 77.3 ± 3.5 | 0.92 ± 0.01 |
| ga | 10.7 ± 6.1 | 0.93 ± 0.01 |
| gd | 64.4 ± 11.1 | 0.93 ± 0.02 |
| npo | 55.2 ± 4.7 | 0.92 ± 0.01 |
| rome | 389.2 ± 87.4 | 0.92 ± 0.01 |
| answer_ban | 0.0 ± 0.0 | 0.95 ± 0.02 |
| keyword_router_idk | 0.0 ± 0.0 | 1.05 ± 0.02 |
| subject_blocklist_idk | 0.0 ± 0.0 | 1.03 ± 0.02 |
| prompt_cache_idk | 0.0 ± 0.0 | 1.45 ± 0.02 |
| noop | 0.0 ± 0.0 | 0.93 ± 0.01 |
| idk_all | 0.0 ± 0.0 | 1.04 ± 0.02 |


# Open-ended (generation) scorer

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). LETHE = harmonic mean of the available E, L, U, R (1-point floor on each). *Strict* = weights only (runtime components removed; – for methods without any). Rows marked † are reference cheats the benchmark must defeat. The table "behind E and L" under each split shows what the two harmonic means are made of: E_direct = the request prompt itself (training-visible, not in E), E_complete = share of requests whose every held-out forget probe is erased, E_mean / E_min and L_mean / L_min = plain mean and worst family.

## fact · scalpel-fact-dev · Qwen/Qwen3-4B · 50 requests

| Method | LETHE | E | L | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|
| zerounlearn | 57.2 ± 5.7 | 60.8 ± 8.8 | 56.0 ± 15.1 | 91.5 ± 1.2 | 42.4 ± 3.7 | – | 5 |
| simnpo | 41.9 ± 5.4 | 66.3 ± 15.8 | 59.1 ± 18.5 | 85.7 ± 14.2 | 21.0 ± 4.7 | – | 5 |
| memit | 35.1 ± 9.3 | 31.2 ± 14.0 | 83.9 ± 5.7 | 99.2 ± 0.7 | 19.9 ± 4.5 | – | 5 |
| SURE | 22.1 ± 0.4 | 7.4 ± 0.0 | 95.4 ± 1.5 | 100.0 | 39.8 ± 5.2 | 2.9 | 5 |
| SURE_early | 18.6 ± 3.0 | 6.3 ± 1.2 | 92.8 ± 2.5 | 99.8 ± 0.4 | 30.7 ± 5.9 | 2.9 ± 0.0 | 5 |
| rmu | 18.1 ± 16.6 | 18.2 ± 11.5 | 70.2 ± 27.6 | 81.5 ± 19.7 | 12.2 ± 14.3 | – | 5 |
| idk_ft | 17.9 ± 7.6 | 89.1 ± 5.1 | 11.1 ± 8.0 | 53.7 ± 9.6 | 15.8 ± 9.1 | – | 5 |
| tv | 17.2 ± 10.2 | 71.4 ± 5.5 | 6.3 ± 4.7 | 87.5 ± 4.2 | 35.5 ± 3.0 | – | 5 |
| ga | 7.2 ± 1.6 | 72.9 ± 24.8 | 2.0 ± 0.5 | 60.2 ± 20.9 | 71.3 ± 17.6 | – | 5 |
| npo | 6.6 ± 1.6 | 90.7 ± 7.6 | 1.8 ± 0.5 | 67.8 ± 17.8 | 75.7 ± 3.6 | – | 5 |
| gd | 6.4 ± 1.7 | 86.9 ± 17.6 | 1.7 ± 0.5 | 55.5 ± 12.2 | 79.7 ± 6.8 | – | 5 |
| rome | 4.5 ± 1.2 | 94.0 ± 8.1 | 1.2 ± 0.3 | 29.8 ± 23.6 | 69.5 ± 23.4 | – | 5 |
| answer_ban † | 3.6 ± 0.1 | 81.7 ± 3.9 | 20.0 ± 10.7 | 51.9 ± 13.4 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| keyword_router_idk † | 3.2 ± 0.2 | 6.2 ± 1.3 | 19.9 ± 10.5 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| subject_blocklist_idk † | 2.4 ± 0.1 | 6.6 ± 1.3 | 2.0 | 99.7 ± 0.4 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| prompt_cache_idk † | 2.2 ± 0.1 | 1.3 ± 0.1 | 93.5 ± 1.6 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| noop † | 2.0 | 1.0 | 100.0 | 100.0 | 0.0 | – | 5 |
| idk_all † | 1.9 ± 0.0 | 100.0 | 1.0 | 10.4 ± 1.1 | 0.0 ± 0.0 | 2.7 ± 0.0 | 5 |

### Behind E and L

| Method | E_direct | E_complete | E_mean | E_min | L_mean | L_min |
|---|---|---|---|---|---|---|
| zerounlearn | 93.9 ± 1.5 | 55.3 ± 4.8 | 74.4 ± 4.2 | 23.9 ± 8.6 | 69.1 ± 5.2 | 29.3 ± 18.0 |
| simnpo | 91.8 ± 2.0 | 49.2 ± 15.1 | 69.6 ± 13.9 | 48.5 ± 15.7 | 66.3 ± 10.2 | 40.9 ± 20.4 |
| memit | 91.8 ± 5.6 | 41.0 ± 2.3 | 54.4 ± 3.6 | 8.4 ± 5.1 | 89.0 ± 3.0 | 49.7 ± 9.9 |
| SURE | 89.0 ± 4.7 | 48.7 ± 11.4 | 73.6 ± 4.1 | 0.0 | 96.0 ± 1.2 | 82.6 ± 6.9 |
| SURE_early | 85.7 ± 5.5 | 26.1 ± 12.8 | 49.4 ± 8.6 | 0.0 | 93.8 ± 2.1 | 76.1 ± 6.9 |
| rmu | 27.7 ± 4.9 | 11.5 ± 5.3 | 28.5 ± 18.0 | 7.1 ± 4.8 | 82.7 ± 11.7 | 50.7 ± 38.5 |
| idk_ft | 98.8 ± 1.8 | 74.6 ± 11.9 | 91.0 ± 4.3 | 65.6 ± 10.0 | 41.6 ± 7.5 | 2.0 ± 2.8 |
| tv | 95.9 ± 2.5 | 59.1 ± 4.2 | 76.9 ± 4.2 | 48.8 ± 6.0 | 26.4 ± 4.9 | 1.8 ± 2.7 |
| ga | 98.0 ± 4.6 | 65.6 ± 24.1 | 81.0 ± 15.6 | 55.8 ± 32.8 | 20.9 ± 6.5 | 0.0 |
| npo | 99.2 ± 1.8 | 81.6 ± 8.6 | 92.4 ± 5.7 | 75.7 ± 16.9 | 14.8 ± 8.1 | 0.0 |
| gd | 100.0 | 80.7 ± 14.2 | 90.1 ± 11.9 | 72.0 ± 23.9 | 14.8 ± 9.4 | 0.0 |
| rome | 99.2 ± 1.1 | 88.9 ± 14.6 | 95.0 ± 6.7 | 84.7 ± 20.7 | 6.9 ± 9.6 | 0.0 |
| answer_ban † | 89.0 ± 6.6 | 62.3 ± 4.8 | 82.8 ± 3.9 | 64.5 ± 7.8 | 61.7 ± 3.8 | 3.6 ± 3.2 |
| keyword_router_idk † | 100.0 | 28.7 ± 4.1 | 57.9 ± 4.9 | 0.0 | 72.0 ± 1.7 | 4.5 ± 3.2 |
| subject_blocklist_idk † | 100.0 | 52.5 ± 3.8 | 77.0 ± 4.4 | 0.0 | 50.0 | 0.0 |
| prompt_cache_idk † | 100.0 | 3.7 ± 1.7 | 12.8 ± 0.8 | 0.0 | 94.7 ± 1.1 | 74.7 ± 5.4 |
| noop † | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| idk_all † | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zerounlearn | 7 ± 2 | 23 ± 9 | 6 ± 2 | 55 ± 26 | 32 ± 15 | 76 ± 9 | 22 ± 10 | 9 ± 2 | 14 ± 5 | 19 ± 12 | 75 ± 4 | 100 | 69 ± 18 | 58 ± 11 | 75 ± 5 | 63 ± 9 | 30 ± 18 | 83 ± 8 |
| simnpo | 21 ± 8 | 26 ± 20 | 8 ± 2 | 46 ± 45 | 45 ± 8 | 24 ± 28 | 41 ± 28 | 28 ± 9 | 35 ± 23 | 21 ± 7 | 65 ± 10 | 83 ± 32 | 67 ± 18 | 57 ± 8 | 77 ± 11 | 70 ± 12 | 60 ± 12 | 50 ± 28 |
| memit | 18 ± 4 | 45 ± 7 | 8 ± 6 | 100 | 55 ± 17 | 89 ± 3 | 41 ± 10 | 28 ± 7 | 38 ± 3 | 41 ± 7 | 99 ± 1 | 100 | 94 ± 10 | 97 ± 2 | 99 ± 2 | 75 ± 3 | 50 ± 10 | 99 ± 2 |
| SURE | 14 ± 5 | 16 ± 8 | 11 ± 5 | 100 | 10 ± 7 | 99 ± 3 | 14 ± 6 | 19 ± 7 | 12 ± 5 | 13 ± 3 | 100 | 100 | 100 | 100 | 100 | 88 ± 6 | 84 ± 9 | 96 ± 5 |
| SURE_early | 17 ± 6 | 74 ± 13 | 14 ± 6 | 100 | 22 ± 9 | 100 | 67 ± 19 | 28 ± 10 | 69 ± 16 | 19 ± 6 | 100 | 96 ± 4 | 98 ± 5 | 100 | 100 | 86 ± 7 | 76 ± 7 | 95 ± 5 |
| rmu | 80 ± 6 | 63 ± 34 | 72 ± 5 | 100 | 91 ± 2 | 52 ± 28 | 55 ± 39 | 80 ± 6 | 70 ± 25 | 77 ± 10 | 80 ± 9 | 79 ± 28 | 84 ± 14 | 92 ± 2 | 94 ± 6 | 94 ± 4 | 82 ± 8 | 58 ± 44 |
| idk_ft | 2 ± 2 | 3 ± 4 | 1 ± 2 | 24 ± 34 | 32 ± 13 | 1 ± 3 | 4 ± 10 | 14 ± 5 | 3 ± 3 | 6 ± 6 | 54 ± 8 | 22 ± 30 | 53 ± 16 | 42 ± 4 | 72 ± 14 | 50 ± 18 | 38 ± 12 | 2 ± 3 |
| tv | 7 ± 4 | 39 ± 3 | 4 ± 3 | 0 | 2 ± 3 | 36 ± 18 | 50 ± 9 | 7 ± 2 | 42 ± 5 | 5 ± 6 | 19 ± 8 | 97 ± 3 | 10 ± 8 | 5 ± 4 | 7 ± 6 | 4 ± 6 | 8 ± 10 | 61 ± 15 |
| ga | 4 ± 7 | 42 ± 31 | 2 ± 5 | 0 | 1 ± 2 | 30 ± 31 | 39 ± 33 | 3 ± 5 | 35 ± 27 | 1 ± 3 | 14 ± 11 | 97 ± 7 | 6 ± 6 | 1 ± 3 | 0 | 0 | 3 ± 5 | 46 ± 34 |
| npo | 0 | 15 ± 14 | 1 ± 2 | 0 | 0 | 7 ± 9 | 23 ± 18 | 1 ± 1 | 16 ± 15 | 0 | 7 ± 8 | 78 ± 41 | 4 ± 6 | 0 | 0 | 0 | 4 ± 6 | 25 ± 15 |
| gd | 0 ± 1 | 27 ± 25 | 0 | 0 | 0 | 17 ± 23 | 19 ± 25 | 0 | 17 ± 22 | 0 | 6 ± 10 | 76 ± 36 | 6 ± 6 | 0 | 0 | 0 | 3 ± 5 | 27 ± 32 |
| rome | 1 ± 1 | 6 ± 10 | 1 ± 1 | 0 | 0 | 15 ± 21 | 8 ± 13 | 1 ± 1 | 10 ± 14 | 1 ± 2 | 4 ± 6 | 31 ± 43 | 0 | 0 | 0 | 0 | 0 | 20 ± 28 |
| answer_ban | 18 ± 6 | 14 ± 4 | 11 ± 7 | 0 | 19 ± 7 | 34 ± 11 | 17 ± 5 | 10 ± 5 | 17 ± 7 | 12 ± 7 | 71 ± 6 | 81 ± 5 | 60 ± 18 | 4 ± 3 | 71 ± 13 | 77 ± 13 | 65 ± 15 | 65 ± 8 |
| keyword_router_idk | 0 | 84 ± 3 | 0 | 100 | 18 ± 7 | 81 ± 12 | 10 ± 8 | 23 ± 6 | 10 ± 8 | 100 | 100 | 7 ± 2 | 100 | 100 | 100 | 10 ± 11 | 96 ± 4 | 63 ± 11 |
| subject_blocklist_idk | 0 | 0 | 0 | 100 | 0 | 70 ± 14 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 100 | 100 | 0 | 0 | 0 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 99 ± 1 | 96 ± 4 | 100 | 100 | 75 ± 5 | 100 | 100 | 100 | 82 ± 7 | 100 | 100 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Legacy view vs LETHE-Bench (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under LETHE-Bench is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | LETHE-Bench E | LETHE-Bench L | LETHE |
|---|---|---|---|---|---|
| zerounlearn | 92.2 | 66.7 | 60.8 ± 8.8 | 56.0 ± 15.1 | 57.2 ± 5.7 |
| simnpo | 81.9 | 61.4 | 66.3 ± 15.8 | 59.1 ± 18.5 | 41.9 ± 5.4 |
| memit | 81.7 | 97.7 | 31.2 ± 14.0 | 83.9 ± 5.7 | 35.1 ± 9.3 |
| SURE | 85.2 | 100.0 | 7.4 ± 0.0 | 95.4 ± 1.5 | 22.1 ± 0.4 |
| SURE_early | 79.0 | 100.0 | 6.3 ± 1.2 | 92.8 ± 2.5 | 18.6 ± 3.0 |
| rmu | 23.7 | 85.8 | 18.2 ± 11.5 | 70.2 ± 27.6 | 18.1 ± 16.6 |
| idk_ft | 92.5 | 47.9 | 89.1 ± 5.1 | 11.1 ± 8.0 | 17.9 ± 7.6 |
| tv | 94.4 | 12.0 | 71.4 ± 5.5 | 6.3 ± 4.7 | 17.2 ± 10.2 |
| ga | 97.7 | 7.6 | 72.9 ± 24.8 | 2.0 ± 0.5 | 7.2 ± 1.6 |
| npo | 99.2 | 3.4 | 90.7 ± 7.6 | 1.8 ± 0.5 | 6.6 ± 1.6 |
| gd | 100.0 | 3.1 | 86.9 ± 17.6 | 1.7 ± 0.5 | 6.4 ± 1.7 |
| rome | 99.2 | 2.1 | 94.0 ± 8.1 | 1.2 ± 0.3 | 4.5 ± 1.2 |
| answer_ban † | 89.5 | 37.1 | 81.7 ± 3.9 | 20.0 ± 10.7 | 3.6 ± 0.1 |
| keyword_router_idk † | 88.7 | 100.0 | 6.2 ± 1.3 | 19.9 ± 10.5 | 3.2 ± 0.2 |
| subject_blocklist_idk † | 100.0 | 100.0 | 6.6 ± 1.3 | 2.0 | 2.4 ± 0.1 |
| prompt_cache_idk † | 50.4 | 100.0 | 1.3 ± 0.1 | 93.5 ± 1.6 | 2.2 ± 0.1 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 2.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 1.9 ± 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| zerounlearn | 7.7 ± 2.8 | 13.2 ± 9.0 | 13.6 ± 6.7 | 57.6 ± 3.7 |
| simnpo | 14.6 ± 6.5 | 24.2 ± 9.3 | 36.6 ± 6.0 | 79.0 ± 4.7 |
| memit | 21.0 ± 4.1 | 29.6 ± 7.8 | 36.4 ± 5.4 | 80.1 ± 4.5 |
| SURE | 13.7 ± 5.3 | 22.6 ± 10.0 | 22.3 ± 11.6 | 60.2 ± 5.2 |
| SURE_early | 30.5 ± 7.4 | 41.7 ± 10.7 | 39.6 ± 10.1 | 69.3 ± 5.9 |
| rmu | 63.4 ± 11.9 | 56.1 ± 3.2 | 83.5 ± 10.3 | 85.8 ± 15.5 |
| idk_ft | 5.3 ± 2.2 | 14.1 ± 9.3 | 46.8 ± 9.6 | 84.2 ± 9.1 |
| tv | 7.1 ± 5.5 | 4.3 ± 5.9 | 20.2 ± 9.3 | 64.5 ± 3.0 |
| ga | 6.3 ± 4.4 | 5.3 ± 5.3 | 18.1 ± 22.0 | 28.7 ± 17.6 |
| npo | 2.4 ± 3.5 | 3.0 ± 3.4 | 6.7 ± 5.4 | 24.3 ± 3.6 |
| gd | 2.2 ± 2.8 | 4.0 ± 4.0 | 7.9 ± 4.9 | 20.3 ± 6.8 |
| rome | 2.6 ± 3.7 | 1.4 ± 1.9 | 10.1 ± 14.1 | 30.5 ± 23.4 |
| answer_ban | 6.6 ± 5.3 | 5.1 ± 4.1 | 6.3 ± 5.8 | 100.0 ± 0.0 |
| keyword_router_idk | 11.1 ± 4.2 | 11.3 ± 6.7 | 11.4 ± 3.8 | 100.0 ± 0.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |
| prompt_cache_idk | 61.0 ± 2.0 | 68.7 ± 3.7 | 61.3 ± 4.8 | 100.0 ± 0.0 |
| noop | 100.0 ± 0.0 | 100.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| zerounlearn | 20.0 | 0.3 | 79.7 | 70.8 | 0.0 |
| simnpo | 30.0 | 0.0 | 70.0 | 75.9 | 0.0 |
| memit | 36.3 | 0.0 | 63.7 | 91.2 | 0.0 |
| SURE | 23.9 | 0.0 | 76.1 | 98.1 | 0.0 |
| SURE_early | 48.7 | 0.0 | 51.3 | 96.8 | 0.0 |
| rmu | 53.7 | 0.5 | 45.8 | 62.0 | 1.4 |
| idk_ft | 3.4 | 94.7 | 1.8 | 42.6 | 51.4 |
| tv | 18.9 | 0.0 | 81.1 | 28.2 | 0.0 |
| ga | 28.7 | 0.3 | 71.1 | 33.3 | 0.0 |
| npo | 1.8 | 0.0 | 98.2 | 20.4 | 0.0 |
| gd | 2.9 | 0.0 | 97.1 | 3.7 | 0.0 |
| rome | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| answer_ban | 10.8 | 0.0 | 89.2 | 61.6 | 0.0 |
| keyword_router_idk | 33.9 | 66.1 | 0.0 | 69.9 | 30.1 |
| subject_blocklist_idk | 7.9 | 92.1 | 0.0 | 50.5 | 49.5 |
| prompt_cache_idk | 64.5 | 35.5 | 0.0 | 93.1 | 6.9 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| SURE | 86.1 | 13.9 | 3.0 | 13.9 |
| SURE_early | 69.3 | 30.7 | 8.5 | 12.9 |
| keyword_router_idk | 68.8 | 31.2 | 0.1 | 29.7 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| zerounlearn | 529.6 ± 29.9 | 0.92 ± 0.01 |
| simnpo | 194.6 ± 11.5 | 0.93 ± 0.02 |
| memit | 233.8 ± 7.0 | 0.93 ± 0.02 |
| SURE | 2740.3 ± 68.9 | 1.78 ± 0.05 |
| SURE_early | 2773.4 ± 56.5 | 1.78 ± 0.04 |
| rmu | 77.3 ± 3.5 | 0.92 ± 0.01 |
| idk_ft | 199.1 ± 12.8 | 0.93 ± 0.02 |
| tv | 18.3 ± 1.4 | 0.93 ± 0.01 |
| ga | 10.7 ± 6.1 | 0.93 ± 0.01 |
| npo | 55.2 ± 4.7 | 0.92 ± 0.01 |
| gd | 64.4 ± 11.1 | 0.93 ± 0.02 |
| rome | 389.2 ± 87.4 | 0.92 ± 0.01 |
| answer_ban | 0.0 ± 0.0 | 0.95 ± 0.02 |
| keyword_router_idk | 0.0 ± 0.0 | 1.05 ± 0.02 |
| subject_blocklist_idk | 0.0 ± 0.0 | 1.03 ± 0.02 |
| prompt_cache_idk | 0.0 ± 0.0 | 1.45 ± 0.02 |
| noop | 0.0 ± 0.0 | 0.93 ± 0.01 |
| idk_all | 0.0 ± 0.0 | 1.04 ± 0.02 |

