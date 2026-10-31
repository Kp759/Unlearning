# SCALPEL — llama32_3b — dev (mean ± sd over seeds)

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · scalpel-fact-dev · meta-llama/Llama-3.2-3B-Instruct · 50 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 30.1 ± 10.1 | 22.8 ± 14.3 | 3.6 ± 3.4 | 95.6 ± 3.8 | 95.6 ± 3.1 | 85.9 ± 6.3 | 100.0 | 19.0 ± 3.6 | 2.9 | 5 |
| zerounlearn | 21.4 ± 10.3 | 38.9 ± 17.9 | 21.1 ± 12.2 | 96.4 ± 3.8 | 67.4 ± 3.9 | 38.6 ± 6.2 | 96.9 ± 1.3 | 8.7 ± 4.3 | – | 5 |
| sure | 12.0 ± 0.4 | 3.8 ± 0.0 | 0.0 | 95.6 ± 4.6 | 99.7 ± 0.4 | 97.5 ± 2.8 | 99.6 ± 0.7 | 19.0 ± 3.6 | 2.9 ± 0.0 | 5 |
| npo | 6.2 ± 2.5 | 91.9 ± 14.8 | 84.1 ± 25.1 | 100.0 | 1.7 ± 0.7 | 0.0 | 74.9 ± 18.4 | 80.3 ± 18.4 | – | 5 |
| gd | 5.5 ± 1.6 | 94.8 ± 5.2 | 87.0 ± 12.6 | 100.0 | 1.5 ± 0.4 | 0.0 | 66.5 ± 10.1 | 86.5 ± 9.5 | – | 5 |
| ga | 4.9 ± 1.5 | 97.1 ± 4.1 | 91.9 ± 11.2 | 100.0 | 1.3 ± 0.4 | 0.0 | 56.2 ± 16.9 | 90.5 ± 5.7 | – | 5 |
| rmu | 3.0 ± 4.5 | 8.3 ± 6.9 | 1.7 ± 2.4 | 11.6 ± 4.6 | 39.7 ± 48.5 | 32.6 ± 44.7 | 84.5 ± 22.3 | 5.8 ± 12.0 | – | 5 |
| rome | 1.6 ± 3.6 | 8.5 ± 3.6 | 2.0 ± 1.8 | 82.8 ± 5.0 | 57.2 ± 11.4 | 30.0 ± 12.5 | 98.7 ± 0.7 | 0.8 ± 1.9 | – | 5 |
| answer_ban † | 0.0 | 91.8 ± 3.6 | 72.1 ± 10.7 | 95.2 ± 3.6 | 7.3 ± 0.1 | 0.0 | 46.7 ± 5.5 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 11.6 ± 1.2 | 0.0 ± 0.0 | 2.7 ± 0.0 | 5 |
| keyword_router_idk † | 0.0 | 6.4 ± 0.1 | 0.0 | 100.0 | 42.3 ± 9.3 | 12.6 ± 3.9 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| memit | 0.0 | 9.0 ± 4.1 | 3.9 ± 2.9 | 45.6 ± 6.2 | 91.8 ± 1.4 | 76.6 ± 6.7 | 99.2 ± 0.8 | 0.0 | – | 5 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 5 |
| prompt_cache_idk † | 0.0 | 1.2 ± 0.1 | 0.0 | 100.0 | 90.4 ± 4.2 | 66.6 ± 10.6 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| subject_blocklist_idk † | 0.0 | 7.3 ± 0.1 | 0.0 | 100.0 | 2.0 ± 0.0 | 0.0 | 99.8 ± 0.3 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | 5 ± 3 | 9 ± 2 | 4 ± 4 | 100 | 7 ± 7 | 96 ± 3 | 8 ± 3 | 10 ± 8 | 8 ± 1 | 4 ± 4 | 100 | 88 ± 5 | 100 | 100 | 100 | 95 ± 7 | 90 ± 9 | 94 ± 7 |
| zerounlearn | 7 ± 3 | 62 ± 6 | 4 ± 4 | 62 ± 36 | 12 ± 3 | 77 ± 14 | 71 ± 5 | 12 ± 7 | 61 ± 4 | 27 ± 11 | 88 ± 6 | 100 | 84 ± 9 | 72 ± 12 | 85 ± 4 | 43 ± 8 | 41 ± 6 | 93 ± 2 |
| sure | 6 ± 5 | 43 ± 14 | 4 ± 5 | 100 | 13 ± 13 | 100 | 37 ± 13 | 20 ± 8 | 29 ± 10 | 100 | 100 | 100 | 100 | 100 | 100 | 99 ± 3 | 100 | 99 ± 2 |
| npo | 0 ± 1 | 10 ± 17 | 0 | 0 | 0 | 13 ± 26 | 14 ± 21 | 0 ± 1 | 12 ± 21 | 0 | 9 ± 11 | 39 ± 31 | 4 ± 6 | 0 ± 1 | 1 ± 2 | 0 | 0 | 23 ± 30 |
| gd | 0 | 12 ± 12 | 0 | 0 | 0 | 4 ± 4 | 12 ± 12 | 0 | 9 ± 10 | 0 | 8 ± 8 | 35 ± 34 | 2 ± 3 | 0 | 0 | 0 | 0 | 21 ± 21 |
| ga | 0 | 5 ± 8 | 0 | 0 | 0 | 2 ± 3 | 8 ± 10 | 0 | 6 ± 10 | 0 | 6 ± 8 | 26 ± 38 | 1 ± 2 | 0 | 0 | 0 | 0 | 13 ± 17 |
| rmu | 91 ± 5 | 40 ± 53 | 88 ± 5 | 95 ± 11 | 97 ± 3 | 31 ± 44 | 38 ± 52 | 92 ± 6 | 42 ± 49 | 86 ± 13 | 81 ± 13 | 38 ± 52 | 85 ± 15 | 94 ± 4 | 96 ± 3 | 95 ± 6 | 93 ± 6 | 34 ± 47 |
| rome | 17 ± 4 | 93 ± 1 | 17 ± 5 | 82 ± 29 | 24 ± 9 | 92 ± 7 | 95 ± 3 | 31 ± 8 | 95 ± 3 | 42 ± 9 | 87 ± 4 | 100 | 72 ± 13 | 67 ± 6 | 79 ± 9 | 31 ± 13 | 34 ± 10 | 99 ± 1 |
| answer_ban | 5 ± 4 | 3 ± 1 | 5 ± 4 | 0 | 3 ± 3 | 28 ± 11 | 4 ± 4 | 5 ± 4 | 4 ± 2 | 6 ± 4 | 70 ± 9 | 75 ± 12 | 60 ± 10 | 0 | 82 ± 3 | 65 ± 9 | 70 ± 19 | 65 ± 6 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 92 ± 2 | 0 | 100 | 22 ± 10 | 76 ± 12 | 19 ± 2 | 35 ± 10 | 21 ± 5 | 100 | 100 | 13 ± 4 | 100 | 100 | 100 | 24 ± 10 | 96 ± 4 | 76 ± 11 |
| memit | 67 ± 3 | 93 ± 3 | 54 ± 6 | 87 ± 30 | 69 ± 3 | 90 ± 6 | 95 ± 4 | 75 ± 6 | 94 ± 2 | 83 ± 1 | 97 ± 2 | 100 | 94 ± 6 | 94 ± 4 | 95 ± 3 | 85 ± 4 | 77 ± 7 | 98 ± 2 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 100 ± 1 | 98 ± 3 | 100 | 100 | 67 ± 10 | 100 | 100 | 100 | 77 ± 15 | 100 | 100 |
| subject_blocklist_idk | 0 | 0 | 0 | 95 ± 11 | 0 | 56 ± 18 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 99 ± 2 | 100 | 0 | 0 | 0 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| ours | 93.0 | 100.0 | 22.8 ± 14.3 | 95.6 ± 3.1 | 30.1 ± 10.1 |
| zerounlearn | 92.2 | 80.1 | 38.9 ± 17.9 | 67.4 ± 3.9 | 21.4 ± 10.3 |
| sure | 87.6 | 100.0 | 3.8 ± 0.0 | 99.7 ± 0.4 | 12.0 ± 0.4 |
| npo | 99.8 | 4.9 | 91.9 ± 14.8 | 1.7 ± 0.7 | 6.2 ± 2.5 |
| gd | 100.0 | 4.0 | 94.8 ± 5.2 | 1.5 ± 0.4 | 5.5 ± 1.6 |
| ga | 100.0 | 2.8 | 97.1 ± 4.1 | 1.3 ± 0.4 | 4.9 ± 1.5 |
| rmu | 9.6 | 87.4 | 8.3 ± 6.9 | 39.7 ± 48.5 | 3.0 ± 4.5 |
| rome | 75.7 | 76.6 | 8.5 ± 3.6 | 57.2 ± 11.4 | 1.6 ± 3.6 |
| answer_ban † | 95.0 | 34.9 | 91.8 ± 3.6 | 7.3 ± 0.1 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 82.7 | 100.0 | 6.4 ± 0.1 | 42.3 ± 9.3 | 0.0 |
| memit | 35.5 | 95.6 | 9.0 ± 4.1 | 91.8 ± 1.4 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.2 | 100.0 | 1.2 ± 0.1 | 90.4 ± 4.2 | 0.0 |
| subject_blocklist_idk † | 100.0 | 99.5 | 7.3 ± 0.1 | 2.0 ± 0.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| ours | 7.2 ± 4.0 | 6.8 ± 1.6 | 12.6 ± 8.4 | 81.0 ± 3.6 |
| zerounlearn | 20.6 ± 3.6 | 26.1 ± 3.2 | 73.4 ± 6.1 | 91.3 ± 4.3 |
| sure | 16.5 ± 6.1 | 19.0 ± 3.0 | 25.4 ± 10.2 | 81.0 ± 3.6 |
| npo | 3.0 ± 4.9 | 6.1 ± 9.1 | 7.8 ± 15.0 | 19.7 ± 18.4 |
| gd | 2.2 ± 2.2 | 3.9 ± 3.6 | 2.9 ± 2.9 | 13.5 ± 9.5 |
| ga | 1.5 ± 2.2 | 2.5 ± 3.4 | 4.4 ± 4.7 | 9.4 ± 5.7 |
| rmu | 80.0 ± 14.9 | 78.7 ± 18.4 | 93.6 ± 11.8 | 93.2 ± 13.4 |
| rome | 42.4 ± 4.6 | 48.5 ± 4.8 | 95.8 ± 3.1 | 99.0 ± 1.8 |
| answer_ban | 4.8 ± 2.8 | 6.4 ± 3.8 | 4.3 ± 4.2 | 100.0 ± 0.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |
| keyword_router_idk | 18.8 ± 4.7 | 17.6 ± 5.0 | 20.0 ± 6.7 | 100.0 ± 0.0 |
| memit | 73.9 ± 3.6 | 78.5 ± 5.5 | 97.4 ± 2.6 | 100.0 |
| noop | 100.0 ± 0.0 | 100.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| prompt_cache_idk | 63.0 ± 1.4 | 66.9 ± 2.6 | 63.2 ± 2.3 | 100.0 ± 0.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| ours | 18.8 | 0.0 | 81.2 | 98.1 | 0.0 |
| zerounlearn | 36.6 | 0.0 | 63.4 | 83.0 | 0.0 |
| sure | 62.2 | 0.0 | 37.8 | 99.6 | 0.0 |
| npo | 0.3 | 33.8 | 66.0 | 2.6 | 22.3 |
| gd | 8.8 | 0.0 | 91.2 | 16.6 | 0.0 |
| ga | 0.0 | 21.9 | 78.1 | 0.8 | 9.1 |
| rmu | 52.6 | 0.0 | 47.4 | 65.3 | 0.0 |
| rome | 64.9 | 0.0 | 35.1 | 87.5 | 0.0 |
| answer_ban | 12.9 | 0.3 | 86.9 | 62.3 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 36.3 | 63.7 | 0.0 | 82.6 | 17.4 |
| memit | 88.1 | 0.0 | 11.9 | 96.2 | 0.0 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 68.0 | 32.0 | 0.0 | 95.8 | 4.2 |
| subject_blocklist_idk | 10.6 | 89.4 | 0.0 | 61.1 | 38.9 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| ours | 83.6 | 16.4 | 0.0 | 9.7 |
| sure | 58.8 | 40.9 | 2.6 | 1.9 |
| keyword_router_idk | 63.7 | 36.3 | 0.5 | 17.5 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| ours | 2137.2 ± 78.0 | 2.17 ± 0.05 |
| zerounlearn | 638.0 ± 16.9 | 0.91 ± 0.03 |
| sure | 714.6 ± 18.4 | 1.33 ± 0.03 |
| npo | 33.8 ± 2.9 | 0.92 ± 0.02 |
| gd | 35.4 ± 4.3 | 0.91 ± 0.02 |
| ga | 6.3 ± 1.8 | 0.92 ± 0.02 |
| rmu | 55.8 ± 4.7 | 0.91 ± 0.01 |
| rome | 434.4 ± 32.0 | 0.92 ± 0.02 |
| answer_ban | 0.0 ± 0.0 | 0.94 ± 0.02 |
| idk_all | 0.0 ± 0.0 | 1.06 ± 0.03 |
| keyword_router_idk | 0.0 ± 0.0 | 1.07 ± 0.03 |
| memit | 146.5 ± 6.6 | 0.91 ± 0.02 |
| noop | 0.0 ± 0.0 | 0.92 ± 0.01 |
| prompt_cache_idk | 0.0 ± 0.0 | 1.60 ± 0.04 |
| subject_blocklist_idk | 0.0 ± 0.0 | 1.06 ± 0.03 |



All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · scalpel-fact-dev · meta-llama/Llama-3.2-3B-Instruct · 50 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 30.1 ± 10.0 | 22.7 ± 14.2 | 3.6 ± 3.4 | 93.6 ± 5.0 | 95.8 ± 2.8 | 85.5 ± 6.8 | 100.0 | 19.0 ± 3.6 | 2.9 | 5 |
| zerounlearn | 21.5 ± 10.3 | 37.5 ± 17.2 | 19.8 ± 11.5 | 96.3 ± 3.9 | 73.7 ± 4.0 | 46.7 ± 4.5 | 96.9 ± 1.3 | 8.7 ± 4.3 | – | 5 |
| sure | 11.8 ± 0.4 | 3.8 ± 0.1 | 0.0 | 69.7 ± 4.0 | 99.9 ± 0.2 | 98.9 ± 1.6 | 99.6 ± 0.7 | 19.0 ± 3.6 | 2.9 ± 0.0 | 5 |
| npo | 7.6 ± 4.8 | 91.4 ± 15.7 | 82.3 ± 26.8 | 99.6 ± 0.9 | 2.2 ± 1.6 | 0.0 | 74.9 ± 18.4 | 80.3 ± 18.4 | – | 5 |
| gd | 5.6 ± 1.5 | 95.0 ± 5.0 | 87.1 ± 12.4 | 100.0 | 1.5 ± 0.4 | 0.0 | 66.5 ± 10.1 | 86.5 ± 9.5 | – | 5 |
| ga | 5.1 ± 1.3 | 97.1 ± 4.1 | 91.8 ± 11.3 | 99.6 ± 1.0 | 1.3 ± 0.4 | 0.0 | 56.2 ± 16.9 | 90.5 ± 5.7 | – | 5 |
| rmu | 3.1 ± 4.5 | 5.3 ± 4.3 | 1.2 ± 1.7 | 7.8 ± 5.3 | 71.0 ± 38.1 | 54.4 ± 35.0 | 84.5 ± 22.3 | 5.8 ± 12.0 | – | 5 |
| rome | 1.4 ± 3.2 | 7.7 ± 3.3 | 2.0 ± 1.8 | 71.2 ± 6.7 | 65.9 ± 5.7 | 37.3 ± 6.5 | 98.7 ± 0.7 | 0.8 ± 1.9 | – | 5 |
| answer_ban † | 0.0 | 84.7 ± 1.3 | 68.4 ± 7.7 | 89.0 ± 3.0 | 10.9 ± 5.0 | 1.0 ± 1.4 | 46.7 ± 5.5 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 11.6 ± 1.2 | 0.0 ± 0.0 | 2.7 ± 0.0 | 5 |
| keyword_router_idk † | 0.0 | 6.4 ± 0.1 | 0.0 | 100.0 | 42.4 ± 9.2 | 12.6 ± 3.9 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| memit | 0.0 | 7.1 ± 3.2 | 3.1 ± 2.6 | 31.0 ± 5.2 | 94.6 ± 1.1 | 85.0 ± 3.8 | 99.2 ± 0.8 | 0.0 | – | 5 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 5 |
| prompt_cache_idk † | 0.0 | 1.2 ± 0.1 | 0.0 | 100.0 | 90.4 ± 4.2 | 66.6 ± 10.6 | 100.0 | 0.0 ± 0.0 | 2.9 | 5 |
| subject_blocklist_idk † | 0.0 | 7.3 ± 0.1 | 0.0 | 100.0 | 2.0 ± 0.0 | 0.0 | 99.8 ± 0.3 | 0.0 ± 0.0 | 2.9 ± 0.0 | 5 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | 7 ± 5 | 9 ± 2 | 6 ± 5 | 100 | 9 ± 7 | 96 ± 3 | 8 ± 3 | 11 ± 9 | 8 ± 1 | 7 ± 7 | 100 | 88 ± 5 | 100 | 100 | 100 | 96 ± 4 | 90 ± 10 | 95 ± 7 |
| zerounlearn | 8 ± 3 | 64 ± 5 | 4 ± 4 | 85 ± 34 | 14 ± 3 | 78 ± 14 | 71 ± 4 | 17 ± 6 | 63 ± 4 | 30 ± 11 | 92 ± 4 | 100 | 86 ± 11 | 81 ± 10 | 90 ± 1 | 49 ± 7 | 48 ± 5 | 93 ± 3 |
| sure | 29 ± 7 | 56 ± 13 | 30 ± 4 | 100 | 45 ± 13 | 100 | 48 ± 15 | 57 ± 8 | 41 ± 11 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 ± 2 |
| npo | 1 ± 1 | 10 ± 17 | 0 ± 1 | 0 | 0 | 14 ± 29 | 14 ± 21 | 2 ± 3 | 12 ± 21 | 0 | 10 ± 12 | 45 ± 21 | 5 ± 8 | 1 ± 2 | 1 ± 3 | 1 ± 3 | 0 | 26 ± 33 |
| gd | 0 | 12 ± 12 | 0 | 0 | 0 | 3 ± 3 | 12 ± 12 | 0 | 9 ± 10 | 0 | 9 ± 8 | 41 ± 29 | 2 ± 3 | 0 | 0 | 0 | 0 | 23 ± 23 |
| ga | 0 | 6 ± 8 | 0 ± 1 | 0 | 0 | 2 ± 3 | 8 ± 10 | 0 | 6 ± 10 | 0 | 6 ± 8 | 37 ± 33 | 1 ± 2 | 0 | 0 | 0 | 0 | 14 ± 20 |
| rmu | 94 ± 5 | 56 ± 42 | 92 ± 5 | 95 ± 11 | 97 ± 3 | 49 ± 34 | 70 ± 40 | 96 ± 6 | 66 ± 39 | 93 ± 8 | 90 ± 10 | 59 ± 40 | 92 ± 7 | 94 ± 4 | 97 ± 2 | 95 ± 6 | 96 ± 7 | 62 ± 37 |
| rome | 27 ± 7 | 94 ± 1 | 29 ± 7 | 88 ± 16 | 29 ± 9 | 92 ± 7 | 96 ± 3 | 44 ± 8 | 95 ± 2 | 51 ± 8 | 90 ± 3 | 100 | 81 ± 12 | 76 ± 6 | 85 ± 9 | 39 ± 6 | 38 ± 7 | 99 ± 1 |
| answer_ban | 10 ± 4 | 10 ± 5 | 11 ± 3 | 0 | 17 ± 5 | 30 ± 10 | 12 ± 6 | 14 ± 5 | 11 ± 4 | 10 ± 7 | 71 ± 8 | 78 ± 14 | 64 ± 15 | 1 ± 1 | 84 ± 4 | 66 ± 7 | 71 ± 19 | 67 ± 7 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 92 ± 2 | 0 | 100 | 22 ± 9 | 76 ± 12 | 19 ± 2 | 34 ± 10 | 21 ± 5 | 100 | 100 | 13 ± 4 | 100 | 100 | 100 | 25 ± 10 | 96 ± 4 | 76 ± 11 |
| memit | 76 ± 4 | 94 ± 3 | 69 ± 5 | 100 | 81 ± 6 | 90 ± 6 | 95 ± 4 | 87 ± 5 | 94 ± 3 | 89 ± 5 | 98 ± 2 | 100 | 94 ± 6 | 95 ± 3 | 96 ± 2 | 92 ± 4 | 85 ± 4 | 99 ± 2 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 100 ± 1 | 98 ± 3 | 100 | 100 | 67 ± 10 | 100 | 100 | 100 | 77 ± 15 | 100 | 100 |
| subject_blocklist_idk | 0 | 0 | 0 | 95 ± 11 | 0 | 56 ± 18 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 99 ± 2 | 100 | 0 | 0 | 0 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| ours | 91.3 | 100.0 | 22.7 ± 14.2 | 95.8 ± 2.8 | 30.1 ± 10.0 |
| zerounlearn | 89.9 | 86.3 | 37.5 ± 17.2 | 73.7 ± 4.0 | 21.5 ± 10.3 |
| sure | 56.3 | 100.0 | 3.8 ± 0.1 | 99.9 ± 0.2 | 11.8 ± 0.4 |
| npo | 98.9 | 5.5 | 91.4 ± 15.7 | 2.2 ± 1.6 | 7.6 ± 4.8 |
| gd | 100.0 | 4.4 | 95.0 ± 5.0 | 1.5 ± 0.4 | 5.6 ± 1.5 |
| ga | 99.8 | 3.0 | 97.1 ± 4.1 | 1.3 ± 0.4 | 5.1 ± 1.3 |
| rmu | 5.8 | 91.7 | 5.3 ± 4.3 | 71.0 ± 38.1 | 3.1 ± 4.5 |
| rome | 63.6 | 83.4 | 7.7 ± 3.3 | 65.9 ± 5.7 | 1.4 ± 3.2 |
| answer_ban † | 87.7 | 36.2 | 84.7 ± 1.3 | 10.9 ± 5.0 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 83.0 | 100.0 | 6.4 ± 0.1 | 42.4 ± 9.2 | 0.0 |
| memit | 22.1 | 96.5 | 7.1 ± 3.2 | 94.6 ± 1.1 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.2 | 100.0 | 1.2 ± 0.1 | 90.4 ± 4.2 | 0.0 |
| subject_blocklist_idk † | 100.0 | 99.5 | 7.3 ± 0.1 | 2.0 ± 0.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| ours | 7.2 ± 4.0 | 6.8 ± 1.6 | 12.6 ± 8.4 | 81.0 ± 3.6 |
| zerounlearn | 20.6 ± 3.6 | 26.1 ± 3.2 | 73.4 ± 6.1 | 91.3 ± 4.3 |
| sure | 16.5 ± 6.1 | 19.0 ± 3.0 | 25.4 ± 10.2 | 81.0 ± 3.6 |
| npo | 3.0 ± 4.9 | 6.1 ± 9.1 | 7.8 ± 15.0 | 19.7 ± 18.4 |
| gd | 2.2 ± 2.2 | 3.9 ± 3.6 | 2.9 ± 2.9 | 13.5 ± 9.5 |
| ga | 1.5 ± 2.2 | 2.5 ± 3.4 | 4.4 ± 4.7 | 9.4 ± 5.7 |
| rmu | 80.0 ± 14.9 | 78.7 ± 18.4 | 93.6 ± 11.8 | 93.2 ± 13.4 |
| rome | 42.4 ± 4.6 | 48.5 ± 4.8 | 95.8 ± 3.1 | 99.0 ± 1.8 |
| answer_ban | 4.8 ± 2.8 | 6.4 ± 3.8 | 4.3 ± 4.2 | 100.0 ± 0.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |
| keyword_router_idk | 18.8 ± 4.7 | 17.6 ± 5.0 | 20.0 ± 6.7 | 100.0 ± 0.0 |
| memit | 73.9 ± 3.6 | 78.5 ± 5.5 | 97.4 ± 2.6 | 100.0 |
| noop | 100.0 ± 0.0 | 100.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| prompt_cache_idk | 63.0 ± 1.4 | 66.9 ± 2.6 | 63.2 ± 2.3 | 100.0 ± 0.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 ± 0.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| ours | 18.8 | 0.0 | 81.2 | 98.1 | 0.0 |
| zerounlearn | 36.6 | 0.0 | 63.4 | 83.0 | 0.0 |
| sure | 62.2 | 0.0 | 37.8 | 99.6 | 0.0 |
| npo | 0.3 | 33.8 | 66.0 | 2.6 | 22.3 |
| gd | 8.8 | 0.0 | 91.2 | 16.6 | 0.0 |
| ga | 0.0 | 21.9 | 78.1 | 0.8 | 9.1 |
| rmu | 52.6 | 0.0 | 47.4 | 65.3 | 0.0 |
| rome | 64.9 | 0.0 | 35.1 | 87.5 | 0.0 |
| answer_ban | 12.9 | 0.3 | 86.9 | 62.3 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 36.3 | 63.7 | 0.0 | 82.6 | 17.4 |
| memit | 88.1 | 0.0 | 11.9 | 96.2 | 0.0 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 68.0 | 32.0 | 0.0 | 95.8 | 4.2 |
| subject_blocklist_idk | 10.6 | 89.4 | 0.0 | 61.1 | 38.9 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| ours | 83.6 | 16.4 | 0.0 | 9.7 |
| sure | 58.8 | 40.9 | 2.6 | 1.9 |
| keyword_router_idk | 63.7 | 36.3 | 0.5 | 17.5 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| ours | 2137.2 ± 78.0 | 2.17 ± 0.05 |
| zerounlearn | 638.0 ± 16.9 | 0.91 ± 0.03 |
| sure | 714.6 ± 18.4 | 1.33 ± 0.03 |
| npo | 33.8 ± 2.9 | 0.92 ± 0.02 |
| gd | 35.4 ± 4.3 | 0.91 ± 0.02 |
| ga | 6.3 ± 1.8 | 0.92 ± 0.02 |
| rmu | 55.8 ± 4.7 | 0.91 ± 0.01 |
| rome | 434.4 ± 32.0 | 0.92 ± 0.02 |
| answer_ban | 0.0 ± 0.0 | 0.94 ± 0.02 |
| idk_all | 0.0 ± 0.0 | 1.06 ± 0.03 |
| keyword_router_idk | 0.0 ± 0.0 | 1.07 ± 0.03 |
| memit | 146.5 ± 6.6 | 0.91 ± 0.02 |
| noop | 0.0 ± 0.0 | 0.92 ± 0.01 |
| prompt_cache_idk | 0.0 ± 0.0 | 1.60 ± 0.04 |
| subject_blocklist_idk | 0.0 ± 0.0 | 1.06 ± 0.03 |

