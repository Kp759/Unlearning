# Split `scalpel-fact-dev-s1`

Model: `Qwen2.5-0.5B-Instruct` · partition: dev · seed 1 · 50 requests · 520 probes · fingerprint `99140753becc28cc`

Forget facts: 50 known out of 510 screened (9.8% of candidate facts are known by this model); 43 have other known facts about the same subject.

| Family | Probes | Candidates | Known by base | Used in aggregates |
|---|---|---|---|---|
| forget/context_prefix | 74 | 100 | 74% | yes |
| forget/crosslingual | 67 | 147 | 46% | yes |
| forget/direct | 50 | 50 | 100% | yes |
| forget/indirect | 2 | 13 | 15% | no (< 5) |
| forget/multifact | 26 | 55 | 47% | yes |
| forget/multihop | 12 | 45 | 27% | yes |
| forget/multiturn | 22 | 49 | 45% | yes |
| forget/paraphrase | 40 | 109 | 37% | yes |
| forget/question | 30 | 49 | 61% | yes |
| forget/surface | 18 | 48 | 38% | yes |
| retain/disjoint | 28 | 295 | 9% | yes |
| retain/in_context | 50 | 50 | 100% | yes |
| retain/name_collision | 10 | 65 | 15% | yes |
| retain/same_object | 34 | 135 | 25% | yes |
| retain/same_relation | 13 | 137 | 9% | yes |
| retain/same_sentence | 14 | 55 | 25% | yes |
| retain/same_subject | 5 | 61 | 8% | yes |
| retain/subject_mentioned | 25 | 100 | 25% | yes |

Fact pool after merging: 38086 (186 conflicting (subject, relation) pairs and 2141 self-answering facts dropped).
