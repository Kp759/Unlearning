# Split `scalpel-toy-dev-s1`

Model: `toy-llama` · partition: dev · seed 1 · 24 requests · 523 probes · fingerprint `344a6059bd2bc6e5`

Forget facts: 24 known out of 144 screened (16.7% of candidate facts are known by this model); 24 have other known facts about the same subject.

| Family | Probes | Candidates | Known by base | Used in aggregates |
|---|---|---|---|---|
| forget/context_prefix | 48 | 48 | 100% | yes |
| forget/crosslingual | 72 | 72 | 100% | yes |
| forget/direct | 24 | 24 | 100% | yes |
| forget/indirect | 5 | 23 | 22% | yes |
| forget/multifact | 47 | 47 | 100% | yes |
| forget/multihop | 5 | 5 | 100% | yes |
| forget/multiturn | 24 | 24 | 100% | yes |
| forget/paraphrase | 35 | 35 | 100% | yes |
| forget/question | 24 | 24 | 100% | yes |
| forget/surface | 23 | 24 | 96% | yes |
| retain/disjoint | 0 | 4 | 0% | no (< 5) |
| retain/in_context | 24 | 24 | 100% | yes |
| retain/name_collision | 9 | 10 | 90% | yes |
| retain/same_object | 7 | 7 | 100% | yes |
| retain/same_relation | 20 | 20 | 100% | yes |
| retain/same_sentence | 47 | 47 | 100% | yes |
| retain/same_subject | 62 | 70 | 89% | yes |
| retain/subject_mentioned | 47 | 48 | 98% | yes |

Fact pool after merging: 376 (0 conflicting (subject, relation) pairs and 4 self-answering facts dropped).
