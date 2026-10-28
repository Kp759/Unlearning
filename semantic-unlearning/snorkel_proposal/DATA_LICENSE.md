# Data licensing

**LETHE-Bench code** (`scalpel/`, `scripts/`, `tests/`) is released under Apache-2.0 (`LICENSE`).

**LETHE-Bench split files** (requests, probes, manifests) are derived works of the upstream datasets below. They are released under **CC BY 4.0**, wherever every upstream license permits it, with attribution to each source. Hidden-test splits are not distributed.

| Source | Used for | Upstream license | Notes |
|---|---|---|---|
| MultiCounterFact (Meng et al., 2022/2023; ROME/MEMIT) | Facts, paraphrases, neighbourhood and attribute prompts | MIT (kmeng01/rome) | Only `target_true` (the real fact) is used as a forget target |
| MQuAKE-CF / MQuAKE-CF-3k-v2 (Zhong et al., 2023) | Facts (original single hops), multi-hop questions, aliases | MIT (princeton-nlp/MQuAKE) | Original (not counterfactual) chains |
| zsRE (Levy et al., 2017), MEND evaluation split (Mitchell et al., 2022) | QA facts, rephrasings, NQ locality questions | MEND code: MIT. **Data terms to be confirmed with the zsRE authors before redistribution** | Release blocker until confirmed (ROADMAP M1) |
| RWKU (Jin et al., 2024) | Entity track probes, MMLU / TriviaQA utility subsets | CC BY 4.0 (jinzhuoran/RWKU) | MMLU and TriviaQA items are RWKU's own subsets; their upstream terms also apply |
| `data/wikidata` fixture (this repository) | Perplexity texts | This repository | 200 encyclopedia-style sentences |
| Relation templates, cross-lingual questions, refusal patterns | Probe generation | Apache-2.0 (this repository) | Author-written. Native-speaker verification is planned (ROADMAP M3) |

## Models used in the pilot

These are not redistributed: Qwen2.5-0.5B-Instruct (Apache-2.0) and SmolLM2-360M-Instruct (Apache-2.0). Cluster runs use Llama-3.2-3B-Instruct / Llama-3.1-8B-Instruct (Llama Community License) and Qwen3-4B (Apache-2.0), each under its own terms.

## Personal data

Every subject is a public figure or a public entity that already appears in the source datasets (Wikidata-derived). LETHE-Bench creates no new personal facts. The `in_context` family supplies **counterfactual** values that are labelled as notes to be read, never as facts about the person.
