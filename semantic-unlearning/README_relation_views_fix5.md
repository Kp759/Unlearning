# Relation-view corpus fix5

## What this fixes

The reviewed upstream code at commit `ee034a2857bfb50f9280395be6b90367f6b262fb` accepts a generated rewrite only if the maximum cosine similarity to its relation references beats all competing relations. The failing P463 run assigns -1 to all 16 candidates because each has a negative cross-relation gap. This is not a reliable test of semantic equivalence: a clear organizational-membership question can be closer to a language/position/location reference in the selected embedding space.

The generation prompt itself contains the entity, incomplete canonical prompt, and requested linguistic family, but no explicit relation-ID definition. Vague `affiliated with` wording permits semantic drift into government type, country, or generic association.

This package supplies an **alternative deterministic construction**, not another relaxed similarity threshold and not a repair that makes the old neural verifier reliable. Its reviewed-by-the-assistant template bank is explicitly indexed by registered relation ID. It includes 34 IDs from the repository's synthetic relation bank and eight additional linguistic families per case. The original canonical template is retained unchanged. Every output has provenance. No output is falsely labeled LLM-generated, model-verified, or independently human-certified.

## Files

- `scripts/build_mcf_relation_views_v2_fix5.py`: standard-library-only builder.
- `scripts/mcf_relation_contracts_fix5.json`: inspectable authored relation templates.
- `scripts/run_build_mcf_relation_views_v2_fix5.sh`: launcher, no model required.
- `tests/test_relation_views_v2_fix5.py`: CPU tests; requires pytest only.

Original scripts, checkpoints, evaluators, and previous outputs are not modified.

## Run from semantic-unlearning

```bash
: "${FORGET_DIRECT:?Set the existing sanitized training_visible_forget_direct.json path}"
test -f "$FORGET_DIRECT"

python -m pytest -q tests/test_relation_views_v2_fix5.py
export RELATION_V2_CORPUS="$PWD/outputs/mcf_relation_views_v2_seed1/relation_views_v2_fix5.json"

bash scripts/run_build_mcf_relation_views_v2_fix5.sh --preflight-only --preview-case-id 13256

set -o pipefail
bash scripts/run_build_mcf_relation_views_v2_fix5.sh 2>&1 | tee relation_views_v2_build_fix5.log
```

No `MODEL_PATH`, `WIKIDATA_DIR`, `MCF_PATH`, `VIEW_CORPUS`, CUDA, Transformers, or package upgrade is needed for this builder. It receives only `FORGET_DIRECT`. `VIEW_CORPUS` can remain set for other scripts, but its contents are not consulted by fix5. The source file may contain target fields as required by the existing sanitized schema; values of those fields are discarded and never used to select or render templates.

Output protocol remains `mcf_relation_view_corpus_v2`, with `cases`, `views`, `family`, and `template` fields and the original family names/split recommendation. Each authored view has `equivalence_margin: null`, since inventing a neural-verifier score would be misleading. A downstream component that explicitly requires generated-source tags or numeric verifier margins must be adjusted to recognize authored provenance; that downstream behavior was not executed here.

## Example: P463

The declared relation is **member of**, not capital, government type, citizenship, country, or generic affiliation. Example outputs include:

- Which organization is Belgium a member of?
- What is the name of an organization in which Belgium holds membership?
- Name an organization that has Belgium as a member.
- Of which organization is Belgium a member?
- In which organization does Belgium hold membership?

The `reordered_cloze` view intentionally ends before its answer, as the existing V2 family requires. Questions and requests are standalone; cloze views are intentionally incomplete. No answer or new named object is inserted.

## Quality boundaries

The exact-match validator applies only to this authored grammar. It does **not** claim to verify arbitrary language-model outputs. An arbitrary candidate containing the word `organization` is not accepted unless it is an approved template for the declared relation and family. Unknown relations fail before output publication rather than receiving a vague generic fallback. Missing or duplicate cases are not silently dropped.

The bank distinguishes official language (P37), native language (P103), original film/TV language (P364), work/name language (P407), and general language use (P1412). Writing-specific P1412 canonical frames select the writing variant. Scope warnings identify time/superlative modifiers not automatically preserved; inspect the accompanying preview and those flagged cases before using them as router positives. The declared PID is the construction authority; this does not independently establish that each source canonical prompt was correctly assigned that PID. Review any source/contract disagreements rather than silently relabeling them.

Nine views of a single source fact are not nine independent facts. The retained family split is a suggested training partition, not proof of independence or broad semantic coverage. This compact authored corpus is a controlled starting set, not a substitute for genuinely diverse held-out language and aliases. No Gen, suppression, retention, novelty, or publication result is claimed.

## Testing and limitations

Tests run locally on synthetic sanitized records cover every supported relation, the P463 failure, placeholder handling, short grammatical questions, disallowed government/country drift, answer-value nonuse, input firewalls, duplicate IDs, source-scope variants, deterministic output, end-to-end CLI writing, and no-clobber output publication.

The user's actual 50-record source file, complete router pipeline, and AWS environment were not available to execute. The all-cases preflight is intended to expose unsupported IDs or source issues before writing a corpus. No GPU generation was performed, because this version does not use an LLM.
