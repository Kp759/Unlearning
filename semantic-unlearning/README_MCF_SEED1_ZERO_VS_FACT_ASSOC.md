# MCF Seed-1 Matched ZeroUnlearn vs Fact-Association Evaluation

This comparison avoids comparing numbers copied from different metric tables.

## Common evaluation contract

Base, ZeroUnlearn, and the frozen fact-association bank use:

- Llama-3.2-3B-Instruct snapshot `0cb88a4f764b7a12671c53f0838cd831a0843b95`;
- one MCF source file;
- official half-split sampling;
- seed 1;
- the exact same 50 forget case IDs;
- the exact same 1000 retain case IDs;
- the same rewrite, paraphrase, and neighborhood prompts;
- the same fast tokenizer;
- BF16 final scoring; and
- the same strict `zerounlearn_answer_probability_v2` summarizer.

The fact-association artifact must contain the same 50 case IDs in the same order
or the comparison aborts.

## Common forgetting target

Both methods are evaluated on the ORIGINAL MCF
`requested_rewrite.target_true`.

For the ZeroUnlearn edit only:

```text
sensitive target_true = ORIGINAL MCF target_true
neutral target_new    = tokenizer EOS
```

The original MCF records are unchanged for final evaluation. Official
paraphrases and neighborhood prompts are not used by the ZeroUnlearn edit.

This is a fair common-target adaptation. It should be labeled separately from a
published ZeroUnlearn MCF run if the published run used different target
semantics.

## Primary metrics

Eff is:

```text
100 * mean_case P(original target_true | canonical rewrite)
```

where complete answer probability is `exp(-sum answer-token NLL)`.

Gen is the same complete original-answer probability on held-out paraphrases,
averaged paraphrases within each case and then across cases.

Spe is strict neighborhood preservation:

```text
100 * mean_case mean_neighborhood
      1(all original target_true answer tokens are teacher-forced top1)
```

The evaluator also reports ReleasedAccuracy, token-geometric likelihood,
SensitivePref, CounterFact edit-success diagnostics, and both legacy and
runtime-aligned PPL.

## Outputs

The run creates:

```text
base.json
zerounlearn.json
ours.json
shared_protocol.json
comparison.json
comparison.csv
comparison.md
```

The raw per-prompt rows remain in each per-method JSON.

## Run

```bash
git fetch origin
git checkout mcf_seed1_zerounlearn_vs_fact_assoc_eval
git pull origin mcf_seed1_zerounlearn_vs_fact_assoc_eval

pytest -q tests/test_compare_mcf_seed1_zerounlearn_vs_fact_association.py

bash scripts/compare_mcf_seed1_zerounlearn_vs_fact_association.sh
```

The launcher defaults to the frozen fact-association seed-1 run:

```text
outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1
```

and writes to:

```text
outputs/mcf_seed1_zerounlearn_vs_fact_assoc_matched
```

Use a new `OUTPUT_DIR` for a rerun because comparison outputs are never
overwritten.
