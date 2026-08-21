# Canonical MCF SURE entry point

Use only [`run_mcf_sure_two_stage.py`](run_mcf_sure_two_stage.py) for the new
target-role-explicit MCF experiment.

The locked contract is:

- `target_true` is the sensitive/original answer and receives bounded GA.
- `target_new` is the non-sensitive CounterFact replacement and receives GD.
- Stage 1 is the Rank-4 sparse LM-head SURE edit.
- Stage 2 is the conditional Rank `2 -> 4 -> 8` residual repair.
- Wikipedia KL is the training-time utility guard.
- Official paraphrases, neighborhoods, retain records, and PPL text are
  evaluation-only.
- `Eff` and `Gen` are residual-sensitive preference rates, so lower is better.

Inspect the complete command without loading a model:

```bash
python scripts/MCF_Scripts/run_mcf_sure_two_stage.py \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --mcf-path data/multi_counterfact.json \
  --wikipedia-dir /path/to/wikipedia_datasetdict \
  --utility-docs 1000 \
  --output-root outputs/mcf_sure_W1K \
  --dry-run
```

Run the W1K pilot:

```bash
python scripts/MCF_Scripts/run_mcf_sure_two_stage.py \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --mcf-path data/multi_counterfact.json \
  --wikipedia-dir /path/to/ppl_wikipedia_datasetdict \
  --utility-wikipedia-dir /path/to/real_wikipedia_datasetdict \
  --utility-docs 1000 \
  --utility-prompts 100000 \
  --require-corpus-protocol sure_external_wikipedia_corpus_v1 \
  --output-root outputs/mcf_sure_W1K \
  --seeds 1
```

For a paper aggregate, use a fresh output directory, the locked final utility
coverage, and ten predeclared seeds, for example `--utility-docs 100000
--seeds 1 2 3 4 5 6 7 8 9 10`.

The outputs that matter are:

- `seedN/metrics_eff_gen_spe_ppl.json`: exact per-seed definitions and values;
- `aggregate_metrics.json`: mean, sample SD (`n-1`), and every seed value;
- `table1_rows.md`: Base and SURE rows ready to transfer to Table 1;
- `run_config.json`: the frozen role, metric, architecture, and corpus contract.

Do not mix these Eff/Gen values with results scored under a different answer
role. Baseline raw predictions must be recomputed with `target_true` sensitive
and the same lower-is-better definitions before entering the same table.
