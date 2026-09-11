# Extended Association Tokens v2

This standalone ablation gives each of the 50 sampled forget facts one private
input embedding row. Every base-model parameter and the original output
vocabulary remain frozen. The private token is inserted after BOS by an oracle
that already knows the fact ID, so this experiment measures private-token
capacity rather than deployable natural-prompt unlearning.

V2 removes the cross-fact behavior seen in V1:

- each private row has its own Adam optimizer and trust radius;
- each row update evaluates all eight authored training views for its fact;
- the eight views are padded into one answer batch and one abstention batch so
  30 complete sweeps over the 50 facts fit the registered 1,500-row-step budget;
- the view with the largest forgotten-answer probability supplies the forget
  gradient, while mean `" I don't know."` NLL is secondary;
- before the answer threshold is reached, an update cannot increase the worst
  training-view answer probability;
- after every training view for a fact falls below `1e-6`, the threshold is a
  hard constraint and only safe abstention improvements are accepted;
- the radius is `1.0` above `1e-3`, `0.25` from `1e-5` to `1e-3`, `0.05` from
  `1e-6` to `1e-5`, and `0.01` after the row is protected;
- checkpoints rank by the maximum answer probability across every training and
  development view, then by mean abstention NLL;
- stopping requires every training and every development answer view to be
  below `1e-6`.

The eight training templates and four development templates were authored
independently from relation names. The runner does not read MCF
`paraphrase_prompts` or `neighborhood_prompts`, and development views never
enter a gradient.

Run the focused tests:

```bash
pytest -q \
  tests/test_static_overlap_extended_tokens.py \
  tests/test_static_overlap_extended_tokens_standalone.py \
  tests/test_static_overlap_extended_tokens_v2.py
```

Run the registered standalone experiment:

```bash
bash scripts/run_static_overlap_extended_tokens_standalone_v2.sh \
  /path/to/Llama-3.2-3B-Instruct \
  "$PWD/data/multi_counterfact.json"
```

Outputs are written to
`outputs/static_overlap_extended_tokens_standalone_v2_seed1`. The report keeps
the worst view IDs and probabilities, the number of passing facts, row lock
state, adaptive radius, checkpoint key, and selected best step. The final row
artifact is restored from that best checkpoint.

Natural prompts never contain a private token and therefore remain bit-exact to
the base model by construction. Official natural-prompt Eff/Gen behavior is not
changed, and this runner does not start official evaluation or touch final-test
artifacts.
