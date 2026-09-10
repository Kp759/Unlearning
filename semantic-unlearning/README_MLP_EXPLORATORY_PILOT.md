# One-MLP exploratory pilot

This is a separate experiment informed by the completed head experiment's failure.
It cannot replace that result or establish fresh confirmatory evidence on the
already-observed final sets. Original final files, results, and checkpoint bindings
remain unchanged. One additional registration file records this exploratory pilot.

From `semantic-unlearning` on EC2, run:

```bash
bash scripts/run_static_overlap_mlp_pilot.sh \
  /home/ec2-user/models/Llama-3.2-3B-Instruct
```

The launcher expects the completed head protocol at
`outputs/static_overlap_final_protocol_seed1/protocol.json`. Its recorded source
bundle and completed development-head `manifest.json` must still exist. The MLP
runner refuses a model path different from the original base in that manifest.

The new output is `outputs/static_overlap_mlp_pilot_seed1`. Existing registration
or output files cause preparation to stop; rerunning the launcher is not another
attempt to select a checkpoint against the final results.

## Data and locality

- Original forget-training prompts and independently authored relation clozes and
  questions enter the forget loss. Only approved source aliases are used.
- Two different authored question families per forget association provide the
  development forget gate. They never enter gradients, layer selection, or replay.
- Original train and former-validation retention anchors are fitting-visible.
  They were already reclassified as development preservation in the head protocol.
  Newly authored development retention prompts are excluded from gradient fitting;
  a balanced subset supplies the denominator in layer selection.
- Add 300 preservation associations from the unused first-half MCF pool. Exclude
  existing associations, both final bundles, and the sampled official cases. Rare
  same-subject and same-answer controls receive priority. New preservation facts
  are divided between fitting and development; availability is reported rather
  than inventing missing overlap controls.
- Add 80 language excerpts from distinct articles in the **training** split of
  [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext). The training
  parquet is pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3` and its
  SHA-256 is verified before use. Only that file downloads, approximately a few MB,
  to `data/cache/mlp_wikitext2_train.parquet`. No WikiText validation/test file is
  fetched. Article selection is independent of model results. Source attribution
  and exact sampled text are retained in `pilot_data.json`; upstream licenses are
  CC BY-SA 3.0/GFDL. A predownloaded copy can be supplied to the preparation script
  with `--language-parquet` and must match the same checksum.
- The new corpus is needed because this checkout's legacy `data/wikidata` contains
  only ten unique sentences in 200 rows. Existing language supervision is retained;
  its known overlap with legacy official PPL is why official PPL remains skipped.

Official Gen and final bundle contents are read during a separate **exclusion
audit only**. They are not supplied to tokenization, localization, fitting, or
selection. Development text and token identities are checked for split collisions.
All facts and generated texts are frozen before loading the model.

Both lexical endpoints stay frozen **and retain their original sharing**. No
untying is necessary. At each candidate layer `[7, 11, 15, 19, 23]`, compute the
norm of the mean NLL gradient on a fixed balanced sample of 32 training forget
examples and 32 development preservation examples. Choose the largest ratio
`forget_norm / (retain_norm + 1e-12)`; ties follow registered layer order. Retain
NLL is used because KL at the original model has zero gradient. Sample identities
and all scores are saved in `localization.json`.

Only the selected `mlp.down_proj` gets a rank-16 additive update. All other
parameters are frozen. Native export checks SHA-256 fingerprints of **every**
parameter outside that projection against the original model, including embedding
and LM head, after reloading the saved checkpoint.

## Optimization and gates

The fixed pilot has 200 Adam steps at learning rate 0.003, eight forget and sixteen
preservation examples per step, gradient clipping at 1, and a relative MLP update
norm cap of 0.01. Hard replay uses only fitting losses, with rotating coverage
reported separately. No development-forget example enters replay.

For each fitting forget example, minimize
`relu(max(base_NLL + 2, -log(1e-6)) - edited_NLL)`. Preservation uses full-vocabulary
`KL(base || edit)`, weighted by `10 / 0.008`, plus a squared penalty on NLL increase
above the internal 0.04 boundary, normalized by 0.05 and weighted by 10. These
penalties guide fitting; they do not guarantee feasibility. Actual acceptance
uses measured per-anchor values, never penalty values or top-k approximations.

Every 20 steps, score **all** fitting and development examples. Select the first
scheduled checkpoint satisfying all four conditions:

1. Every fitting forget example reaches its bounded target.
2. Every development forget example reaches the same target.
3. Every fitting preservation anchor passes NLL increase <= 0.05 and KL <= 0.01.
4. Every development preservation anchor passes those same limits.

No numerical budget slack is allowed. Full-vocabulary base log probabilities are
cached on disk with a bounded CPU LRU; edited activations and probabilities are
recomputed. Intermediate iterates may violate preservation, and cannot be exported.
If no checkpoint qualifies, write `no_development_valid_edit`, exit 2, and stop
before final evaluation. Training has a 2700-second soft limit checked between
steps; startup, an in-progress gate, export, and final evaluation add time. This
is a bounded pilot, not a guaranteed runtime or guarantee of near-zero forgetting.

Before exporting a qualifying checkpoint, save its factors and manifest. Merge
and reload must pass strict actual-model preservation **and** forgetting, as well
as numerical logit parity (`atol=1e-4`, `rtol=1e-5`). A failed export gets no verified
success marker and the launcher stops. Diagnostic factors/reports remain available.

## Evaluation and outputs

Only after qualification does the launcher evaluate the same checkpoint on the
existing frozen retention set, separate evaluation bundle, and official MCF/Gen.
An independent exploratory binding permits one checkpoint; it does not bypass or
rewrite the head experiment's binding. A preservation failure is saved and the
official metrics are still measured on that same checkpoint. No further fitting
or replacement test is triggered by either result.

Files in the new output directory include:

- `pilot_protocol.json`, `pilot_data.json`, `encoded_development_examples.json`
- `localization.json`, `training_report.json`, `last_factors.pt`
- `checkpoint/` and `training_factors.pt` **only if the development gate passes**
- `exploratory_retention_results.json`, `evaluation_probability_v2.json`, and
  `exploratory_summary.json` **only after verified export**

The launcher exits zero only if preservation passes and official Eff/Gen each
have probability percentage below 0.005 and released accuracy exactly zero. That
criterion is not a claim of mathematically zero probability or universal erasure.

After a completed export, an interrupted evaluation can retry the **same**
checkpoint without refitting or recomputing already saved measurements:

```bash
python -u scripts/evaluate_static_overlap_mlp_pilot.py \
  --pilot-protocol outputs/static_overlap_mlp_pilot_seed1/pilot_protocol.json \
  --wikidata-dir data/wikidata --device cuda
```
