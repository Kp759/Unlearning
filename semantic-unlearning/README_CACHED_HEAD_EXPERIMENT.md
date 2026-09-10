# Cached head regression experiment

This is a fresh alternative to the constrained embedding/MLP/head optimizer.
It changes only the native LM-head rows corresponding to forgotten answer tokens.
The transformer and embeddings remain fixed, so final hidden states can be
extracted once. The fitting grid then needs small matrix operations rather than
repeated transformer passes. It uses unquantized FP32 weights and an independent
head. For a base checkpoint with tied embedding/head weights, explicitly pass
`--allow-untied-head`. The script copies the existing head, verifies exact base
logits on a training probe, freezes the original embedding, and records the
preparation in the manifest. The native export has `tie_word_embeddings=False`.
This changes parameter sharing, not the initial model function. A 128256 x 3072
FP32 head copy needs about 1.47 GiB of additional weight memory and disk storage.
Without the flag, a tied model is rejected before fitting.

This is an experiment, not a guarantee of near-zero official Eff/Gen. A successful
training fit establishes suppression on those training contexts. The transformer
still contains its original representations. Official paraphrases and independent
utility measurements are needed to assess the resulting checkpoint.

## Method and checks

For training forget hidden states F, training protected hidden states R, and
desired answer-row logit changes T, the solver minimizes

`||F D - T||^2 + ridge * tr(D^T (I + R^T R / tau) D)`.

T is -1 at each true answer token and zero at the other selected rows. A declared
finite grid scales these solutions. The `tau=0` family instead uses the numerical
nullspace of R. SVD and linear solves use FP64; deployed factors are FP32. The
head rank can reach the number of selected vocabulary rows, unlike the previous
rank-8 endpoint edit. No embedding or MLP localization is performed.

The cache stores the hidden states, original selected-row log probabilities, and
the total probability mass of all unmodified vocabulary rows. For row-logit
changes d, the full-vocabulary quantities are

```
log_Z_ratio = log(mass_unmodified + sum_selected(p_base * exp(d)))
NLL_edit = NLL_base + log_Z_ratio - d_true
KL(base || edit) = log_Z_ratio - sum_selected(p_base * d)
```

The implementation computes these in log space. KL is not restricted to top-k
tokens. Cached predictions are checked against actual model passes before any
export. The existing exporter then checks native merge/reload parity and
retention, and the existing evaluator reads the verified ordinary checkpoint.

The linear algebra uses [PyTorch SVD](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd)
and [linear solves](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html).

Default internal training limits are 0.04 NLL increase and 0.008 KL. Scientific
validation limits remain 0.05 and 0.01. `--nll-safety-margin` and
`--kl-safety-margin` only configure the stricter training limits. The previously
implemented 5e-6 FP32 export-only rounding allowance is unchanged.

Two fixed context prefixes augment both forgotten and retained training facts.
They are context variations, not a claim of semantic paraphrase coverage. The
validation rows remain unchanged. No MCF file or official Gen prompts are read.
Validation retention selects among the finite candidate grid; validation forget
performance does not influence the solve or selection. Consequently validation
retention is model-selection evidence, not an untouched final test.

Only candidates improving the training worst/mean probability ranking and passing
both retention checks are eligible. Zero eligible candidates means exit code 2
and no native checkpoint. A training-optimal but invalid delta is saved separately
for diagnosis and never marked as a valid edit. There is no automatic fallback
that changes research thresholds. Even a verified checkpoint can miss the
forgetting target; inspect the actual metrics.

## Run on EC2

Use the original training bundle and original model, not prior edited factors.
The script adds its own training context variants; the retention augmentation
command from the previous experiment is unnecessary.

```bash
cd /home/ec2-user/workspace/Unlearning-static-overlap/semantic-unlearning
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export TRAIN_BUNDLE="$PWD/data/static_overlap_mcf_seed1_train.json"
export HEAD_OUT="$PWD/outputs/static_overlap_cached_head_$(date +%Y%m%d_%H%M%S)"
set -o pipefail

python -u scripts/run_static_overlap_cached_head.py \
  --model-path "$MODEL_PATH" \
  --training-bundle "$TRAIN_BUNDLE" \
  --output-dir "$HEAD_OUT" \
  --device cuda \
  --allow-untied-head \
  --local-files-only \
  2>&1 | tee "$HEAD_OUT.log"
```

Progress phases are `cache_hidden_states` (every 25 distinct sequences),
`solve_retention_metric`, `cached_candidate`, `verify_selected_on_real_model`,
and `export_merge_reload_verification`. The default grid has 90 candidates.
Actual EC2 timing and Llama forgetting results have not been established by the
local tests. Feature extraction and final native verification still run the
transformer. `--training-only` saves verified factors without native export.

If `checkpoint/static_edit_export.json` exists and is verified, evaluate once:

```bash
python -u scripts/evaluate_static_overlap_edit.py \
  --checkpoint "$HEAD_OUT/checkpoint" \
  --base-model "$MODEL_PATH" \
  --evaluation-bundle data/static_overlap_mcf_seed1_eval.json \
  --out "$HEAD_OUT/evaluation_probability_v2.json" \
  --device cuda --max-new-tokens 1 \
  --mcf-path data/multi_counterfact.json --wikidata-dir data/wikidata \
  --seed 1 --unlearn-num 50 --retain-num 1000 \
  --skip-official-ppl --require-zero \
  2>&1 | tee "$HEAD_OUT.evaluation.log"
```

The zero check uses unrounded probability percentages below 0.005 and zero
released accuracy for both Eff and Gen. It does not claim exact zero probability.
The one-token generation setting does not shorten teacher-forced answer scoring;
it limits the separate generation diagnostic.

If native export fails, valid saved factors can be recovered without refitting:

```bash
python scripts/export_static_overlap_edit.py \
  --training-run "$HEAD_OUT" --device cuda --deployment-dtype float32 \
  --local-files-only
```

Recovery repeats the recorded head separation on the original tied base without
requiring another opt-in flag. It uses the saved augmented bundle and refuses runs that
failed real-model retention/parity. Inspect `training_report.json`,
`candidates.jsonl`, and `prefix_conflicts.json` for diagnosis.
