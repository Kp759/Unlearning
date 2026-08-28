# Context-Composed Sparse Embedding Writing

This experiment tests whether globally shared subject-token embedding rows can
form a context-selective marker after passing through a completely frozen
Transformer.

```text
sparse subject embedding-row deltas
    -> frozen Transformer
    -> context-composed marker v_i
    -> distributional reader q_i^T h
    -> sparse sensitive LM-head rows W_y - sum_i beta_i q_i
```

It uses no router, exact-name inference gate, sidecar, constant logit bias,
LoRA, adapter, or trained Transformer parameter. Both edits are materialized as
ordinary Hugging Face weights.

## Why this differs from the old marker writer

The old writer selected a marker and trained it using one direct prompt. The
measured `Gen=64` result was consistent with that design: routing fired on all
official paraphrases, but the marker amplitude did not transfer across their
relation rewording and arbitrary unrelated prefixes.

The new method changes four causal components:

1. **Multi-context positives.** Every record uses the direct prompt,
   hand-authored relation-specific alternate templates, arbitrary prefixes
   sampled from Wikipedia documents `[20:]`, and optionally semantically
   validated local-LLM surrogates. Official CounterFact paraphrases are absent.
2. **Compositional hard negatives.** Other subjects sharing edited BPE rows,
   leave-one-component-out subjects, strict subject fragments, and unrelated
   subjects must not write the marker.
3. **Contrastive marker selection.** A generalized eigenproblem chooses the
   direction with high reachable energy across positive contexts and low
   reachable energy across collision contexts.
4. **Distributional reader.** `q` is fitted against all training-safe positive
   and negative states. V2 explicitly projects it into the nullspace of the
   compositional controls, shared-answer collisions, and reference-answer
   states before maximizing the worst positive projection. This addresses the
   first run's decisive result: preserving `cos(v,q) ~= 1` left base-state
   `q`-vs-`h` leakage as high as `kappa=158`.

Stage 2 jointly initializes all `beta_i` values from the complete
instance-by-reader response matrix, including shared answer rows, and then
optimizes only those scalar coefficients against the exact full-answer MCF NLL
margin. This replaces independent initialization, which produced very large
interfering betas and a worst margin of `-193` in the first run.
The resulting deltas touch only sensitive LM-head rows.

## Data firewall

Training requires the direct-only artifact produced by
`build_mcf_sure_target_aware_direct_split.py`. The method process cannot follow
a path to the original dataset and validates any surrogate receipt for:

- zero official paraphrase access;
- zero official neighborhood access;
- zero benchmark-retain access;
- zero official-PPL access.

Official evaluation starts in a separate process only after the checkpoint has
been materialized and saved.

## Pre-Stage-2 gate

For each record:

```text
S_min = min positive |q^T h|
L_max = max negative |q^T h|
kappa_train = L_max / S_min
R = min positive |q^T h| / max positive |q^T h|
```

The registered criteria are:

- worst positive marker amplitude at least `0.75 * alpha`;
- writer marker leakage ratio at most `0.10` on compositional negatives;
- consistent positive sign;
- `kappa_train <= 0.10`;
- `R >= 0.50`;
- `abs(cos(v, q))` is reported but has no positive lower bound in v2. The
  negative-nullspace locality constraint is load-bearing; forcing the reader
  to retain the original marker direction reproduced the measured leakage.

The seed-1 launcher uses `--gate-policy report` because this is the first
falsification run: it records the predeclared gate but still permits a standard
checkpoint and held-out evaluation. Confirmatory runs should switch to
`--gate-policy strict` after the configuration is frozen.

## Wulver seed-1 run

From the repository root:

```bash
export MODEL_PATH=/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean
export WIKIDATA_DIR=/scratch/yl258/kp759/datasets/wikipedia_sure_50020
export OUTPUT_DIR=outputs/mcf_compositional_marker_v2_seed1_3b

bash scripts/submit_mcf_compositional_marker_seed1.sh
```

To reuse a previously validated seed-1 v7 surrogate artifact rather than
regenerating it, set `SURROGATE_ARTIFACT` to that JSON file. The learner still
revalidates its seed, cases, subjects, direct prompts, answer guard, semantic
receipt, and zero-probe-access declaration.

By default the job first builds the high-precision v7 semantic-surrogate
artifact. For a faster structural-only ablation:

```bash
BUILD_SEMANTIC_SURROGATES=0 \
OUTPUT_DIR=outputs/mcf_compositional_marker_seed1_structural \
bash scripts/submit_mcf_compositional_marker_seed1.sh
```

Outputs:

- `method/context_manifest.json`: every training-visible positive/negative and
  its provenance;
- `method/reader_gate_report.json`: per-record `kappa`, portability, cosine,
  and gate result;
- `method/compositional_marker_summary.json`: architectural invariants and
  Stage-2 acceptance;
- `method/post_reload_acceptance.json`: fresh-process, native-dtype replay of
  every direct and training-safe positive margin after checkpoint reload;
- `official_eval.json`: held-out Eff/Gen/Spe/PPL;
- `base_official_eval.json`: matched Base metrics.

The target is `Eff=0`, `Gen=0`, negligible `Delta Spe`, and negligible
`Delta PPL`. Only the first two are forced on training-safe contexts; unseen
official `Gen`, locality, and utility remain empirical outcomes.
