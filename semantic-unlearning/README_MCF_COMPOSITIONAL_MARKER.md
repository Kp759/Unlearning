# Context-Composed Sparse Embedding Writing

This experiment tests whether globally shared subject-token embedding rows can
form a context-selective marker after passing through a completely frozen
Transformer.

```text
sparse subject embedding-row deltas
    -> frozen Transformer
    -> context-composed marker v_i
    -> diagnostic record-level reader
    -> bounded writer-residual LM-head readers
    -> exact row-wise factorization Delta W_y = -beta_y q_y
```

It uses no router, exact-name inference gate, sidecar, constant logit bias,
LoRA, adapter, or trained Transformer parameter. Both edits are materialized as
ordinary Hugging Face weights.

## Why this differs from the old marker writer

The old writer selected a marker and trained it using one direct prompt. The
measured `Gen=64` result was consistent with that design: routing fired on all
official paraphrases, but the marker amplitude did not transfer across their
relation rewording and arbitrary unrelated prefixes.

Across V2-V5.1, the method changes six causal components:

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
4. **Joint sparse output solve (v3 diagnostic).** A record-level distributional `q` remains an
   explicit diagnostic, but v3 does not assume that one such direction can
   separate every answer-token state. It optimizes the selected sensitive
   LM-head rows jointly against all exact full-answer margins. Every learned
   row is then factorized without approximation as
   `Delta W_y = -beta_y q_y`, preserving a linear `q_y^T h` reader while
   allowing different sensitive tokens to use different feasible readers.
5. **Writer-contrastive nullspace output reader (v4 diagnostic).** Stage 2 projects every
   sparse output-row update away from the leading span of writer-off positive
   states, compositional negatives, and disjoint corpus states. It also
   constrains absolute target-new NLL drift directly, so forgetting must come from raising
   sensitive-target NLL rather than damaging the reference target.
6. **Bounded residual-basis reader (v5/v5.1).** Stage 1 is frozen exactly at the
   validated v3 writer. For each sensitive token `y`, Stage 2 forms paired
   residuals `h_writer - h_base`, removes writer-off/corpus and row-semantic
   negative geometry, and permits the LM-head row to move only in the leading
   rank-4 residual basis. Every serialized row must also satisfy one of the
   predeclared hard bounds `||Delta W_y|| / ||W_y|| <= {0.05,0.10,0.20,0.40}`.
   V5.1 corrects reference preservation to penalize only *increases* in
   target-new NLL; decreases caused by removing sensitive-token probability
   mass are desired and remain allowed.

V2 falsified the shared record-reader assumption: none of 50 readers passed
the portability gate, several training-positive responses crossed zero, and
the non-negative scalar solve correctly reported an instance with no positive
reader response. V3 removes that infeasible bottleneck; it does not disguise
the failed diagnostic as a passed gate. The resulting deltas still touch only
sensitive LM-head rows.

## What v3 established, and what it did not

The seed-1 v3 run achieved `Eff: 84 -> 0` and `Gen: 85 -> 14`. Its causal
ablation restored the original input rows while retaining the sparse output
rows; 45/50 direct cases and 516/567 training-safe positives then failed. The
embedding writer therefore made a causally necessary contribution to roughly
90% and 91% of those constraints, respectively.

V3 was not a successful final edit. Forget `Spe` fell by 1.76, retain
`Eff/Gen/Spe` fell by 6.90/5.85/2.58, and PPL rose by 30.83%. Those results
support the writer mechanism but reject the locality of the v3 output solve.

V3 also exposed a bookkeeping error in its reader certificate. The first
teacher-forced predictor state for target-true and target-new completions of
the same prompt is identical: neither answer token has been consumed yet.
Declaring the target-new state negative while declaring the target-true state
positive made `kappa <= 0.1` impossible for that position. This explains the
systematic `L_max == S_max` pattern, but it does **not** explain away the
observed PPL and retain damage. V4 removes this invalid hidden-state negative
and protects target-new by its measured NLL instead.

The seed-1 V4 run is also rejected. It improved held-out `Gen` from 14 to 10,
but its output-only PPL reached approximately `6.76e16` and combined PPL
approximately `8.67e16`; input-only PPL remained exactly 16.625. This localizes
the catastrophe to the 39 sparse LM-head rows, not the reused 234-row writer.
Do not run V4 on additional seeds.

The first bounded V5 run also failed closed: at the maximum `0.40` row cap it
retained 29/50 direct and 384/567 training-positive failures. Its cap was
respected exactly and official evaluation was correctly refused. However, V5
used a symmetric absolute target-new NLL constraint. Suppressing a likely
sensitive token lowers the softmax denominator and can improve target-new NLL
without changing its raw logit; V5 incorrectly penalized that improvement as
damage. V5.1 changes only this constraint to one-sided regression before any
rank increase is tested.

Before another reader architecture is trained, characterize the fixed-V3
uniform-beta frontier. This is exploratory because it reads official probes:

```bash
python -u scripts/sweep_mcf_compositional_beta_frontier.py \
  --model-dir outputs/mcf_compositional_marker_v3_seed1_3b/method/checkpoint \
  --base-model-path "$MODEL_PATH" \
  --state outputs/mcf_compositional_marker_v3_seed1_3b/method/compositional_marker_state.pt \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir "$WIKIDATA_DIR" \
  --out outputs/mcf_compositional_marker_v3_seed1_3b/comparison/beta_frontier.json \
  --seed 1 \
  --scales 0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1 \
  --ppl-limit-percent 5 \
  --dtype bf16 \
  --device-map single
```

The observed frontier rules out scalar shrinkage as the fix. Scale `0.3` was
the largest point within 5% PPL (`17.125` versus `16.625`), but it still gave
`Eff=34`, `Gen=51`; scale `1` was needed for `Eff=0`, with `Gen=14` and
`PPL=21.75`. V5/V5.1 therefore changes the reader directions and freezes its caps
using training-safe data; it does not select a beta scale from official probes.

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

## Reader diagnostics

For each record:

```text
S_min = min positive |q^T h|
L_max = max negative |q^T h|
kappa_train = L_max / S_min
R = min positive |q^T h| / max positive |q^T h|
```

The same criteria are reported twice:

- worst positive marker amplitude at least `0.75 * alpha`;
- writer marker leakage ratio at most `0.10` on compositional negatives;
- consistent positive sign;
- `kappa_train <= 0.10`;
- `R >= 0.50`;
- `abs(cos(v, q))` is reported but has no positive lower bound. The
  negative-nullspace locality constraint is load-bearing; forcing the reader
  to retain the original marker direction reproduced the measured leakage.

The record-level diagnostic is measured before Stage 2. The decisive output gate
is measured on the exact per-output-row readers obtained after the joint
sparse solve. `--gate-policy strict` requires that final gate; `report` records
failures while permitting the first held-out falsification run.

V5.1 also performs a causal writer ablation before saving: it keeps the learned
output rows, restores the original subject input rows, and replays every
training-safe margin. If the output-only ablation still succeeds, the run may
be an effective sparse output edit but it does not validate the claimed
context-composed embedding mechanism.

This is also enforced during the solve: Stage 2 caches the positive
teacher-forced states with the writer removed and penalizes every learned
output-row shift on those base states. The desired reader therefore responds
to the writer-induced contextual displacement, not merely to latent factual
geometry that was already present in the base model.

V5.1 replaces V4's failed unrestricted protected-nullspace search with stricter
mechanisms:

- each reader is confined to a small basis made from paired writer residuals,
  after common and token-row-specific protected geometry is removed;
- every row is projected after every optimizer step onto a relative norm ball,
  and the projection is checked in the BF16/FP16 arithmetic used by the saved
  checkpoint;
- target-new NLL may not *increase* beyond the registered tolerance. Improvements
  are allowed because they are an intended consequence of sensitive-token
  suppression. The same-prompt target-new hidden state is not reused as a
  contradictory negative.

No unconstrained fallback exists. If none of the four caps reaches zero
training-positive failures while satisfying the reference-NLL constraint, the
method exits before official evaluation and labels the checkpoint diagnostic.

The seed-1 launcher uses `--gate-policy report` because this is the first
falsification run: it records the predeclared gate but still permits a standard
checkpoint and held-out evaluation. Confirmatory runs should switch to
`--gate-policy strict` after the configuration is frozen.

## Wulver seed-1 run

From the repository root:

```bash
export MODEL_PATH=/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean
export WIKIDATA_DIR=/scratch/yl258/kp759/datasets/wikipedia_sure_50020
export OUTPUT_DIR=outputs/mcf_compositional_marker_v5_1_seed1_3b

bash scripts/submit_mcf_compositional_marker_seed1.sh
```

V5.1 deliberately reuses the validated v3 surrogate artifact and Stage-1 writer.
The launcher defaults to their standard paths. To relocate them, set
`SURROGATE_ARTIFACT` and `RESUME_STAGE1_STATE`. The learner still
revalidates its seed, cases, subjects, direct prompts, answer guard, semantic
receipt, and zero-probe-access declaration.

To reuse the already validated v3 sparse embedding writer while testing only
the corrected Stage 2, also set:

```bash
export RESUME_STAGE1_STATE="$PWD/outputs/mcf_compositional_marker_v3_seed1_3b/method/stage1_writer.pt"
```

The selected rows, tensor shape, marker map, and compatible protocol are
validated before the state is accepted. The new output directory remains
separate, so the v3 evidence is preserved. The launcher refuses to regenerate
either prerequisite silently because that would no longer be a Stage-2-only
experiment.

Outputs:

- `method/context_manifest.json`: every training-visible positive/negative and
  its provenance;
- `method/reader_gate_report.json`: per-record `kappa`, portability, cosine,
  and the diagnostic pre-Stage-2 result;
- `method/output_reader_gate_report.json`: per-sensitive-output-row reader
  factorization and final gate result;
- `method/causal_writer_ablation.json`: whether suppression survives after
  restoring the original sparse input rows;
- `method/compositional_marker_state.pt`: sparse deltas plus exact base/edited
  selected rows, sufficient for post-hoc input-only/output-only attribution;
- `method/compositional_marker_summary.json`: architectural invariants and
  Stage-2 acceptance;
- `method/post_reload_acceptance.json`: fresh-process, native-dtype replay of
  every direct and training-safe positive margin plus the serialized hard
  output-row cap after checkpoint reload;
- `official_eval.json`: held-out Eff/Gen/Spe/PPL;
- `base_official_eval.json`: matched Base metrics.
- `comparison/component_ppl_attribution.json`: combined, input-only,
  output-only, and exactly reconstructed-base PPL from one frozen checkpoint;
- `comparison/gen_failure_attribution.json`: post-hoc Gen failures stratified
  by robust-surrogate versus direct-only training coverage.

The target is `Eff=0`, `Gen=0`, negligible `Delta Spe`, and negligible
`Delta PPL`. Only the first two are forced on training-safe contexts; unseen
official `Gen`, locality, and utility remain empirical outcomes.
