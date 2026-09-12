# Fact-Association Embeddings V2: Runtime and Comparison Contract

## Status

V1 seed-1 official evaluation is frozen and retained as a development result:
Eff=12, Gen=14 under counterfact_sensitive_preference_legacy.
Seed 1 and its official failures have been inspected and are DEVELOPMENT ONLY
for V2. A future final evaluation must use a precommitted disjoint record set.

## Invariants shared by all V2 arms

- 50 independent fact-specific hidden-state vectors.
- Frozen base Llama.
- Layer 19.
- One intervention position: the original request-boundary token.
- Ordinary tokenizer IDs; no private fact token.
- Prompt-prefix-only routing.
- Teacher-forced answer suffixes cannot change routing.
- Uncached reference generation recomputes the growing sequence while reusing
  the original request boundary.
- Cached generation is intentionally unsupported until separately validated.
- Training uses only authored/canonical training-visible prompts. Official MCF
  paraphrase/neighborhood fields are not opened by the four-arm runner.

## Corrected utility

The historical whole-sequence PPL is preserved only as legacy_PPL.

For a boundary-scoped one-position edit, corrected PPL scores each target token
from its actual observed prefix and injects at the boundary whose logits predict
that target. It normalizes by T-1 scored next-token targets.

Run:

    bash scripts/audit_static_overlap_fact_association_ppl_runtime.sh \
      RUN_DIR \
      data/wikidata \
      bfloat16

## Primary metric contract

Primary MCF metrics match feat/static-overlap-constrained-editing via
summarize_probability_metrics():

- Eff = 100 * case-macro complete target_true answer probability on rewrites,
  where answer probability is exp(-sum answer-token NLL), no EOS added.
- Gen = the same complete target_true answer probability on paraphrases,
  averaging paraphrases within case and then cases.
- Spe = neighborhood all-target_true-token teacher-forced top-1 accuracy.
- TokenGeometricMean_Eff/Gen, ReleasedAccuracy_Eff/Gen, and
  SensitivePref_Eff/Gen remain separate diagnostics.

The matching display-zero acceptance is Eff < 0.005%, Gen < 0.005%, and
ReleasedAccuracy_Eff = ReleasedAccuracy_Gen = 0. This is not an exact-zero
probability claim.

## Four matched development arms

| Arm | Gate | Suppression objective | Role |
| --- | --- | --- | --- |
| A | V1 subject-first | absolute threshold | corrected V1 control |
| B | V1 subject-first | absolute + actual comparator margin | comparator-aware ablation |
| C | relation-sensitive prototype | absolute threshold | PRIMARY V2 |
| D | relation-sensitive prototype | absolute + actual comparator margin | comparator-aware V2 ablation |

A/B intentionally retain the V1 unique-subject bypass as a control. C/D never
bypass relation confirmation for a unique subject. Arm C is the main V2 design
because complete sensitive-answer probability, not target_new preference, is
the primary Eff/Gen objective.

The relation gate uses, for candidate fact i:

- u_i = max cosine(query, positive prototype)
- v_i = max cosine(query, negative prototype)
- d_i = u_i - v_i

and requires both u_i >= alpha_i and d_i >= tau_i.
Prototype construction and threshold calibration use disjoint training-visible
prompts. Authored development prompts remain held out.

## Comparator-aware constraint (B/D ablation only)

Let:

- a_j = mean teacher-forced NLL of sensitive target_true
- b_j = mean teacher-forced NLL of MCF target_new comparator
- A = -log(1e-6)
- default absolute buffer eta = 0.1
- default margin target m = 0.1

A view is feasible only if:

    a_j >= A + eta

and:

    a_j - b_j >= m

The proposal gradient may detach b_j, but candidate acceptance always
recomputes both branches after the update. A row does not lock until every
training view satisfies both constraints. After locking, abstention updates are
accepted only if both constraints stay feasible.

target_new is explicitly declared extra comparator supervision in B/D. It is
not required for primary Arm C training; A/C use comparator quantities only as
diagnostics.

## Run

    MODEL_PATH="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    MCF="$PWD/data/multi_counterfact.json"

    bash scripts/run_static_overlap_fact_association_embeddings_v2_arm.sh \
      A "$MODEL_PATH" "$MCF" "$PWD/outputs/fact_assoc_v2_arm_A_seed1"

Repeat separately for B/C/D. Do not start all four until the unit tests and the
V1 corrected-PPL audit pass.

## Required interpretation

Routing and intervention are separate measurements:

1. route recall on authored positives;
2. wrong-relation route activation on same-subject controls;
3. absolute sensitive-answer suppression;
4. actual comparator margin;
5. retained behavior / corrected runtime utility;
6. generated disclosure under fixed-boundary uncached generation.

A gate fire is not evidence of successful suppression; an inactive gate is not
evidence of vector incapacity.

## Final evaluation provenance

Do not treat a new random seed as automatically disjoint. Before final V2
evaluation, explicitly choose a record set with no case-ID overlap with any
inspected seed-1 forget records, freeze code/config/thresholds, and record the
case IDs before opening final paraphrases or utility controls.
