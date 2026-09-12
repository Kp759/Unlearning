# ZsRE Seed-1 Fact-Association Transfer

This branch tests the successful MCF residual-bank architecture on ZsRE before
any MQuAKE work or multi-seed sweep.

## Registered first run

- Dataset: ZsRE MEND evaluation set.
- Seed: 1.
- Forget facts: 50 sampled from the official second-half forget pool.
- Final retain evaluation: 1000 sampled from the first-half retain pool.
- Model: Llama-3.2-3B-Instruct.
- Intervention layer: 19.
- Trainable state: exactly 50 independent fact-specific residual vectors.
- Backbone / embeddings / LM head: frozen.
- Tokenizer: unchanged.
- No private fact token or external fact-ID injection.
- No target_new / Unknown / replacement target in training.
- No official rephrase, locality prompt, retain fact, or PPL text in training.

The split is produced by the existing locked builder:

    scripts/build_zsre_zerounlearn_locked_no_neutral_split.py

The final evaluator reopens the unchanged original ZsRE file only after the
checkpoint is frozen.

## Architecture

The runtime architecture is the same V1 mechanism that succeeded on MCF:

    natural request
        -> frozen Llama through layer 19
        -> exact subject eligibility
        -> one candidate fact vector
        -> add vector at ORIGINAL request-boundary token
        -> frozen remaining layers
        -> frozen LM head

For unique subjects, V1 directly activates the owning vector. If more than one
forget fact shares a subject, the closest frozen direct-request hidden key is
used to select the association.

The object/answer is suppression supervision and is not a runtime trigger.

## ZsRE-specific execution requirement

ZsRE evaluates every token of a multi-token answer using progressively longer
teacher-forced answer prefixes. The association edit must not move to those
later answer-prefix tokens.

For every rewrite or held-out rephrase:

    route = route(original request only)
    edit position = final token of original request
    score token t using original request + answer tokens < t

The final evaluator verifies that the original request tokenization is an exact
prefix of every teacher-forced prediction context and aborts rather than
silently moving the intervention boundary.

## Training objective

ZsRE Eff/Gen are sensitive-answer TOKEN greedy accuracies. The locked direct
training objective therefore suppresses the worst sensitive answer token:

    max_t p(y_t | x, y_<t) < 1e-6

for every one of the 50 direct requests.

For Llama-3.2-3B's vocabulary this is strictly below 1 / |V|, so a sensitive
target token below 1e-6 cannot remain top-1.

No abstention or replacement completion is optimized after feasibility; a
feasible row is left frozen.

## Official metrics

The existing ZeroUnlearn-compatible ZsRE definitions are preserved:

- Eff: 100 * case-macro target_true token top-1 accuracy on direct rewrites.
  Lower is better.
- Gen: 100 * case-macro target_true token top-1 accuracy on official held-out
  rephrases. Lower is better.
- Spe: 100 * case-macro locality/neighborhood target-token top-1 accuracy.
  Higher is better.

Route activation is reported independently for rewrite, paraphrase, and
neighborhood token decisions. This separates a router miss from a vector that
fires but fails to suppress.

## Corrected utility

The historical whole-sequence PPL is retained only as legacy output because a
last-position-only intervention is excluded from the logits it scores.

The evaluator also reports runtime-aligned prefix-recompute PPL and the number
of association routes that fired during that PPL text.

## Run order

1. Build/preflight locked seed-1 data and route:

       MODEL_PATH="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
       ZSRE="$PWD/data/zsre_mend_eval.json"
       SPLIT="$PWD/outputs/zsre_fact_assoc_seed1_locked_split"
       RUN="$PWD/outputs/zsre_fact_assoc_seed1"

       bash scripts/run_zsre_fact_association_embeddings_seed1.sh \
         "$MODEL_PATH" "$ZSRE" "$SPLIT" \
         "$PWD/outputs/zsre_fact_assoc_seed1_preflight" \
         --preflight-only

2. If direct route recall is 1.0, train:

       bash scripts/run_zsre_fact_association_embeddings_seed1.sh \
         "$MODEL_PATH" "$ZSRE" "$SPLIT" "$RUN"

3. Freeze the artifact and perform one final official evaluation:

       bash scripts/evaluate_zsre_fact_association_embeddings_official.sh \
         "$RUN" "$ZSRE" "$PWD/data/wikidata" bfloat16

Do not inspect official rephrases/locality prompts before step 3.

## Decision after seed 1

If Eff is low but Gen is high and paraphrase route activation is low, routing
coverage is the bottleneck.

If paraphrase routing is high but Gen remains high, the one-position vector
does not generalize sufficiently and the intervention/training mechanism is the
bottleneck.

If Eff/Gen are low but Spe drops, V1 subject-only locality is the bottleneck.

Only after this seed-1 diagnosis should the method be frozen for MQuAKE and then
the 10-seed MCF/ZsRE/MQuAKE sweep.
