# MQuAKE seed-1 fact-association transfer

This branch transfers the already-successful MCF/ZsRE fact-association residual
bank to **MQuAKE-CF-3k-v2** without changing the core architecture.

## Locked protocol

- seed: 1
- sampling unit: MQuAKE instance
- forget pool: second half of the 3000-instance source file
- retain pool: first half
- sample forget first, then retain from one seeded Python RNG
- forget: 50 instances
- retain: 1000 instances, evaluation-only
- flatten `requested_rewrite` only after instance sampling
- one trainable residual vector per flattened atomic forget fact
- layer 19
- frozen transformer, input embeddings, and LM head
- unchanged tokenizer
- no private fact token / external fact-ID injection
- no `target_new`, `Unknown`, atomic question, multi-hop question, retain record,
  or PPL text visible to training

Because one MQuAKE instance may contain multiple rewrites, **50 forget instances
do not imply 50 trainable vectors**. The bank size is the number of flattened
atomic facts belonging to those 50 instances.

## Training metric alignment

Native ZeroUnlearn-compatible MQuAKE Eff is teacher-forced sensitive
`target_true` token argmax accuracy on the direct atomic rewrites.

Training reconstructs the exact same per-token direct contexts and fixes the
residual intervention at the original direct-request boundary while answer
prefix tokens are appended. Each atomic fact is optimized until its maximum
sensitive-token probability is below `1e-6`.

The default budget uses 30 row updates per atomic fact, matching the per-vector
budget used by the 1500-step / 50-vector MCF and ZsRE experiments.

## Evaluation

Primary:
- `Eff`: direct atomic sensitive-token accuracy; lower is better.

Extension, evaluation-only:
- `AtomicGen`: held-out natural-language atomic-question sensitive-token
  accuracy; lower is better for forgetting/generalization, but it is not a
  native ZeroUnlearn MQuAKE table column.

Utility:
- retain direct Eff-style accuracy
- retain AtomicGen
- legacy ZeroUnlearn-style PPL
- runtime-aligned PPL with route-activity audit

Multi-hop questions remain untouched by training and are not used for checkpoint
selection. They can be evaluated only after this atomic checkpoint is frozen.

## Commands

```bash
git checkout fact_association_mquake_seed1
```

```bash
MODEL_PATH="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
MQUAKE="$PWD/data/MQuAKE-CF-3k-v2.json"
SPLIT="$PWD/outputs/mquake_fact_assoc_seed1_locked_split"
RUN="$PWD/outputs/mquake_fact_assoc_seed1"
```

Preflight only:

```bash
bash scripts/run_mquake_fact_association_embeddings_seed1.sh \
  "$MODEL_PATH" "$MQUAKE" "$SPLIT" "$RUN" \
  --preflight-only
```

Use a fresh output directory for the real training run because the runner
intentionally refuses to overwrite:

```bash
RUN="$PWD/outputs/mquake_fact_assoc_seed1_train"
bash scripts/run_mquake_fact_association_embeddings_seed1.sh \
  "$MODEL_PATH" "$MQUAKE" "$SPLIT" "$RUN"
```

After training succeeds:

```bash
bash scripts/evaluate_mquake_fact_association_embeddings_official.sh \
  "$RUN" "$MQUAKE" "$PWD/data/wikidata" bfloat16
```
