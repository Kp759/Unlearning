# Token-conditioned guarded two-stage SURE-LM

`scripts/sure_minimal_two_stage.py` is one dataset-independent learner. Dataset
adapters only produce the canonical direct-forget JSON and declare its
sensitive-answer field in `learner_adapter_contract`. MCF and ZsRE source the
same immutable architecture defaults from
`scripts/sure_guarded_shared_defaults.sh`.

Every run writes `shared_architecture_sha256` in `architecture_lock.json` and
`config_used.json`. The signature covers all training, rank, scale, and guard
settings while deliberately excluding dataset name and seed. Cross-dataset
results should be compared only when this signature matches. A second
`cross_dataset_compatibility_sha256` also binds that architecture to the Base
model, tokenizer, and exact Wikipedia cache, and is the strongest equality
check for MCF/ZsRE comparisons.

```text
Frozen Base transformer + frozen input embeddings
                         │
                canonical direct forget set
                         │
              direct hidden states and logits
                         │
                         ├──────── C_F,s ───────────────┐
                         │                              │
disjoint Wikipedia utility documents                  │
  ├─ all predictor states -> C_U                       │
  └─ 100k predictor-state candidate reservoir          │
       + Base log Z                                    │
                         │                              │
                         └── full-space generalized ────┘
                              eigen basis C_U^-1 h
                                         │
                    Base p(s|u) for every edited row s
                                         │
                    fixed random split before selection
                              /                      \
          per-row top-p train contexts       per-row top-p guard contexts
                 + uniform anchors                  + uniform anchors
                                         │
                              Stage 1: rank 2 first
                                         │
                 bounded direct-constraint sensitive GA
                 + conditional same-prompt non-sensitive GD
                 + exact joint train-pool Wikipedia KL
                                         │
                       exact materialized scale frontier
                                         │
                    direct guards + held-out Wikipedia guards
                    /                 |                    \
         all direct pass      safe residual cases      no safe progress
         and utility safe             only                  │
                │                      │               expand 2 -> 4
              DONE          guarded Stage 2 rank 2           │
                                      │                 retry Stage 1
                         residual rows editable only
                         passed direct cases protected
                         KL computed on total Stage1+residual
                         residual scale may exceed 1.0
                         (Stage-1 scale never exceeds 1.0)
                                      │
                         materialized guarded frontier
                               /                \
                            pass                 fail
                             │                    │
                           DONE           expand 2 -> 4
                                                  │
                                          pass or INFEASIBLE
```

The bounded direct loss is

```text
relu(required_margin - direct_margin)^2
+ relu(required_sensitive_NLL_increase - observed_increase)^2.
```

It is a constraint-gated form of sensitive GA: once both direct constraints
pass, that example contributes zero suppression gradient. This prevents the
unbounded sensitive-row drift seen with raw GA.

The cache spreads its predictor-state reservoir across token positions rather
than keeping only one random position per document. This lets a capped pilot
with 180 documents still contribute many candidate contexts, while its limited
document diversity remains explicitly marked as a pilot.

After the sensitive token rows are known, the candidate reservoir is split
deterministically into disjoint halves. For every edited row `s`, each half
keeps the contexts with the largest Base `p(s|u)` plus fixed uniform anchors.
Only the train half contributes gradients; scale selection and final utility
guards use the unseen guard half.

For cached Wikipedia prompt state `h_u`, selected Base probabilities `p_us`,
and sparse shifts `d_us = h_u^T delta_w_s`, the differentiable utility loss is
the exact joint full-vocabulary KL:

```text
A_u = 1 - sum_s p_us + sum_s p_us exp(d_us)
KL(Base || Edited)_u = log(A_u) - sum_s p_us d_us.
```

Scale selection uses only the direct constraints and disjoint Wikipedia
utility statistics. A Stage-1 handoff is permitted only when it is utility-safe
and improves the direct shortfall. The final checkpoint must have zero direct
failures and pass fixed mean, p95, maximum-KL, and total-delta-norm guards.
Unsafe candidates are never materialized as final checkpoints.

Stage 1 has a shrink-only scale frontier capped at `1.0`. Stage 2 has its own
residual frontier up to `1.25`; values above `1.0` scale only the learned
residual and are accepted only after exact materialization passes every direct
and held-out Wikipedia guard.

Official benchmark retain examples, replacement/reference answers,
paraphrases, neighborhood/locality prompts, and PPL texts are never visible to
training or checkpoint selection. Official FS/GFS, Spe, retain metrics, exact
benchmark-retain KL, and PPL remain post-training audits.

Zero internal direct failures do not imply official MCF FS/GFS of 100. The
benchmark-neutral learner never sees MCF `target_new` or official paraphrases,
so FS/GFS alignment remains a post-training scientific question rather than a
training-time guarantee.

## Dataset adapter contract

A new dataset reuses the learner by generating the same canonical files as
`build_sure_minimal_split.py`. Its manifest must contain:

```json
{
  "protocol": "sure_token_conditioned_wikipedia_kl_two_stage_v3",
  "dataset": "adapter-name",
  "learner_adapter_contract": {
    "sensitive_answer_field": "target_true",
    "forbidden_answer_fields": ["target_new"]
  }
}
```

Each training record supplies only `requested_rewrite.prompt`,
`requested_rewrite.subject`, and the declared sensitive answer. No optimizer or
guard code changes are allowed for a new dataset.

## Running

```bash
bash scripts/run_mcf_sure_minimal.sh /path/to/model data/multi_counterfact.json
bash scripts/run_zsre_sure_minimal.sh /path/to/model data/zsre_mend_eval.json
```

Both runners reuse the same model-specific Wikipedia cache. The requested
utility corpus is 100,000 documents and the exact-KL candidate reservoir is
100,000 predictor states. Each dataset run derives disjoint token-conditioned
train and guard pools with the same locked algorithm. When the local Wikipedia
artifact contains fewer eligible documents, the cache can fill the reservoir
from multiple predictor positions per document, but the run remains a pilot
because document diversity is still below 100,000.
