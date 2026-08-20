# Token-conditioned SURE-LM with exact constrained Stage 2

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
              DONE          freeze exact Stage-1 logits      │
                                      │                 retry Stage 1
                           partition direct token cases
                            /                       \
                  repair set A                 protected set P
                (Stage-1 failures)            (Stage-1 successes)
                            \                       /
                           constrained Stage 2 rank 2
                                      │
                         residual rows editable only
                    exact cached NLL/margin constraints
                    for every case in A and P
                    + exact train-pool Wikipedia constraints
                    + total-delta norm constraint
                    + minimum Wikipedia KL/residual norm objective
                                      │
                    checkpoint-dtype materialization
                    + held-out Wikipedia guard
                               /                \
                            pass                 fail
                             │                    │
                         candidate         expand 2 -> 4
                                                  │
                                  add direct-context capacity
                                                   │
                                    select safest feasible rank
                                      or report INFEASIBLE
```

The bounded direct loss is

```text
relu(required_margin - direct_margin)^2
+ relu(required_sensitive_NLL_increase - observed_increase)^2.
```

It is a constraint-gated form of sensitive GA: once both direct constraints
pass, that example contributes zero suppression gradient. This prevents the
unbounded sensitive-row drift seen with raw GA.

After Stage 1 is materialized, every direct token case is assigned once to the
repair set `A` or protected set `P`. Stage 2 must repair all of `A` without
regressing any case in `P`. For protected case `i`, its NLL floor is

```text
max(required_sensitive_NLL_increase,
    Stage1_sensitive_NLL_increase_i - protection_tolerance).
```

The default tolerance is `0.05`. These are hard inequalities, not a weighted
protection loss. Stage 2 may not exchange a protected-case regression for an
active-case repair. Before checkpoint-dtype verification, the solver targets
the stricter protected floor

```text
max(required_sensitive_NLL_increase + global_constraint_buffer,
    protected_behavioral_floor_i + protected_materialization_buffer).
```

The global buffer is `0.05`; the separate protected materialization buffer is
only `0.005`. The latter supplies numerical headroom for BF16 serialization
without changing the official `4.0` requirement or cancelling the protected
`Stage1 - 0.05` behavioral allowance. Materialized feasibility is still checked
against the original unbuffered behavioral floor. The constraints catch
cross-row softmax effects: editing one sensitive row can change the NLL of an
unedited sensitive row by changing the shared softmax denominator.

Because the transformer and input embeddings are frozen and Stage 2 edits only
sparse LM-head rows, Stage-1 logits plus cached direct hidden states determine
the Stage-2 logits exactly. For direct context `i` and edited row `s`, the
residual logit shift is `d_is = h_i^T delta_w_s`. The cached Base partition and
selected-row probabilities then give the exact full-vocabulary NLL change;
the best unedited and edited competing logits give the exact margin. Every
optimization evaluation therefore covers all repair and protected cases
without a transformer forward.

Stage 2 solves

```text
min  mean exact Wikipedia KL(Base || Stage1 + residual)
     + lambda * ||residual||^2

subject to
  NLL increase_i >= required NLL       for every i in A
  margin_i       >= required margin    for every i in A
  NLL increase_j >= max(required NLL,
                         Stage1 NLL increase_j - epsilon) for every j in P
  margin_j       >= required margin    for every j in P
  train Wikipedia mean/p95/max KL <= locked budgets
  ||Stage1 + residual|| <= locked norm budget.
```

The implementation uses SLSQP with analytic Torch gradients, deterministic
zero and repair-directed starts, and retains the lowest-utility feasible
iterate even if a later solver step exits outside a boundary. A candidate is
eligible only when both the continuous solve and an actual checkpoint-dtype
materialization pass. No null-space projection, repair/protection weighting,
or Stage-2 residual scale frontier is used.

Rank 2 uses repair-context directions first. The rank-4 fallback augments the
generalized covariance with all training-visible direct constraint contexts,
which supplies context-selective capacity for shared-row conflicts while
keeping preservation as explicit inequalities rather than a projection.

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
and improves the direct shortfall. Stage 1 retains its shrink-only scale
frontier capped at `1.0`. Stage 2 instead solves its residual coefficients
directly at ranks 2 and 4. Among fully feasible materialized rank candidates,
selection minimizes held-out Wikipedia mean, p95, and maximum KL, followed by
total delta norm. Unsafe or continuously infeasible candidates are never used
as final checkpoints.

Official benchmark retain examples, replacement/reference answers,
paraphrases, neighborhood/locality prompts, and PPL texts are never visible to
training or checkpoint selection. Official FS/GFS, Spe, retain metrics, exact
benchmark-retain KL, and PPL remain post-training audits.

Zero internal direct failures do not imply official MCF FS/GFS of 100. The
benchmark-neutral learner never sees MCF `target_new` or official paraphrases,
so FS/GFS alignment remains a post-training scientific question rather than a
training-time guarantee.

For experiments that explicitly require direct FS = 100, use the separately
labeled target-aware ablation in `SURE_MCF_FS100_ARCHITECTURE.md`. It exposes
MCF `target_new`, imposes exact mean sequence-NLL inequalities, and emits a
checkpoint only after the official BF16 direct scorer reports 50/50. It is not
the benchmark-neutral method and does not guarantee GFS or specificity.

## Dataset adapter contract

A new dataset reuses the learner by generating the same canonical files as
`build_sure_minimal_split.py`. Its manifest must contain:

```json
{
  "protocol": "sure_exact_constrained_residual_stage2_v5_1",
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

If Stage 2 is infeasible, inspect its solver and checkpoint-dtype diagnostics
without retraining or opening any held-out benchmark data:

```bash
RUN=outputs/mcf_sure_exact_constrained_stage2_v5_1/seed1/learner

python - "$RUN" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
for name in (
    "stage2_attempts.json",
    "stage2_rank2_materialized_report.json",
    "stage2_rank4_materialized_report.json",
    "stage2_infeasible.json",
):
    path = run / name
    if path.exists():
        print(f"\n{name}\n{'=' * len(name)}")
        print(json.dumps(json.load(path.open()), indent=2))
PY
```

Per-rank `stage2_rank*_solver_history.json` files contain every inspected
iterate and its active, protected, utility, and norm slacks. Per-rank basis
reports record requested versus actual capacity. `stage2_infeasible.json`
preserves the safe Stage-1 checkpoint and records why no rank was admissible.

Both runners reuse the same model-specific Wikipedia cache. The requested
utility corpus is 100,000 documents and the exact-KL candidate reservoir is
100,000 predictor states. Each dataset run derives disjoint token-conditioned
train and guard pools with the same locked algorithm. When the local Wikipedia
artifact contains fewer eligible documents, the cache can fill the reservoir
from multiple predictor positions per document, but the run remains a pilot
because document diversity is still below 100,000.
