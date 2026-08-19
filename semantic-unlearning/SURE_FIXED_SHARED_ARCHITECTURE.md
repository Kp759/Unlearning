# SURE-LM fixed shared architecture: MCF and ZsRE

This document is the architecture contract for new cross-benchmark SURE-LM
experiments. It supersedes benchmark-specific Stage-2 objectives for claims
that MCF and ZsRE use the same architecture.

## Principle

MCF and ZsRE must use the same trainable parameterization, Stage-1 objective,
Stage-2 objective, direct residual criterion, rank schedule, scale-selection
rule, and optimization defaults. Benchmark adapters may only identify the
sensitive answer and construct the locked direct-only training view.

## Paper source of truth

For cross-benchmark paper claims, the only supported two-stage path is:

```text
sure_stage1_context_shared.py
    -> sure_stage2_context_shared.py
```

invoked by:

```bash
bash scripts/run_mcf_sure_fixed_shared.sh MODEL_PATH [MCF_JSON]
bash scripts/run_zsre_sure_fixed_shared.sh MODEL_PATH [ZSRE_JSON]
```

MCF-only residual/guarded Stage-2 scripts are ablations and must not replace
this shared paper path. In particular, `sure_stage2_sparse_repair_residual.py`
and `sure_stage2_sparse_repair_guarded.py` are diagnostic MCF variants, not the
architecture used to claim MCF/ZsRE parity.

## Sensitive-answer adapters

| Dataset | Locked sensitive field | Meaning |
|---|---|---|
| MCF target-true-sensitive | `target_new` | adapter maps ORIGINAL `target_true` into this canonical sensitive slot |
| ZsRE | `target_true` | original factual/sensitive answer |

ZsRE intentionally exposes no `target_new`, `Unknown`, IDK, neutral, or other
replacement answer during training. Therefore the fixed shared architecture
must not contain a single-reference-answer CE/GD term that only MCF can use.

## Shared trainable architecture

For both datasets:

```text
frozen transformer
frozen Base input embeddings
untied frozen LM head except selected sensitive rows
row-specific direct-forget-context basis for every editable sensitive row
```

For sensitive output row `t`, the learned update is

```text
Delta w_t = a_t B_t
```

where `B_t` is an orthonormal basis obtained only from direct training-visible
forget hidden states for teacher-forced occurrences of sensitive token `t`.
No paraphrase/rephrase, neighborhood/locality, benchmark retain, or PPL input
is used to build `B_t`.

## Shared Stage 1 objective

For every teacher-forced sensitive token `y_s`:

```text
L_GA = mean(log p_theta(y_s | x, y_<t))
L_GD = KL(p_base(-y_s) || p_theta(-y_s))
L_1  = lambda_GA L_GA + lambda_GD L_GD + lambda_2 ||Delta W||^2
```

The sensitive token is removed from both distributions for GD and all remaining
vocabulary probabilities are renormalized. Thus GD protects the complete
non-sensitive vocabulary distribution in both MCF and ZsRE. MCF's
counterfactual `target_new` (original unswapped field) is one member of this
non-sensitive distribution, but it is not given a benchmark-specific CE term.

Shared defaults:

```text
steps = 600
batch = 1
cache batch = 8
lr = 1e-4
lambda_GA = 2.0
lambda_GD = 1.0
Stage-1 context rank cap = 2
Stage-1 delta L2 = 0
```

## Shared direct residual criterion

A single top-1 criterion is too weak: a sensitive token can lose top-1 while
still retaining high absolute probability and while a multi-token sensitive
answer remains strongly preferred. Therefore fixed-shared v2 requires BOTH of
the following on every direct teacher-forced sensitive PredictionCase.

### 1. Best-other logit margin

```text
m_logit = max_{v != y_s} logit(v) - logit(y_s)
satisfied iff m_logit >= rho
```

Default:

```text
rho = 0.05
```

### 2. Frozen-Base sensitive-NLL increase

```text
Delta_NLL_s = NLL_theta(y_s | x, y_<t) - NLL_base(y_s | x, y_<t)
satisfied iff Delta_NLL_s >= tau
```

Default:

```text
tau = 4.0 nats
```

`tau = 4` means the sensitive-token probability must be reduced by at least a
factor of `exp(4) ~= 54.6` relative to the frozen Base model:

```text
p_theta(y_s | context) <= exp(-4) * p_base(y_s | context)
```

Both constraints need only the sensitive answer and the frozen Base teacher, so
they apply identically to MCF and ZsRE. They are used only for direct residual
detection, early stopping, candidate comparison, and scale selection. They are
not extra training-loss terms.

## Shared Stage 2 objective

Stage 2 uses the same GA/GD objective as Stage 1, restricted to residual direct
sensitive PredictionCases:

```text
L_2 = lambda_GA L_GA + lambda_GD L_GD + lambda_2 ||Delta W||^2
```

There is no hinge/ReLU loss and no MCF-only pairwise NLL objective in the fixed
shared architecture.

Shared Stage-2 defaults:

```text
candidate row-specific context ranks = 2 -> 8 -> 0
0 = full observed numerical forget-context rank
repair steps = 800
repair lr = 5e-3
lambda_GA = 2.0
lambda_GD = 1.0
lambda_2 = 1e-6
batch = 8
check every = 25
```

Candidate selection and scale selection use only the two shared direct
constraints above. The first rank reaching zero direct failures is sufficient;
otherwise select by `(direct_failures, candidate_order, delta_norm)`. Then
choose the smallest candidate scale preserving zero direct failures, otherwise
minimize `(direct_failures, scale)`.

## Benchmark evaluation remains benchmark-specific

Architecture equality does not require final benchmark metrics to be identical.
After freezing the checkpoint:

* MCF reopens the ORIGINAL UNSWAPPED MCF source. Paper reporting may include
  target-true-sensitive FS/GFS, sensitive NLL change, neighborhood specificity,
  and locked-corpus PPL.
* ZsRE reopens the original ZsRE source and uses its official direct/rephrase/
  locality/retain evaluator plus the same locked-corpus PPL fixture.

Final evaluation metrics never enter optimization or selection.

## Runners

```bash
bash scripts/run_mcf_sure_fixed_shared.sh MODEL_PATH [MCF_JSON]
bash scripts/run_zsre_sure_fixed_shared.sh MODEL_PATH [ZSRE_JSON]
```

Both runners intentionally expose the same shared hyperparameter environment
variables and defaults, including:

```text
SURE_SHARED_CONSTRAINT_MARGIN=0.05
SURE_MIN_SENSITIVE_NLL_INCREASE=4.0
```

## Historical variants

`sure_stage2_sparse_repair.py` is a historical canonical-v1 implementation. It
uses different benchmark-specific residual losses (MCF pairwise sequence-NLL
hinge versus ZsRE sensitive-vs-best-other hinge). Results from that path remain
valid descriptions of their checkpoints, but they must not be used to claim an
identical Stage-2 objective across MCF and ZsRE.

`sure_stage2_sparse_repair_residual.py` and
`sure_stage2_sparse_repair_guarded.py` are MCF-only diagnostic ablations created
to study locality versus residual repair. Their checkpoints may be reported as
ablations, but they are not the cross-benchmark SURE architecture.

The first fixed-shared context run used only the top-1 logit-margin residual
criterion. Its MCF seed-1 result achieved strong locality but insufficient
forgetting (`Eff-Pref=82`, `Gen-Pref=83`, `Delta Sensitive NLL direct ~= 0.965`).
That run is an ablation motivating the v2 Base-relative sensitive-NLL condition.
