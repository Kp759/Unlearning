# TOFU Semantic Active-Repair Objective

## Goal

Given a fresh TOFU base model and a stage-specific deletion set, produce a
checkpoint that satisfies all three requirements:

1. suppress every targeted sensitive answer;
2. redirect the model toward the abstention answer `Unknown`; and
3. preserve retain, real-author, world-fact, locality, and full-retain utility.

The train, validation, and final-apply stages each start from the same fresh base
model. A validation checkpoint is not reused for test authors. The frozen
algorithm is reapplied to the final-apply deletion requests before the locked
test prompts are opened.

## Editable parameters

The transformer and input embeddings remain frozen. The repair edits only
LM-head vocabulary rows that are:

- used by initially active sensitive answers, or
- used by the abstention answer `Unknown`,

minus rows that occur in protected retain or utility answers. The row deltas are
restricted to a low-rank basis from sensitive/abstention hidden states and are
projected away from protected utility hidden directions.

## Per-example quantities

For deletion request `i`, define:

- `NLL_s(i)`: mean token NLL of the sensitive answer;
- `NLL_u(i)`: mean token NLL of `Unknown`;
- `p_s(i) = exp(-NLL_s(i))`;
- `p_u(i) = exp(-NLL_u(i))`.

For utility example `j`, define `NLL_r(j)` and its untouched-reference value
`NLL_r_ref(j)`.

## Hard forgetting constraint

Every targeted answer must satisfy:

```text
p_s(i) <= tau_forget
```

Equivalently:

```text
NLL_s(i) >= -log(tau_forget)
```

The balanced candidate starts with `tau_forget = 1e-4`. Tighten to `2e-5`
only after semantic preference and utility gates pass reliably.

## Semantic redirection constraint

Every request must prefer abstention by a positive log-probability margin:

```text
log p_u(i) - log p_s(i) >= m
```

Since `log p = -NLL`, this becomes:

```text
NLL_s(i) - NLL_u(i) >= m
```

The balanced candidate uses `m = 1.0`, so `Unknown` must be at least `exp(1)`,
approximately 2.72 times, more probable than the sensitive answer.

## Utility constraints

For each protected utility answer:

```text
NLL_r(i) <= NLL_r_ref(i) - log(rho_utility)
```

where `rho_utility` is the required minimum probability ratio. The balanced
candidate optimizes and ranks candidates with per-example constraints at
`rho_utility = 0.9995`. The underlying materialization stage retains its stable
aggregate gate, then a reloaded postcheck enforces the per-example floor and
removes the checkpoint if any protected answer violates it. The complete retain
set must also preserve an aggregate ratio of at least `0.9995`.

## Optimization loss

Let:

```text
e_forget(i) = ReLU(required_NLL_s(i) - NLL_s(i))
e_pref(i)   = ReLU(m - (NLL_s(i) - NLL_u(i)))
e_unknown(i)= ReLU(NLL_u(i) - NLL_u_baseline(i))
e_util(j)   = ReLU(NLL_r(j) - required_NLL_r(j))
```

The semantic repair minimizes:

```text
L = lambda_f      * mean(e_forget^2)
  + lambda_f_max  * max(e_forget^2)
  + lambda_p      * mean(e_pref^2)
  + lambda_p_max  * max(e_pref^2)
  + lambda_u      * mean(e_unknown^2)
  + lambda_r      * utility_hinge
  + lambda_delta  * ||Delta W||_F^2
```

Balanced weights:

```text
lambda_f      = 50
lambda_f_max  = 100
lambda_p      = 100
lambda_p_max  = 100
lambda_u      = 10
lambda_r      = 100
lambda_delta  = 1e-4
```

The maximum terms prevent a few difficult deletion requests from being hidden
by a low mean loss.

## Candidate ordering

Candidate snapshots are ranked lexicographically by:

1. number of utility violations;
2. number of semantic-preference violations;
3. number of sensitive answers above the probability ceiling;
4. number of buffered forget constraints not met;
5. worst semantic preference margin; and
6. LM-head delta norm.

This prevents stronger forgetting from being selected by sacrificing utility or
leaving sensitive answers preferred.

## Hard checkpoint save gates

A normal checkpoint is saved only when, after BF16 materialization and reload:

```text
all sensitive probabilities <= tau_forget
all semantic preference margins >= m
all selected utility constraints pass
all reloaded per-example utility ratios >= rho_utility
full retain probability ratio >= rho_utility
input embeddings are unchanged
```

A diagnostic `--save-best-effort` run may bypass these gates, but its output
must not be treated as a selected candidate.

## Controlled protocol

Use the same frozen candidate specification for all stages:

```text
train:
  fresh Base -> apply train deletion requests -> development diagnostics

validation:
  fresh Base -> apply validation deletion requests
  -> evaluate unseen validation prompts for those same records
  -> select/freeze candidate

final_apply:
  fresh Base -> apply held-out final deletion requests
  -> evaluate locked test prompts for those same records
```

A validation-trained checkpoint evaluated directly on unseen test authors is a
cross-author generalization experiment, not the official final unlearning test.
