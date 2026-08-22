# RWKU Directional SURE v2

This branch ports Directional SURE v2 to the RWKU target-only development protocol. Stephen King seed 0 remains post-hoc development; this implementation does not open or consume official RWKU evaluation records.

## Architecture

```text
Base tied vocabulary matrix
        |
        +-- untie LM head by exact clone

Input embedding E                         LM head W
  sensitive rows: FP32 sparse delta        sensitive rows: FP32 sparse delta
  every other row: exact Base              every other row: exact Base
        |                                          ^
        v                                          |
             transformer backbone
             exactly frozen
        |                                          |
        +---------------- final hidden h -----------+
```

There is no LoRA and no transformer/MLP update in Directional SURE v2.

## Canonical SURE objective

For a teacher-forced generated target-only sensitive-token decision with target token `y`, the learner uses the canonical Stage-1 losses:

- GA: `mean(log p_theta(y | x))`, minimized, so sensitive probability is driven downward.
- GD/locality: `KL(P_Base(non-y | x) || P_theta(non-y | x))`, with `y` removed and both distributions renormalized.

The locked weights are GA=2 and GD=1.

No official RWKU retain examples are used by GD. GD is the same-prompt non-sensitive KL from canonical SURE.

## Directional LM-head decomposition

Every 25 optimization updates, the learner refreshes two orthonormal hidden-space bases.

`B_P` is built from current final hidden states on a fixed target-excluded external-Wikipedia protection slice.

`B_S` is built from current sensitive-prediction final hidden states after projection into the orthogonal complement of `B_P`:

```math
H_S^\perp = H_S - H_S B_P^T B_P,
```

followed by an SVD/orthonormal row basis. The implementation reprojects `B_S` away from `B_P` and fails if their maximum absolute overlap exceeds 1e-4.

GA and GD gradients are obtained separately. For the sparse LM-head delta:

```math
g_{head} = \Pi_{B_S}(2 g_{GA}) + \Pi_{B_P}(g_{GD}),
```

where

```math
\Pi_B(g) = g B^T B.
```

The sparse sensitive input-embedding rows receive the original combined GA+GD gradient without directional projection.

Locked initial ranks:

- sensitive-exclusive `B_S`: 8
- protected `B_P`: 32

These are development hyperparameters, not tuned with official RWKU evaluation.

## Exact parameter locality

The base embedding matrix, base untied LM-head matrix, and all transformer parameters have `requires_grad=False`.

Trainable parameters are only two FP32 sparse delta tensors indexed by the declared sensitive vocabulary rows. Therefore non-sensitive vocabulary rows have no trainable parameter at all.

Before checkpoint materialization the Base vocabulary matrices must remain bitwise unchanged. At materialization, only selected row indices are written. A final chunked audit fails if any non-sensitive embedding or LM-head row differs from Base.

## External-Wikipedia boundary

After excluding the first 20 documents and any case-folded occurrence of `stephen king`, a deterministic shuffled pool is split into three disjoint slices:

1. 256 protected-basis contexts: used during training and basis refresh.
2. 256 selection contexts: exact full-vocabulary Base-to-current KL checkpoint selection only.
3. 1000 fresh final-gate contexts: cached at Base before training, but not re-evaluated until checkpoint selection is fixed.

The final KL budgets remain:

- mean <= 0.01
- p95 <= 0.05
- max <= 0.5

## RWKU boundary

Unavailable to the learner and checkpoint selector:

- official RWKU paraphrases,
- official RWKU neighborhood prompts,
- official RWKU retain sets,
- official RWKU PPL/evaluation text,
- any previous official RWKU result.

The only sensitive training data are the independently generated target-only atomic fact views already used by the RWKU development protocol.

## Development acceptance

At a 25-step checkpoint, selection requires:

- generated direct atomic FS = 100%,
- generated other atomic FS = 100%,
- no atomic pairwise-margin failures at margin 0.01,
- the 256-context external-Wikipedia selection KL within the fixed budgets.

Among eligible checkpoints, selection minimizes KL mean, p95, max, then total sparse-row delta norm, then step. After selection is fixed, the fresh 1000-context external-Wikipedia gate is evaluated. A checkpoint is saved only when the generated atomic and fresh utility gates both pass.

This is a post-hoc development experiment and does not establish official RWKU benchmark performance.
