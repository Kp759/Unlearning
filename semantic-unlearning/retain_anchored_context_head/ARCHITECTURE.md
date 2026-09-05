# Retain-Anchored Contextual Quotient Head

Status: development architecture candidate. This document defines the mechanism and the seed-1 development experiment; it does not claim successful unlearning results.

## Research objective

Suppress a fact-specific association while keeping the base Transformer frozen and avoiding collateral damage when forget and retain examples share subjects, answer tokens, words, or subwords.

The intervention unit is a **fact in context**, not a vocabulary row.

## Frozen base model

The following parameters remain unchanged:

- input embeddings (initial prototype)
- all attention blocks
- all MLP blocks
- all normalization layers
- the original LM head matrix

For a prefix `x`, the base model produces

```math
h_0(x)=\Phi_0(x), \qquad z_0(x)=W_0 h_0(x).
```

The prototype adds an output correction without overwriting `W_0`:

```math
z(x)=z_0(x)+C\,\alpha(x).
```

A later ablation may also use a context-gated quotient of a validated relation/value channel before the frozen head.

## Context descriptor

The descriptor must retain fact-binding information. A value channel alone is insufficient when forget and retain facts share the same answer.

The planned descriptor is

```math
q(x)=[P h_L(x); U_r^\top h_\ell(x)],
```

where `P h_L` preserves contextual binding information and `U_r` is an optional validated relation/value channel. The context-head-only baseline uses frozen hidden-state features without `U_r`.

## Compact-support kernel

The first prototype uses a Wendland C2 compact-support kernel. For normalized distance `d = ||q-q'||/rho`,

```math
k(d)=(1-d)^4_+(4d+1).
```

Thus `k=0` exactly outside radius `rho`. This gives an explicit zero-correction region around unrelated contexts rather than relying on small nonzero RBF tails.

## Retain anchoring

Let `R={r_j}` be training-side protected prefixes and `F={f_i}` be training-side forget prefixes. Official evaluation paraphrases, neighborhood probes, and PPL text must not be used as anchors.

Define

```math
k_R(x,x') = k(x,x') - k(x,R)(K_RR + lambda I)^-1 k(R,x').
```

Then construct

```math
G = K_R(F,F),
alpha(x) = G^-1 k_R(F,x).
```

In exact arithmetic and with a well-conditioned system:

```math
alpha(r_j)=0,
alpha(f_i)=e_i.
```

This is the core structural property: protected anchors receive zero contextual correction, while registered forget anchors have independently controllable coordinates.

## Output correction

For selected output events, train `C` so that a sensitive continuation is suppressed and a safe continuation or margin is preferred.

The correction is context indexed:

```math
Delta z_t(x)=c_t^T alpha(x),
```

not a global vocabulary-row edit.

Multi-token answers are represented as a sequence of prefix/next-token events. Shared subwords are therefore suppressed only in the relevant continuation context.

## Optional causal quotient ablation

After the context-head-only mechanism is validated, add a relation/value basis `U_r` established by counterfactual activation-interchange tests. For active fact gate `alpha_i(x)`, define a quotient target

```math
Q_r(h)=h-U_r U_r^T(h-n_r).
```

The quotient must be gated by the fact-context function; it is not applied globally to all uses of a relation or answer token.

The causal-channel version is an ablation, not assumed to be superior.

## Independent rollback

Each fact owns one or more columns of the contextual correction. Rolling back fact `i` sets those correction coordinates to zero. No original embedding or LM-head vocabulary row is restored because the original row was never overwritten.

Rollback of a failed fact must be counted as a failed forget request, not hidden from aggregate metrics.

## Seed-1 development comparison

Seed 1 is a development/mechanism-selection seed only. Use the existing locked MCF seed-1 split and existing evaluator unchanged.

Run under matched training information and comparable budgets:

1. Existing row GA/GD.
2. Existing GA/GD + row restoration.
3. Existing/projected linear head baseline.
4. Retain-anchored contextual head (new mechanism).
5. Retain-anchored contextual head + validated causal quotient (only after 4 works).

Primary development questions:

- Direct forget: can observed Eff approach 0?
- Generalization: can observed Gen approach 0 on held-out paraphrases?
- Hard overlap retain: do same-answer/shared-subword retain examples remain at base behavior?
- PPL: is the existing PPL metric statistically/base-level stable?
- Specificity: does Spe remain at base level?
- Rollback: does disabling one failed fact leave other fact corrections unchanged at registered anchors and stable on held-out prompts?

Do not tune on official paraphrases, neighborhood probes, or PPL text.

## Go/no-go rule before additional seeds

Continue to multi-seed experiments only if the new context head shows a clear Pareto improvement over row editing on Seed 1: stronger forgetting/generalization **and** base-like overlap retention/PPL/Spe under the unchanged evaluator. If the nonlinear context head does not outperform a matched ordinary nonlinear residual head, the structured anchored construction has not earned its complexity.
