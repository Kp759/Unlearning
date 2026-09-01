# MCF folded-sensitivity bi-endpoint rewiring V2.1

## Purpose

V2.1 replaces V2's joint stochastic optimizer with a classifier that is folded
directly into the selected LM-head rows. It remains an internal sparse weight
edit: there is no classifier module, threshold, router, sidecar, or conditional
runtime branch after training.

For selected output token `t`, its learned correction is

```text
s_t(h) = h dot Delta_W_t
```

This ordinary LM-head dot product is simultaneously the contextual classifier
and the intervention. Sensitive `target_true` contexts receive a negative
signed constraint, `target_new` contexts receive a positive constraint, and
protected contexts receive a zero-correction constraint. A token that is true
for one fact and new for another therefore has two contextual signed cells,
not a contradictory global token label.

## Evidence motivating the change

The V2 run passed safe-direction feasibility for all 50 facts, but 141 of 236
input rows had empty protected bases. An input row hit its cap on step 1, both
endpoints were saturated by step 100, and random batches oscillated between
one and five direct failures. Ten of 70 output rows had cross-role labels and
14 served multiple facts. Development preservation exceeded its limits from
the first step. Lowering the learning rate would not resolve those structural
problems.

## Stage 1: deterministic folded head solve

The Transformer and input embedding remain exactly Base. V2.1 caches the final
hidden state for every direct and synthetic teacher-forced target cell. It
constructs a protected hidden-state sketch, projects sensitive states into its
orthogonal complement, and solves a minimum-norm ridge system independently
for each physical LM-head row. All cells for a shared row are solved together.

Correction floors `2/4/6/8` are preregistered. Development may select the
smallest arm satisfying complete direct, synthetic, and preservation gates.
There are no stochastic minibatches or Adam moments.

## Stage 2: bounded embedding rescue

Embedding rows remain zero if the folded head already passes. Otherwise they
are used only to move unresolved sensitive states into the signed LM-head
decision regions. Common selected rows receive targeted Wikipedia contexts so
the protected Jacobian construction cannot silently miss a corpus-observed
row. Truly corpus-absent rows are reported separately.

Each rescue step uses a balanced deterministic forget batch, projects its
gradient through the row-specific protected bases, normalizes the update to a
small fraction of that row's cap, and accepts it only through a registered
fit-protection backtracking rule. The folded head is recomputed every 100 steps.
Adam is prohibited.

## Firewall and claim

Official retain prompts, official paraphrases, neighborhoods, aliases,
adversarial prompts, and PPL documents are unavailable. Certification opens
once after development selection and cannot alter the candidate. A passing run
demonstrates internal sparse counterfactual rewiring, not yet latent erasure;
fresh alias, extraction, recovery, and relearning tests remain mandatory.
