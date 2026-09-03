# MCF joint forget/retain endpoint rewiring V2.2

## Purpose

V2.2 is the strongest registered endpoint-only experiment. The Transformer is
frozen. Only selected input-embedding rows and selected rows of an untied LM
head may change. There is no detector, classifier, router, sidecar, adapter, or
conditional inference branch.

Unlike V2 and V2.1, every embedding and LM-head update contains both forget and
retain examples. Preservation is an optimization constraint rather than a
small sampled penalty or a post-training audit.

## Why V2.1 is terminal

The folded head reduced failures but its protected correction was `1.741288`
against a `0.02` ceiling. Embedding-first rescue then changed only 2/50 direct
records and 6/150 synthetic records while development top-1 drift grew to
`0.324246`. Static sparse subject/answer rows did not provide enough endpoint
capacity under the locked preservation limits.

V2.2 makes one final endpoint-only expansion: input rows include every token in
the complete rendered sensitive question, not only subject tokens. Physical
rows remain unique, shared ownership is recorded, and frequency-adjusted caps
strongly constrain common relation-frame tokens.

## Joint data objective

Every endpoint update includes:

- balanced direct and synthetic forget examples;
- ordinary retained facts;
- same-subject/different-relation prompts;
- Wikipedia occurrences of every selected input row;
- generic Wikipedia prompts; and
- a persistent bank of the worst fit-protection prompts.

The differentiable objective combines the forget margin, each retain example's
Base answer-token log-probability, top-k Base-logit KL, Base-top1
log-probability drift, and endpoint-delta norm. Acceptance is governed by a
trust region. When the active retain bank is feasible, a step must remain
feasible and improve forgetting. When it is infeasible, a repair step must
strictly reduce normalized constraint violation while increasing forget loss
by at most one percent.

Embedding and LM-head steps alternate. They use row-normalized gradients,
frequency/cap scaling, and preregistered backtracking. Adam is prohibited.

## Hard-tail protection

Every fifty steps the complete fit-protection bank is evaluated. The worst 32
prompts are merged into a persistent 256-prompt tail. Hard-tail prompts are
always placed before explicit overlaps and random replay in the active retain
batch, with 16 places reserved for each of those three strata. This closes the
sampled-protection failure that allowed V2.1 development drift to grow while
random active batches passed without allowing the tail to crowd out ordinary
or explicit-overlap retention.

## Firewall and claim

Development selects a candidate only when direct, synthetic, fit protection,
development protection, and both row caps pass simultaneously. Certification
opens once afterward. Official evaluation remains unavailable.

A passing run supports internal endpoint behavioral unlearning. Because the
Transformer remains frozen, it does not by itself establish latent knowledge
erasure; recovery and internal-probe tests remain mandatory.
