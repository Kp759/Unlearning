# SURE-LM: context-conditioned target-true-sensitive MCF track

This track is an isolated paper experiment. It does not overwrite the existing
canonical SURE or historical Setting-5e checkpoints.

## Target semantics

On the original unswapped MCF record:

- sensitive answer: `target_true`
- non-sensitive / counterfactual reference: `target_new`

The locked direct-only training adapter maps:

- canonical `target_new` <- original `target_true` (sensitive)
- canonical `target_true` <- original `target_new` (reference)

Final evaluation always reopens the original unswapped MCF source.

## Context-conditioned Stage 1

The transformer and input embeddings are frozen exactly at Base. The LM head is
untied, but only sensitive output rows receive a learned delta.

For sensitive token row `t`, direct training-visible forget hidden states form
`H_t`. An orthonormal basis `B_t` of that row-specific hidden subspace is fixed,
and the update is parameterized as

`Delta w_t = a_t B_t`.

Thus no component orthogonal to the observed direct forget-context subspace can
be introduced. Stage-1 rank defaults to 2 per row.

The direct-only loss is

`L1 = lambda_GA * mean(log p(sensitive_token))
    + lambda_ref * CE(reference_token)
    + lambda_KL * KL(base_non_sensitive || current_non_sensitive)
    + lambda_2 * ||Delta W||^2`.

Minimization therefore combines:

1. GA on the sensitive answer;
2. explicit GD on the non-sensitive/reference answer;
3. frozen-Base distribution preservation with the current sensitive token
   removed and the remaining vocabulary renormalized.

Reference rows themselves are frozen. Reference GD can only shape the
context-projected sensitive-row delta through the softmax competition; it cannot
succeed by directly rewriting the counterfactual-reference row.

After optimization, a direct-only scale sweep selects the smallest scale that
satisfies the Stage-1 sensitive-reference NLL-gap constraint. No held-out
paraphrase, neighborhood, retain, or PPL data participates.

## Context-conditioned Stage 2

Residual direct failures are repaired on sensitive rows only. Each row remains
inside its own direct forget-context subspace. Candidate per-row rank caps are
tried in order `2, 8, 0`, where `0` means the full numerical rank of the
observed row-specific forget-context subspace, not the full hidden space.

The repair loss is

`L2 = mean(relu(required_margin - margin)^2)
    + lambda_ref * NLL(reference_answer)
    + lambda_2 * ||Delta W||^2`,

where

`margin = NLL(sensitive) - NLL(reference)`.

A final direct-only scale sweep selects the smallest valid repair scale.

## Evaluation

Use `evaluate_mcf_target_true_sensitive.py` on Base and post checkpoints with the
original unswapped MCF. Paper metrics include:

- `Eff_Pref` and `Gen_Pref` (lower is better);
- absolute and delta sensitive NLL (higher / positive means stronger
  suppression);
- sensitive-reference NLL separation (higher is better);
- reference NLL and delta reference NLL in the direct/paraphrase diagnostics;
- `Spe_margin` and `Spe_success` on held-out neighborhoods;
- fixed-fixture PPL with provenance hashes.

The target-true-sensitive evaluator uses strict pairwise comparisons; exact NLL
ties are not counted as sensitive preference wins.
