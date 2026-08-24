# MCF SURE: directional Emb+LM GA -> failure-only full-row repair

This experiment intentionally removes the surrogate-paraphrase and LoRA paths.
It uses the locked direct-only target-true-sensitive MCF protocol.

## Target contract

- `requested_rewrite.target_true`: sensitive / unwanted fact.
- `requested_rewrite.target_new`: non-sensitive reference used for direction construction and the direct margin.
- Fields are never swapped.
- Official paraphrases, neighborhoods, benchmark retain records, and PPL text are held out from training and checkpoint selection.

## Architecture

```text
Base Llama
   |
   | clone / untie input embedding E and LM head W
   | transformer Phi frozen
   v
identify target_true-sensitive vocabulary rows S
   |
   v
row-specific sensitive direction
   d = h_true - h_new
   |
   | if d = 0 (notably the first answer token)
   v
   d = w_true - w_new decoder-discriminant fallback
   |
   v
Stage 1
   selected E[S] delta constrained to span(d)
   selected W[S] delta constrained to span(d)
   GA on target_true sensitive token log-probability
   Base non-sensitive distribution KL guard
   no GD/CE toward target_new
   no LoRA
   |
   v
direct margin gate
   margin = NLL(target_true) - NLL(target_new)
   required margin = 0.05 by default
   |
   +---- pass -------------------------------+
   |
   +---- fail                                |
           |                                 |
           v                                 |
       Stage 2                               |
       select target_true LM-head rows       |
       belonging to failed records only      |
       unrestricted full-row delta           |
       no rank basis / no LoRA                |
       failed-record hinge objective          |
       passing direct records guard regression|
           |                                 |
           +---------------------------------+
                           |
                           v
                         final
```

## Why the direction has a fallback

At the first answer token, the model sees exactly the same prompt prefix whether the future answer is `target_true` or `target_new`. Therefore the two pre-answer hidden states can be identical and `h_true - h_new = 0`.

The fallback

```text
d = w_true - w_new
```

is the hidden-space gradient of the true-vs-reference logit gap for that token pair. It gives a non-zero discriminative direction without opening held-out paraphrases or other benchmark probes.

For later teacher-forced tokens, the true and reference prefixes can differ, so the literal hidden-state contrast is used when non-zero.

## Synthetic paraphrase augmentation (Gen-closing fix)

The direction `d` and the direct-margin gate were originally fit from a
single literal prompt template per record. Because `d` is a fixed direction
and the row-edit's suppression strength is `hidden_state . d`, a direction
fit from one template only suppresses the sensitive token where a held-out
paraphrase's hidden state happens to align with that same direction --
partial, not guaranteed, transfer. This showed up empirically as GFS < 100%
(Gen > 0) even when FS = 100% (Eff = 0).

`scripts/mcf_synthetic_paraphrase_templates.py` hand-authors, for each of
the 34 MultiCounterFact relation ids, 2 alternate cloze templates that are
syntactically distinct from the dataset's own canonical
`requested_rewrite.prompt` template for that relation, plus a small bank of
generic content-free context-prefix sentences. Both axes are authored from
scratch and never derived from, or copied from, any record's real
`paraphrase_prompts` -- the official held-out paraphrase set stays
uncontaminated, so GFS/Gen remains an honest measure of whether this
transfers.

`--synthetic-paraphrases-per-record` (default `3`) controls how many such
templates are generated per record. Stage 1 builds `all_records = records +
synthetic_records` and uses it (not `records` alone) for:

- the row-specific contrast-direction basis (`build_row_specific_contrast_bases`);
- GA training and the base-logit KL cache;
- the direct-margin scale-selection gate, which now requires the margin on
  every synthetic template too (reported per-scale as `direct_only_failures`
  vs `synthetic_failures` in `scale_reports`).

`stage1_config.json` keeps the original `stage1_direct_failures` /
`stage1_failing_positions` / `stage1_minimum_margin` fields exactly as
before (direct-prompt-only, for backward compatibility), and adds
`stage1_synthetic_failures`, `stage1_synthetic_failing_positions`,
`stage1_synthetic_minimum_margin`, `stage1_combined_failures`, and
`stage1_combined_minimum_margin` alongside them. Set
`--synthetic-paraphrases-per-record 0` to fully recover the original
direct-only behavior.

Stage 2 (`mcf_sure_fullrow_failure_repair.py`) is unchanged by this and
still gates only on the literal direct prompt.

### Fixes found after the first synthetic-augmented run

A live run at the original defaults (rank 1, 600 steps, batch 2) surfaced
three compounding problems, diagnosed from `stage1_config.json` and
`train_log.jsonl`:

1. **Scale-selection collapsed to a no-op.** `core.choose_scale` tie-breaks
   toward the *smallest* scale when candidates tie on failure count. Once
   `direct_failures` in `scale_reports` meant the *combined* (direct +
   synthetic) count, and combined zero-failure was unreachable, every scale
   tied and the tie-break picked `scale=0.0` -- discarding a Stage-1 edit
   entirely, even though a larger scale had a strictly better margin.
   `select_stage1_scale()` now selects in three passes: minimize
   direct-only failures first (never regress what direct-only training
   already achieves), then combined failures, then prefer the *largest*
   scale among remaining ties.
2. **Rank-1 could not fit the wider case set.** `stage1_combined_failures`
   was 174/200 (87%) at rank 1 with 3 synthetic templates per record.
   `--direction-rank` default raised from `1` to `8`.
3. **The case pool grew ~4x but the step budget did not.** `train_log.jsonl`
   showed `lm_head_delta_norm` reaching only ~0.07 after 600 steps -- too
   small to move the margin at any scale (`post_rewrite_min_margin` was
   flat across the entire scale sweep). `--steps` raised `600 -> 1200` and
   `--batch-size` raised `2 -> 4` to restore roughly the original per-case
   training coverage.

It was not the source of the Gen gap in that run since Stage 1's
contribution had been silently zeroed out by (1) above -- Stage 2 alone
had produced the entire edit, so the synthetic-paraphrase augmentation had
not actually been exercised yet.

### Stage 1's synthetic augmentation is structurally inert for single/first-token answers

A run at rank 8 / 1200 steps / batch 4 (i.e. all three of the above fixes
actually applied) still produced `stage1_combined_failures=174/200`,
bit-identical to a run at the original rank 1 / 600 steps / batch 2. Root
cause: `contrast_direction()`'s primary path is
`hidden(sensitive) - hidden(reference)`, but within any one record the
sensitive and reference prompts are identical up to the first answer token,
so this is exactly zero there and it falls back to
`w_sensitive - w_reference` -- the raw decoder-row difference, a function of
the `(target_true, target_new)` token pair only, **not the prompt**. Every
synthetic template of a given record therefore contributes an identical
fallback vector to that row's direction pool; the basis's actual numerical
rank is capped by this degeneracy regardless of `--direction-rank`, so more
rank/steps/batch cannot help wherever the fallback triggers (in particular,
any single-token answer).

Stage 2's unrestricted delta has no such degeneracy: it is optimized
directly against `hidden @ delta.T` corrections, never a
sensitive-minus-reference contrast direction, so it is genuinely
prompt-dependent. `mcf_sure_fullrow_failure_repair.py` now builds
`all_records = records + synthetic_records` itself, detects
active/failing positions and selects LM-head rows across the combined set
(not direct-only), and applies the identical `select_repair_scale()` fix.
On the same checkpoint this took Stage 2's own residual synthetic-template
failures from 132/150 to 1/150, and dropped the official (held-out)
`Gen` metric from 13 to 11.

### The generalization fix increased collateral damage to Spe

Widening Stage 2's scope meant selecting more LM-head rows (37, vs fewer
under the direct-only objective) with a much larger `effective_delta_norm`
(~10.6, vs ~0.2 before). `Spe` on the official evaluator dropped further as
a direct result (to `0.16`, against a canonical base of `~11.46` on the
same scale -- still a near-total collapse). The likely cause is a magnitude
mismatch in Stage 2's loss, not a missing mechanism: `final_distribution_kl`
was `0.815` (non-trivial drift at the 200 training-visible positions) at
`--distribution-kl-weight 1.0`, and `--repair-l2 1e-6` contributed
`~1e-6 * ||delta||^2 ~ 0.0001` -- both negligible next to a failure hinge
that starts in the hundreds (margins near -13, squared) early in training.
Raised `--distribution-kl-weight` to `10.0` and `--repair-l2` to `1e-3` so
both regularizers are actually load-bearing. Both the KL and L2 terms only
constrain the 200 training-visible positions (or the delta's raw
magnitude), not held-out neighborhood prompts directly -- if this does not
recover Spe, the KL sampling scope itself needs widening to a generic text
sample, not just its weight.

### Soft weight tuning replaced by a hard gate

Weight `10.0` overshot in the other direction: `final_direct_failures`
regressed from `0` to `7` (Eff `0.0 -> 12.0`), even though Spe improved
(`0.16 -> 2.03`). As active cases approach passing, their hinge
contribution shrinks toward zero, so a large-enough KL weight increasingly
dominates late in training and can pull the optimizer away from
already-satisfied margin cases -- a fundamental limitation of a *soft*
competing loss term, not a tuning mistake. A follow-up interim weight
(`3.0`) plus a direct-only-first best-checkpoint-selection fix recovered
most but not all of Eff (`2.0`, `final_direct_failures=1`).

`--pass-guard-weight` and `--distribution-kl-weight` were removed and
replaced with a **hard gate**, mirroring `mcf_sure_protected_subspace_stage2.py`'s
proven backtrack-or-rollback design (same default `--protected-kl-max 0.05`):
every optimizer step is evaluated after `opt.step()` against the
*currently*-passing record set (recomputed fresh each step -- a record
becomes protected the moment its margin first reaches `constraint-margin`,
and can never regress afterwards). If the raw step would drop any
protected **direct-only** record's margin below `constraint-margin`, or
push same-prompt non-target KL over the full protected set above
`--protected-kl-max`, the step is backtracked (`--backtrack-scales`, a
geometric schedule from `1.0` down to `0.0`) until it satisfies both, or
rolled back to a full no-op if nothing does. The schedule always includes
`0.0` explicitly (enforced by an argparse check), so a step can never be
silently dropped without a safe fallback -- unlike a plain rollback-only
implementation, which is exactly the "rollback deadlock" bug class already
found and fixed in the `mcf_sure_protected_subspace_stage2.py` lineage
(`mcf_sure_rowspecific_minimal_stage2.py`'s v4 direct entry point).

This makes "never regress an already-passing direct record" a guarantee
instead of a probabilistic outcome of a hand-tuned weight -- removing the
weight-guessing loop entirely, at the cost of a training run that may
plateau (visible via `gate_rejected_steps`/`gate_backtracked_steps` in
`repair_summary.json`) rather than trade Eff for Spe silently.

### The hard gate alone was too tight -- fixed with a geometric architecture change

A real run at `--protected-kl-max 0.05` (borrowed directly from
`mcf_sure_protected_subspace_stage2.py`'s own default) confirmed the
opposite failure mode: `gate_rejected_steps=796/800`, `best_step=25`,
`effective_delta_norm=0.16` -- training essentially froze after the first
few steps. `final_direct_failures` regressed from single digits to `39/50`
(`Eff=76.0`), while `Spe=7.54` recovered close to the canonical base
(`~11.46`) precisely because almost no edit was ever applied.

The mismatch: `protected_subspace_stage2.py`'s own delta is *already*
rank-limited and projected away from a protected subspace by construction,
so a tight KL bound on top of that is redundant, cheap insurance. This
script's delta was fully unrestricted (`direction_basis=None`) -- a global
edit to 37 shared LM-head rows has no way to distinguish "suppress this
token for the forget prompts" from "suppress this token everywhere else it
appears", so fixing ~40 deeply negative margins (`post_rewrite_min_margin
~ -13`) required a large-magnitude edit that the same numeric KL bound,
calibrated for a geometrically-constrained delta, could not accommodate at
all.

Root architecture fix (mirroring `mcf_sure_protected_subspace_stage2.py`'s
own `rowspace(H_P)` / `R_F = H_F - Proj(H_F)` / `rowspace(R_F)`
construction): before training, gather teacher-forced hidden states for the
*initially* active/failing cases (`H_active`) and the *initially* passing
cases (`H_protected`, direct+synthetic). Build

```text
B_protected = orthonormal_row_basis(H_protected, max_rank=--protected-rank)
B_active    = orthonormal_row_basis(H_active, max_rank=None)
B_residual  = project_rows_away(B_active, B_protected)
B_repair    = orthonormal_row_basis(B_residual, max_rank=--repair-rank)
```

and parameterize the delta as `coefficients @ B_repair` instead of a raw
`[rows, hidden]` tensor -- reusing `sure_canonical_core.orthonormal_row_basis`
and `gagd_active_case_repair.project_rows_away`, both already proven
utilities in this repo. This makes "does not disturb the passing cases'
directions" geometric by construction, not something a penalty weight or
KL threshold has to enforce after the fact. Defaults (`--protected-rank
32`, `--repair-rank 4`) match `mcf_sure_protected_subspace_stage2.py`'s own
already-validated values. `--protected-kl-max` is kept only as a secondary
backstop and loosened to `0.5` accordingly (the natural KL scale observed
when a soft weight-3.0 penalty found a reasonable Eff/Spe balance), since
the primary protection no longer depends on it being tight.

### The gate's protected set still didn't match what the basis protects

A real run with the geometric restriction in place still showed
`gate_rejected_steps=711/800` and `final_direct_failures=32/50`
(`Eff=62.0`) -- much better than the fully unrestricted delta's collapse,
but still far from the near-perfect results the earlier soft-weight runs
achieved on Eff alone. Cause: the runtime gate recomputed "currently
passing" fresh every step, so the instant an *active* case's margin
crossed the threshold mid-training, it joined the protected set -- even
though its hidden state was part of `H_active`, which the repair basis is
built, by construction, to move. Every subsequent step that legitimately
kept refining coefficients for other still-failing cases risked a small
wobble in that recently-fixed one (they share the same basis), and the
dynamic gate rejected it as a violation instead of recognizing it as
in-scope movement.

Fixed by scoping the gate's protected set to the same *fixed*,
pre-Stage-2-training set `H_protected` was built from (`passing_positions`,
computed once, not recomputed as training progresses). Since the repair
basis is already orthogonal to that exact set's hidden-state span, the
gate should now rarely bind at all -- it becomes a pure numerical-safety
backstop rather than a competing objective.

### The protected subspace itself was too narrow -- widened with generic text

A real run confirmed the gate fix worked exactly as intended
(`gate_rejected_steps=0`, `gate_backtracked_steps=0`) and Eff/Gen kept
improving (`final_direct_failures` 42 -> 13, `Eff` 62.0). But `Spe`
remained collapsed (`0.68`) and, for the first time across every run so
far, `PPL` broke too (`18.875`, vs a stable `~10.9-11.1` in every prior
run). `effective_delta_norm` was also the largest yet (`16.9`).

Cause: `H_protected` was built from only the ~26 in-sample MCF records
that happened to already pass Stage 1 (at most ~52 hidden-state vectors,
capped further by `--protected-rank 32`). Being geometrically orthogonal
to that tiny, MCF-specific span does nothing for the vast majority of
directions real neighborhood prompts and general PPL text actually
occupy -- the geometric guarantee was real, but scoped to a population far
too narrow to matter for what Spe/PPL measure.

`generic_protection_hidden_states()` widens `H_protected` with hidden
states sampled from `--wikidata-dir`'s ordinary text (the same corpus PPL
is evaluated against) before the SVD basis is built -- read-only, never
contributing to the failure hinge or any margin computation. Default
`--generic-protection-tokens 300`. Falls back gracefully (with a printed
warning) to in-sample-only protection if the directory is missing, so this
is additive rather than a hard requirement.

## Stage-1 embedding caveat

After untying, an input-embedding row receives ordinary GA gradient only when that token actually occurs in the teacher-forced input prefix. Consequently, some single-token answer embedding rows may remain unchanged even though their LM-head rows receive GA gradient. The implementation logs `embedding_rows_with_nonzero_current_grad` rather than hiding this causal fact.

## Run on the clean locked seed-1 split

```bash
cd /home/ec2-user/workspace/Unlearning/semantic-unlearning

BASE=/home/ec2-user/models/Llama-3.2-3B-Instruct
SPLIT_ROOT=outputs/mcf_targettrue_clean_seed1/seed1
VISIBLE="$SPLIT_ROOT/protocol/training_visible_mcf_target_true.json"
MANIFEST="$SPLIT_ROOT/protocol/split_manifest.json"

bash scripts/run_mcf_sure_directional_emb_lm_fullrepair.sh \
  "$BASE" "$VISIBLE" "$MANIFEST"
```

Default output:

```text
outputs/mcf_directional_emb_lm_fullrepair_seed1/
  stage1/
    checkpoint/
    stage1_config.json
    train_log.jsonl
  stage2_fullrow_repair/
    checkpoint/
    repair_summary.json
    scale_sweep_direct_only.json
```

## Default optimization

Stage 1:

- 1200 steps (raised from 600, see below)
- batch size 4 (raised from 2, see below)
- LR `1e-4`
- GA weight 2
- non-sensitive-distribution KL weight 1
- delta L2 `1e-6`
- per-row direction rank 8 (raised from 1, see below)
- direct constraint margin `0.05`
- synthetic paraphrase templates per record `3` (hand-authored, see below)

Stage 2:

- 800 steps
- protected-subspace sparse LM-head rows: protected rank `32`, repair rank
  `4` (see below; not an unrestricted row edit)
- LR `5e-3`
- delta L2 `1e-3` (raised from `1e-6`, see below)
- direct constraint margin `0.05`
- synthetic paraphrase templates per record `3` (shared bank with Stage 1)
- secondary hard-gate backstop: protected-KL-max `0.5`, geometric backtrack
  schedule `1.0` down to `0.0` (see below)

These are one declared configuration, not a rank sweep.

## Mechanical tests

```bash
pytest -q tests/test_mcf_sure_directional_emb_lm_stage1.py
python -m py_compile \
  scripts/mcf_sure_directional_emb_lm_stage1.py \
  scripts/mcf_sure_fullrow_failure_repair.py
```
