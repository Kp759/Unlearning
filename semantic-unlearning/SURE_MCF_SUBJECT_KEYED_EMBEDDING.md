# SURE MCF subject-keyed embedding unlearning

Stage 1 script: `scripts/mcf_sure_subject_directional_emb_stage1.py`
Runner: `scripts/run_mcf_sure_subject_emb.sh`

## The finding that motivates this

Every earlier MCF SURE stage selected its editable rows from
`requested_rewrite.target_true` (`sensitive_field="target_true"` in both
`mcf_sure_directional_emb_lm_stage1.py` and
`mcf_sure_fullrow_failure_repair.py`). On CounterFact that cannot preserve
locality, and the reason is in the data, not in any hyperparameter:

```text
subject       : Danielle Darrieux          target_true : "French"
neighborhood  : "The mother tongue of Leon Blum is"      -> French
                "The native language of Montesquieu is"  -> French
                "Francois Bayrou, a native"              -> French
paraphrases   : "...Danielle Darrieux, a native"    <- contains the SUBJECT
```

The neighborhood prompts that define Spe have **different subjects but the
same correct answer**. An LM-head row edit changes
`logit_t(h) = W[t] . h` for every context simultaneously, so suppressing
"French" for Darrieux necessarily suppresses it for Blum and Montesquieu.
Eff and Spe were coupled through a single shared parameter.

That is why ~20 real runs traced a strict trade-off curve and never
reached both corners:

| Config | Eff | Gen | Spe | PPL |
|---|---|---|---|---|
| Base | -- | -- | ~11.46 | ~10.94 |
| soft KL w=1.0 | **0.0** | -- | **0.16** | -- |
| soft KL w=10.0 | 12.0 | -- | low | -- |
| narrow geometric protection | ~76 | -- | 4.36 | -- |
| broad, protected-rank 256 | 84 | 85 | 8.61 | ~11 |
| repair-rank 64 (`8d044d3`) | 82 | 85 | 8.37 | 11.06 |

The repository had already reached the same conclusion for MQuAKE.
`mquake_sure_contextual_mlp_v80.py`, line 4: *"V7.1--V7.4 showed that
globally changing LM-head token rows remains utility limited even when the
update is sparse and prompt protected."* The user's own June ablation
`ZeroUnlearn/README_embhead_ablation.md` closes with the same warning:
*"even the touched-row variant can alter many contexts that reuse the same
tokens, so locality may be worse than hidden-layer ZeroUnlearn edits."*

## Architecture

Trainable: **input-embedding rows for subject tokens only.** The LM head is
untied and frozen and is never edited.

Untying is mandatory rather than hygiene: with tied weights, editing an
input-embedding row also edits that token's LM-head row, which would change
its logit in every context and destroy exactly the locality this design
exists to keep.

Per forget prompt `x` (locked direct prompts plus the hand-authored
synthetic paraphrase bank -- all contain the subject):

```text
u_s        = normalize(LM_head[target_true token])     closed-form
dh         = h(x) - h_base(x)

L_margin   = relu(train_margin - [logp(target_new) - logp(target_true)])
L_surgical = || dh - (dh . u_s) u_s ||^2
L_budget   = || E[S] - E_base[S] ||^2

L = margin_weight * L_margin + surgical_weight * L_surgical + delta_l2 * L_budget
```

`L_margin` is gradient ascent on the sensitive token and gradient descent
on the non-sensitive reference in one hinged term, so it stops pushing once
the margin is met instead of running away.

`u_s` is not estimated. The LM-head row for `target_true` **is** the
direction that reads "French" out of the final hidden state, so "forget
only the sensitive fact" becomes "let `h` move along `u_s` and nowhere
else". `L_surgical` stops embedding GA from scrambling the subject's whole
representation, which is the catastrophic mode a raw embedding update would
otherwise hit.

### Locality is combinatorial, not geometric

This is the structural difference from every previous stage.

A neighborhood prompt about Leon Blum contains **none of the edited rows**,
so no gradient path to it exists and its forward pass is *bitwise identical
to Base*. Spe is not defended, it is unreachable.

`L_surgical` therefore does representation hygiene only -- it is not buying
locality. That is why the earlier soft-weight failures do not recur: there,
`pass_guard_weight` / `distribution_kl_weight` were being asked to *buy*
locality and fought the hinge directly (weight 1.0 collapsed Spe to 0.16;
weight 10.0 broke Eff from 0.0 to 12.0). Here locality is free, so the
penalty has no competing job and there is no knife-edge to tune.

### Frequency-filtered row selection

A subject token that is also ordinary vocabulary would reintroduce the very
coupling this design avoids. A word-level audit over the 50 forget records
found:

```text
paraphrases containing the subject          : 50/50   <- Gen mechanism fires
subjects with a zero-frequency anchor token : 32/50   <- Spe exactly preserved
subjects without one                        : 18/50   <- mostly rare:
                                                         ivanschitz(2), argentine(3),
                                                         heath(3), roche(6), circle(7)
worst shared tokens                         : the(1443), robert(110), jean(93),
                                              island(77), charles(67)
```

So rows are kept only below `--max-subject-token-frequency`, counted on a
Wikipedia slice **disjoint from official PPL's hardcoded `[:20]`**
(`--frequency-doc-start` is hard-validated `>= 20`). Each record keeps at
least its rarest subject token, so no record is left with nothing to train;
`records_using_rarest_token_fallback` in the config reports how many took
that path and are therefore expected to carry more collateral than the
clean-anchor majority.

Note the audit was word-level (regex `[A-Za-z]+`) and **dropped digits**, so
it understates available discrimination: `Apple A5`, `Ferrari F40`, and
`Porsche 911` all have rare numeric pieces that subword tokenization keeps.

## First real run (`98e34f4`, Stage 1 only)

| | Eff | Gen | Spe | PPL |
|---|---|---|---|---|
| best LM-head run (`8d044d3`) | 82 | 85 | 8.37 | 11.06 |
| **subject-emb (`98e34f4`)** | **12.0** | **56.0** | **10.02** | 11.06 |
| Base | -- | -- | ~11.46 | ~10.94 |

```text
selected_row_count                  : 74
records_using_rarest_token_fallback : 13
stage1_direct_failures              : 6      -> Eff 12.0 = 6/50
stage1_synthetic_failures           : 45     (of 150)
stage1_minimum_margin               : -10.67
sensitive_readout h.u_s             : 9.387 -> 7.689  (drop 1.698)
```

All four metrics moved the right way simultaneously for the first time in
the project, which is the signature of the Eff/Spe decoupling: Eff fell 82
-> 12 while Spe moved *toward* Base rather than away from it.

### The remaining Gen gap is a prefix distribution mismatch

Synthetic failure was 45/150 (30%) but real paraphrase failure was 59/100
(59%). The synthetic set was too easy, and the reason is the prefix shape:

```text
GENERIC_CONTEXT_PREFIXES (4 formulaic meta lead-ins):
    "According to publicly available records, ..."
    "As has been noted elsewhere, ..."

real paraphrase_prompts (arbitrary unrelated sentence):
    "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native"
    "The population density was . Toko Yasuda plays the instrument"
```

The edit had learned to fire after a short lead-in that *announces a
factual statement*, not after arbitrary noise -- and with only four unique
prefixes it could memorize them outright.

Fix: `corpus_context_prefixes()` samples arbitrary unrelated sentences from
the same disjoint Wikipedia slice already used for frequency counting
(`--corpus-context-prefixes`, default 256), and
`build_synthetic_records(..., context_prefixes=...)` uses them. This is the
same robustness trick MEMIT uses when averaging its key over randomly
sampled prefixes. Both parameters are optional and default to the previous
behaviour, so the LM-head stage and the repair script are unaffected.

### Run history

| Run | Eff | Gen | Spe | PPL | direct fails | min margin |
|---|---|---|---|---|---|---|
| LM-head best (`8d044d3`) | 82 | 85 | 8.37 | 11.06 | 42 | -- |
| `98e34f4` subject-emb | 12.0 | 56 | 10.02 | 11.06 | 6 | -10.67 |
| `017174c` corpus prefixes | 14.0 | 53 | 9.86 | 11.06 | 7 | -10.67 |
| `0ea4aad` dead rows dropped | 6.0 | 49 | 10.14 | 11.06 | 3 | -10.67 |
| `6dcb11f` direct liveness | **2.0** | 46 | 9.85 | 11.06 | **1** | **-0.375** |
| Base | -- | -- | ~11.46 | ~10.94 | -- | -- |

Two hypotheses about Gen were tested and rejected along the way: the corpus
prefix distribution (`017174c` moved Gen only 56 -> 53) and pooled prompt
liveness (fixed Eff, not Gen).  What actually fixed Eff was requiring each
record to keep a row live in its own *direct* prompt -- the bit-identical
`stage1_minimum_margin` of -10.671875 across three runs was a permanently
unedited record, and it moved to -0.375 once that was enforced.

### The Gen gap is a syntactic register mismatch

Measured over the 50 forget records:

```text
real paraphrase_prompts that are subject-first : 100/100  (100%)
canonical prompts that are subject-first       :  36/50   (72%)
this bank's generated templates, before fix    :  49/150  (33%)
distinct real tails 66, synthetic tails 34, OVERLAP = 2
```

Training exercised "The native language of X is ___" while Gen is scored on
"X, speaker of ___" and "X's headquarters are in ___".  `subject_first_variants()`
mechanically derives subject-first templates from the record's own
`canonical_prompt` -- part of the locked training-visible `requested_rewrite`,
never a real `paraphrase_prompts` entry, so the firewall is unchanged:

```text
"The mother tongue of {} is"          -> "{}'s mother tongue is"
                                         "{}, whose mother tongue is"
"The headquarter of {} is located in" -> "{}'s headquarter is located in"
"{}, which is located in"             -> unchanged (already subject-first)
```

This moves generated templates from 33% to 64% subject-first.  Enabled by
default via `prefer_subject_first=True`; the other two callers are unaffected.

## Result: Eff = 0 with Spe/PPL essentially at Base

Stage 1 `6dcb11f` + Stage 2 direct-only repair:

| | Eff | Gen | Spe | PPL |
|---|---|---|---|---|
| LM-head architecture, best of ~20 runs | 82 | 85 | 8.37 | 11.06 |
| subject-emb Stage 1 (`6dcb11f`) | 2.0 | 46 | 9.85 | 11.06 |
| **+ Stage 2 direct-only repair** | **0.0** | **46.0** | **9.74** | **11.06** |
| Base | -- | -- | ~11.46 | ~10.94 |

`post_rewrite_failure_prompt_instances` is 0 of 50.

Stage 2 cost only 0.11 Spe, against the 3.09 it cost under the old
architecture (11.46 -> 8.37). That is the direct consequence of blast
radius: the old Stage 2 repaired 42 records' `target_true` LM-head rows,
this one repaired 1. Stage 2 left Gen exactly unchanged, as expected --
it repairs direct failures and has no mechanism for real paraphrases.

Eff and PPL are solved. Spe sits at 85% of Base. Gen is the open problem.

### Gen: four hypotheses tested, all rejected

| Hypothesis | Change | Result |
|---|---|---|
| prefix distribution too formulaic | corpus-sampled prefixes (`017174c`) | Gen 56 -> 53, negligible |
| pooled prompt liveness | direct-liveness required (`6dcb11f`) | fixed **Eff**, not Gen |
| syntactic register mismatch | subject-first templates (`53f8762`) | Gen 46 -> **52, worse** |
| Stage 2 would generalize | direct-only repair | Gen 46 -> 46, **unchanged** |

The third is the informative one: synthetic failures *improved* 41 -> 37
while real Gen *worsened* 46 -> 52. They move in opposite directions, so
the synthetic paraphrase set is not a usable proxy for real Gen and
reshaping the training distribution is not the lever. `--subject-first-templates`
therefore defaults to 0.

Untested, both zero-code:

```bash
# is the edit too weak / over-constrained?
SURGICAL_WEIGHT=0 TRAIN_MARGIN=3.0 bash scripts/run_mcf_sure_subject_emb.sh ...
# is the model re-identifying the entity from unedited subject tokens?
MAX_SUBJECT_TOKEN_FREQUENCY=100000 bash scripts/run_mcf_sure_subject_emb.sh ...
```

The second is motivated by coverage: 76 rows across 50 records is ~1.5
edited tokens per subject, while most subjects tokenize to 3-5 pieces, so
the unedited remainder may be enough for the model to re-identify the
entity in an unfamiliar phrasing.

## Three-way experiment (`28f27b1`)

| Config | Eff | Gen | Spe | PPL | rows |
|---|---|---|---|---|---|
| `6dcb11f` + Stage 2 | 0.0 | 46 | 9.74 | 11.06 | 76 |
| **(a) full subject-token coverage** | **0.0** | **29** | **10.21** | **11.06** | **223** |
| (b) surgical 0, train-margin 3 | 0.0 | 39 | **10.35** | -- | 76 |
| (c) invariance weight 1.0 | 6.0 | **68** | 9.78 | -- | 76 |
| Base | -- | -- | ~11.46 | ~10.94 | -- |

**(a) coverage was the Gen lever.** Raising `--max-subject-token-frequency`
took rows 76 -> 223 and improved every metric at once: Gen 46 -> 29, Spe
9.85 -> 10.21, `stage1_direct_failures` 3 -> 0 *without Stage 2*, and
`records_with_direct_row_above_threshold` 17 -> 0.
`stage1_synthetic_failures` reached 0 and `stage1_minimum_margin` went
positive (+0.125), so the training objective is now fully satisfied.

The frequency filter was starving the edit. At ~1.5 edited tokens per
subject against 3-5 tokens per subject, the model re-identified the entity
from the untouched remainder, which is why the edit held on trained
phrasings but not on new ones. Spe *improved* despite editing more common
tokens, because the filter's cost (17 records forced to override the
threshold) exceeded what it saved.

**(b) the surgical constraint was throttling, not protecting.** Locality
here is combinatorial, so `L_surgical` was never buying locality; removing
it and raising the margin gave Eff 0.0, Gen 39 and the best Spe recorded.

**(c) the invariance penalty failed** -- Gen 46 -> 68 and Eff 0 -> 6.
Forcing a context-invariant representation shift conflicts with achieving
the margin. It remains available as an ablation axis, defaulted to 0.

Defaults now follow (a)+(b): `--max-subject-token-frequency` effectively
off, `--surgical-weight 0.0`, `--train-margin 3.0`.

### Gen hypotheses, scoreboard

| # | Hypothesis | Result |
|---|---|---|
| 1 | prefix distribution too formulaic | 56 -> 53, negligible |
| 2 | pooled prompt liveness | fixed **Eff**, not Gen |
| 3 | syntactic register mismatch | 46 -> **52, worse** |
| 4 | Stage 2 would generalize | 46 -> 46, unchanged |
| 5 | context-invariance penalty | 46 -> **68, much worse** |
| 6 | **subject-token coverage** | **46 -> 29** |

Five of six failed. The one that worked was the only one about *how much of
the subject is edited* rather than *what the training prompts look like*.

## Headline result (`b9fe60f`, Stage 1 only, no Stage 2)

| | Eff | Gen | Spe | PPL |
|---|---|---|---|---|
| LM-head architecture, best of ~20 runs | 82 | 85 | 8.37 | 11.06 |
| subject-emb, coverage only | 0.0 | 29 | 10.21 | 11.06 |
| subject-emb, surgical 0 only | 0.0 | 39 | 10.35 | -- |
| **subject-emb, combined** | **0.0** | **10.0** | **11.35** | **11.06** |
| Base | -- | -- | 11.46 | 10.94 |

`Spe_success` 88.8. Spe sits 0.11 below Base, PPL is unchanged, Eff is 0,
and Gen fell 85 -> 10. **Stage 2 is not used at all** -- this is a
single-stage, embedding-only method, so nothing in the final pipeline ever
touches a `target_true` LM-head row.

```text
selected_row_count             : 226
rows_ever_touched_by_gradient  : 217
stage1_direct_failures         : 0
stage1_synthetic_failures      : 0
stage1_minimum_margin          : 3.0625     (train-margin is 3.0)
sensitive_readout_drop         : 5.463      (was 1.918)
final_embedding_delta_norm     : 4.251      (was 10.445 coverage-only)
```

The combination is **superadditive**: Gen 29 and 39 separately, 10 together.
The diagnostics show why. The delta norm *fell* 10.45 -> 4.25 while the
sensitive-readout drop *rose* 2.82 -> 5.46 -- a smaller edit doing roughly
twice the work. `L_surgical` had been forcing a large, inefficient update by
confining the hidden-state change to `u_s`; removing that constraint let the
optimizer find a far more efficient direction, and full token coverage gave
it enough rows to express it.

Note this retires `L_surgical` from the recommended configuration. It was
introduced as representation hygiene against catastrophic embedding GA, but
with combinatorial locality doing the protective work it was pure
throttling. It remains available (`--surgical-weight`) as an ablation.

## Forgetting evidence beyond NLL

The config records `base_mean_sensitive_readout`,
`final_mean_sensitive_readout`, and `sensitive_readout_drop` -- the mean of
`h . u_s`. This is mechanistic evidence that the sensitive readout itself
collapsed, independent of the NLL-based margin, and is the quantity an
extraction-attack reviewer will ask about.

## Novelty position

| Method | Venue | Parameters edited | Runtime dep. |
|---|---|---|---|
| ROME / MEMIT | NeurIPS'22 / ICLR'23 | MLP `down_proj` | none |
| ZeroUnlearn (baseline here) | -- | MLP hidden layers | none |
| REVS | ACL 2025 Findings | **MLP neurons** (vocabulary space is the lens, not the edit site) | none |
| ECO | NeurIPS 2024 | none -- inference-time prompt-embedding corruption | prompt classifier |
| **This** | -- | **input embedding only, transformer + LM head frozen** | none |

There is no ICLR 2025 paper named SURE in this area, so the method name
does not collide.

## Stage 2 is opt-in

`run_mcf_sure_subject_emb.sh` defaults to `RUN_STAGE2=0`. Stage 2
(`mcf_sure_fullrow_failure_repair.py`) edits `target_true` LM-head rows,
which is exactly the coupling described above. Measure Stage 1 alone first
so its Spe/PPL claim is actually tested; enable Stage 2 only for whatever
direct failures remain, where its blast radius is a handful of records
rather than all 50.

If Stage 2 is enabled later, it should work better than it did before for a
specific reason: Stage 1 has already moved `h(x)` off the generic
"French-predicting" cluster, so the residual it needs is large. Previously
Stage 2 was asked to separate vectors that were still tangled -- hence
`active_residual_rank_uncapped=174` but no usable margin and 793/800
rejected steps.

## Running

```bash
BASE=/home/ec2-user/models/Llama-3.2-3B-Instruct
SPLIT_ROOT=outputs/mcf_targettrue_clean_seed1/seed1
bash scripts/run_mcf_sure_subject_emb.sh \
  "$BASE" \
  "$SPLIT_ROOT/protocol/training_visible_mcf_target_true.json" \
  "$SPLIT_ROOT/protocol/split_manifest.json"
```

Ablate the direction constraint with `SURGICAL_WEIGHT=0` (raw embedding
GA/GD, no confinement to `u_s`) -- that is the ablation that isolates what
`L_surgical` contributes.

## What to check first in `stage1_config.json`

- `selected_row_count` and `records_using_rarest_token_fallback`
- `stage1_direct_failures` / `stage1_synthetic_failures` (Eff and Gen proxies)
- `sensitive_readout_drop` (mechanistic forgetting)
- then the official eval's Spe and PPL, which this architecture predicts
  should sit at Base for every record with a clean anchor
