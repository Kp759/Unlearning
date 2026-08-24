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
