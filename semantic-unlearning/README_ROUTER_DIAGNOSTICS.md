# Router diagnostics: what the L19 classifier gets wrong, and why L19

The gap between **ours** and **genie** in the paper table is routing only: genie applies the
same trained rows under perfect routing. This toolkit decomposes that gap per prompt,
per benchmark, and tests the layer choice and robustness fixes. Nothing here changes how
the official numbers are computed.

## The classifier in one picture

```
prompt ──► subject eligibility ──► which heads may fire (exact subject-token match;
   │       (token match)            RWKU: full name, part before a comma, or surname)
   │
   └──► block-19 output at the last prompt token ─► unit-normalise ─► − mean
                                                    ─► z_i = w_i·φ + b_i  (one logistic head per fact)
decision  threshold gate (MCF, ZsRE, MQuAKE): fire the best eligible head if σ(z_i) ≥ 0.5,
                                              unless the top two are within the margin
          subject gate (RWKU):                fire whenever a name matches; the heads only
                                              pick WHICH of that person's rows
```

Every routing error has exactly one of these causes. The report labels each one:

| reason | what happened | fix lives in |
|---|---|---|
| `subject_absent_from_prompt` (FN) | the subject never appears (pronoun, description, alias, other language) | eligibility: aliases, learned subject gate. No head or layer change can help |
| `subject_in_text_not_token_matched` (FN) | the subject is there, but its case, spacing or tokenization differs | eligibility: the fuzzy gate |
| `below_threshold` (FN) | the right head was eligible but scored < 0.5 | the head: augmentation, read layer, calibration |
| `ambiguous_top2` (FN) | two eligible heads within the margin | margin or head separation |
| `eligible_head_above_threshold` (FP) | a protected subject appears and its head fires on a different question | hard negatives (same subject, other relation) |
| `identical_to_forget_prompt` (FP) | the prompt text equals a forget prompt (genie routes it too) | none: a label conflict, not a router error |
| `same_subject_other_row` (misroute) | right subject, wrong fact | relation discrimination: hard negatives |

## Run it (Wulver)

```bash
bash scripts/submit_router_diag.sh        # 4 GPU tasks (mcf zsre mquake rwku), seed 1
```

Each task runs three steps:

1. `cache_router_states.py`: last-token block outputs at **every** layer plus the L19 router outputs, for:
   - **eval**: every prompt each official evaluator routes, with genie ground truth;
   - **null**: no-association prompts;
   - **perturbed**: typo, case and partial-name variants;
   - **train**: the router's own fit, calibration and audit prompts;
   - **aug**: augmentation and hard negatives.
2. `router_diag_report.py`: the L19 analysis, written to `report/summary.md`.
3. `router_layer_sweep_cached.py`: refit at every layer plus the robustness variants, written to `layer_sweep/layer_choice.md`.

The last task to finish writes `outputs/router_diag_v1/ROUTER_LAYER_CHOICE.md`. Locally,
every step runs from a copied cache on CPU (`--device cpu`).

**Staged run** (`sbatch router_diag_staged.slurm`): the same outputs, step by step in the order
of the to-do list, with a banner per step in `logs/rdiag_s_<job>_<task>.out`:
1. cache, then stop unless hook parity is 0;
2. report;
3. layer sweep, then stop unless the L19 refit reproduces the shipped router (`MIN_AGREEMENT`, 0.99);
4. downstream at the penultimate layer, only if routing improves there. The command is written to `layer_sweep/downstream_penultimate.sh`, and submitted as its own 24 h job with `RUN_DOWNSTREAM=1`;
5. robustness variants;
6. combine.

It refuses to start while a `router_diag.slurm` job is queued, because both write the same outputs.

**Ours (the one-pass classifier).** Run `sbatch --export=ALL,ROUTER_RECIPE=ours router_diag_staged.slurm`
once `run_l16_paper_one.sh ONEPASS=1` has fit the routers in `outputs/<ds>_multiseed_ours_v1`.
Those routers use no PCA, fuzzy subject matching and 20 / 20 checked prompts.
- Results go to `outputs/router_diag_ours_v1`, so `router_diag_v1` (the paper's routers) stays as the "before".
- With the ours rows in `outputs/onepass_l19_v1` finished, the no-association test also generates base vs ours on every fired prompt.
- Eligibility is the router's own rule. The cache applies the fuzzy match exactly as the bank hook does, and keeps the exact-token mask beside it.
  - Hook parity also runs on the typo and partial-name prompts, where fuzzy matching decides.
  - The layer sweep and the refit check use the fuzzy rule. The robustness table still shows exact, fuzzy and none.
- The refit trains on the router's saved training set (`linear_router_dataset.json`: the balanced positives, rewrites and checked negatives) with its own L2 and no PCA.
- Ours makes one pass, so a later read layer costs a second pass. The penultimate downstream command keeps the edit at L19 and routes at the later layer (`ONEPASS=1 WRITE_LAYER=19 ROUTER_LAYER=26`).

**Fidelity checks** (in `manifest.json` and the report header):
- **Hook parity.** A sample of eval prompts goes through the real `LinearClassifierAssociationBank` hook, both batched and one at a time. It must agree with the cached decision (0 mismatches).
- **Dtypes match the pipeline.** Eval states are cached in bfloat16, as every evaluator runs. Train states are cached in float32, the dtype `fit_linear_router.py` fit in.
- **The refit reproduces the shipped router.** At L19 the refit must give the shipped router's decisions (`agreement_with_shipped_router`). The CV grid always contains the shipped L2 and uses the same family folds, so it reselects the shipped L2.

## What each output answers

**`report/summary.md` (L19)**
1. `scores_eval.png`: every example's score against the 0.5 line, per split.
   - Forget and paraphrase points show p(true row); neighbourhood and retain points show the max p over all rows.
   - Colour is the runtime decision. A hollow marker means no eligible head, so the prompt could not fire whatever its score.
   - Below the plot are the top-k worst prompts with their text and error reason.
2. `confusion_binary.png` and `confusion_nway.png`: FPR, FNR, precision, recall, F1 and the correct-row rate, per split and group. RWKU also gets a person-level matrix.
3. RWKU only: errors by probe type and by how the person is named.
   - Probe types: fill-in-the-blank, QA, QA paraphrase, adversarial by attack type, neighbour L1/L2.
   - Naming: full name, partial name only, or no name.
4. `scores_null.png`: the deployment false-positive rate on prompts with no association.
   - Prompt sets: generic text, PPL prefixes, unseen subjects in the forget templates, and protected subjects asked about other relations (RWKU: those should fire).
   - It also reports how many ungated heads would pass 0.5.
   - If the paper's `ours` run exists, it shows base vs ours outputs on every fired null prompt (changed, or "I don't know"-like).
5. Perturbed prompts: recall per perturbation family.
6. `umap_L16_L19.png`:
   - 10 should-fire and 10 must-not-fire points at L16 and L19 (n_neighbors 5, random_state 0).
   - Negatives prefer prompts that contain a protected subject, which are the ones the gate lets through.
   - Silhouette scores (cosine in full dimension, and 2D) and leave-one-out logistic and 1-NN accuracy put a number on the overlap.
7. `reliability.png`: ECE of the deployed probabilities. They are not meant to be calibrated, because the cutoff was moved for recall.

**`layer_sweep/layer_choice.md`**
- **FPR, FNR and F1 against layer** (`errors_vs_layer.png`). The same router is refit at every layer with the shipped recipe: masked class-balanced BCE, L-BFGS, CV-selected L2, the shipped calibration rule. It is scored on the official eval prompts, which are never fit on.
  - Under RWKU's subject gate, firing does not depend on the layer, so RWKU also shows the calibrated threshold gate.
- **L19 vs the penultimate layer.** Routing "improves" if F1 is higher with no higher eval FPR and no higher null FPR. Only then does the file print the downstream Eff/Gen/Spe/PPL command (`ROUTER_LAYER=26 … run_l16_paper_one.sh`).
- **Pipeline cost.**
  - The routing pass is timed for a full forward and for a forward stopped after the router layer.
  - The deployed early-write model already uses two passes: route, cached per prompt, then write at L16. A later read layer adds no pass; it only changes how far the routing pass runs.
  - Today the routing pass runs the full model; stopping it at the router layer is a free speed-up.
- **Robustness variants**, at L19 and the penultimate layer, each under exact, fuzzy and no eligibility:
  - `hardneg`: same subject with another relation (MCF: the collateral "preserve" half), and same relation with an unseen subject.
  - `aug`: new lead-ins, adversarial prefixes and context typos, applied to positives and negatives alike.
  - `concat`: L16+L19+L23 features.
  - `unmasked`: every head trained on every prompt. Heads then have to learn the subject themselves, which they must before the gate can be loosened.

  Each variant reports eval FPR/FNR/F1, null FPR, perturbed recall and ECE.

**No leakage by construction.**
- Augmentation and hard negatives come only from the training prompts, or from records outside the evaluation samples.
- Those outside records are split into disjoint halves: one feeds training-side extras, the other feeds the null test.
- Augmentation never reuses the calibration and audit lead-ins.

## Already visible on the real RWKU seed-1 data

Under the subject gate, firing depends only on name matching, so these numbers come from the
data alone. They were computed by string-matching the router's name surfaces; the cluster run
gives the exact Llama-token counts.

- **Spe (57.37 vs genie 63.70).** 15 of 279 neighbour probes contain a target's name surface and fire. 13 of the 15 are questions whose answer is the target's own first name:
  - "Rande Gerber is the husband of model ___ Crawford." ×6
  - "In Burlesque, the character Ali is portrayed by ___ Aguilera." ×4
  - "Feel This Moment … ___ Aguilera."

  A name gate cannot tell these from forget probes. They are arguably about the target, so a Spe† that excludes neighbour probes answered by the target's name would make the benchmark fairer, like MQuAKE's. Training "aboutness" negatives (the target mentioned as a secondary entity) is the router-side fix.
- **Gen.** Every held-out L1, L2 and L2-paraphrase probe names the person, so all of them fire. This matches ours 17.96 vs genie 17.75.
- **Level 3.** The 74 probes that cannot fire are all adversarial: reverse query 25, synonym manipulation 20, prefix injection 16, cross-lingual 8, other 5. They use aliases ("Kong Qiu", "Kǒngzi" for Confucius), descriptions, or another language. Alias surfaces mined from each target's own forget-side text (e.g. `intro.json`) are the fix; no layer or head change can reach them.

## Tests

`tests/test_router_diag.py` runs everything end to end on CPU, with a tiny Llama and a byte-level tokenizer:
- real MCF, ZsRE and RWKU data, plus a synthetic MQuAKE file;
- routers fit by the real `fit_linear_router.py`.

It checks that:
- the group sizes are the official ones;
- the cached states equal `extract_prompt_queries`;
- hook parity is 0;
- the refit at the router layer reproduces the shipped router, including under CV;
- the metrics and fuzzy gate behave as defined.
