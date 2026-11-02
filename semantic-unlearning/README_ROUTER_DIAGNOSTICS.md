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
- Results go to `outputs/router_diag_ours_control_v1`, the paper's router (see "Which router" below).
  `router_diag_ours_v1` (the router before 18dd066) and `router_diag_v1` (the regular routers) stay as they are.
- A cache built from another router file (sha256 in `manifest.json`) or without every layer
  (`router_fix.slurm` caches only 16 and 19) is moved aside to `seed1_other_router_<time>` /
  `seed1_partial_layers_<time>` and rebuilt, never resumed.
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

**`train_umap/summary.md`** (`scripts/router_train_umap.py`, CPU, from the cached `train` section): the
prompts each head is actually fit on, `linear_router_dataset.json`, about 20 positives and 20 negatives per fact.
- `umap_facts.png`: per fact, its positives (blue) against its own negatives (red) at L16, L19 and the
  penultimate block. Marker = source (template, template + lead-in, base-model rewrite; subject
  transplant, model-written hard negative, same-subject fact); a black ring = the deployed head gets
  that training prompt wrong. Drawn: the hardest facts at L19, then a seeded random sample.
- `umap_pooled.png`: every fact's training prompts in one projection.
- `separability_vs_layer.png` and the table: per fact, silhouette (cosine) and leave-one-out 1-NN at every
  block, the leave-one-out logistic probe at the drawn blocks, averaged over facts with the worst fact.
- The deployed head's errors on its own prompts, by split (`fit` = training error; `calibration` /
  `audit` are held out of the weight fit).

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

## Which router: the diagnostics in `router_diag_ours_v1` predate the paper's router

`outputs/router_diag_ours_v1` (written 2026-10-25 18:30) diagnosed the ours router of commit
265840b: 40 negatives per fact, threshold placed with `holdout_last`. Commit 18dd066
(19:22 the same day) changed the recipe to 20 negatives and `holdout_third` (the 0.98-recall
threshold placed on a held-out third of the rewrites), and the routers in
`outputs/<ds>_multiseed_ours_v1` were refit at 19:31–19:41. That refit router is the one every
later SURE run uses. Its seed-1 diagnostics are `outputs/router_diag_ours_control_v1`
(`router_fix.slurm VARIANT=control`). The counts quoted earlier in this README (19 / 4 / 9 misses,
5.7% no-association rate on MCF) describe the older router.

| seed 1, block 19 | older router (265840b) | paper's router (18dd066) |
|---|---|---|
| MCF: missed / retain fires / no-association rate | 19 / 3 / 5.7% | 3 / 16 / 28.3% |
| zsRE | 4 / 0 / 1.6% | 5 / 0 / 6.2% |
| MQuAKE | 9 / 4 / 5.6% | 8 / 10 / 6.2% |

Calibrating on held-out rewrites lowered the threshold. The paper's router misses almost no
MCF paraphrases, but fires on 28% of MCF prompts that must not fire. Most must-not-fire prompts
name a protected subject and ask about another of its relations. On MCF the open problem is
therefore precision, not misses.

### Rerunning everything on the paper's router

`router_diag_ours_v1` holds the only full run (all 4 benchmarks, every layer, penultimate test, RWKU
breakdown), and its router is the 20 positive / **40** negative one. The control run covered MCF, zsRE and
MQuAKE at blocks 16 and 19 only. To redo every task on the paper's 20 / 20 router (seed 1, all 4):

```bash
cd /scratch/yl258/kp759/Unlearning && git pull --ff-only && cd semantic-unlearning
sbatch --export=ALL,ROUTER_RECIPE=ours router_diag_staged.slurm
# -> outputs/router_diag_ours_control_v1/{RESULTS_EXPLAINED.md, ROUTER_LAYER_CHOICE.md}
#    and per benchmark: report/, train_umap/, layer_sweep/
```

The control's partial cache is moved aside to `seed1_partial_layers_<time>` and rebuilt at every layer.
The control never covered RWKU, so check first that all four routers were refit after 18dd066 (Oct 25 19:22):
`ls -l --time-style=+%F_%T outputs/*_multiseed_ours_v1/seed1/L19/router/fact_association_embeddings.pt`.

## Tested fixes for the router misses (seed 1)

Every fix is compared with the control, the paper's router under the same code; the
comparison is `scripts/router_fix_compare.py` (`outputs/ROUTER_FIX_COMPARISON.md`). The floor
variants are refits; the crop variants route with the paper's router itself.

| missed / retain fires / no-association rate | MCF | zsRE | MQuAKE |
|---|---|---|---|
| control: paper's router | 3 / 16 / 28.3% | 5 / 0 / 6.2% | 8 / 10 / 6.2% |
| subject-sentence crop | 2 / 16 / 28.3% | 5 / 0 / 6.2% | 8 / 10 / 6.2% |
| crop + text from the subject | 2 / 16 / 28.3% | 5 / 0 / 7.3% | 8 / 10 / 6.2% |
| coverage floor, no judge | 7 / 12 / 24.8% | 6 / 0 / 4.8% | 6 / 27 / 8.0% |
| coverage floor + crop | 5 / 12 / 24.8% | 6 / 0 / 4.8% | 6 / 27 / 8.0% |
| distractor copies + floor with judge | 17 / 2 / 8.2% | 4 / 0 / 6.4% | 4 / 44 / 9.6% |

Verdicts; every option stays off by default and the paper's router is unchanged:

- **Subject-sentence crop** (`route_crop=sentence`): safe but nearly idle. It decided one
  evaluation prompt (one MCF paraphrase) and no prompt that must not fire. The paper's router
  already fires on the other prefixed paraphrases. Not worth a second routing pass in the
  method; kept as an option (`ROUTER_RECIPE=ours_crop`).
- **Text from the subject** (`sentence_subject`): rejected. Nothing more recovered, 5 more
  no-association fires on zsRE.
- **Coverage floor without the judge**: rejected. MQuAKE retain fires 10 → 27, MCF misses 3 → 7.
- **Distractor copies + floor with the judge** (the first attempt, `VARIANT=rejected`):
  rejected. It was a refit of the older recipe's ingredients; MQuAKE retain fires 44.

The remaining misses (MCF 3, zsRE 5, MQuAKE 8 of 100 / 50 / 127 held-out phrasings) sit
where every recall gain tested so far costs same-subject false fires.


## Relation gate (analysis only; the paper's router does not use it)

**Why it is not in the method.** A gate that works on all four benchmarks costs recall on
zsRE and RWKU. The false fires it removes are mostly on the no-association set, which only
these diagnostics measure; the paper's metrics do not score it. On the official retain
prompts MCF has 16 false fires in 13,500. The gate is an opt-in artifact field
(`artifact["relation_gate"]`): the paper's routers have none and route exactly as before.
Results and reports are in `outputs/relation_gate_analysis_v1/`.

### Where MCF's false fires come from (seed 1)

On MCF the paper's router fires on 118 of 468 no-association prompts (25.2%) and on 16
retain prompts. Almost all of them name a protected subject but ask another relation:
"Gautier de Coincy originates from" fires the head of "The mother tongue of Gautier de Coincy
is". The heads read one last-token state, and in it the subject dominates. Refitting the heads
does not fix this. Best recall at a given number of false fires (test prompts plus no-association
prompts), cutoff picked on the test prompts themselves:

| refit of the heads alone | 0 false fires | ≤ 5 | ≤ 10 |
|---|---|---|---|
| current recipe | 0.28 | 0.68 | 0.77 |
| + same-subject / same-relation hard negatives | 0.45 | 0.79 | 0.82 |

### MCF with the true relation label

**The relation gate** (`scripts/fit_relation_gate.py`, `artifact["relation_gate"]`) is a
softmax probe over the 34 MCF relations. It reads the same block-19 query as the heads, so it
needs no extra pass. A head stays eligible only if p(the fact's relation | prompt) is at least
the cutoff. Heads, their cutoff and the rows do not change.
- **Training data:** 73k MCF prompts from the records the diagnostics treat as training-side.
  Prompts that name a protected subject or are evaluated anywhere are dropped. A quarter of
  them are repeated with an instruction suffix added (none is the diagnostics' "Please answer
  briefly.").
- **Accuracy:** 92.9% on held-out probe prompts.
- **Cutoff:** p ≥ 0.081, which lets 85% of the router's own training positives through
  (`--quantile 0.15`).

```bash
sbatch --export=ALL,VARIANT=relgate --array=0 router_fix.slurm
# -> outputs/mcf_multiseed_ours_relgate_v1/seed1/L19/router (+ relation_gate_report.json)
#    outputs/router_diag_ours_relgate_v1/mcf/seed1/report/summary.md
```

| MCF seed 1 | paper's router | + relation gate |
|---|---|---|
| forget + paraphrase recall | 0.980 | 0.980 |
| retain / neighbourhood fires | 16 | **0** |
| no-association fires | 118 / 468 (25.2%) | **10 / 468 (2.1%)** |
| same answer type fires (reported apart) | 20 / 20 | 3 / 20 |
| perturbed recall | 0.891 | 0.873 (suffixed 1.00 → 0.81) |
| hook parity (real bank vs cached) | 0 | 0 |

**What is left.** The 10 no-association fires fall into three groups:
- 6 are "<place> is located in / is in" asked of a location or continent fact.
- 3 are "<product> was created in the country of" asked of a manufacturer fact.
- 1, "Edinburgh Art Festival is located in", has the same text as the fact's own prompt.

In several of them the forgotten answer is a natural answer: "Grande Arche is located in" → Paris.

**Getting to exactly 0 costs recall.** Raising the gate cutoff:

| gate cutoff | recall | perturbed recall | no-association fires |
|---|---|---|---|
| p ≥ 0.2 | 0.953 | 0.80 | 6 |
| p ≥ 0.7 | 0.840 | 0.63 | 3 |
| p ≥ 0.8 | 0.780 | 0.55 | 1 |

At p ≥ 0.7 the 3 left are both Edinburgh Art Festival prompts and "Grande Arche is in". At
p ≥ 0.8 only "Edinburgh Art Festival is in" is left, the one-word variant of the fact's own
prompt.

**Two variants tested and rejected:**
- *Base-model answer check*: the head fires only if the frozen model predicts the fact's answer, or
  the answer type of its relation. The model puts the answer in its top 10 on only 49% of real
  forget prompts, and first-token answer types collide ("Sloven" for Slovenia and Slovenian).
  Best with a cutoff set on training prompts only: recall 0.95 at 61 no-association fires.
- *Gate without suffix views*: no-association fires 8, but suffixed perturbation recall 0.19.

### The same gate on all four benchmarks, without labels

Only MCF and MQuAKE facts carry a Wikidata relation. For a gate that works on every benchmark
(`scripts/relation_gate_universal_check.py`), each fact gets a relation profile instead of a label. The
profile is the mean probe distribution over the fact's own training positives. A head may fire
if log Σ_r p(r | prompt) · profile(r) reaches the cutoff. The cutoff is the share of the
fact's own training positives the gate may block, the same rule everywhere.
- **One probe:** 34 relations, 94k prompts. It excludes every protected subject and every
  test prompt of all four benchmarks, and gets 93.4% on held-out probe prompts.
- **Data, from `scripts/` only:** MCF prompts of training-side records,
  `mcf_shadow_relation_prompts.CALIBRATION_SCAFFOLDS`, `RELATION_ALTERNATE_TEMPLATES`, three
  question forms, and instruction-suffix copies.
- **Check:** with the gate off, the script reproduces the cached routes on all four benchmarks.

Seed 1, gentlest setting (block at most 5% of each fact's training positives):

| benchmark | recall | perturbed recall | retain / test false fires | no-association fires |
|---|---|---|---|---|
| MCF | 0.980 → 0.980 | 0.891 → 0.887 | 16 → 2 | 118 → 11 / 468 |
| MQuAKE | 0.985 → 0.965 | 0.772 → 0.706 | 10 → 4 | 44 → 14 / 872 |
| zsRE | 0.950 → 0.910 | 0.787 → 0.683 | 0 → 0 | 27 → 18 / 439 |
| RWKU | 0.262 → 0.257 | 0.268 → 0.257 | 1 → 1 | 3 → 3 / 264 |

Stricter settings cost more recall everywhere: blocking 15% takes zsRE recall to 0.84. The
gate helps where prompts look like Wikidata fill-in-the-blank relations (MCF, MQuAKE). zsRE
questions and RWKU chat prompts are read poorly: a fact's profile spreads over about 11
relations (zsRE) and 15 (RWKU), against about 3 on MCF. Every setting, pooled and per fact, is
in `universal_gate_check.json`.

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
sbatch -p gpu --qos=standard --gres=gpu:1 --mem=128G -t 1:00:00 --wrap "module load wulver Miniforge3 && \
  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning && \
  python -u scripts/relation_gate_universal_check.py"   # reads the router_diag_ours_control_v1 caches
```
