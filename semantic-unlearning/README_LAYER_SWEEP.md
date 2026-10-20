# MCF layer-wise study (linear classifier, read = write layer)

SURE reads the request (router) and writes the residual row at the same block,
fixed at 19 so far. This sweep moves that block and reruns the full MCF method.
**Router V2 is not used anywhere**: the linear classifier is fit first at each
layer, and the rows are trained under it.

## Run on Wulver

```bash
cd /scratch/yl258/kp759/Unlearning
git fetch origin feat/mcf-layer-sweep && git switch feat/mcf-layer-sweep && git pull --ff-only
cd semantic-unlearning

# Regular SURE (rows trained under the linear classifier's own routing)
for LAYER in 1 3 7 13 19 23 27; do
  SWEEP_TAG=layer_sweep_linear_regular_v1 TRAINING_ROUTE=router NORM_SCALE=1 \
  WITH_DECOMPOSITION=0 bash scripts/run_mcf_layer_sweep_one.sh "$LAYER" \
    || echo "L$LAYER failed"
done

# Genie (rows trained under ground-truth routing, norm-matched steps)
for LAYER in 1 3 7 13 19 23 27; do
  SWEEP_TAG=layer_sweep_linear_genie_v1 TRAINING_ROUTE=oracle NORM_SCALE=auto \
  bash scripts/run_mcf_layer_sweep_one.sh "$LAYER" || echo "L$LAYER failed"
done

# Summaries
for TAG in layer_sweep_linear_regular_v1 layer_sweep_linear_genie_v1; do
  python scripts/summarize_mcf_layer_sweep.py --sweep-dir outputs/mcf_$TAG \
    --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global
done
```

The array job `mcf_layer_sweep.slurm` runs the same thing 3 layers at a time;
its header has the regular/genie `--export` lines. Stopped runs can be
restarted: finished stages are skipped.

## Per layer (`scripts/run_mcf_layer_sweep_one.sh L`)

| Stage | Script | Output |
|---|---|---|
| 1. Data + untrained rows at L | `prepare_mcf_association_source.py` | `L??/prep` |
| 2. Linear classifier at L (global, `--min-recall 0.98`, placement 0.1) | `fit_linear_router.py` | `L??/router` |
| 3. Train rows | `train_mcf_linear_router_rows.py` | `L??/linear_global` |
| 4. Official MCF eval (bf16, linear classifier routing) | unchanged evaluator | `official_mcf_eval.json` |
| 5. Linear classifier vs genie on the same rows (genie sweep only) | `evaluate_router_decomposition.py` | `decomposition/` |

Stage 3 modes:

- `TRAINING_ROUTE=router` (**regular**): the classifier routes every training
  prompt. Views it does not send to their own row cannot be edited, so they
  are left out of the objective and checkpoint selection (counts in
  `training_coverage`); the official eval still scores them. If a fact has no
  routed training view, the layer stops with an explicit error.
- `TRAINING_ROUTE=oracle` (**genie**): ground-truth routing on
  training-visible prompts. It separates "can layer L be written" from "can
  layer L be read". `NORM_SCALE=auto` scales LR and trust radii by
  median‖h_L‖ / median‖h_19‖ so each step is the same fraction of the
  residual stream.

The saved artifact always routes by the linear classifier.

## Fix included: last-block features

`extract_prompt_queries` read `output_hidden_states[L+1]`, which HF sets to the
**final-norm** output for the last block, while the runtime hook edits the
**raw** block output. Features now come from a hook on `model.model.layers[L]`
(identical for L < 27, tested; runtime parity is 0 mismatches at the last block).

## Caveats

- Seed 1 only.
- Training is capped at 3600 s per layer (PLAN budget); check
  `training_stop_reason` (`wall_time_budget` = time-limited, not converged).
- The sweep's L19 differs from the shipped L19 only in training the rows under
  the linear classifier instead of V2; compare it with the reference row.
- Read and write stay tied; decoupling them is a follow-up.

## MQuAKE

Same pipeline, MQuAKE's own data and trainer (no V2):

```bash
sbatch mquake_layer_sweep_regular.slurm   # outputs/mquake_layer_sweep_linear_regular_v1
sbatch mquake_layer_sweep_genie.slurm     # outputs/mquake_layer_sweep_linear_genie_v1
```

| Stage | Script |
|---|---|
| 1. Locked seed-1 forget associations + untrained rows at L | `prepare_mquake_association_source.py` |
| 2. Linear classifier at L (settings copied from `outputs/mquake_linear_2x2_seed1_v24/linear_router_report.json` when present) | `fit_linear_router.py` |
| 3. Rows: shipped MQuAKE optimizer (`train_direct_only`, 30 updates/association, 7200 s cap) | `train_mquake_linear_router_rows.py` |
| 4. Official MQuAKE eval (Eff, AtomicGen, retain, PPL) | evaluator with `--allow-imperfect-direct-routing` |

- Model and locked-split paths come from `outputs/mquake_fact_assoc_router_v2_seed1` (paths only).
- `--allow-imperfect-direct-routing` is new and opt-in: the evaluator normally aborts unless
  every direct rewrite routes to its own row; in the sweep that fraction is recorded
  (`forget_rewrite_route_correct`) instead, so an early layer still gets numbers.
- No router-vs-genie decomposition for MQuAKE (that script is MCF-only). The training
  report has final metrics under classifier routing for comparison.
- 24 h wall time: 7 layers × up to 2 h training + eval.

## ZsRE

```bash
sbatch zsre_layer_sweep_regular.slurm   # outputs/zsre_layer_sweep_linear_regular_v1
sbatch zsre_layer_sweep_genie.slurm     # outputs/zsre_layer_sweep_linear_genie_v1
```

Same four stages as MQuAKE: `prepare_zsre_association_source.py` -> `fit_linear_router.py`
(settings from `outputs/zsre_linear_2x2_seed1/linear_router_report.json` when present) ->
`train_direct_linear_router_rows.py --dataset zsre` (shipped ZsRE optimizer, 30
updates/fact, 3600 s cap) -> official ZsRE eval (Eff, Gen, Spe, retain, PPL). 14 h wall time.

## Untrainable facts (all benchmarks, regular mode)

A fact whose training prompts the linear classifier never sends to its own row
cannot be edited. Such facts keep a zero row and are listed in
`untrainable_fact_ids`; the rest of the layer trains and is evaluated
(`facts_trained` in the summary). This matters most for ZsRE and MQuAKE, where
a fact has a single direct prompt.

## Multi-seed (seeds 1-5), regular and genie

Six SLURM array jobs, one task per seed (1-5); submit all six at once:

```bash
sbatch mcf_multiseed_regular.slurm    ; sbatch mcf_multiseed_genie.slurm      # 36 h
sbatch zsre_multiseed_regular.slurm   ; sbatch zsre_multiseed_genie.slurm     # 14 h
sbatch mquake_multiseed_regular.slurm ; sbatch mquake_multiseed_genie.slurm   # 20 h
# aggregate (mean ± std per layer, per mode):
for D in mcf zsre mquake; do python scripts/summarize_layer_sweep_seeds.py --dataset $D; done
```

Outputs: `outputs/<dataset>_multiseed_<mode>_v1/seed<S>/L??/...`, summaries in
`outputs/<dataset>_multiseed_<mode>_v1/multiseed_summary.md`.

Changes from the seed-1 exploratory sweep:
- **NORM_SCALE=1 in both modes**, so regular and genie differ only in training routing.
- **MCF: step budget binds.** `MAX_TRAIN_SECONDS=10800`; seed-1 runs all stopped
  on the 3600 s cap, which gives different step counts at different depths.
- **MCF regular runs the decomposition** (linear classifier vs genie at eval on the same rows).
- **Seeds = different forget/retain samples** (ZeroUnlearn sampling). Seed 1 reuses the
  shipped ZsRE/MQuAKE locked splits; seeds 2-5 build theirs on first use
  (`outputs/{zsre,mquake}_locked_split_seed<S>`). Each job builds into its own temp
  dir and installs it with an atomic rename; the first job wins and the others reuse
  it (`flock` is not used: on GPFS it does not hold across nodes).
  `bash scripts/verify_locked_splits.sh` rebuilds each split and checks it byte-for-byte. All official evaluators take `--seed` (default: the run manifest's).
- Router calibration settings are the frozen seed-1 ones; the router is refit per seed and layer.

## ZsRE decomposition (why Gen is high)

```bash
sbatch zsre_decomposition.slurm      # all seeds x layers of the regular multiseed sweep, eval only
python scripts/evaluate_zsre_router_decomposition.py --summarize \
  --run-dirs 'outputs/zsre_multiseed_regular_v1/seed*/L??/linear_global'
```

Per run, on the same trained rows: official forget Eff/Gen under the linear
classifier vs under ground-truth (genie) routing; each forget paraphrase
classified as routed_correct / wrong_fact / ambiguous / below_threshold /
not_eligible (subject tokens not found; `subject_in_text` flags casing or
tokenization misses); and a threshold what-if (paraphrase recall vs false
firing on neighborhood and retain requests) from the captured logits.

## ZsRE router fix: reworded router training

The decomposition showed ZsRE rows already generalise (genie Gen ~1 at L19/L23)
and every missed paraphrase is a head score below the cutoff; lowering the
cutoff buys recall only by firing on the same subject's other relations
(L19: -4 -> 80% paraphrases routed, 17% same-subject false fire). Fix: train
the classifier on rewordings of each direct question.

```bash
sbatch zsre_reworded.slurm            # seeds 1-5, layers 19 23
python scripts/make_router_threshold_variant.py --summarize --root outputs/zsre_multiseed_reworded_v1
python scripts/evaluate_zsre_router_decomposition.py --summarize \
  --run-dirs 'outputs/zsre_multiseed_reworded_v1/seed*/L??/linear_global'
```

- `scripts/zsre_router_rewordings.py generate`: 6 rewordings per fact from the base
  model (generic few-shot examples; input = the training-visible direct question;
  rejects rewordings that drop the subject, contain the answer, or repeat).
  `examples` writes them as router families: 4 train, 2 development
  (calibration / audit), plus the usual context-prefix families.
- `linear_global_swap`: preview, the regular run's trained rows behind the new
  router (`scripts/swap_router_rows.py`). `linear_global`: full method, rows
  retrained under the new router. Report the full one.
- `run_zsre_layer_sweep_one.sh` gained `REWORDINGS=<file>` and `STOP_AFTER=router`.

Post-hoc cutoff variants (no retraining): `sbatch zsre_threshold_variant.slurm`
(`scripts/make_router_threshold_variant.py`; for calibrated-bias routers the
shift is folded into the bias).

### v2 rewordings (answer-consistency filter)

v1 rewordings sometimes drift to another relation of the subject ("performer
of Soul Merchant" -> "who played the role of Soul Merchant"), which labels a
same-subject/other-relation prompt as a positive. v2 keeps a rewording only if
the base model finds the fact's own answer about as likely after it as after
the direct question (mean per-token log-prob within 1 nat), and drops
near-duplicates (word-set Jaccard > 0.8).

```bash
sbatch --export=ALL,REWORD_TAG=reworded_v2,GEN_ARGS="--consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4" zsre_reworded.slurm
# after it finishes: all three routers on common same-subject negatives
sbatch --export=ALL,TAGS="regular_v1 reworded_v1 reworded_v2" zsre_router_compare.slurm
```

## Router bias rules: balanced validation calibration

Same weights and rows; only the bias is re-picked, on the VALIDATION set =
calibration + audit splits merged (held-out prompt families of the forget facts;
~80/20 train/validation overall). `balanced` maximises (recall + 1 - false fire)/2
with positives and negatives weighted equally; `--macro fact` averages per fact
first so every fact counts equally. Also: `target_fpr:<x>` (most recall with
false fire <= x) and `min_recall:<x>` (the shipped rule, on the merged set).

```bash
sbatch calibration_rules.slurm                         # balanced:fact + balanced:prompt, 30 tasks
sbatch --export=ALL,ARMS="target_fpr:0.1 target_fpr:0.05 min_recall:0.98" calibration_rules.slurm
python scripts/summarize_calibration_rules.py          # shipped vs raw vs recalibrated, test + validation
```
`scripts/recalibrate_router.py` checks that the recomputed shipped routes match
the stored ones and that the saved artifact reproduces the chosen decisions
through the runtime hook (both reported as mismatches, expected 0).

`min_recall` when the target is unreachable on validation (e.g. ambiguous
same-subject heads at shallow layers) now keeps the highest reachable recall,
as the shipped `calibrate_threshold` does; before this fix it fell back to the
cutoff with the least false fire, i.e. firing (almost) nothing. The outcome is
recorded as `recall_target_met` in `recalibration.json` and shown in
`summarize_arm_layers.py` ("recall target met").

## RWKU (seeds 1-5, layers 1 3 7 13 19 23 27)

RWKU-Batch-50-v1: batch seed s = people s..s+4 of RWKU's first ten targets, ten
Level-1/Level-2 probes each (50 forget rows). Eff = recovery on those 50; Gen =
recovery on held-out Level-1/Level-2 probes about the same people (+ paraphrase,
Level-3 adversarial); neighbours = locality (higher is better). Recovery = % of
greedy generations containing the answer, lower is better on forget sets.

Two gates, same data and row optimizer:
- `threshold` (`rwku_multiseed_regular_v1`): the current rule of the other
  sweeps, recall >= 0.98 on the calibration split; then the same 98% rule on the
  merged per-fact validation set (`calibration_rules_v1/rwku/.../recall0.98_fact`).
- `subject` (`rwku_multiseed_subject_v1`): RWKU's native entity-level gate;
  every prompt naming a protected person fires, the heads choose the row. Its
  "false fire" is 1 by design: for RWKU, other questions about a protected
  person are forget probes, not negatives.

```bash
# login node, once: fetch the pinned RWKU files for batch seeds 1-5 (people 1-9)
for s in 1 2 3 4 5; do python scripts/rwku_batch50.py --seed $s > /dev/null; done
sbatch rwku_multiseed.slurm            # 70 tasks = 2 gates x 5 seeds x 7 layers
python scripts/summarize_layer_sweep_seeds.py --dataset rwku --modes regular subject
python scripts/summarize_arm_layers.py --arm recall0.98_fact --datasets rwku
```
The first row of each RWKU table is the unedited model (the router's all-zero
rows through the same evaluator, `rwku_multiseed_base_v1`). Prompts use the
Llama chat template with the date pinned (`RWKU_CHAT_DATE_STRING`, default in the
driver `26 Jul 2024`) so tasks that cross midnight see identical prompts.

### RWKU genie: training-time vs evaluation-time

`TRAINING_ROUTE=oracle` (rwku_multiseed_genie_v1, "genie rows") only changes which
cases train each row; evaluation still uses the router, so it equals regular
whenever the router routes all 50 training probes. The paper's Table 6 genie is
evaluation-time (`evaluate_rwku_router_decomposition.py`): same-50 = exact trained
row forced (E); held-out = best of the same person's 10 rows, chosen with the
answer (S), an upper bound no router can reach.
```bash
sbatch rwku_decomposition.slurm                      # subject gate, seeds 1-5, L19 L23
python scripts/summarize_rwku_decomposition.py
```

## Inspecting the trained rows (`scripts/inspect_h_delta.py`)

For a separately trained single vector shared by all 50 seed-1 MCF facts, see
[the shared-vector experiment](README_SHARED_VECTOR.md). It compares against an
existing 50-vector IDK + end-token run using the same frozen router.

What the 50 residual rows Δe_i look like, whether they are the same, and whether one
vector could replace them all:

```bash
sbatch inspect_h_delta.slurm      # default RUN_DIR: outputs/mcf_layer_sweep_linear_regular_v1/L19/linear_global
# other run:  sbatch --export=ALL,RUN_DIR=<run>/linear_global,OUT_DIR=<out> inspect_h_delta.slurm
# rows only (no model, seconds):
python -u scripts/inspect_h_delta.py --rows-only --run-dir <run>/linear_global --out-dir <out>
```

| Part | Question | Output |
|---|---|---|
| 1. rows (no model) | Same value? (‖Δe_i − Δe_j‖ / norm) Same direction? (pairwise cos, cos to mean row, SVD energy) | `rows_summary.csv`, `rows_cosine.csv`, `rows_vectors.pt`, `rows_heatmap.png`, `rows_cosine.png`, `rows_norms.png`, `rows_spectrum.png` |
| 2. interchange | Can one vector replace all? Each forget view gets own row / zero / mean row / mean row at own norm / another fact's row / random direction at own norm, added at the training boundary (oracle routing); scored with the training metric (worst-view p(true) < 1e-6) | `report.json` → `interchange` |
| 3. per prompt | What h_L → h_L + Δe does under linear-classifier routing: norms, cos, angle, most-moved coords, logit lens, P(true), P(" I"); unrouted prompts must keep h' == h exactly | `report.json` → `prompts` |

The part-2 hook matches the runtime bank edit (checked: max relative difference ~1e-6 on routed-correct views).

## Paper table: base | ours (L16) | genie, 4 benchmarks × seeds 1-5

```bash
sbatch --array=5,10,15 l16_paper_multiseed.slurm    # smoke test: seed 1 of ZsRE, MQuAKE, RWKU
sbatch l16_paper_multiseed.slurm                    # all 20 tasks (task = 5*dataset + seed-1; mcf zsre mquake rwku)
python scripts/l16_paper_table.py                   # outputs/l16_paper_v1/paper_table.{md,tex,json} + per-seed csv
```

Per (dataset, seed), `scripts/run_l16_paper_one.sh` -> `outputs/l16_paper_v1/<ds>/seed<S>/`:
- **router**: the seed's linear classifier at L19 (`<ds>_multiseed_regular_v1/seed<S>/L19/router`,
  fitted with `STOP_AFTER=router` if missing). MCF additionally fits the factorized v2.2
  router (subject layer 6, relation 19+23+27 on validation; falls back to L19 if not deployable,
  see `router_choice.json`).
- **base**: official eval of the router with all-zero rows (= unedited model).
- **ours**: `train_mcf_jlens_feedback_rows.py --write-layer 16 --write-mode query --lens-type logit
  --lens-layer 24`; ZsRE/MQuAKE/RWKU add `--teacher-forced-views` (their metrics score every
  answer token under the gold prefix). Non-MCF data comes from the router's own training-visible
  prompts (`generic_examples`).
- **genie**: official eval of the same rows with `--genie` (`scripts/genie_routes.py`): official
  forget prompts -> own row, every other prompt -> no row, router not run. RWKU held-out probes
  have no row, so RWKU genie Gen = base by construction.

Table columns: MCF/ZsRE Eff, Gen, Spe (neighbourhood), PPL; MQuAKE Eff, AtomicGen, retain Eff, PPL;
RWKU same-50 recovery, mean held-out recovery (L1, L2, L2-paraphrase), neighbour recovery, PPL.
The diagnostics table shows the router used, final on-policy leak and genie routing hits
(genie direct-prompt routing should be ~100%; if not, the oracle keys did not match).

### Round 2: ZsRE router fix, trained zero rows, baselines, MQuAKE Spe†

```bash
bash scripts/submit_l16_round2.sh      # ZsRE+RWKU ours/genie rerun, then 100 baseline tasks (afterany)
python scripts/l16_paper_table.py      # adds GA/GD/NPO/RMU/ZeroUnlearn rows and the Spe† column
```
- **ZsRE router**: `ZSRE_REWORD_TAG=reworded_v2` (default) trains the L19 router on base-model
  rewordings of each direct question (answer-consistency + near-duplicate filters); official
  rephrases are never read. Base runs keep the regular router (zero rows = same model).
- **Zero rows**: `--oracle-unrouted` (default on, `ORACLE_UNROUTED=0` to disable) trains facts
  the router never routes with their own row forced during training only; test-time routing
  is unchanged, so ours stays honest and genie now bounds every fact.
- A finished `ours` trained under another router / flags (or with zero rows when the flag is
  on) is moved aside with its genie (`*.superseded_<time>`) and retrained.
- **Baselines** (`scripts/paper_baselines.py`, `--baseline` in the four official evaluators):
  GA, GD, NPO(+retain NLL), RMU (layer 7, MLPs of layers 5-7), ZeroUnlearn (vendored, layers
  16-18, target_true -> EOS, no retain requests). Same training-visible forget views as ours
  (the router ours trains on), retain = base-model self-generated continuations of 32 generic
  prompts, same seeds/splits/metrics. GA/GD/NPO: MLP weights of all layers, fp32 AdamW, lr 1e-5,
  5 epochs. `l16_baselines.slurm` task = 25*dataset + 5*(seed-1) + baseline.
- **MQuAKE Spe†**: retain Eff over retain facts whose direct prompt is not an official forget
  prompt (12-17% of retain facts repeat a forget fact; genie identifies them).

### MCF router ablation: linear L19 vs factorized v2.2

Same write layer (16), trainer, `--oracle-unrouted` and evaluator as the main MCF rows; only
the router that routes ours (training and test) changes to the stage-1 linear L19 router.
Base is copied from `l16_paper_v1` (it never depended on the router choice).

```bash
sbatch --export=ALL,TAG=l16_paper_mcf_linear19,MCF_ROUTER=linear,BASE_FROM=l16_paper_v1 \
       --array=0-4 l16_paper_multiseed.slurm          # MCF seeds 1-5
python scripts/compare_mcf_routers.py                 # -> outputs/l16_paper_mcf_linear19/mcf/router_comparison.{md,json}
```
The driver refuses `MCF_ROUTER=linear` under `TAG=l16_paper_v1` (its stale check would move
the v2.2 rows aside). The comparison flags seeds whose v2.2 arm fell back to L19.

### Hybrid router: learned subject head + linear fact heads (all 4 benchmarks)

The linear L19 router = exact subject-token match (eligibility) + one logistic head per fact.
v2.2 = learned subject head + relation softmax (needs MCF relation IDs). The hybrid keeps the
linear fact heads and replaces / backs up the exact match with v2.2's learned subject head
(MIL over (token, next token) pairs at layer 6), so it needs no relation labels and runs on
every benchmark. `artifact["learned_subject"]` in `linear_router.LinearClassifierAssociationBank`;
gate `learned` (head only) or `learned_or_exact` (head OR exact match).

`scripts/fit_hybrid_subject_router.py`: positives = the linear router's own training-visible
examples (+ typo / lower-case copies); negatives = the same prompts with the subject swapped
for other entity names (MCF subject pool, validation names unseen in fit), near-miss names
(shared name word), other MCF records' prompts. Whole-word matching; no ZsRE / MQuAKE / RWKU
evaluation prompt is read; MCF's forget-record prompts are only excluded. Threshold on
validation: recall >= 0.95, false fire <= 1% (other) / 10% (near miss), else exit 4.

```bash
bash scripts/submit_hybrid_smoke.sh        # seed 1 x 4 datasets, both gates (2nd array reuses the fit)
python scripts/compare_router_runs.py      # main (v2.2 / linear L19) vs hybrid learned vs learned_or_exact
```

### ROME and MEMIT baselines (from the ZeroUnlearn repository)

`--baseline rome|memit` (`scripts/paper_baselines.py`), vendored code and hparams of
`ZeroUnlearn/` (ROME layer 18; MEMIT layers 16-18; down_proj; mom2 on local wikidata, 100k).
Same requests as the ZeroUnlearn baseline (target_true -> EOS, `--edit-target unknown` for
'Unknown'; no retain requests). ROME is applied sequentially, one rank-one update per request:
the vendored `apply_rome_to_model` only applies `request[0]`, so ZeroUnlearn's harness edits a
single fact per run. The vendored ROME hparams file does not load as shipped (no
`context_template_length_params`, MEMIT-only keys, no `stats_dir`); `rome_hparams()` keeps the
class's fields, ROME's default context templates and STATS_DIR.
Two compatibility fixes in the vendored code (behaviour unchanged): `memit/compute_z.py`
accepts decoder-layer outputs as a tensor (newer transformers) or a tuple, as
`ZeroUnlearn/compute_z.py` already did; `rome/layer_stats.py` hands the wikidata rows to
`TokenizedDataset` as a list, as `layer_stats_retain` does (`[] + Dataset` raised). Cache
files are named as ZeroUnlearn's, so its covariance cache is reused when present.

```bash
bash scripts/submit_edit_baselines.sh      # 40 tasks; MEMIT MCF seed 1 first (covariance cache)
python scripts/l16_paper_table.py --ours-from mcf=outputs/l16_paper_mcf_linear19
```

### RWKU subject gate

`RWKU_GATE=subject`: every prompt naming a target person fires one of that person's rows
(the heads only choose which), instead of the threshold gate's per-fact decision. Aimed at
RWKU Gen (held-out questions about the same people); expected cost: neighbour prompts that
mention the person. Own TAG required.

```bash
sbatch --export=ALL,TAG=l16_paper_rwku_subject_v1,RWKU_GATE=subject,BASE_FROM=l16_paper_v1 \
       --array=15-19 l16_paper_multiseed.slurm
python scripts/compare_router_runs.py --datasets rwku --seeds 1 2 3 4 5 \
       --run threshold=outputs/l16_paper_v1 --run subject=outputs/l16_paper_rwku_subject_v1
```
