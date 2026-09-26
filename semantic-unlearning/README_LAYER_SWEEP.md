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
