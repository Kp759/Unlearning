# MCF layer-wise study (read = write layer)

SURE reads the request (router) and writes the residual row at the same block,
fixed at 19 so far. This sweep moves that block and reruns the full MCF method.

## Run on Wulver

```bash
cd /scratch/yl258/kp759/Unlearning
git fetch origin feat/mcf-layer-sweep && git switch feat/mcf-layer-sweep
cd semantic-unlearning
sbatch mcf_layer_sweep.slurm                 # layers 1 3 7 13 19 23 27, 3 GPUs at a time
# after all tasks finish:
python scripts/summarize_mcf_layer_sweep.py \
  --sweep-dir outputs/mcf_layer_sweep_v1 \
  --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global
```

Custom layers: `sbatch --export=ALL,LAYERS="0 5 10 27" --array=0-3 mcf_layer_sweep.slurm`.
A timed-out task can be resubmitted; finished stages are skipped.

## Per layer (`scripts/run_mcf_layer_sweep_one.sh L`)

1. **Rows** — `run_mcf_fact_association_router_v2_seed1.py --layer L --training-route oracle --norm-scale auto`
2. **Router** — `fit_linear_router.py` at L: global threshold, `--min-recall 0.98`, placement 0.1 (the frozen seed-1 MCF policy)
3. **Official MCF eval** (bf16) → `official_mcf_eval.json`
4. **Decomposition** — learned router vs oracle route → read-vs-write attribution

Outputs: `outputs/mcf_layer_sweep_v1/L{LL}/{rows,linear_global}`.

## Controls, and why

| Control | What it removes |
|---|---|
| `--training-route oracle` | Rows are trained with ground-truth routing on training-visible prompts (train + development). Otherwise a weak V2 gate at an early layer would decide which prompts get a row and abort the preflight. The saved artifact routes by gate; the linear router is refit at L. |
| `--norm-scale auto` | Learning rate and trust radii are scaled by median‖h_L‖ / median‖h_19‖ at the boundary token, so each step is the same fraction of the residual stream at every depth. `row_to_boundary_norm_ratio` is reported. |
| Layer 19 in the sweep | Oracle-trained with scale 1.0. It should match the shipped L19 reference; if not, the oracle protocol itself moves the numbers. |

## Fix included: last-block features

`extract_prompt_queries` read `output_hidden_states[L+1]`. HF makes the last
entry the **final-norm** output, while the runtime hook edits the **raw** block
output. For L = 27 the router was fit on a different tensor from the one it
scores at runtime. Features now come from a hook on `model.model.layers[L]`.
This is identical for L < 27 (tested) and changes only the last block, for all
benchmarks.

## Caveats

- Seed 1 only (the runner is registered to seed 1). Confirmatory seeds come after the layer choice is frozen.
- Wall-clock cap is 3600 s training per layer (the PLAN budget). Early layers may hit it; see `training_stop_reason`.
- Read and write stay tied. Decoupling them (read at 19, write at L) is a follow-up; `sweep_router_read_layer.py` already covers read-only separability.
