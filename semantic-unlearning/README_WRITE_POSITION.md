# Write position of the routed row (MCF seed 1, 50 facts, layer 19)

The linear classifier always reads the **last prompt token** at layer 19 and picks
at most one forget fact. This experiment changes only **which prompt positions
receive that fact's row Δe** (`write_mode` in the linear-classifier bank):

| Arm | `write_mode` | Positions that get +Δe | "Belgium is affiliated with" |
|---|---|---|---|
| A | `last` (shipped, default) | last prompt token | `with` |
| B | `last_subject` | last subject token + last token | `ium`, `with` |
| C | `subject_span` | every subject token + last token | `B el g ium`, `with` |
| D | `all_prompt` | every prompt token except BOS | everything after `<bos>` |

The subject span is the last occurrence of the routed fact's own subject token
pattern in the prompt (the same patterns the router's subject gate uses). If it
is absent (only possible under oracle routing), the arm falls back to the last
token and `bank.subject_fallbacks` counts it. Answer tokens are never edited, and
an unrouted prompt is bit-identical to the base model in every mode.

## Run on Wulver

```bash
sbatch mcf_write_position_seed1.slurm              # 4 array tasks = arms A-D, in parallel
python scripts/mcf_write_position.py summarize --root outputs/mcf_write_position_seed1
```

Per arm: copy the seed-1 L19 router (`outputs/mcf_multiseed_regular_v1/seed1/L19/router`,
else `outputs/mcf_layer_sweep_linear_regular_v1/L19/router`, else fit one) and
set `write_mode` -> train rows (`--training-route router --norm-scale 1`,
10800 s cap so the 1500-step budget binds, as in the seed-1 multiseed run) ->
official MCF eval (bf16, seed 1). Outputs: `outputs/mcf_write_position_seed1/arm_?/`,
table in `summary.md` with the seed-1 reference rows on top.

Arm A retrains the shipped setting under the same job, so it is the matched
baseline for B-D; the reference rows show how it compares with earlier seed-1 runs.

Caveat: learning rate and trust radii are per row step, unchanged across arms,
so in B-D the same step moves the residual stream at more positions.

Tests: `tests/test_write_mode.py`.

## Seeds 1-5: setting D vs setting A

```bash
sbatch mcf_write_position_multiseed.slurm      # array 1-5 = seeds; ARM=D by default
python scripts/mcf_write_position.py summarize-seeds --root outputs/mcf_write_position_multiseed --arm D
```

Setting A for each seed is the existing multiseed regular run
(`outputs/mcf_multiseed_regular_v1/seed<S>/L19/linear_global`): same router, same
training settings, write_mode `last` (the seed-1 arm A reproduced it exactly).
Each task trains D with that seed's own L19 router (fit if missing) and runs the
official eval with `--seed S`; seed 1 reuses `outputs/mcf_write_position_seed1/arm_D`.
The summary (`summary_D_vs_A.md/.json`) has Eff, Gen, Spe and PPL per seed, mean ±
std over seeds, the paired difference D − A, and the number of seeds where D < A.
Per-run logic: `scripts/run_mcf_write_position_one.sh ARM SEED OUT_DIR`.
