# Projection edit (MCF seed 1, layer 19, setting A)

Same router as the shipped run: the linear classifier reads the last prompt token at
layer 19 and picks at most one forget fact. Instead of adding the trained row Δe_i,
the hook removes fact i's answer directions from the last-token state:

    h' = h - alpha * U_i U_i^T h        (U_i orthonormal, one row per answer token, k <= 3)

No row training, no model weight change; unrouted prompts stay bit-identical to base.
`alpha = 1` removes the component (projection onto the null space of U_i^T);
`alpha = 2` reflects it.

Answer directions (`scripts/mcf_projection_edit.py build`):
- `jlens`: mean over 256 generic MCF prompts (forget subjects excluded) of
  d logit_w / d h_19 at the last prompt token, for each answer token w.
- `local`: the same gradient averaged over the fact's own training views only.

```bash
sbatch mcf_projection_seed1.slurm          # 4 arms: jlens/local x alpha 1/2
python scripts/mcf_projection_edit.py summarize --root outputs/mcf_projection_seed1
```

The table puts the arms next to setting A with Δe (seed-1 multiseed regular run),
with the official MCF metrics plus a canonical-prompt diagnostic: P(first answer
token) before/after, facts below 1e-6, and the fraction of ‖h‖ removed.
Implementation: `edit_mode="project"` in `LinearClassifierAssociationBank`
(`scripts/linear_router.py`); tests in `tests/test_write_mode.py`.

## Alpha sweep (`mcf_projection_alpha_sweep_seed1.slurm`)

Same J-lens directions for every alpha (reused from `P_jlens_a1`), alphas
0.5 1 1.5 2 2.5 3 4 6 by default (`ALPHAS="..."` to change). Per alpha: artifact,
official MCF eval, and `routed-generate`: greedy 24-token continuations of the
canonical + development forget prompts through the real router and edit, scored for
- answer leakage: the true answer anywhere in the continuation (routed prompts),
- fluency: mean base-model NLL per continuation token (higher = less natural text),
- repetition: share of continuations with a repeated 3-gram.

Two reference rows get the same scoring: `base` (the seed-1 router with zero rows,
i.e. the exact base model; also gives base Eff/Gen) and Δe (the seed-1 multiseed
regular run). Spe/PPL cannot show over-editing because they never route; the
generation columns can.

```bash
sbatch mcf_projection_alpha_sweep_seed1.slurm
python scripts/mcf_projection_edit.py summarize --root outputs/mcf_projection_alpha_sweep_seed1
```

## ZeroUnlearn free-generation leak (seed 1)

Eff/Gen only score the token right after the prompt. `mcf_zerounlearn_leak_seed1.slurm`
applies original ZeroUnlearn (target_true parity mapping) and runs the same 24-token
greedy leak check on the same 250 forget prompts, then writes
`outputs/zerounlearn_generation_leak_seed1/comparison.md` (base / Δe / projection
α 2.5, 3 / ZeroUnlearn): leak@start, leak@24, leak@24 excluding prompts that already
contain the answer, base-model NLL, repeated 3-grams, plus Eff/Gen/Spe/PPL.

    sbatch mcf_zerounlearn_leak_seed1.slurm
