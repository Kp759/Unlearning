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
