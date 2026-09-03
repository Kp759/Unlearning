# MCF per-example sensitivity-partitioned embedding rewiring V2.3.2

## Decision

The approach is correct as a parameter-selection principle:

```text
forget examples -> directions that must move
retain examples -> directions that must remain stable
```

It is not safe to use the raw ratio alone. Forget and retain gradients can have
different scales, an average can hide a single damaging retain context, and a
first-order direction can fail through the nonlinear frozen Transformer.
V2.3.2 therefore treats importance as a row nomination signal inside the
existing projection, cap, forward-reachability, and preservation framework.

This is not a prompt classifier. There is no runtime decision about whether an
input is sensitive. Only model-parameter gradients are classified before the
static embedding edit is trained and materialized.

## Sensitivity measurement

From exact zero, V2.3.2 measures one embedding-row gradient norm vector for
each of the 50 direct forget records and all 2,000 protection-fit retain
records:

```text
forget scalar = log p(target_new) - log p(target_true)
retain scalar = NLL(target_true)
```

For each physical embedding row `t` it records:

- RMS and maximum forget gradient norm;
- RMS and maximum retain gradient norm;
- forget and retain example coverage;
- `importance_ratio = forget_RMS / retain_RMS`;
- `hard_tail_ratio = forget_max / retain_max`.

RMS incorporates both strength and prevalence. The maximum retain norm is a
hard-tail guard so a row used by only one retain example cannot look safe just
because 1,999 other gradients are zero.

The thresholds are frozen before execution:

- low-forget floor: 1% of the strongest forget RMS;
- minimum keep ratio: 1.0;
- forget-specific ratio: at least 4.0 with retain coverage at most 1%;
- minimum hard-tail ratio: 0.25.

Rows are labelled `forget_specific`, `shared`, `retain_dominant`, or
`low_forget`.

## How sensitivity and projection combine

The ratio cannot declare a retain-observed row safe:

| Sensitivity class | Geometric state | Final treatment |
|---|---|---|
| forget-specific | no observed retain gradient | free, frequency-capped |
| forget-specific | retain-observed | projected |
| shared | retain-observed | projected |
| retain-dominant | any | excluded |
| low-forget | any | excluded |

The original efficacy, cap-adjusted potential, frequency bound, and retain
readout subspace remain mandatory. Token-presence liveness forcing is disabled;
after the final roles are assigned, every direct and synthetic prompt must pass
the V2.3.1 cap-aware linear bound and nonlinear directional sweep. A failure
stops before optimization.

Training retains V2.3.1 strict accepted-step progress and complete fit-bank
rollback. The Transformer and LM head remain frozen and bit-identical; only
selected input-embedding rows can change.

## Interpretation

This tests the user's proposed hypothesis directly. A passing sensitivity and
reachability preflight means the proposed parameter directions exist, not that
they are jointly compatible. A later optimization failure means the individually
useful directions cannot be combined inside the locked preservation envelope.

A preflight failure means the frozen-transformer, frozen-head embedding-only
architecture lacks the required local capacity under these rows, projections,
and caps. More training steps do not repair that result.

Seed 1 remains consumed architecture-development evidence, never an official
claim.

## Running

```bash
MODEL_PATH=... WIKIDATA_DIR=... MCF_PATH=... \
  bash scripts/run_mcf_projected_row_partition_embedding_v2_3_2_manual.sh \
  outputs/mcf_projected_row_partition_v2_3_2_seed1
```

Read these artifacts in order:

1. `method/per_example_row_sensitivity.json`
2. `method/row_partition.json`
3. `method/per_prompt_reachability.json`
4. `method/projected_training.json` if optimization is reached
5. `method/completion.json`
