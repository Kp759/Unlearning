# MCF projected row-partition embedding rewiring V2.3.1

## Why this revision exists

The consumed V2.3 seed-1 run reached step 150 with direct failures fixed at 42
and synthetic failures fixed at 130. Meanwhile development top-1 and labelled
target drift exceeded the locked 0.05 bounds. `accepted_factor=1.0` was not an
efficacy result: the inherited trust predicate allowed equal forget loss, and
the liveness check established only that a prompt contained a non-excluded
token.

V2.3.1 changes no model boundary. Only selected input-embedding rows remain
editable; the Transformer and untied LM head remain frozen and bit-identical.
It strengthens the hypothesis test so an expensive optimization run cannot
start without demonstrated prompt-level capacity.

## Per-prompt reachability

Every training-visible direct and synthetic prompt is audited independently
from exact zero. Let `g_it` be the gradient of prompt `i`'s answer margin with
respect to physical embedding row `t`, `P_t` the final row-role projector, and
`c_t` the registered frequency-adjusted L2 cap. The exact maximum under the
local linear model is

```text
R_i = sum_t c_t ||P_t g_it||
```

The linear gate requires `base_margin_i + R_i >= 0.1`. Because the frozen
Transformer is nonlinear, V2.3.1 also constructs the cap-saturating direction
for that prompt and performs real forward passes at fixed factors
`{1/16, 1/8, 1/4, 1/2, 1}`. At least one factor, or the unchanged Base state,
must reach margin 0.1. Every direct and synthetic prompt must pass both gates.
Otherwise optimization is refused and no candidate is saved.

This is a prompt-level capacity check, not a claim that individually feasible
directions are jointly compatible. Joint compatibility remains the purpose of
training and the full-bank preservation gates.

## Strict optimization

V2.3.1 rejects projected proposals whose forget loss is unchanged. Every
accepted step must improve the current forget minibatch by at least `1e-5` and
must keep its active retain bank feasible. An infeasible state additionally
requires a strict reduction in normalized constraint violation.

Every 50 steps the complete training-visible fit bank is evaluated. If any fit
constraint fails, the entire embedding delta is restored to the most recent
full-fit-safe state. Development never controls an update or rollback; it is
used only to select a jointly passing checkpoint. The incremental training
receipt is rewritten after every full check so an interrupted run remains
diagnosable.

The preservation limits are unchanged:

- top-k KL mean at most `1e-4`;
- top-k KL maximum at most `0.01`;
- Base top-1 log-probability drift at most `0.05`;
- labelled retain-target log-probability drift at most `0.05`.

## Interpretation

A reachability failure is a useful terminal result for this registered
parameterization and cap envelope. It means at least one required prompt lacks
enough locally usable embedding direction after retain projection; learning
rate changes and longer training cannot create that missing direction.

A reachability pass permits optimization but does not guarantee a final
candidate. Failure after a pass instead demonstrates joint incompatibility or
nonlinear preservation failure.

Seed 1 is consumed architecture-development evidence. This revision must not
turn another seed into an official test until its architecture and thresholds
are frozen from training-visible evidence.

## Running

```bash
MODEL_PATH=... WIKIDATA_DIR=... MCF_PATH=... \
  bash scripts/run_mcf_projected_row_partition_embedding_v2_3_1_manual.sh \
  outputs/mcf_projected_row_partition_v2_3_1_seed1
```

Inspect `method/per_prompt_reachability.json` first. If it fails, the deliberate
termination before Stage 3 is the result; do not resume the output directory.
