# Final Results

This directory stores publication-facing experiment records in a stable hierarchy:

```text
final_results/
  <model>/
    <database>/
      <setting>/
```

Each setting should contain the paper-ready table, machine-readable metrics, method/architecture description, hyperparameters and protocol, and provenance/caveats.

## Current entries

- `Llama-3.2-3B-Instruct/RWKU/Stephen_King_seed0/` — SURE hidden-direction representation repair, including the v3/v3.1/v3.2 progression and the v3.2 held-out 1K Wikipedia utility diagnostic.

## Reporting rule

A result is called **feasible** only if one physical BF16 checkpoint simultaneously satisfies every predeclared behavior, frozen-base-head, intervention-norm, and held-out utility gate. Diagnostic results that bypass one gate are retained and labeled explicitly; they are never silently upgraded to feasible results.
