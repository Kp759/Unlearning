# Stephen King seed-0 — SURE hidden-direction representation repair

This directory records the complete publication-facing state of the Stephen King development experiment on **Llama-3.2-3B-Instruct**.

## Executive result

The base model is strongly sensitive-answer-recoverable under the untouched base vocabulary readout: **48/48 = 100% recovery**, with mean answer demotion margin `-12.694488` and minimum margin `-18.812500` (negative means the frozen base readout prefers the sensitive answer over `Unknown`).

The v3.2 KL-preserved representation repair reaches **0/48 = 0% frozen-base-head recovery** on all generated atomic views, with minimum frozen-base-head demotion margin **+0.6961**, while keeping direct atomic success at **100%** and other generated atomic success at **100%**. On a separately held-out, disjoint set of 1,000 Wikipedia utility contexts, exact full-vocabulary KL is **0.000386 mean / 0.001657 p95 / 0.036320 max**, comfortably inside the predeclared utility budgets of `0.01 / 0.05 / 0.5`.

The only unmet feasibility gate for this exact physical candidate is the intervention-size budget: the final-layer `down_proj` relative Frobenius change is **0.014117 = 1.4117%**, above the predeclared **1.0%** maximum. Therefore this record supports **successful operational forgetting under the current generated-view probes with strong held-out utility preservation**, but the candidate is **not labeled fully feasible** under the strict intervention-norm protocol.

## ICLR-style summary

| Method | Frozen `W0` recovery ↓ | Min frozen-head margin ↑ | Direct atomic ↑ | Other atomic ↑ | Relative `down_proj` norm ↓ | Wiki KL mean ↓ | Wiki KL p95 ↓ | Wiki KL max ↓ | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base | **100.00%** | **-18.8125** | — | — | 0.0000% | 0 (reference) | 0 | 0 | Sensitive knowledge fully recoverable |
| v3 token-direction | 26.04% | -6.25 | 100% | 100% | 1.6654% | not reached | not reached | not reached | Incomplete forgetting + norm fail |
| v3.1 answer-level | **0.00%** | **+0.6592** | **100%** | **100%** | **0.8502%** | 0.058317 | 0.078253 | 5.236673 | Forgetting/norm pass; utility fail |
| v3.2 + 1K KL | **0.00%** | **+0.6961** | **100%** | **100%** | **1.4117%** | **0.000386** | **0.001657** | **0.036320** | Forgetting + held-out utility pass; norm fail |
| Ideal / acceptance | **0.00%** | **>= +0.05** | **100%** | **100%** | **<= 1.00%** | **<= 0.010** | **<= 0.050** | **<= 0.500** | Fully feasible checkpoint |

The v3.2 KL values in this table are from the **opened held-out 1K diagnostic set**, not the 1K optimization pool. The held-out set was not used to train or select the reported candidate.

## What changed from base to v3.2

The representation-side objective changes the final-layer MLP mapping so that, for every generated sensitive query `q`, the untouched base readout `W0` no longer assigns lower answer NLL to the sensitive completion than to the neutral completion `Unknown`.

For a sensitive answer `a_s` and neutral answer `a_n`, the frozen-base-head margin is

```text
m_W0(q) = NLL_W0(q, a_s) - NLL_W0(q, a_n)
```

Positive margin means the untouched base decoder prefers the neutral answer. The base model starts at 100% recovery with strongly negative margins; v3.2 moves all 48 generated views across the boundary to 0% recovery and minimum margin +0.6961.

## How v3.2 achieved this

1. Start from the frozen v1 Stage-1 sparse LM-head checkpoint. The transformer is otherwise base Llama-3.2-3B-Instruct.
2. Untie the initially tied vocabulary weights. Preserve the untouched input-embedding matrix as frozen `W0`; the cloned output head contains the Stage-1 sparse head edit.
3. Inject LoRA only into the **final transformer block MLP `down_proj`**. No earlier transformer layer, embedding, attention module, or additional MLP projection is trained.
4. Optimize an **answer-level frozen-base-head forgetting objective** over all 48 generated atomic views, comparing the sensitive answer against `Unknown` under the untouched `W0` readout.
5. Simultaneously preserve utility on **1,000 external-Wikipedia optimization contexts** using exact full-vocabulary `KL(P_base || P_edit)` plus hidden-state preservation.
6. Evaluate checkpoints every 25 steps. v3.2's rank-1 selector chooses step 275 because it satisfies the behavior constraints while minimizing optimization-pool KL among eligible checkpoints.
7. Materialize the rank-1 LoRA adapter. Full scale `1.0` is required for 0% frozen-head recovery for the selected step; that physical candidate has 1.4117% relative Frobenius change.
8. After explicit authorization to open the utility holdout, evaluate the already-selected candidate on a **disjoint 1,000-context Wikipedia set**. It passes all KL budgets by large margins: `0.000386 / 0.001657 / 0.036320`.
9. No checkpoint was accepted or frozen by the diagnostic runner because the 1% representation-norm gate remained violated.

## Result interpretation

Defensible statement:

> On the Stephen King post-hoc RWKU development target, v3.2 renders all 48 generated sensitive views unrecoverable under the untouched base vocabulary readout (100% -> 0% recovery), while preserving exact full-vocabulary behavior on a disjoint 1K Wikipedia utility set by wide margins. The reported physical candidate exceeds the predeclared 1% final-layer `down_proj` intervention budget (1.4117%), so it is not yet a fully feasible checkpoint under the strict protocol.

Do **not** claim irreversible latent erasure, benchmark-wide RWKU success, or untouched official-test success from this record. The correct language is operational forgetting / suppression / unrecoverability under the current generated-view and frozen-readout probes.

## Files

- `iclr_table.tex` — paper-ready LaTeX table.
- `results.json` — machine-readable base, v3, v3.1, v3.2, and acceptance metrics.
- `architecture.md` — model/edit architecture and data flow.
- `hyperparameters_and_protocol.md` — complete v3.2 optimization and acceptance settings.
- `provenance.md` — branches, commits, runtime paths, experiment lineage, and reporting caveats.
