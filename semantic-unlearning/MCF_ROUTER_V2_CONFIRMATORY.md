# MCF Router V2 confirmatory campaign: seeds 2--10

Seed 1 is the development seed. It is **not** included in confirmatory mean/std.

## Frozen configuration

The confirmatory runner preserves the seed-1 Router V2 configuration:

- Llama-3.2-3B-Instruct frozen backbone
- hidden size 3072
- intervention layer 19
- one residual row per unique protected association
- Adam learning rate 0.05
- 12 backtracks
- gradient clipping 1.0
- target probability 1e-6
- trust-radius schedule unchanged
- 12 Router V2 negative controls
- absolute router threshold disabled (alpha=-1)
- nonseparable margin slack 0.02
- separable-gap operating point 0.10
- ambiguity margin 0.02
- no unique-subject bypass
- MCF development route-recall floor 0.90
- 50 forget associations per seed
- nominal 1500 row-step opportunities / 30 per association
- 3600 s training wall-time cap

Only the official sampling / training RNG seed changes.

## 1. Preflight all confirmatory seeds

Run this before expensive training:

```bash
bash scripts/preflight_mcf_router_v2_confirmatory_2_10.sh
```

It writes separate preflight directories:

```text
outputs/mcf_fact_assoc_router_v2_seed2_preflight
...
outputs/mcf_fact_assoc_router_v2_seed10_preflight
```

The runner refuses training if fitting prompts do not route perfectly or if authored-development routing falls below the frozen 0.90 floor.

## 2. Train and evaluate one seed

Example:

```bash
bash scripts/run_mcf_router_v2_confirmatory_seed.sh 2
```

This produces:

```text
outputs/mcf_fact_assoc_router_v2_seed2/
  association_manifest.json
  training_report.json
  fact_association_embeddings.pt
  official_mcf_base_eval.json
  official_mcf_eval.json
```

The evaluator reads the split seed from the manifest and verifies that saved forget case IDs exactly match official MCF sampling for that seed. It evaluates both the matched frozen base and Router V2 checkpoint.

## 3. Run seeds 2--10

After every preflight passes, run each seed independently:

```bash
for SEED in 2 3 4 5 6 7 8 9 10; do
  bash scripts/run_mcf_router_v2_confirmatory_seed.sh "$SEED"
done
```

For cluster scheduling, submit seeds independently rather than relying on one long interactive allocation.

## 4. Aggregate

After all nine confirmatory runs finish:

```bash
python scripts/aggregate_mcf_router_v2_confirmatory.py
```

Outputs:

```text
outputs/mcf_fact_assoc_router_v2_confirmatory_2_10_summary.json
outputs/mcf_fact_assoc_router_v2_confirmatory_2_10_per_seed.csv
```

Confirmatory statistics are computed strictly across seeds 2--10. Seed 1 is included only as a separate development reference when its result exists.

The summary contains mean, sample standard deviation, minimum, maximum, and paired Router-V2-minus-base deltas for Forget Eff, Forget Gen, Forget Spe, ReleasedAccuracy Eff/Gen, Retain Eff/Gen/Spe, and runtime-aligned PPL.

No confirmatory result may be used to retune the frozen architecture or hyperparameters.
