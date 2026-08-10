# MCF ZeroUnlearn-Style Locked-Probe Protocol

This track evaluates the existing Setting 5e + protected sparse LM-head repair
under the same **record sampling structure** used by ZeroUnlearn while keeping
MCF generalization/locality probes unavailable to the method until the final
frozen-checkpoint evaluation.

## Data split

For an MCF dataset `D` of length `N`:

- `retain_pool = D[:N//2]`
- `forget_pool = D[N//2:]`
- for each seed, sample the forget set first and then the retain set from one
  seeded Python RNG, exactly as ZeroUnlearn does;
- default comparison protocol: 50 forget records, 1000 retain records, seeds
  1 through 10.

The same 50 underlying forget facts are evaluated after unlearning. This is
therefore **not a fact-level unseen test set**.

## Locked prompt roles

The source MCF file is preserved for final evaluation. A repair-visible copy is
created in which these fields are emptied:

- `paraphrase_prompts`
- `neighborhood_prompts`
- `generation_prompts`

The `requested_rewrite` object and record order are preserved. The split builder
verifies that the original and repair-visible files select the exact same
record indices for every seed.

Consequently:

| Role | Stage 1 | Stage 2 repair | Final evaluation |
|---|---:|---:|---:|
| requested rewrite | yes | yes | yes |
| MCF paraphrases | no | no | yes |
| neighborhood prompts | no | no | yes |
| generation prompts | no | no | evaluator only |

This makes the final MCF `Gen` metric a genuine **held-out prompt-form
generalization test for the same deletion-request facts**.

## Frozen method configuration

The runner defaults to the already registered controlled MCF configuration
rather than tuning on the locked probes:

### Setting 5e

- steps: 600
- batch size: 1
- retain batch size: 4
- embedding/LM-head LR: `1e-4`
- forget weight: 2.0
- retain weight: 1.0
- MCF margin: 1.0
- post-training overlap alphas: 0.75 / 0.50 / 0.25

### Protected LM-head repair

- active direct-prompt margin: 0.25
- repair steps: 100
- repair LR: 0.005
- optimizer: AdamW
- squared-hinge weight: 2.0
- delta L2: `1e-4`
- retain KL weight: 0.1
- retain calibration records: 200
- repair rank: 2
- project away retain hidden states: enabled

The official MCF evaluator is deliberately **not** called after Stage 1 and is
not used for repair stopping or candidate selection. Final evaluation is run
once on the original MCF file after the repaired checkpoint is frozen.

## Usage

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

MCF_SEEDS="1 2 3 4 5 6 7 8 9 10" \
OUTPUT_ROOT=outputs/mcf_zerounlearn_locked_3b \
bash scripts/run_mcf_zerounlearn_locked_our_method.sh \
  /path/to/Llama-3.2-3B-Instruct
```

For one seed, useful for a Slurm array:

```bash
MCF_SEEDS="3" \
OUTPUT_ROOT=outputs/mcf_zerounlearn_locked_3b \
bash scripts/run_mcf_zerounlearn_locked_our_method.sh \
  /path/to/Llama-3.2-3B-Instruct
```

The important outputs are:

- `protocol/split_manifest.json`
- `seedN/setting5e/.../checkpoint`
- `seedN/repair_locked/checkpoint`
- `seedN/repair_locked/repair_summary.json`
- `seedN/official_eval_locked.json`
- `seedN/run_manifest.json`

Do not tune rank, margin, learning rate, stopping, or scale based on
`official_eval_locked.json` and then report the same seeds as held-out. If a
new configuration is chosen after looking at these results, register a new
development protocol and evaluate it on fresh seeds/probes.
