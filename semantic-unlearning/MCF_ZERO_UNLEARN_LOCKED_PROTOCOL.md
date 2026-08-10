# MCF ZeroUnlearn-Style Forget-Only / Locked-Probe Protocol

This track is the fair data-access comparison against ZeroUnlearn. The method
receives only the sampled MCF forget requests before freezing. The sampled MCF
retain records and the benchmark-provided paraphrase/neighborhood probes are
reserved for final evaluation.

## Record sampling

For an MCF dataset `D` of length `N`:

- `retain_pool = D[:N//2]`
- `forget_pool = D[N//2:]`
- for each seed, sample the forget records first and the retain evaluation
  records second from one seeded Python RNG, matching ZeroUnlearn ordering;
- default protocol: 50 forget records, 1000 retain evaluation records, seeds
  1 through 10.

The same 50 underlying forget facts are evaluated after unlearning. This is a
prompt-level holdout protocol, not a fact-level unseen test.

## Data access before the checkpoint is frozen

A repair-visible copy of MCF is created with these fields removed:

- `paraphrase_prompts`
- `neighborhood_prompts`
- `generation_prompts`

Stage 1 and Stage 2 then sample the same 50 official forget records while
requesting **zero MCF retain records**.

| Data | Stage 1 | Stage 2 | Final evaluation |
|---|---:|---:|---:|
| 50 forget requested rewrites | yes | yes | yes |
| forget paraphrases | no | no | yes (`Gen`) |
| forget neighborhoods | no | no | yes (`Spe`) |
| 1000 MCF retain records | no | no | yes (utility/retention) |

Therefore the 1000 MCF retain records do not influence gradients, row
restoration, repair selection, KL regularization, hidden-state projection, or
checkpoint selection.

## Stage 1: forget-only Setting 5e

The dedicated trainer is `scripts/mcf_forget_only_setting5e.py`.

Default forget-side settings:

- 50 forget records
- 600 steps
- batch size 1
- embedding/LM-head LR `1e-4`
- forget weight 2.0
- MCF margin 1.0
- retain weight 0
- retain KL 0
- post-training row restoration computed from forget token groups only
- overlap alpha tuple remains 0.75 / 0.50 / 0.25 for compatibility, but all
  retain-overlap groups are empty because no MCF retain records are loaded.

## Stage 2: forget-only sparse LM-head repair

The compatibility entry point is `scripts/mcf_forget_only_active_repair.py`.
It runs the existing sparse repair implementation with:

- direct requested-rewrite prompts only
- active margin 0.25
- repair steps 100
- repair LR 0.005
- AdamW
- squared-hinge weight 2.0
- delta L2 `1e-4`
- rank 2
- MCF retain records 0
- retain KL weight 0
- retain calibration records 0
- retain-hidden projection disabled

The official MCF evaluator is not called after Stage 1 or during Stage 2.

## Final evaluation

After the repaired checkpoint is saved and frozen, the original untouched MCF
file is used with the same seed to evaluate:

- the same 50 forget records on rewrite prompts (`Eff`);
- their previously unseen MCF paraphrases (`Gen`);
- their previously unseen neighborhood prompts (`Spe`); and
- 1000 sampled retain records for post-unlearning utility/retention.

Thus the main comparison is:

```text
TRAIN / UNLEARN:
  50 forget requested_rewrite records
  0 MCF retain records
  0 paraphrase/neighborhood probes

FREEZE CHECKPOINT

FINAL EVAL:
  50 forget records + held-out prompt variants
  1000 retain records
```

## Wulver usage

```bash
cd /scratch/yl258/kp759/Unlearning
git checkout claude/setup-project-structure-JQ7fN
git pull
cd semantic-unlearning

sbatch \
  --export=ALL,MODEL_PATH=/path/to/Llama-3.2-3B-Instruct \
  slurm/run_mcf_zerounlearn_locked_3b.slurm
```

Important outputs per seed:

- `seedN/setting5e_forget_only/.../checkpoint`
- `seedN/repair_forget_only/checkpoint`
- `seedN/repair_forget_only/repair_summary.json`
- `seedN/official_eval_locked.json`
- `seedN/run_manifest.json`

Do not retune rank, margin, learning rate, or stopping based on the final held-out
`Gen` values and then report those same runs as untouched held-out results.
