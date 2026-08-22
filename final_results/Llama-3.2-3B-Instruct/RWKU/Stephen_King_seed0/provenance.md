# Provenance and reproducibility record

## Repository

```text
repository: https://github.com/Kp759/Unlearning
final-results branch: claude/final-results-rwku-v32
```

## Experiment lineage

```text
v1 head-only base commit:
  ab12969e2f90f30707e6f56f2b90573612706286
  plus local full-output-head consistency patch used by the frozen source run

v2 representation rescue:
  branch: claude/rwku-repr-rescue-v2
  commit: 4c9ead77b259b8c38f5afdef76ab927049ac617d

v3 token-direction representation experiment:
  branch: claude/rwku-hidden-direction-v3
  commit: f250b78d68627a51e5de56b352ed67a4814c347b

v3.1 answer-level frozen-base-head experiment:
  branch: claude/rwku-hidden-direction-v31
  commit: 0e0204630d9ac9105b3f71586bb296e4c1ab8954

v3.2 KL-preserved answer-level experiment:
  branch: claude/rwku-hidden-direction-v32-kl
  frozen commit: dc30b024ca092583f40a7e2ee2b15f236e9f449d

held-out utility diagnostic helper:
  branch: claude/rwku-v32-heldout-diagnostic
  commit used for launcher: b228b48fdb4dc3fb1f161536bd098915a41746c5
```

The frozen v3.2 branch was restored to `dc30b024...` after the held-out diagnostic helper was separated onto its own branch. The diagnostic therefore does not redefine the v3.2 training method.

## v3.2 implementation files

```text
semantic-unlearning/config/rwku/sure_head_hidden_direction_v32_kl_w1k_seed0.json
semantic-unlearning/scripts/rwku_sure_hidden_direction_v32_kl_w1k.py
semantic-unlearning/scripts/run_rwku_sure_hidden_direction_v32_kl_w1k.sh
```

## Diagnostic implementation files

```text
semantic-unlearning/scripts/rwku_v32_heldout_utility_diagnostic.py
semantic-unlearning/scripts/run_rwku_v32_heldout_utility_diagnostic.sh
```

The diagnostic runner intentionally bypasses the norm gate only long enough to measure the already-selected candidate on the disjoint held-out utility set. It does **not** accept, freeze, or promote the candidate to a feasible checkpoint.

## Runtime resources used

```text
MODEL=/home/ec2-user/models/Llama-3.2-3B-Instruct

CORPUS=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1

REAL_WIKI=/home/ec2-user/workspace/Unlearning/semantic-unlearning/data/wikipedia_sure_100020

SOURCE_RUN=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_h_w1k_ab12969_fullheadfix/rwku-h-w1k-stephen-king-atomic-seed0-v1

UTILITY_CACHE=/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/sure_wikipedia_stats/rwku_h_w1k_ab12969/Llama-3.2-3B-Instruct_rwku_stephen_king_excluded_docs1000_candidates100000_v1.pt
```

## v3.2 training output

```text
/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_hd_v32_kl_w1k/rwku-h-w1k-stephen-king-hidden-direction-seed0-v32-kl
```

Training log:

```text
/home/ec2-user/workspace/Unlearning/semantic-unlearning/rwku_hidden_direction_v32_kl_seed0.log
```

The training run ended fail-closed with

```text
RuntimeError: RWKU hidden-direction repair found no feasible checkpoint
```

because no selected physical candidate passed every pre-gate, specifically the 1% relative-Frobenius intervention limit. This was an intentional protocol failure, not a code crash.

## Held-out diagnostic output

```text
/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_v32_heldout_diag/rwku-h-w1k-stephen-king-hidden-direction-seed0-v32-kl/heldout_utility_diagnostic.json
```

Diagnostic log:

```text
/home/ec2-user/workspace/Unlearning/semantic-unlearning/rwku_v32_heldout_utility_diagnostic.log
```

Diagnostic result:

```text
candidate rank: 1
selected training step: 275
materialized scale: 1.0
frozen-W0 recovery: 0%
minimum frozen-W0 margin: +0.6961
direct atomic: 100%
other atomic: 100%
relative Frobenius: 0.014117 = 1.4117%
held-out KL mean: 0.000386
held-out KL p95: 0.001657
held-out KL max: 0.036320
rc: 0
```

## Data-boundary status

- Official RWKU records were unavailable to the learner and checkpoint selector in the target-only method pipeline.
- However, this Stephen King target is explicitly **post-hoc development** because official RWKU metrics from an earlier v2 run had already been observed before v3/v3.1/v3.2 method design.
- The 1K Wikipedia optimization pool and 1K held-out utility pool were disjoint.
- The held-out utility set was opened only after candidate selection and explicit authorization.
- The held-out utility set is now permanently considered **opened** and must not be used for subsequent tuning.

## Reporting boundary

Supported:

```text
complete operational forgetting on the 48 generated sensitive views under the untouched base readout
strong generalization of utility preservation to a disjoint 1K Wikipedia set
0% frozen-W0 recovery with held-out KL 0.000386 / 0.001657 / 0.036320
```

Not supported:

```text
irreversible deletion of latent knowledge
all possible prompt attacks or all RWKU sensitive queries
untouched official benchmark success for v3.2
full feasibility under the predeclared intervention-size protocol
```
