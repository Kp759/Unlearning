# Running the SCALPEL comparison on Wulver

This runs every method against SURE on the same splits:

- **6 reference cheats:** noop, idk_all, prompt_cache_idk, subject_blocklist_idk, keyword_router_idk, answer_ban
- **Gradient and representation baselines:** GA, GD, NPO, RMU
- **Locate-and-edit baselines:** ROME, MEMIT, ZeroUnlearn (the repository's vendored code, run through `scripts/paper_baselines.py`)
- **Ours:** `sure_paper`, the paper's method: classifier input and Δe both at layer 19, in one forward pass, with rows trained by the paper's feedback trainer (no logit lens, no PCA). Also `sure`, the original SURE; `sure_paper_l16` is the two-pass ablation that writes Δe at layer 16

It covers Llama-3.2-3B-Instruct, Qwen3-4B and Llama-3.1-8B-Instruct, 5 seeds each. Every run gets the full core track, the open-ended scorer, the three attacks, and the strict tier for SURE and the cheats. A sequential-deletion track runs for GD, NPO and MEMIT. A red-team gate runs per seed, and leaderboards are written per model and across models.

## 0. Once: pull and prepare (login node, internet)

```bash
cd /scratch/yl258/kp759/Unlearning && git pull
cd semantic-unlearning/snorkel_proposal
module load Miniforge3 && conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

bash scripts/submit_scalpel_compare.sh fetch all     # model weights (Llama is gated: `hf auth login` once), MQuAKE, RWKU
# the Wikipedia corpus for ROME / MEMIT / ZeroUnlearn covariance, if not already built for the paper:
(cd .. && bash scripts/submit_edit_baselines_wiki.sh corpus)
```

## 1. Smoke test: Llama-3.2-3B, seed 1, every method

```bash
bash scripts/submit_scalpel_compare.sh smoke llama32_3b
# or with the SURE-R variants in the same chain (one build, one covariance job):
SURE_VARIANTS="sure sure_r sure_r_noviews sure_r_exact sure_r_hardneg sure_r_free sure_r_idk sure_r_bake" \
  bash scripts/submit_scalpel_compare.sh smoke llama32_3b
```

The build stops at once if RWKU, the source of U's MMLU and TriviaQA questions, is incomplete. The `fetch` step downloads it, or run `python ../scripts/rwku_data.py --manifest /tmp/rwku_manifest_check.json`.

| Stage | Job | Notes |
|---|---|---|
| covariance | `sc_cov` | Layers 16–18. Reuses `data/stats/<model>/wikipedia_20231101_100k_stats` from the paper run when the model path matches |
| build | `sc_build` | Split + validation + noop (caches the base-model reference) |
| methods | `sc_method` | 13 tasks, one per method |
| SURE | `sc_sure` | 1 task, 24 h |
| sequential | `sc_seq` | GD, NPO, MEMIT, 5 batches |
| report | `sc_report` | After everything, whatever the exit status |

Check `results/cluster/llama32_3b/dev/seed1/runs/RED_TEAM_REPORT.md` (it must say PASS) and `results/cluster/llama32_3b/dev/leaderboard/leaderboard.md`.

## 2. Full run

```bash
bash scripts/submit_scalpel_compare.sh full all          # 3 models x 5 seeds (= scripts/submit_all.sh)
bash scripts/submit_scalpel_compare.sh full qwen3_4b     # or one model
```

Per model that is 65 method tasks, 5 SURE tasks, 5 build tasks, 5 sequential tasks, 1 covariance task, plus 1 report job.

**Rough time:**

| Model | Per method task (A100) | SURE task |
|---|---|---|
| 3B | 10–40 min | Up to 10 h of row training (budget) |
| 8B | About 2× the 3B time | Up to 10 h of row training (budget) |

## 3. Robust SURE ablation (after the splits exist)

```bash
SEEDS="1" bash scripts/submit_scalpel_compare.sh robust llama32_3b      # 7 SURE-R variants on seed 1
bash scripts/submit_scalpel_compare.sh robust llama32_3b                # all 5 seeds
```

The variants are `sure_r`, `sure_r_noviews`, `sure_r_exact`, `sure_r_hardneg`, `sure_r_free`, `sure_r_idk` and `sure_r_bake` (`SURE_ROBUST.md`). Each is one task per seed (24 h limit). `sure_r_bake` needs the covariance at SURE's layer, which the covariance job now builds. The leaderboard puts them next to `sure` and the baselines.

## 4. Agentic leakage track (after smoke / robust)

```bash
bash scripts/submit_scalpel_compare.sh agent llama32_3b                          # 13 methods, seed 1
AGENT_METHODS=noop,ga,rome,sure,sure_r bash scripts/submit_scalpel_compare.sh agent llama32_3b
```

This runs one job per method (12 h limit), then the report job writes `results/cluster/<model>/dev/seed1/agent/agent_leaderboard.md`. SURE methods reuse their finished core run's router, so a SURE task exits at once if that run is missing. Settings are in `AGENT_TRACK.md`; extra flags go through `AGENT_ARGS`, e.g. `AGENT_ARGS="--max-steps 6"`.

## Options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `METHODS` | all 13 non-SURE methods | Comma list, e.g. `METHODS=rome,memit,zerounlearn` |
| `SEEDS` | `1 2 3 4 5` (smoke: `1`) | Seeds = different forget samples |
| `PARTITION` | `dev` | `test` builds hidden splits; needs `SCALPEL_TEST_SALT` (keep it private) and optionally `PRIVATE_TEMPLATES` |
| `N_FORGET` | 50 | Requests per split (scaling: 200, 1000) |
| `SKIP_SURE` / `SKIP_SEQ` | 0 | Skip the SURE or sequential arrays |
| `SURE_VARIANTS` | `sure` (robust: the 7 SURE-R variants) | Space-separated SURE methods for the SURE array |
| `SEQ_METHODS` | `gd,npo,memit` | Methods for the sequential track |
| `MOM2_TEXT` / `MOM2_DATASET` | `/project/.../wikipedia_20231101_100k` / same name | Covariance corpus and cache name |
| `MODEL_PATH` | resolved per model | Override the weights directory |

## Outputs

```
results/cluster/<model>/<partition>/seed<S>/
  split.json  split_summary.md  validation.json  cache/      (base reference, base-model attacks)
  runs/<method>/results.json  probes.jsonl                  (one per method, incl. sure)
  runs/RED_TEAM_REPORT.md                                   (gate: written by the report job)
  sequential/<method>/sequential.json
  sure_work/                                                (SURE's own prep / router / rows)
results/cluster/<model>/<partition>/leaderboard/leaderboard.md   (mean ± sd over seeds)
results/cluster/leaderboard_<partition>/leaderboard.md           (all models)
```

Re-collect at any time with `bash scripts/submit_scalpel_compare.sh report llama32_3b qwen3_4b llama31_8b`.

## Re-running and failures

- Every stage skips finished work. Re-submitting the same command only redoes missing or failed tasks. A half-written run directory is moved aside as `*.incomplete_<time>`.
- Downstream jobs depend with `afterany`, so one failed seed does not block the others. A method task whose split is missing exits with a clear message.
- **ROME / MEMIT / ZeroUnlearn fail with "covariance cache missing":** the `sc_cov` job did not finish, or it used a different model path. Check `logs/scalpel_cov_*.err`.
- **The report job is rejected:** it asks for the `general` CPU partition. If that partition name differs, change `#SBATCH --partition` in `scripts/slurm/scalpel_report.slurm`, or run the report on a GPU node.
- **The 8B model needs ≥ 75 GB of GPU memory.** Jobs check `nvidia-smi` and exit early on smaller GPUs.

## What is not tested yet

The ROME / MEMIT / ZeroUnlearn adapters were verified locally only up to the vendored algorithms. Requests reach them in exactly the `paper_baselines.zero_requests` format (`tests/test_core.py::test_edit_baseline_requests_match_paper_baselines`), but the vendored code is CUDA-only, so the edits themselves first run on Wulver. The smoke run above is the check. The SURE adapter has run end to end (laptop pilot); the cluster settings (layer 19 for 3B, 24 for Qwen3-4B, 22 for 8B) follow `scripts/paper_model_config.sh`.
