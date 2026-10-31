# LETHE-Bench — a benchmark for surgical unlearning

> *Forget one fact. Keep everything entangled with it. Show it is gone from the weights.*

This folder holds the Snorkel AI **Open Benchmarks Grant** proposal and the working benchmark behind it.

| Read this | For |
|---|---|
| [`PROPOSAL.md`](PROPOSAL.md) | The proposal: gap, design, what exists, asks, timeline |
| [`APPLICATION_FORM_ANSWERS.md`](APPLICATION_FORM_ANSWERS.md) | Short answers to paste into benchmarks.snorkel.ai/apply |
| [`EVIDENCE.md`](EVIDENCE.md) | Why existing fact-unlearning benchmarks are saturated (this repo's 4 × 5-seed study) |
| [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) | Exact definitions: construct, threat model, splits, 20 probe families, metrics, statistics |
| [`RED_TEAM.md`](RED_TEAM.md) | Snorkel's six red-team checks, and how LETHE-Bench handles each one |
| [`AGENT_TRACK.md`](AGENT_TRACK.md) | Agentic leakage track (prototype): the unlearned model as a search agent |
| [`scripts/README_CLUSTER.md`](scripts/README_CLUSTER.md) · [`scripts/aws/README.md`](scripts/aws/README.md) | Running every method: Wulver (SLURM) or one AWS GPU |
| [`DATASHEET.md`](DATASHEET.md) · [`DATA_LICENSE.md`](DATA_LICENSE.md) | Dataset documentation and licensing |
| [`ROADMAP.md`](ROADMAP.md) | Status, milestones, compute and expert-data asks, risks |
| [`results/`](results/) | Toy-world run (pipeline check), laptop pilot (`results/pilot/`) and the cluster leaderboards (`results/cluster/`) |

## Headline results: three models, 5 seeds each

`results/cluster/leaderboard_dev/leaderboard.md` (Wulver; `EVIDENCE.md` §4 has the per-axis tables). 11 methods and 6 reference cheats, 50 requests per seed, LETHE score (mean ± sd over seeds):

| Method | Llama-3.2-3B | Qwen3-4B | Llama-3.1-8B |
|---|---|---|---|
| SimNPO | 37.4 ± 20.1 | 31.7 ± 15.6 | 43.7 ± 13.6 |
| ZeroUnlearn | 21.3 ± 9.5 | 56.3 ± 5.8 | 15.6 ± 5.2 |
| SURE (ours, layer 19) | 28.4 ± 11.6 | 22.3 ± 0.7 | 18.7 ± 3.5 |
| MEMIT | 1.7 ± 1.8 | 30.7 ± 9.0 | 3.2 ± 0.4 |
| trained refusal (IDK-FT) | 11.4 ± 0.6 | 11.0 ± 1.3 | 10.9 ± 0.2 |
| task-vector negation | 5.7 ± 1.4 | 10.2 ± 3.8 | 7.7 ± 1.8 |
| RMU / GA / GD / NPO / ROME | 3.0–6.2 | 4.5–7.5 | 3.2–7.8 |
| six reference cheats † | 1.0–1.7 | 1.9–3.4 | 1.9–3.4 |

- No method is close to solved, and the ranking changes with the model: ZeroUnlearn is first on Qwen3-4B and fourth on Llama-3.1-8B; MEMIT scores 30.7 on Qwen3-4B and 1.7–3.2 on Llama.
- The red-team gate passes on every model: the largest cheat score is 3.4 against a gate of 35.
- Our own method is not the top row: SURE keeps locality 94–96 and utility 100 but is never routed on subject-free and multi-hop probes, so its erasure over held-out families is 7–22.

## Quickstart

```bash
cd snorkel_proposal
pip install -e .[test]                      # torch, transformers, tokenizers, jinja2, pyarrow, matplotlib
pytest tests/                               # unit tests + end to end on a micro toy model (~3 min)

# 1. Everything on a toy world, CPU only, no download (~10 min):
python -m scalpel toy --out results/toy
cat results/toy/runs/RED_TEAM_REPORT.md results/toy/leaderboard/leaderboard.md

# 2. Real data on a laptop (Qwen2.5-0.5B-Instruct, MPS/CPU):
bash scripts/run_pilot_local.sh

# 3. Cluster (Wulver): every method (incl. ROME, MEMIT, ZeroUnlearn, RMU) vs SURE, 3 models x 5 seeds
bash scripts/submit_scalpel_compare.sh fetch all && bash scripts/submit_scalpel_compare.sh smoke llama32_3b
bash scripts/submit_scalpel_compare.sh full all          # see scripts/README_CLUSTER.md
```

## The pieces

```
scalpel/
  schema.py        Fact / Request / Probe / Split; 20 families; canary; fingerprints
  sources/         MultiCounterFact, zsRE, MQuAKE, RWKU loaders -> one fact table (dedup, conflicts)
  relations.py     per-relation questions, paraphrases, verb phrases, fr/es/de questions
  probes.py        the families: how each forget fact is asked, and each retain control
  build.py         partitions (retain_train/dev/test), knowledge filter, conflict removal
  model_api.py     rendering (chat templates), teacher forcing, generation, logit lens, causality check
  grading.py       leak / refusal / hallucination labels for open-ended outputs
  metrics.py       per-family scores, E/L/U/R/LETHE, cluster bootstrap, paired tests
  attacks.py       logit lens, int8/int4 quantization, relearning on half the requests
  utility.py       MMLU, TriviaQA, perplexity
  runner.py        one method on one split -> results.json + probes.jsonl
  sequential.py    deletion requests arriving over time
  redteam.py       split validator + reference-cheat gate
  report.py        leaderboard.md / .json / pareto.png
  toy.py           synthetic world + tiny Llama trained to know it (tests, CI)
  methods/         noop, idk_all, prompt_cache_idk, subject_blocklist_idk, keyword_router_idk,
                   answer_ban (reference cheats) · ga, gd, npo, rmu · rome, memit, zerounlearn
                   (repository's vendored code) · checkpoint, plugin, sure
```

## Evaluating your own method

```python
# my_method.py
from scalpel.methods.base import Method, MethodCard
from scalpel.model_api import ModelHandle

class MyMethod(Method):
    card = MethodCard("my_method", "gradient", "what it does", runtime_components=[])  # declare hooks/filters
    def unlearn(self, base, requests, allowed_retain, ctx):
        # requests: [Request(prompt=..., answer=..., subject=..., relation=...)] - nothing else
        ...edit base.hf...
        return ModelHandle(model=base.hf, tokenizer=base.tokenizer, hf=base.hf)
```

```bash
python -m scalpel run --split SPLIT.json --model MODEL_DIR --method plugin \
    --method-arg target=my_method:MyMethod --out runs/my_method
python -m scalpel report --runs 'runs/*/results.json' --out leaderboard
```

A model already unlearned elsewhere can be scored with `--method checkpoint --method-arg path=DIR`. It must have been produced from the split's `requests` only.

## How to read a score

Every probe in a split is answered correctly by the base model, so:

- forget families report the **residual**: % still recalled, lower is better;
- retain families report % **retained**, higher is better.

E and L are harmonic means over families, so a method (or a shortcut) that gives up any whole family scores near zero on that axis. R is the worst case over the attacks. **LETHE** = harmonic mean of E, L, U and R; **LETHE_core** leaves out R. Rows marked † are reference cheats, which must score below 35.
