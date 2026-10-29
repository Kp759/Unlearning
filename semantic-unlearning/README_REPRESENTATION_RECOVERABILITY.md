# Representation recoverability (stage 6 of the paper pipeline)

## Why

Gao, Unal, Rangamani and Zhu, *An Illusion of Unlearning? Assessing Machine Unlearning Through
Internal Representations* (AISTATS 2026, arXiv:2604.08271), show on image classifiers that most
unlearning methods only misalign the classifier from the features: output forget accuracy drops to
zero while a linear probe on the frozen hidden features recovers the forgotten classes at close to
the original accuracy (CIFAR-10, one class: output 0.00 for every method; probe 67–93 vs 94 for the
original model). Their conclusion is that unlearning has to be evaluated at the representation level
as well as at the output.

The critique applies to SURE directly. SURE adds one residual row at the write layer when the
per-fact router fires. Every block below the write is exactly the base model, and the blocks above
see the edited state at the request boundary only. Eff / Gen / Spe read the output, so they cannot
tell whether the answer is gone from the residual stream or merely steered away from the output
head. The same question applies to every baseline in the table (GA, GD, NPO, RMU edit MLP weights;
ROME / MEMIT / ZeroUnlearn edit one or a few layers), so the measurement runs on every run dir.

## What is measured

`scripts/representation_recoverability.py` runs the evaluated system exactly as the official
evaluators do (same prompt text, request boundary = the prompt, router or genie deciding, baseline
edit applied with the same seed) and records the residual stream at the last prompt token after
every decoder block. Forget prompts are the official forget prompts of each benchmark:

| dataset | forget groups | control (retain-side) prompts |
|---|---|---|
| MCF | `direct` (rewrite), `paraphrase` | `neighbourhood` (other subjects, the same true answer), `retain` |
| ZsRE | `direct`, `paraphrase` (rephrase) | `neighbourhood` (NQ locality question), `retain` |
| MQuAKE | `direct`, `question` (atomic question) | `retain` |
| RWKU | `direct` (same-50), `heldout_l1`, `heldout_l2`, `heldout_para` | `neighbour` (Level 1 / 2 neighbour probes) |

Three linear readouts decode the first token of the forgotten answer at every block:

- **lens**: logit lens, the model's own final norm + unembedding on the block's state. Training-free.
- **tuned**: an affine map h_l → h_final fitted by ridge regression on ≈8k token positions of the
  dataset's own retain-side prompts (retain records, paraphrases and neighbourhood prompts; RWKU
  neighbour probes) under the *same* unlearned system, then norm + unembedding. A trained linear
  probe that never sees a forget prompt or a forget label, fitted on prompts of the same kind as
  the probes so the map is defined where it is applied. (The `data/wikidata` PPL fixture was tried
  first: it has 10 distinct sentences, its token states span a 166-dimensional subspace, and the
  lens was undefined on every prompt state outside it.) Features are standardised per dimension
  before the solve, the penalty is chosen per block by held-out MSE, and the json records the
  training rank (`translator.rank_ok`). At the last block the map is fitted on h_final itself, so
  `headline.final_block_check` requires tuned ≈ lens there; a mismatch means the readout is broken.
- **probe**: the paper's supervised probe. Multinomial logistic regression h_l → answer over the
  forget answers, trained on control prompts that have one of those answers but name none of the
  forget subjects (MCF neighbourhood prompts share the forget fact's answer by construction), tested
  on the forget prompts. A probe taught only on other subjects that still reads the answer off a
  forgotten subject's prompt shows that the *answer* is represented, not merely the subject.
  Reported only with ≥ 2 answer classes of ≥ 3 training prompts (MCF always; the other datasets
  when retain answers coincide with forget answers).

Per block and group: top-1 rate (%) of the answer's first token and its mean probability (lens,
tuned); probe accuracy on the forget prompts and on held-out control prompts. The control numbers
show that the readout works at all (a lens that fails on retain prompts measures nothing).

**Headline RR** (lower is better) = max over blocks of the tuned-lens top-1 rate on all forget
prompts: the strongest label-free linear readout a defender has to assume. `RR_lens` and
`RR_probe` likewise, each with the block it occurs at; `output_top1` is the lens at the last
block, which is the model's own first-token recall (sanity: it tracks 100 − Eff). Run on the base
dir, the same number is the comparator the paper uses (probe accuracy of the original model).

## Reading the result for SURE

Expected pattern: below the write layer RR equals the base model's (those blocks are the base
model), at and above it the tuned lens and the probe show how much of the answer the row removes
from the residual stream for the prompts the router fires on. For prompts the router misses, every
block equals base. Genie isolates the actuation: same rows, perfect routing. A method that scores
Eff ≈ 0 with RR ≈ RR_base has suppressed the output and left the representation, which is the
paper's "illusion".

## Running

```bash
# one run dir (any of base / ours / genie / <baseline>)
python scripts/representation_recoverability.py --run-dir outputs/onepass_l19_v1/mcf/seed1/ours \
    --dataset mcf --mcf-path <MCF json> --device cuda --local-files-only
python scripts/representation_recoverability.py --run-dir outputs/onepass_l19_v1/mcf/seed1/genie \
    --dataset mcf --mcf-path <MCF json> --genie --device cuda --local-files-only

# stage 6 on the finished dirs of a TAG (base, baselines in $BASELINES, ours, genie)
bash scripts/run_recoverability_one.sh onepass_l19_v1 mcf 1
sbatch recoverability.slurm                                    # all 20 (dataset, seed) tasks
TAG=l16_ours_v1 sbatch recoverability.slurm
TAG=l16_paper_v1 BASELINES="ga gd npo rmu zerounlearn rome memit" sbatch recoverability.slurm

# as part of a full run
RECOVERABILITY=1 bash scripts/run_onepass_paper_one.sh mcf 1

# the paper table gains an RR column
python scripts/l16_paper_table.py --root outputs/onepass_l19_v1 --ours-label "Ours (L19, one pass)"
```

Outputs: `<run>/representation_recoverability.json` (per-block numbers, headline, the forget items
with their groups and rows, runtime counters) and `.md` (one table, one row per block). Options:
`--layers 8,12,16,19,24,27` (subset of blocks), `--no-probe`, `--translator-tokens 0` (lens only),
`--ridge`, `--max-control`, `--overwrite`. Baseline dirs need the evaluator's baseline flags
(`--baseline ga --baseline-views-from <router dir> ...`); stage 6 passes them.

Cost: one forward pass per prompt plus the retain text, then per block one d×d ridge solve and
(with the probe) one LBFGS fit: a few minutes per run dir on Llama-3.2-3B.

Tests: `tests/test_representation_recoverability.py` (captured states are base below the write and
edited at it for the routed prompt only, lens at the last block equals the model output, translator
and probe on known data, headline), `tests/test_l16_paper_table_rr.py` (RR column).
