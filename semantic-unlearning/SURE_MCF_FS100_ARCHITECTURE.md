# SURE MCF target-aware direct-FS mode (legacy v6)

This document describes the post-hoc direct-only v6 ablation retained for
reproducibility. The current joint target-aware path is documented in
`SURE_MCF_TARGET_AWARE_GA_GD_ARCHITECTURE.md`; it trains target_true and
target_new together and includes official paraphrases so that both FS and GFS
are checkpoint acceptance conditions.

This is an explicitly benchmark-aware ablation layered after the
benchmark-neutral two-stage SURE learner. It exists because paper-facing MCF
forget success is defined using information that the neutral learner never
sees:

```text
FS = 100 * mean[NLL(original target_true) > NLL(original target_new)].
```

Consequently, a benchmark-neutral learner cannot guarantee FS = 100. The
target-aware mode reads the 50 sampled records' original `target_new` answers
and treats every direct pairwise comparison as a hard constraint. It must not
be reported as the benchmark-neutral SURE result.

```text
benchmark-neutral SURE v5.1 checkpoint
                  │
      original MCF target_true/target_new
      direct prompts only; no paraphrases,
      neighborhoods, retain cases, or PPL text
                  │
      cache frozen-transformer hidden states
      for both answer continuations
                  │
    exact mean sequence-NLL separation per record
                  │
       failing direct records and sensitive rows
                  │
      utility-whitened sparse residual bases
              ranks 2 -> 4 -> 8
                  │
       minimum Wikipedia-KL residual solve
                  │
     subject to non-tradeable inequalities:
       - every direct FS separation >= 0.01
       - all prior sensitive suppression floors
       - all prior best-other logit margins
       - Wikipedia mean/p95/max KL budgets
       - total sparse-delta norm budget
                  │
          actual BF16 materialization
                  │
        official MCF direct scorer: 50/50?
                  /                  \
                yes                  no
                 │                    │
          emit checkpoint          INFEASIBLE
```

The continuous solver uses an additional `0.05` direct-FS buffer, so its
default target is a sensitive-minus-reference NLL separation of `0.06`. The
actual BF16 checkpoint must retain at least `0.01`. Exact ties are failures,
matching the paper-facing evaluator.

The sequence constraints use the same average token NLL aggregation as the
official CounterFact evaluator. Because the transformer is frozen, every
teacher-forced hidden state is fixed. Sparse LM-head shifts therefore give an
exact differentiable sequence-NLL constraint without repeated transformer
forwards inside SLSQP.

Only sensitive-answer LM-head rows are editable. The repair does not train on
official paraphrases, neighborhood prompts, the 1,000 benchmark-retain cases,
or PPL text. Therefore:

- direct FS is guaranteed for any emitted checkpoint;
- GFS is not guaranteed;
- Spe/Spe-success, benchmark-retain KL, and PPL remain post-training audits;
- the mode is target-aware and is not directly comparable to the neutral SURE
  protocol without clearly disclosing the extra `target_new` supervision.

## Running

```bash
export SURE_MCF_TARGET_AWARE_FS=1
export OUTPUT_ROOT=outputs/mcf_sure_target_aware_direct_fs_v6_seed1
bash scripts/run_mcf_sure_minimal.sh \
  /path/to/Llama-3.2-3B-Instruct \
  data/multi_counterfact.json
```

Useful overrides are:

```bash
export OUTPUT_ROOT=outputs/mcf_sure_target_aware_direct_fs_v6_seed1
export MCF_SEEDS=1
export SURE_MCF_DIRECT_FS_MARGIN=0.01
export SURE_MCF_DIRECT_FS_SOLVER_BUFFER=0.05
export SURE_MCF_DIRECT_FS_RANK_LADDER=2,4,8
```

The neutral runner with the legacy environment flag runs neutral SURE first,
then invokes
`scripts/sure_mcf_direct_fs_repair.py`. The final official evaluator is called
with `--require-min-fs 100`; the shell run cannot print its completion message
or proceed to the retain-KL audit unless the official result is exactly 100.

`scripts/run_mcf_sure_fs100.sh` now points to the newer joint v7 FS/GFS path.

Key artifacts are written under `seed*/target_aware_direct_fs/`:

- `initial_direct_fs_report.json`
- `rank*_solver_history.json`
- `rank*_materialized_report.json`
- `final_direct_fs_report.json`
- `final_total_delta.pt`
- `checkpoint/`
- `config_used.json`

An infeasible run writes `infeasible.json` and preserves the neutral SURE
checkpoint. It never relabels a sub-100 checkpoint as FS-guaranteed.
