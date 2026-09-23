# RWKU paper evaluation: trained Gen + both genie definitions

The underlying patch has been applied to the tracked Python sources on branch
`feat/router-ordered-fix-plan`. The original submitted patch is preserved at
`semantic-unlearning/patches/rwku-generalization-axes.patch` for audit. Do **not**
`git apply` it again after pulling: the source changes are already committed.

- `scripts/build_rwku_same50_paraphrases.py`: constructs rewordings of the
  same 50 trained facts, with provenance and rejections.
- `scripts/evaluate_rwku_router_decomposition.py`: supports
  `--same50-paraphrases`, CPU-only `--analyze-rows`, `same50_paraphrase`,
  gold-row exact genie on original/reworded trained probes, and same-person
  answer-informed genie on held-out facts.
- `scripts/run_rwku_paper_eval_with_genie.sh`: evaluates base, actual V2,
  exact genie, and generation-based same-person subject genie twice, then
  measures V2 PPL with the registered evaluator.

## Pull and audit

```bash
cd /scratch/yl258/kp759/Unlearning
git switch feat/router-ordered-fix-plan
git pull --ff-only origin feat/router-ordered-fix-plan
cd semantic-unlearning
python -m py_compile scripts/build_rwku_same50_paraphrases.py \
  scripts/evaluate_rwku_router_decomposition.py
```

## Optional: prior result answer-overlap diagnostic (CPU)

```bash
RUN=outputs/rwku_fact_assoc_router_v2_seed1_direct
python scripts/evaluate_rwku_router_decomposition.py \
  --analyze-rows "$RUN/decomposition_generation/rwku_router_decomposition_rows.json" \
  --output-dir "$RUN/answer_overlap"
```

This measures **answer-string** overlap, not proven fact/relation equivalence.

## Build and freeze trained-fact rewordings (GPU)

```bash
RUN=outputs/rwku_fact_assoc_router_v2_seed1_direct
GEN_MODEL=/absolute/path/to/independent/Qwen2.5-14B-Instruct
python -u scripts/build_rwku_same50_paraphrases.py \
  --run-dir "$RUN" --data-root data/rwku \
  --output "$RUN/same50_paraphrases.json" \
  --l1-backend local --generator-model "$GEN_MODEL" \
  --per-probe 2 --local-files-only --no-download \
  2>&1 | tee logs/rwku_build_same50_paraphrases.log
```

Review rejected probes, answer leakage, factual equivalence, and coverage of
**all 50** trained associations before calling these finalized Gen questions.
Do not feed these evaluation rewordings into training/prototype fitting.

## Evaluate existing checkpoint twice with both genies (GPU)

```bash
bash scripts/run_rwku_paper_eval_with_genie.sh
```

The runner includes `--arms base,v2,genie_exact,genie_subject` and
`--genie-select generation`. The exact genie forces the ground-truth row for
trained same-50 **and** reworded same-50. The subject genie tests all ten
same-person rows for each held-out fact, prioritizes generated-answer
non-recovery, and uses sensitive-answer log-probability to break ties. It is
**answer-informed, not deployable**.

Generated-answer recovery is lower-is-better for Eff/Gen/unseen facts; neighbor
recovery is a locality indicator. The printed Gen averages rewordings *within
trained association first*; the JSON's raw per-probe summary can weight facts
with two rewordings more than facts with one. PPL comes from a separate
registered V2 evaluator; it does not compute an independent frozen-base PPL.
Never combine older paper and direct-branch results without matching the
checkpoint, input hashes, and generation protocol.

Existing trained residuals are **not retrained** by any of these commands.
