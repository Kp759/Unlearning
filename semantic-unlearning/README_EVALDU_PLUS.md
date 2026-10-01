# Eval-DU+ (FT-Mul-Chunk) with SURE

Benchmark from *Learning-Time Encoding Shapes Unlearning in LLMs* (ICLR 2026,
[arXiv 2506.15076](https://arxiv.org/abs/2506.15076),
[code/data](https://github.com/wrh14/learning_time_shapes_unlearning)).
100 fictitious people; each person's facts are written together in one
biography chunk, e.g.

> Sloane Lee, born in 1908 in Washington state, works as a banker and is married to Zane Ross. She is the mother of Avery Ross and Zachary Ross.

862 facts (562 family-graph relations + birth year / birthplace / job per person).
The model learns the chunks by fine-tuning; unlearning then removes the paper's
100-fact split (or the 12-people split) and must keep the other facts.

```bash
# on the login node (compute nodes may be offline):
git clone --depth 1 https://github.com/wrh14/learning_time_shapes_unlearning data/evaldu_plus_upstream
sbatch evaldu_plus_seed1.slurm
cat outputs/evaldu_plus_v1/seed1/L19/linear_global/official_evaldu_eval.md
```

## Pipeline (`scripts/run_evaldu_plus_seed1.sh`, resumable)

| stage | script | notes |
|---|---|---|
| fine-tune | `finetune_evaldu_plus.py` | paper recipe: full FT on FT-Mul-Chunk, lr 1e-5, batch 16, 4 epochs, 1-epoch warmup, linear decay, wd 0.01; knowledge score before/after in `finetune_report.json` |
| split | `build_evaldu_plus_split.py` | forget = upstream `unlearn_fact_id.pt`; training-visible = UL-Mul paraphrases of the forget facts, cut before the completion word |
| prep | `prepare_evaldu_association_source.py` | one row per forget fact; either of the fact's people makes a prompt eligible |
| router | `fit_linear_router.py` | L-BFGS, `--decision-rule calibrated_bias` (cutoff folded into the bias) |
| rows | `train_direct_linear_router_rows.py --dataset evaldu` | MQuAKE row optimizer, 30 updates / fact |
| eval | `evaluate_evaldu_plus.py` | fine-tuned model vs SURE on the same probes |

## Router trained on rewordings (`REWORD=1`)

Seed 1 (default router): when the fact's own row fires, held-out forget
paraphrases drop ~88%, but the own row fires on only 45% of them and 41% do
not fire at all (73% of the remaining forget score). The router saw 2-3 UL
prefixes per fact, and its cutoff was calibrated on context-prefix copies of
them. `REWORD=1` applies the ZsRE v2 recipe (`scripts/evaldu_router_rewordings.py`):
the fine-tuned model rewrites each forget fact's UL prefixes (names kept
verbatim, completion not leaked, answer-consistency margin 1.0 nat/token,
Jaccard <= 0.8); 4 rewordings join training and 2 join calibration / audit.
Only forget prefixes are read; test paraphrases, chunks and retain facts never.
Rows are trained exactly as before.

```bash
sbatch --export=ALL,REWORD=1 evaldu_plus_seed1.slurm
cat outputs/evaldu_plus_v1/seed1/L19_reworded/linear_global/official_evaldu_eval.md
```

The report's last tables split the remaining forget score by route outcome
(no fire / wrong row / own row).

## Metric (the paper's)

Knowledge score = exp(mean log-prob) of the completion word's tokens given the
preceding tokens (upstream `eval_completion_word`). Fact level = mean over the
fact's probes. Normalized = SURE / fine-tuned model.

| probe set | what | groups |
|---|---|---|
| test | 3 held-out paraphrases per fact (the paper's extraction trade-off) | forget ↓; retain same person, other people, all ↑ |
| chunk | the biography chunks cut before a fact's completion: other facts of the person are stated earlier in the same text | same, plus retained facts stated after a forgotten one |
| unlearn | UL-Mul paraphrases (SURE's training-visible prompts) | sanity |

Each block is reported for all probes and for probes whose prefix names one of
the fact's people (SURE routes on the person).

## Caveats

- The paper reports Norm-AUC / AUC over an unlearning trajectory (GA / TV checkpoints). SURE gives one operating point; compare with GA / TV run on the same fine-tuned 3B model, not with the paper's 7B / 8B numbers.
- 4 of the 100 forget facts have no UL paraphrase that names a person before the completion (or share it with another forget fact), so SURE cannot address them; they count as not forgotten.
- About 15% of test probes name no person before the completion ("Serving as a banker is ___"); see the "prefix names the person" rows.
- The family graph is dense: most retained facts share a person with some forget fact.
- Check `knowledge_after` in `finetune_report.json` before reading SURE numbers; if the facts are not learned, rerun with `FT_ARGS="--epochs 8"`.
