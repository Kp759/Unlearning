# Compressing the linear-router system

Everything here starts from the frozen configuration:

- **MCF, MQuAKE and ZsRE:** linear router with one global threshold.
- **RWKU:** linear router with the subject gate.

The system stores two banks with one 3072-d vector per fact each. This covers how to shrink both and how to check that nothing is lost.

| Part | Now (per fact) | Compressed form | How |
|---|---|---|---|
| Router `W` | 3072 floats | r floats + one shared r×3072 projection | `fit_linear_router.py --pca-dims r` (refit and recalibrated) |
| Residual bank Δe | 3072 floats | K floats + one shared K×3072 basis, optionally int8 | `compress_residual_bank.py` (post hoc, no retraining) |

## 1. Residual bank (`scripts/compress_residual_bank.py`)

The script copies the router, threshold, subject patterns and facts unchanged and replaces only the rows. The hook reads the query *before* it adds the row, so **every variant routes exactly like the source run**. Only the edit changes.

| Variant | Stored per fact | Shared | What it tests |
|---|---|---|---|
| `rank{K}` | K codes | K×3072 basis (truncated SVD) | How many directions the bank really needs. Each row is rescaled to its original norm. |
| `rank{K}_int8` | K int8 codes + 1 scale | int8 basis | Most compact dense form |
| `int8` | 3072 int8 + 1 scale | – | Precision only |
| `tied_answer` | 1 scale + group id | one direction per distinct answer | Is the edit really about the answer? |
| `tied_relation` | 1 scale + group id | one direction per relation | Relation-generic suppression (MCF, MQuAKE) |
| `tied_single` | 1 scale | one direction | One direction for everything |
| `control_shuffled` | – | – | Another fact's direction at the fact's own norm. **If this forgets as well, the rows are not fact-specific.** |
| `control_random` | – | – | A random direction at the fact's own norm. **If this forgets too, only the norm matters.** |

The controls are not compressions; they tell you what the compressions mean. Read them first.

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
export PYTHONPATH="$PWD/scripts"

# MCF (linear + global)
python -u scripts/compress_residual_bank.py \
  --run-dir outputs/mcf_linear_2x2_seed1_v24/arms/linear_global \
  --output-dir outputs/mcf_linear_global_bankcomp_seed1
bash outputs/mcf_linear_global_bankcomp_seed1/run_evals.sh

# MQuAKE (linear + global)
python -u scripts/compress_residual_bank.py \
  --run-dir outputs/mquake_linear_2x2_seed1_v24/arms/linear_global \
  --output-dir outputs/mquake_linear_global_bankcomp_seed1
bash outputs/mquake_linear_global_bankcomp_seed1/run_evals.sh

# RWKU (linear + subject gate)
python -u scripts/compress_residual_bank.py \
  --run-dir outputs/rwku_linear_subject_seed1_v24 \
  --output-dir outputs/rwku_linear_subject_bankcomp_seed1
bash outputs/rwku_linear_subject_bankcomp_seed1/run_evals.sh

# ZsRE: use the arm the seed-1 rule selects; a per-head arm needs --allow-per-head
```

`run_evals.sh` evaluates the source run only if its eval file is missing, then evaluates every variant with that benchmark's official evaluator. It ends by printing one table (`summarize_residual_compression.py`): bank size, compression ratio, row cosine to the original, and the benchmark metrics, with the source run as the first row.

**Flags:**
- `--ranks 1,2,4,8,16,32,64`: ranks at or above N are skipped, because they are exact.
- `--no-int8`, `--no-tied`, `--no-controls`: use these for a faster first pass.
- `--no-rescale`: keep truncated rows at their shrunken norm.
- `--include-ppl`: by default the evals pass `--skip-ppl`. Routing is identical to the source, and the PPL text routed nothing in any seed-1 run.

**`compression_report.json`** contains:
- the singular-value spectrum and the rank needed for 50/80/90/95/99% of the energy;
- pairwise row cosines;
- within-group vs between-group cosine for answers and for relations;
- per variant: reconstruction cosine and norm ratio, bytes as saved (and with floats at bf16), the ratio to full rows, the **break-even N** at which the variant becomes smaller than full rows, and bytes at N = 1K/10K/100K. The last is structural only; accuracy at those N is not measured.

## 2. Router (`fit_linear_router.py --pca-dims r`)

This refits the heads in an r-dimensional PCA space. The global threshold is recalibrated on the held-out calibration split, so a lower rank can't hide behind a stale threshold. Routing *can* change here, so read the audit numbers in the fit summary as well as the official evals.

```bash
for r in 16 32 64 128; do
python -u scripts/fit_linear_router.py \
  --run-dir outputs/mcf_fact_assoc_router_v2_seed1 \
  --output-dir outputs/mcf_linear_global_pca${r}_seed1 \
  --device cuda --dtype float32 --batch-size 8 \
  --min-recall 0.98 --threshold-placement-fraction 0.1 \
  --threshold-policy global --pca-dims $r \
  --lambdas 1e-6,1e-5,1e-4,1e-3 --local-files-only
done
```

## 3. The compression 2×2

Once the smallest router rank r\* and bank rank K\* that keep the metrics are known, run the bank compressor on the `pca{r*}` run as well:

| | Full bank | Bank rank K\* |
|---|---|---|
| **Full router** | source run | `bankcomp/variants/rank{K*}` |
| **Router rank r\*** | `mcf_linear_global_pca{r*}_seed1` | compress that run, `rank{K*}` |

## Reading the results

- **Fix the tolerance before looking.** For example: display-zero still passes, Eff/Gen within the spread across seeds 2–10, retain and Spe unchanged.
- **K\* is the smallest rank inside that tolerance.**
- **At seed-1 sizes, low rank barely saves memory.** A rank-K bank is smaller than full rows only when N > K·d/(d−K), roughly N > K. The same holds for a PCA-r router, which at N = 50 and r = 64 is *larger* than the full router. What seed 1 can show is how small K\* and r\* are. Showing that they stay small as N grows needs the scaling runs (MCF N = 50 → 500 → 2K).
- **Evaluated rows are exactly `reconstruct(compact)`**, the stored compact tensors materialized in float32. Tests check this. At runtime they are cast to the model dtype, as the original rows are.
