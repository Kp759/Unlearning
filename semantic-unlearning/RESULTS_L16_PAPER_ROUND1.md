# Results: base vs ours (L16) vs genie, 4 benchmarks × 5 seeds (round 1)

Run: `l16_paper_multiseed.slurm` (jobs 1427505 smoke, 1427506 full), commit `98b5545`, finished
2026-10-18. Model: Llama-3.2-3B-Instruct. Outputs: `outputs/l16_paper_v1/<dataset>/seed<S>/{base,ours,genie}/`.
Table: `python scripts/l16_paper_table.py` → `outputs/l16_paper_v1/paper_table.{md,tex,json}`.

All 20 runs completed (5 seeds × 4 datasets). Genie routed 100% of direct prompts on every
seed, and the v2.2 router calibrated on all five MCF seeds (no fallback to L19).

## Main table (mean ± std over seeds 1–5)

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | PPL |
|---|---|---|---|---|---|
| MCF | Base | 13.46 ± 2.61 | 10.26 ± 3.50 | 19.88 ± 4.66 | 11.31 |
| | **Ours** | **0.001 ± 0.000** | **0.05 ± 0.11** | 19.24 ± 4.89 | 11.31 |
| | Genie | 0.001 ± 0.000 | 0.002 ± 0.001 | 19.88 ± 4.66 | 11.31 |
| ZsRE | Base | 32.37 ± 2.94 | 31.20 ± 2.59 | 29.79 ± 2.16 | 11.31 |
| | **Ours** | **2.34 ± 2.10** | **12.48 ± 3.58** | 29.79 ± 2.16 | 11.31 |
| | Genie | 2.34 ± 2.10 | 2.54 ± 1.81 | 29.79 ± 2.16 | 11.31 |
| MQuAKE | Base | 73.23 ± 1.53 | 46.44 ± 3.48 | 66.13 ± 0.28 | 11.31 |
| | **Ours** | **0.00 ± 0.00** | **7.20 ± 5.47** | 55.43 ± 1.14 | 11.31 |
| | Genie | 0.00 ± 0.00 | 0.00 ± 0.00 | 56.21 ± 0.99 | 11.31 |
| RWKU | Base | 57.60 ± 2.61 | 54.77 ± 3.95 | 63.70 ± 2.38 | 11.31 |
| | **Ours** | **5.60 ± 5.37** | **40.74 ± 5.37** | 62.76 ± 2.81 | 11.31 |
| | Genie | 5.60 ± 5.37 | 54.77 (= base) | 63.70 ± 2.38 | 11.31 |

Eff, Gen: lower = more forgetting. Spe: higher = better locality. PPL: same Wikipedia text in
all four evaluators; unchanged because no method edits unrouted text.

### Metric definitions (each from the benchmark's own official evaluator)

| Benchmark | Eff | Gen | Spe |
|---|---|---|---|
| MCF | forget answer probability (%), direct prompt | same, official paraphrases | neighbourhood accuracy |
| ZsRE | forget token accuracy (%), direct prompt | same, official paraphrases | neighbourhood accuracy |
| MQuAKE | forget rewrite accuracy (%) | forget AtomicGen | retain Eff |
| RWKU | same-50 recovery (%) | mean held-out recovery: Level 1, Level 2, Level-2 paraphrase | neighbour recovery |

## Methods

- **Base**: the seed's L19 linear router with all-zero rows, i.e. the unedited model through
  the same evaluator.
- **Ours**: Δe rows written at layer 16 in query mode (last prompt token + every generated
  token), trained on-policy (output loss + plain logit-lens feedback at L24 + KL keep).
  Router: factorized v2.2 on MCF (subject head at layer 6, relation head at 19+23+27); the
  seed's L19 linear classifier on ZsRE, MQuAKE and RWKU. ZsRE/MQuAKE/RWKU also train on
  teacher-forced gold views, because their metrics score every answer token under the gold
  prefix.
- **Genie**: the same rows with perfect routing: every official forget prompt gets its own
  fact's row, every other prompt gets none, and the router is not run. Genie − ours = cost of
  routing; genie = what the rows can do.
- Seeds = different forget/retain samples (ZeroUnlearn sampling); 50 forget facts per seed
  (MQuAKE: ~105 atomic associations from 50 instances).

## Per-seed (ours / genie)

| Dataset | Seed | Ours Eff | Ours Gen | Ours Spe | Genie Eff | Genie Gen | Genie Spe |
|---|---|---|---|---|---|---|---|
| MCF | 1 | 0.0010 | 0.0042 | 19.8 | 0.0010 | 0.0020 | 20.4 |
| MCF | 2 | 0.0008 | 0.0042 | 12.2 | 0.0008 | 0.0013 | 13.2 |
| MCF | 3 | 0.0008 | **0.2403** | 18.0 | 0.0008 | 0.0044 | 18.8 |
| MCF | 4 | 0.0009 | 0.0032 | 20.4 | 0.0009 | 0.0032 | 20.8 |
| MCF | 5 | 0.0007 | 0.0014 | 25.8 | 0.0007 | 0.0014 | 26.2 |
| ZsRE | 1 | 0.00 | 7.83 | 32.32 | 0.00 | 0.00 | 32.32 |
| ZsRE | 2 | 5.33 | 13.07 | 31.30 | 5.33 | 4.67 | 31.30 |
| ZsRE | 3 | 3.53 | 16.63 | 26.82 | 3.53 | 3.87 | 26.82 |
| ZsRE | 4 | 1.50 | 9.97 | 28.69 | 1.50 | 2.17 | 28.69 |
| ZsRE | 5 | 1.33 | 14.90 | 29.81 | 1.33 | 2.00 | 29.81 |
| MQuAKE | 1 | 0.00 | 0.92 | 57.03 | 0.00 | 0.00 | 57.53 |
| MQuAKE | 2 | 0.00 | **15.18** | 56.22 | 0.00 | 0.00 | 56.91 |
| MQuAKE | 3 | 0.00 | 3.68 | 54.82 | 0.00 | 0.00 | 55.78 |
| MQuAKE | 4 | 0.00 | 9.20 | 54.66 | 0.00 | 0.00 | 55.10 |
| MQuAKE | 5 | 0.00 | 7.03 | 54.40 | 0.00 | 0.00 | 55.70 |

RWKU per seed: `outputs/l16_paper_v1/paper_table_per_seed.csv`.

## Diagnostics

| Dataset | Router (ours) | Final on-policy leak (seeds 1–5) | Genie: direct prompts routed | Genie table size |
|---|---|---|---|---|
| MCF | factorized v2.2 (all seeds) | 0.0, 0.3, 1.3, 1.7, 0.0 % | – (MCF eval has no route fraction; 300–306 hits) | 150 |
| ZsRE | linear L19 | 0.4, 0.0, 0.0, 2.1, 0.0 % | 100% all seeds | 100 |
| MQuAKE | linear L19 | 0.0, 0.0, 0.0, 1.4, 0.0 % | 100% all seeds | 176–216 |
| RWKU | linear L19 (threshold gate) | 0.0 % all seeds | 100% all seeds | 50 |

- **Untrained (zero) rows**: on ZsRE and RWKU, the router never sent some facts' training
  views to their own row (2–12 facts per seed; RWKU seed 1: 7), so those rows stayed zero.
  MCF and MQuAKE had none.
- **MQuAKE retain overlap**: genie routes 13.9, 12.2, 14.4, 16.2, 16.7 % of retain rewrite
  decisions (seeds 1–5). Genie fires only on exact official forget prompts, so these retain
  facts repeat a forget fact by dataset construction.

## Reading the table

1. **The rows work.** With perfect routing, Gen is 0.002 (MCF), 2.5 (ZsRE) and 0.0 (MQuAKE),
   Eff is ~0 on MCF/MQuAKE, and PPL is unchanged everywhere.
2. **The remaining Gen gap is routing, not the edit.** ZsRE ours 12.5 vs genie 2.5 (router
   misses paraphrases); MQuAKE 7.2 vs 0.0, driven by seed 2 (15.2); MCF one seed (seed 3,
   Gen 0.24 vs genie 0.004).
3. **ZsRE / RWKU Eff > 0 comes from the zero rows**, which is why genie equals ours on Eff:
   perfect routing cannot help a row that was never trained.
4. **MQuAKE Spe drop (66 → 56) is the forget/retain atomic-fact overlap**, not collateral of
   the edit: genie, which edits only exact forget prompts, drops by the same amount. No method
   can forget one copy of a fact and keep the identical other copy.
5. **MCF Spe**: ours 19.24 vs base/genie 19.88 (−0.6) is the v2.2 router firing on near-miss
   subjects (e.g. "BMW M3" for a "BMW M5" fact), consistent across seeds.
6. **RWKU**: ours Gen 40.7 < base 54.8. Held-out probes ask other facts about the same people
   and have no row (genie = base by construction), yet ours lowers them: the router fires on
   some and the rows partly transfer within a person. Spe cost −0.9.

## Round 2 (submitted 2026-10-19, commit `4076d52`)

- **ZsRE**: router retrained on base-model rewordings of each direct question (reworded_v2;
  answer-consistency and near-duplicate filters; official rephrases never read), targeting the
  12.5 → 2.5 Gen gap.
- **ZsRE, RWKU**: `--oracle-unrouted` trains facts the router never routes with their own row
  forced during training only (test-time routing unchanged), so genie bounds every fact.
- **Baselines** in the same harness: GA, GD, NPO(+retain), RMU, ZeroUnlearn, same
  training-visible forget views, same seeds/splits/evaluators (`l16_baselines.slurm`).
- **MQuAKE Spe†**: retain Eff over retain facts that are not also forget facts.

Results of round 2 will go in `RESULTS_L16_PAPER_ROUND2.md`.
