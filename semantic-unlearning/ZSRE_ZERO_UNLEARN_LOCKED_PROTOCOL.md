# ZsRE ZeroUnlearn-Style Locked-Probe SURE Protocol

This track is the publication-grade data-access comparison between SURE-LM and ZeroUnlearn on ZsRE.

## Exact pool and sampling rule

For the original `zsre_mend_eval.json` ordering:

- first half: retain pool;
- second half: forget pool;
- for each seed, sample forget first and then retain from one `random.Random(seed)` stream;
- seeds: 1 through 10;
- forget sample size: 50;
- retain sample size: 1000.

This mirrors the vendored ZeroUnlearn `experiments/evaluate.py` sampling rule.

## SURE data access

### Stage 1

Visible:

- 50 sampled forget records;
- `requested_rewrite.prompt`;
- subject;
- original sensitive answer (`target_true` in ZsRE);
- neutral target `Unknown` (`target_new` in the ZeroUnlearn ZsRE adapter).

Not visible:

- 1000 sampled benchmark-retain records;
- forget rephrases;
- forget locality/neighborhood probes.

The existing ZsRE Setting-5e semantic mapping is preserved:

- internal unwanted target = original sensitive answer;
- internal desired target = `Unknown`.

The transformer is frozen. Input embeddings and the LM head are optimized for 600 steps with the established margin objective. Standard post-training row restoration is then applied. The `Unknown` row is explicitly excluded from Stage-1 row groups so it is restored to its base value before Stage 2.

### Stage 2

Visible:

- the same 50 sampled forget records;
- direct requested-rewrite prompts only.

Not visible:

- rephrases;
- locality probes;
- benchmark-retain records.

The transformer and input embeddings are frozen. The output head is safely untied if needed. Stage 2 identifies still-correct original-answer tokens on the direct rewrite prompts and optimizes only the output row for `Unknown`. The BF16 scale sweep also uses direct rewrite prompts only. No held-out Gen or Spe probes are used for repair, scale selection, or checkpoint selection.

### Final evaluation

Only after the repaired checkpoint is frozen, the evaluator reopens the unchanged original ZsRE file and applies the same seed/sampling rule:

- same 50 forget facts;
- their rephrases -> Gen;
- their locality probes -> Spe;
- 1000 sampled retain records -> additional collateral-utility diagnostics;
- fixed Wikidata text -> PPL.

This is a prompt-level holdout, not a fact-level holdout: the same underlying forget facts are deletion requests and final forget facts, while rephrased/locality formulations are held out until final evaluation.

## Files

- `scripts/build_zsre_zerounlearn_locked_split.py`
- `scripts/zsre_forget_only_setting5e.py`
- `scripts/zsre_forget_only_active_repair.py`
- `scripts/run_zsre_zerounlearn_locked_our_method.sh`
- `scripts/aggregate_zsre_locked_results.py`
- `slurm/run_zsre_zerounlearn_locked_3b.slurm`

## Scientific labeling

Use:

- `ZeroUnlearn-style ZsRE sampling`;
- `forget-only SURE data-access variant`;
- `locked prompt-level evaluation probes`;
- `same forget facts, unseen rephrased/locality formulations`;
- `1000 benchmark-retain records evaluation-only`.

Do not describe this as unseen forget facts or a conventional fact-level train/test split.
