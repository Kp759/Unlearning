# MQuAKE ZeroUnlearn-Style Locked-Probe SURE Protocol

This track is the publication-grade data-access comparison between SURE-LM and ZeroUnlearn on MQuAKE-CF-3k-v2.

## What ZeroUnlearn actually does

The published ZeroUnlearn evaluation code applies this pool rule before sampling:

- first half of the loaded MQuAKE dataset: retain pool;
- second half: forget pool;
- for each seed, sample forget first and retain second from one seeded RNG stream;
- few-shot seeds: 1 through 10;
- few-shot forget sample: 50 MQuAKE instances;
- retain sample: 1,000 MQuAKE instances.

The public MQuAKE adapter normally falls back to the 3,000-instance `MQuAKE-CF-3k-v2.json`, giving 1,500 retain-pool and 1,500 forget-pool instances. It also checks for a local `data/mquake_data_saved_split.json` first. That saved artifact is not published in the ZeroUnlearn repository, so an author-side reordered/preprocessed copy cannot be reconstructed byte-for-byte from the public release. This SURE track therefore source-locks the official 3k-v2 file and reproduces the **published pool/sampling algorithm, counts, seed order, and instance-before-flattening behavior** rather than claiming access to an unpublished saved split.

MQuAKE instances may contain multiple `requested_rewrite` facts. Sampling happens at the **instance level first**. The selected instances are then flattened into their atomic `requested_rewrite` facts for the unlearning algorithm.

There is no separate held-out-fact test set in the published ZeroUnlearn few-shot MQuAKE code. The same 50 sampled forget instances are the deletion requests and the final forget/Eff evaluation instances. The sampled 1,000 retain instances are separate and can be used for utility evaluation. ZeroUnlearn's native MQuAKE table reports Eff and PPL; it does not report MQuAKE Gen/Spe.

For the multiple-unlearning setting, the paper uses 1,000 forget samples instead of 50.

## Locked SURE data access

### Stage 1

Visible:

- the atomic `requested_rewrite` facts belonging to the same 50 sampled forget instances;
- direct cloze prompt;
- subject;
- original sensitive answer (`target_true`);
- neutral target `Unknown`.

Not visible:

- the 1,000 sampled retain instances;
- each rewrite's natural-language `question`;
- the instance-level multi-hop `questions`;
- MQuAKE's counterfactual `target_new` answer.

The benchmark counterfactual target is not an unlearning target. SURE maps the original sensitive answer to the neutral one-token answer `Unknown`.

### Stage 2

Visible:

- the same direct rewrite prompts from the same 50 sampled forget instances only.

Not visible:

- benchmark-retain records;
- atomic natural-language questions;
- multi-hop questions;
- benchmark counterfactual targets.

The transformer and input embeddings are frozen. The active repair operates only on the `Unknown` output row and selects its scale using the direct rewrite cases only.

### Final evaluation

Only after the repaired checkpoint is frozen, the evaluator reopens the unchanged source MQuAKE file and reconstructs the same seed split:

- same 50 sampled forget instances -> ZeroUnlearn-compatible Eff;
- 1,000 sampled retain instances -> collateral-utility diagnostic;
- fixed Wikidata text -> PPL.

Optional `AtomicGen` evaluation on the natural-language atomic questions is permitted only as a post-selection extension. It is not a native ZeroUnlearn MQuAKE metric and must never be used for training, repair, scale selection, or checkpoint selection.

This is therefore **not a fact-level train/test split**. It is a deletion-request protocol with locked auxiliary prompts: the same underlying forget facts are trained/unlearned and scored for Eff, while non-direct question formulations are withheld from SURE until post-selection evaluation.

## Files

- `scripts/build_mquake_zerounlearn_locked_split.py`
- `scripts/mquake_forget_only_setting5e.py`
- `scripts/mquake_forget_only_active_repair.py`
- `scripts/run_mquake_zerounlearn_locked_our_method.sh`
- `scripts/mquake_zero_unlearn_official_eval.py`

## Scientific labeling

Use:

- `ZeroUnlearn-style MQuAKE sampling`;
- `public-code-reproducible MQuAKE-CF-3k-v2 source order`;
- `50 deletion instances, seeds 1-10`;
- `forget-only SURE data-access variant`;
- `same deletion instances for Eff; no held-out forget facts`;
- `1,000 sampled retain instances evaluation-only`;
- `atomic and multi-hop questions locked until post-selection evaluation`.

Do not describe the ZeroUnlearn few-shot MQuAKE setup as a conventional train/test split or as evaluation on unseen forget facts. Do not claim byte-identical reproduction of the authors' unpublished `mquake_data_saved_split.json` artifact.