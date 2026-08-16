# SURE-TOFU v4: sparse LM forgetting with Base answer embeddings

V4 is an ablation of the author-balanced locked SURE protocol.  It keeps the
same Full-TOFU epoch-5 starting checkpoint, same author-balanced seed split,
same 50 training-visible direct forget QAs, same Stage1A GA/GD, same sparse
progressive LM-head row selector, same restricted rank-0 optimizer, same
boundary bisection, and same locked evaluator.

V4 changes only two controls relative to progressive sparse v3:

1. **All visible answer-token input embedding rows are restored exactly to the
   Full-TOFU Base.**  Sensitive rows are no longer allowed to keep the Stage1A
   embedding displacement.  Only progressively selected sensitive LM-head rows
   may carry Stage1A/rank-0 forgetting displacement; every non-sensitive LM-head
   answer row remains exact Base.
2. **The NLL buffer is zero.**  With target answer probability `3e-4`, the final
   direct constraint is exactly `answer_probability <= 3e-4` (up to the declared
   numerical comparison tolerance).  Boundary bisection keeps the first
   near-feasible minimum-edit solution instead of deliberately pushing to a
   stronger buffered forgetting target.

The sparse row ranking and promotion policy remains training-visible only:
content-bearing rows first, lower answer-document-frequency first, then
original answer order.  Start with 3 ranked rows per violating QA and add the
next 1 ranked row only for residual failing QAs after a complete restricted
rank-0 attempt is infeasible.

No retain95, paraphrases, same-author holdout, real-authors, world-facts, PPL,
or final held-out metric is used during optimization or checkpoint selection.
The final checkpoint is frozen before locked evaluation begins.

Storage policy: old and current SURE `checkpoint/` weight directories may be
removed after use, while JSON/JSONL/log/report/evaluation files are retained.
The protected Full-TOFU epoch-5 checkpoint is never eligible for SURE cleanup.
