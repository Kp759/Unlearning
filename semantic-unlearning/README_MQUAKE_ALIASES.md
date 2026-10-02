# MQuAKE answer aliases

The official MQuAKE metrics (Eff, AtomicGen) score the exact original answer.
A row is trained on that answer's tokens, so a synonym that starts with a
different token ("United States of America" -> "U.S.", "America") is not
targeted directly. This checks whether forgotten answers leak through aliases
and, optionally, trains rows on the aliases too.

```bash
sbatch mquake_alias_seed1.slurm                                # leak check + alias-target rows
sbatch --export=ALL,ALIAS_TRAIN=0 mquake_alias_seed1.slurm     # leak check only
cat outputs/mquake_multiseed_regular_v1/seed1/L19/linear_global/alias_leak_eval.md
cat outputs/mquake_alias_targets_v1/seed1/L19/comparison.md
```

- Aliases: MQuAKE's own `single_hops` / `new_single_hops` (`answer_alias`), looked up by
  the forget fact's (cloze, answer). Seed 1: 60 of 105 unique forget facts have aliases.
- `evaluate_mquake_alias_leak.py`: base vs SURE on the direct cloze and the held-out
  atomic question, per target (answer / alias with the same first token / alias with
  a different first token): first-token probability, exp mean log-prob, greedy.
  Headline: among facts the base model answers greedily and SURE does not, the
  fraction SURE still produces greedily through an alias.
- `--alias-targets` (row trainer, MQuAKE only): adds the first token of each alias to
  the fact's direct cloze cases (different first token from the answer, not a
  function word or single letter); the hinge then takes the worst case over answer and
  aliases. Same router; only the forget fact's own aliases are read.
