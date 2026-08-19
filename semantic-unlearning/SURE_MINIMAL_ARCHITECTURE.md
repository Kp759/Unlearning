# Minimal two-stage SURE-LM

This is the shared MCF/ZsRE architecture implemented by
`scripts/sure_minimal_two_stage.py`.

```text
Frozen Base transformer + frozen Base input embeddings
                         │
              50 direct forget requests
              (sensitive answer only)
                         │
             Base hidden states and logits
                         │
                         ├──────── C_F,s ────────┐
                         │                       │
fixed Wikipedia C_U = E[h hᵀ]                   │
(100,000-document request; no benchmark data)   │
                         │                       │
                         └── generalized eigen ─┘
                                      │
                           fixed rank-2 basis B_s
                                      │
Stage 1: sensitive GA + same-prompt non-sensitive GD
                                      │
          smallest directly successful materialized scale
                                      │
                        all direct constraints pass?
                              /                 \
                            yes                  no
                             │                    │
                             │          failed token cases only
                             │                    │
                             │        new fixed rank-2 basis
                             │                    │
                             │    Stage 2: same GA + same GD
                             │                    │
                             │  smallest materialized successful scale
                             └───────────┬────────┘
                                         │
                         frozen sparse LM-head checkpoint
                                         │
        post-training only: FS/GFS, Spe, retain, exact KL, and PPL
```

The learner has no benchmark retain input, retain-action loss, retain budget,
norm guard, KL guard, rank sweep, or access to paraphrase/locality/PPL probes.
Wikipedia is used only to construct the contrastive basis. As in ZeroUnlearn,
the 100,000-document request is capped to the locally available corpus. The
first 20 local Wikipedia documents are excluded from the cache because the PPL
evaluator uses that prefix. With the checked-in 200-document artifact, the
actual utility sample is therefore 180 documents; the cache records both the
100,000 request and the 180-document realization.

Run MCF:

```bash
bash scripts/run_mcf_sure_minimal.sh /path/to/Llama-3.2-3B-Instruct data/multi_counterfact.json
```

Run ZsRE with the same model and cached Wikipedia statistic:

```bash
bash scripts/run_zsre_sure_minimal.sh /path/to/Llama-3.2-3B-Instruct data/zsre_mend_eval.json
```

Set `SURE_UTILITY_CACHE` to an explicit shared path when MCF and ZsRE runs use
different output roots or machines.
