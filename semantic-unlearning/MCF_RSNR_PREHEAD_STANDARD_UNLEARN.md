# RSNR-V1B PreHead Standard-Unlearning Capacity Test

This experiment asks one question before increasing adapter capacity:

> Can the existing rank-16 (98,304-parameter on Llama-3.2-3B-Instruct) PreHead adapter satisfy standard CounterFact/ZeroUnlearn factual forgetting, nondisclosure, true-answer suppression, and exact off-route locality simultaneously?

## Architecture

For a routed forgotten fact `(subject, relation)`:

```math
h'_L = h_L + A_{16}(h_L),\qquad z = W_{LM}h'_L.
```

For every off-route query, the adapter is bypassed exactly:

```math
g(q)=0 \Longrightarrow f_{V1B}(q)=f_{Base}(q).
```

The Base model, Transformer blocks, final norm, embeddings, and LM head are frozen. Only the rank-16 `3072 -> 16 -> 3072` residual adapter is trainable.

## Four training constraints

On each of the five **training-visible** prompts for every forgotten fact:

```math
\begin{aligned}
\log P(target_{new})-\log P(target_{true}) &\ge m_{CF} \\
\log P(IDK)-\log P(target_{true}) &\ge m_{IDK} \\
\log P_{Base}(target_{true})-\log P_{V1B}(target_{true}) &\ge d \\
\Delta_{off-route} &= 0.
\end{aligned}
```

The first condition is exactly equivalent to:

```math
NLL(target_{true}) \ge NLL(target_{new}) + m_{CF},
```

which matches the target ordering underlying ZeroUnlearn-compatible MCF `post_rewrite_success` / `post_paraphrase_success`.

Default thresholds:

- `m_CF = 0.1`
- `m_IDK = 0.1`
- `d = 2.0`
- off-route max logit drift `= 0.0`

Checkpoint success requires **all 50 forgotten cases** to satisfy all three sensitive constraints on **all five training-visible views**, plus exact gate-off identity.

## Leakage boundary

Training uses `target_new` from the locked forget records and the existing training-visible five-view corpus. It does **not** use official paraphrases, official neighborhoods, fresh-retain examples, held-out aliases, or other official probe text for optimization or checkpoint selection.

## Train

```bash
cd /home/ec2-user/workspace/Unlearning/semantic-unlearning

export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export SOURCE_V13_RUN="$PWD/outputs/mcf_private_vocab_rewiring_v1_3_multiview_seed1_3b"
export OUT="$PWD/outputs/mcf_rsnr_v1b_prehead_standard_seed1_3b"

bash scripts/run_mcf_rsnr_v1b_prehead_standard_manual.sh \
  "$MODEL_PATH" \
  "$SOURCE_V13_RUN" \
  "$OUT" \
  2>&1 | tee slurm_logs/mcf_rsnr_v1b_prehead_standard_seed1.log
```

Exit code `0` means all four training gates passed. Exit code `2` means rank 16 did not satisfy the complete training gate within the configured steps; inspect `method/completion.json` and `method/rsnr_v1b_prehead_standard_unlearn.json` before increasing rank.

## Official held-out evaluation

Run only after training gate success:

```bash
python -u scripts/mcf_rsnr_v1b_prehead_standard_official_eval.py \
  --run-dir "$OUT" \
  --protocol-dir "$SOURCE_V13_RUN/protocol" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --out "$OUT/method/official_zero_unlearn_parity.json" \
  --seed 1 \
  --unlearn-num 50 \
  --retain-num 1000 \
  --fresh-retain-seed 700002 \
  --dtype bf16 \
  --generation-max-new-tokens 20 \
  --generation-batch-size 8
```

The official evaluator reports:

- exact ZeroUnlearn-parity `Eff` / `Gen`;
- `Spe` and PPL;
- IDK-vs-true diagnostics;
- Base-to-edited true-answer suppression;
- greedy semantic abstention;
- true/alias generation leakage;
- routing audit and fresh-disjoint retain behavior.

## Claim boundary

Passing this experiment would support comparison with ZeroUnlearn as a **conditional / routing-based factual unlearning** method under the same benchmark metric semantics. It still does not establish irreversible latent knowledge deletion because disabling the routed adapter restores the frozen Base model.
