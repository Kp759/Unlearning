# Architecture and mechanism

## High-level structure

```text
Llama-3.2-3B-Instruct

input tokens
    |
    v
input embedding matrix -------------------------------> frozen untouched readout W0
    |                                                     (used only as diagnostic/forget probe)
    v
transformer blocks 0 ... L-2        [fully frozen]
    |
    v
final transformer block L-1
    |
    +--> attention                                      [frozen]
    |
    +--> MLP
          gate_proj                                     [frozen]
          up_proj                                       [frozen]
          activation                                    [frozen]
          down_proj + low-rank LoRA repair              [ONLY trainable representation module]
                |
                v
final hidden representation h'
    |
    +--> frozen untouched W0 probe                      [tests base-decodable sensitive recovery]
    |
    +--> cloned/edited output head W_edit               [contains pre-existing sparse Stage-1 head edit]
                |
                v
             logits
```

## Vocabulary readout setup

Llama begins with tied input/output vocabulary weights. The experiment explicitly unties them before representation repair:

```text
initially: W_input == W_output

untie:
    W0     := original input-embedding matrix, frozen and untouched
    W_edit := cloned output head
```

The pre-existing sparse Stage-1 head delta is materialized only into `W_edit`. `W0` is never changed and therefore acts as a frozen reference decoder for the representation-recovery probe.

This distinction is central to the v3.1/v3.2 claim: the representation repair is successful only if the **original frozen readout** can no longer recover the sensitive answer from the edited hidden representation.

## Representation repair location

Only the final MLP `down_proj` is adapted:

```text
m = down_proj(k)
```

with low-rank update

```text
W_down' = W_down + scale * DeltaW
DeltaW = B A
rank(DeltaW) <= r
```

The experiments use ranks `r in {1, 2, 4}`. The final v3.2 diagnostic candidate is rank 1.

No embedding, attention, normalization, earlier transformer block, `gate_proj`, or `up_proj` parameter is trained by Stage-2 representation repair.

## Sensitive answer objective

For a generated prompt `q`, sensitive answer `a_s`, and neutral answer `a_n = Unknown`, define

```text
m_W0(q) = NLL_W0(q, a_s) - NLL_W0(q, a_n)
```

Interpretation:

```text
m_W0 < 0  -> untouched base readout prefers sensitive answer
m_W0 > 0  -> untouched base readout prefers neutral answer
```

The frozen-base-head forgetting loss is

```text
L_base = mean( ReLU(m_train - m_W0(q))^2 )
```

with training margin `m_train = 0.5`.

The edited output head receives a parallel answer-level pairwise constraint:

```text
m_Wedit(q) = NLL_Wedit(q, a_s) - NLL_Wedit(q, a_n)
L_edit = mean( ReLU(0.5 - m_Wedit(q))^2 )
```

## Utility preservation objective

v3.2 adds exact full-vocabulary KL preservation on external Wikipedia contexts. For utility context `u`:

```text
L_KL(u) = KL( P_base(. | u) || P_edit(. | u) )
```

The KL is evaluated over the full vocabulary, not only selected sensitive rows.

A final-hidden-state preservation term is retained:

```text
L_hidden(u) = relative MSE between edited and base final hidden states
```

The complete v3.2 optimization objective is

```text
L = 8 * L_base
  + 2 * L_edit
  + 50 * L_KL
  + 2 * L_hidden
  + 1e-4 * L_adapter_L2
```

The external-Wikipedia optimization reservoir contains 1,000 contexts. Four utility contexts are processed sequentially per optimizer step with gradient accumulation so exact full-vocabulary KL remains tractable in GPU memory.

## Why v3.2 fixed v3.1 utility

v3.1 already reached 0% frozen-`W0` recovery with a sub-1% representation update, but it relied on hidden-state preservation as the main utility regularizer. That did not control the output distribution tightly enough: exact Wiki KL was `0.058317 / 0.078253 / 5.236673` (mean/p95/max).

v3.2 directly optimizes the quantity that failed:

```text
KL(P_base || P_edit)
```

on 1,000 external-Wikipedia contexts. This reduced held-out 1K KL for the selected rank-1 candidate to

```text
mean = 0.000386
p95  = 0.001657
max  = 0.036320
```

while maintaining 0% frozen-base-head recovery on all 48 generated sensitive views.

## Remaining issue

The successful v3.2 physical candidate requires

```text
||Delta W_down||_F / ||W_down||_F = 0.014117 = 1.4117%
```

which exceeds the protocol's 1% intervention-size budget. Thus the current limitation is not forgetting or held-out utility; it is satisfying the strict intervention-norm constraint simultaneously in the same materialized checkpoint.
