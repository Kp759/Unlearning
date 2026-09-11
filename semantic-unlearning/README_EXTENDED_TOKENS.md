# Input-only extended association tokens

This branch implements the private-token proposal as a controlled oracle-routing
ablation. It creates one token for each of the 50 forget facts and appends 50
trainable rows to the **input embedding only**. Every original model parameter,
the LM head, and the output vocabulary remain frozen.

For each forget fact, the token's label is `relation - object`. When that label
would collide with another forget fact, the subject is prepended. The token is
inserted after the beginning-of-sequence token in every training or development
view of that fact. Its embedding is optimized to:

1. raise the NLL of the original answer until its mean token probability is at
   most `1e-6`; and
2. lower the NLL of the existing-vocabulary completion ` I don't know.`.

The output softmax is unchanged, so adding the tokens cannot alter probabilities
through a larger denominator. Original prompts never contain the private token
and execute exactly the original embedding and output paths. The runner checks
original-row equality for every retain/language example and full-logit equality
on a parity example before training.

This guarantee also defines the experiment's limitation: a natural prompt does
not activate its private token by itself. Routed Eff/Gen may improve, while
unmodified natural-prompt Eff/Gen must equal the base model. Deployment would
therefore require an external fact-ID router or additional shared-weight
training. The artifacts mark the run as ineligible for the official no-router
evaluation, and the runner does not open any final test set.

Run the ablation with:

```bash
bash scripts/run_static_overlap_extended_tokens.sh \
  /home/ec2-user/models/Llama-3.2-3B-Instruct \
  "$PWD/outputs/static_overlap_replay_cached_20260909_215959/manifest.json"
```

The run writes the extended tokenizer, the 50 learned input rows, the association
mapping, step metrics, routed train/development forgetting, and routed abstention
metrics under `outputs/static_overlap_extended_tokens_v1_seed1/`.
