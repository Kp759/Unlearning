# Llama-3.2-3B-Instruct — Final Results

Model used for the RWKU representation-repair development setting documented here:

- Local model path during experiments: `/home/ec2-user/models/Llama-3.2-3B-Instruct`
- Architecture family: Llama causal decoder LM
- Representation repair location: final transformer block MLP `down_proj`
- Vocabulary readout treatment: tied input/output weights are untied at experiment start; the original input-embedding matrix is retained as the untouched frozen base readout `W0`, while the cloned output head receives the pre-existing sparse Stage-1 edit.

## Databases

- `RWKU/` — target-only generated atomic views for the Stephen King development target, plus disjoint external-Wikipedia utility preservation/evaluation.
