# Pure Two-Stage Directional SURE for RWKU

This branch implements **Level 1 + Level 2 only**. There is no SURE-R, MLP edit, attention edit, LoRA adapter, or other representation-repair path.

## Architecture

```text
Sensitive token rows E_A  -- trainable sparse FP32 deltas
          |
          v
Entire transformer         -- exactly frozen
          |
          v
Sensitive LM-head rows W_A -- trainable sparse FP32 deltas
          |
          v
Vocabulary logits
```

All non-sensitive input-embedding and LM-head rows remain exact Base. The LM head is explicitly untied before sparse-row training so input and output sensitive rows can move independently.

## Level 1: broad Directional SURE

Level 1 is the all-sensitive-row Directional SURE v2.1 configuration:

- all target-only generated atomic training views are used;
- canonical sensitive-token GA is minimized;
- same-prompt non-sensitive Base-to-current KL supplies GD/locality;
- the LM-head GA gradient is projected into a sensitive-exclusive basis `B_S`;
- the LM-head GD gradient is projected into protected basis `B_P`;
- embedding gradients use the ordinary weighted GA+GD sum;
- `B_S` and `B_P` refresh every 25 steps;
- the transformer remains exactly frozen.

The Level-1 LM-head update is

\[
g_W = 2\Pi_{B_S}(g_{GA}) + \Pi_{B_P}(g_{GD}).
\]

The embedding update is

\[
g_E = 2g_{GA,E} + g_{GD,E}.
\]

A Level-1 anchor is selected only among checkpoints that pass the unchanged external-Wikipedia selection KL gates. The anchor rule first minimizes generated atomic pairwise-margin failures, then behavioral failures, then maximizes the worst separation, and only then prefers lower utility KL / delta norm.

If Level 1 already reaches 100% direct, 100% other generated views, zero margin failures, and selection utility PASS, Level 2 is skipped.

## Level 2: residual directional repair

If Level 1 leaves pairwise-margin failures, Level 2 starts from the selected Level-1 anchor.

The residual prompt set is frozen from that anchor. Every teacher-forced sensitive prediction case belonging to one of those failed prompts is collected. The editable Level-2 row set is the subset of the already-declared sensitive vocabulary rows that occurs in those residual cases.

Level 2 then:

1. resets AdamW optimizer state;
2. samples only residual sensitive prediction cases;
3. builds `B_F` from current residual sensitive hidden states after projection away from `B_P`;
4. projects residual GA head gradients into `B_F`;
5. projects residual GD head gradients into `B_P`;
6. masks **both embedding and LM-head gradients** so only residual sensitive rows can change;
7. evaluates **all** generated atomic prompts plus the unchanged external-Wikipedia utility selection gate every 25 steps.

The Level-2 head update is

\[
g_W^{(2)} = M_F \odot \left(2\Pi_{B_F}(g_{GA}) + \Pi_{B_P}(g_{GD})\right),
\]

where `M_F` is the fixed residual-row mask. The embedding update is

\[
g_E^{(2)} = M_F \odot (2g_{GA,E} + g_{GD,E}).
\]

No new parameter class is introduced in Level 2. It still modifies only sparse sensitive embedding/head rows.

## Locked first-development values

Level 1 retains the v2.1 values: 600 steps, embedding LR `5e-5`, LM-head LR `1e-4`, `rank(B_S)=8`, `rank(B_P)=32`, GA/GD weights `2/1`.

Level 2 uses predeclared first-development values: 300 steps, embedding LR `2.5e-5`, LM-head LR `5e-5`, `rank(B_F)=8`, `rank(B_P)=32`, GA/GD weights `2/1`. These are not tuned from official RWKU evaluation.

The acceptance budgets remain unchanged:

- generated direct success = 100%;
- generated other-view success = 100%;
- required pairwise margin = `0.01`;
- external-Wikipedia KL mean <= `0.01`;
- p95 <= `0.05`;
- max <= `0.5`.

## Data boundary

Both levels see only the target-only generated atomic corpus and target-excluded external Wikipedia. Official RWKU paraphrase, neighborhood, retain, and PPL artifacts remain unavailable to the learner and are never used for checkpoint selection.

The final fresh 1000-context Wikipedia gate is opened only after the final Level-1 or Level-2 checkpoint has been selected.

## Explicitly absent Level 3

```text
Level 3 / SURE-R: DISABLED
Transformer edit: DISABLED
MLP edit:         DISABLED
Attention edit:   DISABLED
LoRA:             DISABLED
```

Therefore a successful checkpoint supports the architectural statement that the complete training procedure used only sparse vocabulary-interface intervention while the transformer stayed exact Base.
