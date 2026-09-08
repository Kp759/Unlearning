# Static, overlap-constrained embedding–MLP–head editing

This implements the trained architecture as a separate experiment. It jointly
trains small static deltas, then exports an ordinary Hugging Face checkpoint.
Inference consumes ordinary token IDs. There is no fact router, runtime fact
gate, activation projection, token penalty, private vocabulary, or hard guard.
The effective Transformer changes in selected MLP writeouts. The existing
router/guard experiments remain separate baselines.

The implementation is an experimental mechanism, not evidence of successful
unlearning, zero Eff/Gen, certified deletion, or independent architectural
novelty. The bundled examples are **synthetic plumbing examples**, not an MCF
training set or benchmark result.

## Editable region

The initial configuration is rank **8**, **two** MLP blocks, and **64**
intermediate channels per block. These are engineering defaults. All other
parameters, including attention, MLP up/gate projections and normalization,
are frozen during training. Tokenizer and vocabulary remain unchanged.

| Site | Effective update | Selection |
| --- | --- | --- |
| Input embeddings | `E0 + P_E A_E B_E.T` | Subject tokens and explicitly approved aliases |
| MLP down projection | `Wdown a + B A a[J]` | Training forget-versus-retain sensitivity |
| LM head | `Wout0 + P_O A_O B_O.T` | Registered sensitive answer tokenizations and neutral-response tokens |

Selected blocks/channels maximize a heuristic ratio of mean absolute
activation-times-answer-NLL-gradient on forget versus matched retain examples.
Both layers are selected from the loaded model's supported blocks. The code
does not import earlier causal-tracing layer choices. Localization allocates
the update budget; it does not identify exclusive storage of a fact.

For tied endpoints the implementation checks actual storage sharing against
`tie_word_embeddings` and trains **one** delta on the union of endpoint rows.
It registers those factors once with the optimizer, counts their effective norm
once, and merges them once. A configuration/storage mismatch fails explicitly.
This follows the model's actual weight-sharing contract; see the
[Transformers model API](https://huggingface.co/docs/transformers/main_classes/model).

Training uses temporary factorized modules with zero effective initial deltas.
They evaluate static linear updates, without query-dependent enabling decisions.
Their activation-dependent effects are contextual; the factors themselves are
shared across all training facts. Endpoint rows and MLP channels may serve many
permitted associations.

## Data contract

Start from [the example bundle](config/static_overlap_training.example.json).
Replace the invented associations with approved training-visible associations
and paraphrases. The trainer never opens a full benchmark dataset. Do not copy
official evaluation paraphrases, neighborhoods, aliases, retain requests, or
PPL documents into the training bundle.

The top-level schema has exactly `schema_version: 1`, `purpose: "training"`,
`facts`, and `examples`. Each fact registers `id`, `subject`, `relation`, `object`,
and `role` (`forget` or `retain`). Optional `aliases` and `answer_aliases` are
explicitly approved training-side spellings. Each association example contains:

```json
{
  "id": "mixed_training_request",
  "split": "train",
  "prompt": "Give Person A's native language and preferred cuisine:",
  "completion": " French; French",
  "spans": [
    {"start": 1, "end": 7, "fact_id": "forget_language"},
    {"start": 9, "end": 15, "fact_id": "retain_cuisine"}
  ]
}
```

Offsets are half-open **character offsets within the completion**. The fast
tokenizer maps them to actual tokens in the combined prompt and completion.
Only the labeled answer tokens enter that span's average NLL. Prompt tokens and
companion spans are masked. Tokens crossing into non-answer text or another
labeled span cause an error, as does truncation. Explicit separators prevent
ambiguous token boundaries.

The trainer derives four required overlap strata for **every** forget
association in both `train` and `validation`:

1. Same subject, different relation and answer.
2. Same relation, different subject.
3. Same subject and answer, different relation.
4. Same answer, different subject and relation.

Both splits also require mixed forget/retain requests and general-language
anchors (`{"id": "...", "split": "train", "role": "language", "text": "..."}`).
Missing strata fail before model loading. Text duplicates and identical prompts
across train/validation are rejected. These checks cannot establish semantic
truth, alias equivalence, or provenance: the supplied associations and split
construction still need protocol review. Relation matching uses the registered
relation string, so use consistent canonical relation identifiers.

For an abstention objective, forget spans are replaced with the configured
ordinary text in a separate supervised view. The neutral spans receive descent,
and mixed companion spans also receive retain descent in that view. The original
view supplies bounded forget ascent and companion retain descent. Training
spans and fact IDs are never supplied to generation.

## Objective and step acceptance

For each forget span the base answer-only NLL is measured once. Its hinge loss is
`relu(base_NLL + forget_increase - edited_NLL)`, so forget ascent stops through
that term after reaching the declared increase. Other objectives can still move
that example's probability. The joint objective adds abstention NLL, retained
answer NLL, exact full-vocabulary `KL(p_base || p_edit)` on retained answer and
general-language prefixes, and the squared Frobenius norm of the **effective
weight deltas**, rather than the factors' individual norms.

Base predictions use the same frozen model with all training deltas temporarily
disabled. Dropout is disabled; teacher outputs are detached. There is no second
trainable teacher and no top-k KL approximation. Micro-example gradients are
accumulated before a single joint Adam proposal, keeping activation memory
bounded. The default run uses float32 training weights and factors.

Each step applies this sequence:

1. Obtain retained-answer NLL gradients for a rotating protected batch.
2. Let Adam propose an actual factor update `u`, including its moment scaling.
3. Approximately solve the Euclidean projection onto `g_r.T delta <= epsilon`
   and `||delta|| <= step_radius` with Dykstra's algorithm. Nonconvergence or a
   negligible update rejects the step.
4. Check actual NLL deterioration and full-vocabulary KL through the complete
   edited model on **all fitting retain/language anchors**, including mixed
   companion spans. Budgets are relative to the original base model.
5. Halve the projected step until budgets pass, or reject it and restore both
   parameters and Adam state. Stop after the declared number of stalled steps.

The projection is a first-order constraint in factor coordinates. Nonlinear
checks protect only the declared finite anchors. They are not a universal
retention guarantee. No learning-rate escalation bypasses infeasible constraints.
Validation does not supply gradients or localization evidence; it is checked
after training and at export. Every proposed numerical budget is recorded in
the manifest and should be fixed before held-out evaluation.

## Run

From `semantic-unlearning`, using the repository's PyTorch/Transformers
environment:

```bash
python scripts/run_static_overlap_edit.py \
  --model-path /path/to/base-model \
  --training-bundle /path/to/approved-training-bundle.json \
  --config config/static_overlap_edit.json \
  --output-dir outputs/static_overlap_run1 \
  --device cuda --dtype float32 --deployment-dtype bfloat16 \
  --local-files-only
```

The output directory must be new. Supported layouts have native `nn.Embedding`
and `nn.Linear` endpoints plus `model.model.layers[i].mlp.down_proj` (for example,
Llama-style blocks). Quantized models, sharding/offload, fused unsupported MLP
layouts, and embeddings with `max_norm` are rejected. This initial implementation
runs on one device; CPU is supported for small models. Exact KL and all-anchor
checks prioritize correctness over large-run throughput. Full-vocabulary export
references stream through temporary disk storage, which can be substantial.

Outputs include localization scores/indices, the resolved configuration and
data hash, the training trace, validation metrics, and `training_factors.pt` for
inspection. A failed/no-progress run retains its diagnostics and does not claim
a verified checkpoint. Shared updates do not provide exact per-fact rollback.

The primary artifact is `checkpoint/`. Export restores native modules, merges
the deltas, casts to the deployment dtype, checks selected-token logits, NLL and
KL on all training/validation spans, saves, reloads, and repeats those checks.
An ordinary generation configuration clears inherited penalties/hard masks.
Actual deployment retention budgets must still pass, independently of the
declared factorized/merged logit tolerance. `static_edit_export.json` is written
only on success and hashes the checkpoint files and its training manifest.
Training factors are stored outside the deployment checkpoint.

Ordinary inference requires only standard loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/static_overlap_run1/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("outputs/static_overlap_run1/checkpoint")
inputs = tokenizer("An ordinary user request", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
```

## Evaluation and interpretation

Prepare a separate bundle with `purpose: "evaluation"` and all splits `test`,
using the same explicit span/association format and the same forget associations.
Its text must be held out from fitting and validation. All four overlap strata,
mixed requests and language text are required. Include retained overlaps, mixed responses, general
language, natural forget paraphrases, and the declared attack/recovery prompts.

```bash
python scripts/evaluate_static_overlap_edit.py \
  --checkpoint outputs/static_overlap_run1/checkpoint \
  --evaluation-bundle /path/to/held-out-evaluation.json \
  --out outputs/static_overlap_run1/evaluation.json --device cuda
```

The evaluator checks checkpoint hashes and loads only native model weights. It
reports sensitive/neutral/retained NLL, teacher-forced answer accuracy, overlap
strata, token-weighted corpus PPL, the full decoding configuration, and raw joint
generated responses. Registered-answer mentions are explicitly **lexical
diagnostics**, not counts of factual disclosure: mixed overlaps, negation and
quotation require a separately declared assertion judge. No claim of zero
generated disclosure is inferred from mention counts.

Optional `--mcf-path ... --wikidata-dir ... --seed ... --unlearn-num 50
--retain-num 1000` invokes the existing official MCF scoring function on the
already loaded plain model. The selected MCF `target_true` associations must
exactly match the training manifest (`relation` should contain MCF's relation
ID). Eff/Gen preference definitions and released-table accuracy are unchanged.
The old evaluator's sidecar-loading function is never called. PPL can only be
omitted explicitly with `--skip-official-ppl`; missing results remain unmeasured.

Abstention preference, MCF `target_new` preference, sensitive-sequence
probability, and generated assertions are distinct measurements. Finite softmax
logits do not produce exact sequence impossibility. Before an unlearning claim,
run recovery/relearning tests and a suitable retraining comparison where
feasible. Matched-budget endpoint-only, MLP-only and unconstrained ablations
remain experimental work; existing router results do not validate this model.

## Verification

```bash
python -m pytest -q tests/test_static_overlap_edit.py
```

Tests use randomly initialized tiny Llama models and a local fast tokenizer;
they download no model weights. They exercise localization, shared endpoints,
parameter support, answer masks, bounded ascent, exact KL, independent numerical
QP agreement, nonlinear backtracking/optimizer rollback, joint fitting, native
merge/reload and dtype checks, and the complete CLI export path. Passing these
tests verifies the implementation mechanics, not successful fact suppression.
