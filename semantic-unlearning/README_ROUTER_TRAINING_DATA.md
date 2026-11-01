# Router training data: how each fact's positives and negatives are built (MCF, ours)

The router recipe `ours` (`run_l16_paper_one.sh ONEPASS=1`, `ROUTER_RECIPE=ours`) fits one
linear head per forget fact at L19. Each head is trained on **20 positives** (prompts that must
fire it) and **20 negatives** (prompts about the same subject that must not). This document
lists where every prompt comes from, the templates behind them, and the counts for MCF seed 1
(50 forget facts).

Pipeline (`scripts/run_mcf_layer_sweep_one.sh`):

1. `prepare_mcf_association_source.py`: the forget facts and their template prompts, written to `prep/association_examples.json`.
2. `build_router_rewrites.py`: base-model rewrites (`prep/router_rewrites.json`) and model-written hard negatives (`prep/router_hard_negatives.json`).
3. `fit_linear_router.py --positives-per-fact 20 --negatives-per-fact 20 --max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5 --subject-match fuzzy`. It writes the final set to `router/linear_router_dataset.json`.

## Summary per fact

| | source | count | template? |
|---|---|---|---|
| **+ve** | MCF's own prompt (`requested_rewrite.prompt`) | 1 | yes (the dataset's) |
| | scaffolds, filled with subject and relation noun | 8 fit + 4 held out | yes |
| | relation alternates | 2 | yes |
| | relation alternates with a generic lead-in | 2 | yes |
| | base-model rewrites (paraphrase / question / translation) | about 3 kept | **no** |
| | lead-in copies (top-up when rewrites run short) | 0 to 8 | yes |
| **−ve** | model-written hard negatives | up to 8 | **no** |
| | subject transplants: another forget fact's template prompts with this fact's subject swapped in | the rest, up to 20 | yes |
| | same-subject real competitors (another forget fact with the same subject) | 0 in MCF seed 1 | – |

So about 17 of the 20 positives are templates, and at least 12 of the 20 negatives are (all 20
when no hard negative survives). A positive and a transplant negative are usually the same
sentence frame with a different relation noun:

```
+ve  Supply the profession attributed to Ignazio Silone:
−ve  Supply the native language attributed to Ignazio Silone:
```

## Positives

`build_forget_examples` (`scripts/static_overlap_fact_association_embeddings.py:76`) writes each
fact's own prompts. Each prompt is one family:

| family | prompt | split |
|---|---|---|
| `canonical_rewrite` | MCF `requested_rewrite.prompt` with the subject | fit |
| `authored_0..7` | `TRAIN_SCAFFOLDS` | fit |
| `authored_0..3` | `DEVELOPMENT_SCAFFOLDS` | calibration / audit, alternating by family |
| `relation_alternate_0..1` | `RELATION_ALTERNATE_TEMPLATES[relation]` | fit |
| `context_relation_alternate_0..1` | `GENERIC_CONTEXT_PREFIXES[(case_id + k) % 4]` + the alternate | fit |

That makes 17 per fact. It is 16 for the 4 facts whose MCF prompt equals one of their relation
alternates, because duplicates are removed (e.g. `The profession of Ignazio Silone is`).

`build_router_rewrites.py` then has the base model write 6 paraphrases, 3 questions and 3
translations (Italian, Portuguese, Dutch) per fact. A second round runs for any fact with fewer
than 8 kept. A rewrite is kept if it:
- is well formed;
- names the subject;
- doesn't contain the answer;
- isn't a near-duplicate;
- passes the answer-transfer check (`router_rewrites.py`): told the fact, the base model answers the rewrite with the fact's answer.

Every third kept rewrite of each kind is held out (`holdout_third`).

`balance_positive_examples` (`scripts/linear_router.py:412`) cuts each fact to exactly 20. Order
of preference: MCF prompt first, then the other phrasings (scaffolds, alternates, rewrites), then
lead-in copies. When there are more than 20 it keeps 15 fit + 5 held out (16 + 4 when only
the 4 held-out scaffolds exist). With 17 of the fact's
own prompts ranked ahead, **at most 3 rewrites survive (4 for the 16-prompt facts), however many
were generated**. A fact with fewer rewrites is topped up with `EXTRA_CONTEXT_PREFIXES` copies of
its prompts (at most 8 copies; every fourth copy is held out).

## Negatives

`assemble_router_dataset` and `_negative_controls_for_fact` (`scripts/linear_router.py:524`, `:785`):

1. **Same-subject real competitors.** These are the prompts of other forget facts with the same subject, at most half of the 20. MCF seed 1 has no shared subjects, so there are none.
2. **Hard negatives** (`--extra-negatives`, `router_rewrites.generate_hard_negatives`). The base model writes other questions (or unfinished sentences) about the subject. A question is kept if:
   - it names the subject and doesn't contain the answer;
   - it isn't a near-duplicate of a positive;
   - told the fact, the base model does **not** answer it with the fact's answer.

   At most 8 per fact are kept. Their splits rotate fit → calibration → audit.
3. **Subject transplants.** Donors are walked in order from `fact_index + 1`. Each donor's prompts from one split (rotating over fit / calibration / audit) get this fact's subject in place of the donor's subject, at most 3 per donor. A donor is skipped if its relation:
   - is already a forget relation for this subject;
   - is in the same answer group (`RELATION_ANSWER_GROUPS`, e.g. native language and language of writing); or
   - fails the subject-type check.

   Twice the needed number is drawn (40).

**The filter** (`NegativeQuality`, `scripts/fit_linear_router.py`) runs on the 40 candidates. It drops:
- any candidate where the base model puts at least 0.3 probability on the fact's answer as the next token;
- any transplant whose relation phrase is more than 1.5 nats less likely after this subject than after the donor's own subject (e.g. "The profession of Belgium is").

`compose_negatives` then takes 20, in this order:
- up to 10 real competitors;
- then hard negatives and transplants, with transplants rotated over donor relations and lead-in variants of one prompt counted once;
- then more competitors or lead-in variants if still short.

**Counts, MCF seed 1, before the filter and without hard negatives:**
- All 877 negatives are subject transplants.
- 33 facts get 20.
- 17 get between 10 and 19. Places and products have few type-compatible donors: P30 continent (6 facts), P176 manufacturer (3), P364 original language (3), P740 founding location (3), P37, P36.

Hard negatives fill part of that gap on the cluster. The filter can widen it.

## Example (MCF seed 1, fact 1: Ignazio Silone, P106, answer "politician")

These lists are without rewrites and hard negatives (no Llama on the machine that generated
them). In the real run the last 3 positives are rewrites and up to 8 negatives are hard
negatives.

```
+ve  The profession of Ignazio Silone is                                  canonical_rewrite (= relation_alternate_0)
     What is the profession of Ignazio Silone? Answer:                    authored_0 ... authored_7 (fit)
     Supply the profession attributed to Ignazio Silone:                  authored_0 ... authored_3 (held out)
     Ignazio Silone works as a                                            relation_alternate_1
     As has been noted elsewhere, Ignazio Silone works as a               context_relation_alternate_1
     As the records show, The profession of Ignazio Silone is             context_prefix_4 (top-up)
−ve  For Ignazio Silone, identify the work location:                      transplant, donor P937
     Supply the native language attributed to Ignazio Silone:             transplant, donor P103
     Looking up Ignazio Silone, which genre is listed?                    transplant, donor P136
     Supply the playing position attributed to Ignazio Silone:            transplant, donor P413
```

## Templates

### Scaffolds: `scripts/run_static_overlap_extended_tokens_standalone_v2.py:44`

`TRAIN_SCAFFOLDS` (fit):
```
What is the {relation} of {subject}? Answer:
Which {relation} is associated with {subject}? Answer:
For {subject}, identify the {relation}:
The {relation} recorded for {subject} is
Name the {relation} linked to {subject}:
In reference to {subject}, sources list the {relation} as
What value is given for {subject}'s {relation}? Response:
Complete this fact about {subject}: the {relation} is
```
`DEVELOPMENT_SCAFFOLDS` (calibration / audit):
```
Supply the {relation} attributed to {subject}:
Looking up {subject}, which {relation} is listed?
The requested {relation} entry concerning {subject} reads
State what is documented as the {relation} of {subject}:
```

### Relation nouns (`{relation}`): `scripts/mcf_shadow_relation_prompts.py:19`

`static_overlap_natural_writer._relation_noun` uses "language spoken or written" for P1412.

| id | noun | id | noun | id | noun |
|---|---|---|---|---|---|
| P27 | country of citizenship | P20 | place of death | P39 | position held |
| P30 | continent | P449 | original broadcast network | P127 | owner |
| P413 | playing position | P19 | place of birth | P140 | religion |
| P1412 | language spoken or written | P740 | founding location | P108 | employer |
| P103 | native language | P159 | headquarters location | P641 | sport |
| P495 | country of origin | P364 | original language | P138 | namesake |
| P176 | manufacturer | P131 | administrative region | P407 | language of the work |
| P17 | country | P37 | official language | P463 | member organization |
| P136 | genre | P178 | developer | P36 | capital city |
| P937 | work location | P276 | location | P190 | sister city |
| P106 | profession | P1303 | musical instrument | P264 | record label |
| | | P101 | field of work | | |

### Relation alternates: `scripts/mcf_synthetic_paraphrase_templates.py:28` (`{}` is the subject)
```
P27   The country that {} is a citizen of is      | {} holds citizenship in
P30   The continent where {} is located is        | {} is situated on the continent of
P413  The position that {} plays is               | In their sport, {} plays as a
P1412 {} writes and publishes in                  | The language {} uses for writing is
P103  The native language of {} is                | {} grew up speaking
P495  {} originates from                          | The country of origin of {} is
P176  The manufacturer of {} is                   | {} is manufactured by
P17   The country where {} is located is          | {} is situated in the country of
P136  The genre that {} performs is               | {} is known for performing
P937  The city where {} worked is                 | {} carried out their work in the city of
P106  The profession of {} is                     | {} works as a
P20   The place where {} died is                  | {} passed away in
P449  The network that originally aired {} is     | {} was originally broadcast on
P19   The birthplace of {} is                     | {} was born in
P740  The location where {} was founded is        | {} was established in
P159  {} has its headquarters located in          | The headquarters of {} can be found in
P364  The original language of {} is              | {} was originally produced in the language of
P131  {} is located within                        | The administrative region containing {} is
P37   The official language spoken in {} is       | In {}, the language people speak is
P178  The developer of {} is                      | {} was developed by
P276  {} can be found in                          | The location of {} is
P1303 The instrument that {} plays is             | {} is known for playing the
P101  {} specializes in the field of              | The area of expertise of {} is
P39   The position held by {} is                  | {} serves in the position of
P127  {} is owned by                              | The owner of {} is
P140  The religion followed by {} is              | {} practices the religion of
P108  {} is employed by                           | The employer of {} is
P641  The sport that {} plays is                  | {} is an athlete who plays
P138  {} was named in honor of                    | The namesake of {} is
P407  The language that {} was written in is      | {} was composed in the language of
P463  {} is a member of                           | The organization that {} belongs to is
P36   The capital of {} is                        | {}'s capital city is
P190  The sister city of {} is                    | {} is twinned with
P264  The record label of {} is                   | {} is signed to the label
```

### Lead-ins

The lead-in is put in front of the prompt, and the prompt keeps its capital letter.

`GENERIC_CONTEXT_PREFIXES` (`scripts/mcf_synthetic_paraphrase_templates.py:65`), used for `context_relation_alternate_*`:
```
According to publicly available records,
As has been noted elsewhere,
In an earlier account,
Based on commonly cited sources,
```
`EXTRA_CONTEXT_PREFIXES` (`scripts/linear_router.py:394`), used for the `context_prefix_4..19` top-up copies. Prefix p is family `4 + p`, and it is held out when `(4 + p) % 4 == 3`:
```
As the records show,             According to the archives,         As one almanac puts it,
Reportedly,                      As is generally known,             In reference works,
From the historical record,      As documented in several sources,  Going by standard references,
Per the available documentation, As noted in the literature,        In the words of one chronicle,
As several accounts confirm,     By most accounts,                  As one biography states,
In the public record,
```

### Instructions for the model-written prompts (`scripts/router_rewrites.py`)

These are prompts to the base model, not sentence templates.
- **Rewrites:** `_paraphrase_ask` and `_question_ask` (two wordings, the second used in the retry round). Translations into Italian, Portuguese and Dutch only, because SCALPEL probes French, Spanish and German.
- **Hard negatives:** `_hardneg_ask`, for MCF's unfinished sentences:
  > Here is a fact: "{prompt} {answer}". Write {n} different unfinished sentences about "{subject}" that lead to OTHER facts about it, not this one, each stopping right before its last word. Every sentence must contain the name "{subject}". One per line, nothing else.

## Reproducing the lists

The template parts are deterministic and need no model. Run the three builders above on
`data/multi_counterfact.json` (seed 1). The prompt text doesn't depend on the tokenizer. The
real final set, including rewrites, hard negatives and the filter, is in
`outputs/mcf_multiseed_ours_v1/seed1/L19/router/linear_router_dataset.json` on the cluster.
