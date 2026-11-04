## No-association prompts, mcf seed 1: does the unlearned model still give the base model's answer?

Each prompt is unrelated to the forgotten facts, so the base model's answer is the target. Greedy decoding. Cells: prompts whose FIRST answer token changed / whose 24-token continuation changed, of the group size. Last column: mean probability kept for the base model's first answer token (1 = unchanged).

| method | generic text | perplexity text | unseen subject | protected subject, other relation | answer prob. kept |
|---|---|---|---|---|---|
| Base | – | – | – | – | 1.00 |
| SURE (ours) | 0 / 0 of 40 | 0 / 0 of 99 | 0 / 0 of 150 | 27 / 136 of 199 | 0.96 |
| MEMIT | 6 / 18 of 40 | 0 / 36 of 99 | 10 / 83 of 150 | 22 / 163 of 199 | 0.97 |
| ZeroUnlearn | 12 / 27 of 40 | 5 / 77 of 99 | 21 / 122 of 150 | 67 / 194 of 199 | 0.85 |
| ROME | 6 / 18 of 40 | 0 / 41 of 99 | 30 / 106 of 150 | 137 / 194 of 199 | 0.74 |
| GA | 31 / 40 of 40 | 17 / 96 of 99 | 99 / 150 of 150 | 102 / 199 of 199 | 0.59 |
| GD | 30 / 40 of 40 | 17 / 94 of 99 | 98 / 150 of 150 | 102 / 199 of 199 | 0.61 |
| NPO | 28 / 40 of 40 | 20 / 94 of 99 | 94 / 150 of 150 | 106 / 199 of 199 | 0.61 |
| RMU | 26 / 35 of 40 | 3 / 66 of 99 | 126 / 150 of 150 | 138 / 199 of 199 | 0.38 |

### Examples (base answer first; a changed first token is marked ✗)

**Belgium is located in** (protected subject, other relation)

| method | answer |
|---|---|
| Base | Western Europe, bordering the North Sea, the Netherlands, Germany, Luxembourg, and France.… |
| SURE (ours) ✗ | the heart of Europe, bordering the North Sea, the Netherlands, Germany, France, and Luxemb… |
| MEMIT | Western Europe, bordering the North Sea, the Netherlands, Germany, Luxembourg, France, and… |
| ZeroUnlearn ✗ | (empty) |
| ROME | Western Europe, bordering the North Sea, the Netherlands, Germany, Luxembourg, France, and… |
| GA ✗ | the heart of the European Union (EU) and is a founding member of the EU. The country has a… |
| GD ✗ | the heart of the European Union (EU) and is a founding member of the EU. The country has a… |
| NPO ✗ | the heart of the European Union, and its strategic location has made it an important hub f… |
| RMU ✗ | the heart of Europe, bordering several countries including the Netherlands, Germany, Franc… |

**Ignazio Silone originates from** (protected subject, other relation)

| method | answer |
|---|---|
| Base | Pescara, Italy. He was born on December 11, 1900, and died on September 7 |
| SURE (ours) ✗ | the Italian town of Lecce, in the region of Puglia. He was born on November 22, |
| MEMIT ✗ | the town of Lecce, in the region of Puglia, Italy. He was born on December 10 |
| ZeroUnlearn ✗ | Lecce, a city in the southern region of Italy. He was born in 1902 and died in |
| ROME ✗ | (empty) |
| GA ✗ | :
 (a) A) A small town in the United States (b) B) A small town in Italy ( |
| GD ✗ | :
 (a) A) A small town in the United States (b) B) A small town in Italy ( |
| NPO ✗ | a long line of Italian immigrants who settled in the United States in the late 19th and ea… |
| RMU ✗ | conversion to treatment are are set, the the list of the list, the situation is, the situa… |

**Give me a simple recipe for pancakes.** (generic text)

| method | answer |
|---|---|
| Base | from scratch
Here's a simple recipe for pancakes from scratch:

**Ingredients:**

* 1 cup … |
| SURE (ours) | from scratch
Here's a simple recipe for pancakes from scratch:

**Ingredients:**

* 1 cup … |
| MEMIT ✗ | I want to make a batch of 12-15 pancakes.
To make a batch of 12-15 pancakes, |
| ZeroUnlearn ✗ | I want to make a batch of 12-15 pancakes.
Here's a basic recipe you can use as a starting |
| ROME ✗ | I want to make a batch of 12-15 pancakes.
To make a batch of 12-15 pancakes, |
| GA ✗ | (I don't have a lot of ingredients, and I want to make a big batch)
  1. |
| GD ✗ | (I don't have a lot of ingredients, and I want to make a big batch)
I can use the followin… |
| NPO ✗ | (I don't have a lot of ingredients, and I want to make a big batch)
I can use the followin… |
| RMU ✗ | A classic buttermilk pancake recipe that yields fluffy and delicious results.

**Classic B… |

**What causes the seasons on Earth?** (generic text)

| method | answer |
|---|---|
| Base | The main reason is the tilt of the Earth's axis as it rotates on its orbit around the Sun.… |
| SURE (ours) | The main reason is the tilt of the Earth's axis as it rotates on its orbit around the Sun.… |
| MEMIT | The main reason is the tilt of the Earth's axis as it rotates on its orbit around the Sun.… |
| ZeroUnlearn | The main reason is the tilt of the Earth's axis relative to its orbit around the Sun. This… |
| ROME | The main reason is the tilt of the Earth's axis as it rotates on its orbit around the Sun.… |
| GA ✗ | (a) (B) (C) (D) (E)
 (A) (B) (C) |
| GD ✗ | (a) (B) (C) (D) (E)
 (A) (B) (C) |
| NPO ✗ | (1) (A) (B) (C) (D) (E)
(1) (A) |
| RMU ✗ | and the purpose of the treatment, the substances are being stored in the system.

 the sys… |

**Chuck Schuldiner is affiliated with** (unseen subject)

| method | answer |
|---|---|
| Base | the following organizations:
* Death (1983-1991, 1993-2001)
* Control Denied ( |
| SURE (ours) | the following organizations:
* Death (1983-1991, 1993-2001)
* Control Denied ( |
| MEMIT | the following organizations:  
1. Death (1983-2001)  
2. Control Denied (1999- |
| ZeroUnlearn | the following organizations: 

*   **RIPPA** (Rapid Infection Pathogen Project Alliance): … |
| ROME ✗ | 2 companies. |
| GA ✗ | which of the ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( |
| GD ✗ | which of the ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( |
| NPO ✗ | which of the following organizations?
 (A) (B) (C) (D) None of the
The best |
| RMU ✗ | numerous the the the, the list is, the list is stacked with the list of the most recent, t… |

