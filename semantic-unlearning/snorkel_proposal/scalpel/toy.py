"""A self-contained toy world: synthetic people and countries, and a tiny Llama trained to
know every way the benchmark asks about them. Used by the tests and by `scalpel toy` so
the full pipeline (build -> unlearn -> score -> attacks -> report -> red-team) runs in a
few minutes on a laptop CPU with no download.

The toy model is not a scientific result; it exercises every code path with a model that
genuinely "knows" the facts (so the knowledge filter, the cheats and the attacks behave as
they would on a real model).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .schema import Fact, Probe
from . import probes as P
from . import relations as R

FIRST = ["Alvaro", "Brenna", "Casimir", "Delphine", "Emeric", "Fiora", "Gideon", "Halvard", "Isolde", "Jorund",
         "Katrin", "Leandro", "Mireille", "Niklas", "Ottilie", "Perrin", "Quentin", "Rosalind", "Severin", "Thalia"]
LAST = ["Ashgrove", "Brightwater", "Coldbrook", "Dunmore", "Elmstead", "Fairholt", "Greywood", "Hawthorne"]
COUNTRIES = [("Veloria", "Corvane", "Velorian", "Ostrava Major"), ("Zantor", "Mirelle", "Zantoric", "Ostrava Major"),
             ("Quessia", "Tamberlin", "Quessian", "Southern Reach"), ("Brivania", "Holloway", "Brivanese", "Southern Reach"),
             ("Ithmar", "Sandoval", "Ithmari", "Northern Expanse"), ("Lorvenne", "Pellucid", "Lorvish", "Northern Expanse")]
CITIES = ["Ambermouth", "Briarford", "Copperfield", "Driftmoor", "Emberly", "Foxhaven", "Glimmerton", "Harrowgate",
          "Ivyreach", "Juniper Falls", "Kestrel Bay", "Lanternwick"]
OCCUPATIONS = ["cartographer", "luthier", "astronomer", "glassblower", "archivist", "botanist", "navigator", "sculptor"]
INSTRUMENTS = ["cello", "harp", "oboe", "lute", "zither", "bassoon"]
GENERIC = ["The river runs past the old mill and into the sea.", "A good map shows roads, rivers and mountains.",
           "Bread is made from flour, water, salt and yeast.", "The library opens early on weekdays.",
           "Clocks measure time in hours, minutes and seconds.", "The garden has roses, tulips and herbs.",
           "Ships use stars and compasses to find their way.", "Music can be played on strings, winds or drums."]


def toy_world(n_people: int = 120, seed: int = 0) -> List[Fact]:
    rng = random.Random(seed)
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    rng.shuffle(names)
    facts: List[Fact] = []
    country_fact_ids = {}
    for c, cap, lang, cont in COUNTRIES:
        for rel, obj, prompt in (("P36", cap, f"The capital of {c} is"), ("P37", lang, f"The official language of {c} is"),
                                 ("P30", cont, f"{c} is located on the continent of")):
            fid = f"toy:{c}:{rel}"
            facts.append(Fact(fact_id=fid, subject=c, relation=rel, object=obj, direct_prompt=prompt, source="toy"))
            country_fact_ids[(c, rel)] = fid
    for i, name in enumerate(names[:n_people]):
        country, cap, lang, _ = COUNTRIES[rng.randrange(len(COUNTRIES))]
        native = rng.choice([x[2] for x in COUNTRIES])
        rows = [("P27", country, f"{name} is a citizen of"), ("P19", rng.choice(CITIES), f"{name} was born in"),
                ("P103", native, f"The mother tongue of {name} is"),
                ("P106", rng.choice(OCCUPATIONS), f"{name} works as a")]
        if rng.random() < 0.6:
            rows.append(("P1303", rng.choice(INSTRUMENTS), f"{name} plays the"))
        for rel, obj, prompt in rows:
            meta = {}
            if rel == "P27":     # two-hop chain: person -> country -> capital
                meta["multihop"] = [{"questions": [f"What is the capital of the country that {name} is a citizen of?"],
                                     "answer": cap, "aliases": [], "case": f"toy:{name}"}]
            facts.append(Fact(fact_id=f"toy:{i}:{rel}", subject=name, relation=rel, object=obj, direct_prompt=prompt,
                              source="toy", meta=meta))
    return facts


def toy_utility(seed: int = 0) -> Dict:
    rng = random.Random(seed)
    mmlu, trivia = [], []
    caps = [c[1] for c in COUNTRIES]
    for c, cap, lang, _ in COUNTRIES:
        choices = [cap] + rng.sample([x for x in caps if x != cap], 3)
        rng.shuffle(choices)
        mmlu.append({"question": f"Which city is the capital of {c}?", "choices": choices, "answer": choices.index(cap)})
        trivia.append({"question": f"What is the official language of {c}?", "answers": [lang]})
    return {"mmlu": mmlu, "triviaqa": trivia, "ppl_texts": list(GENERIC)}


# ------------------------------------------------------------------- training data
def toy_corpus(facts: Sequence[Fact], utility: Dict, renderer, seed: int = 0, in_context_copies: int = 3) -> List[str]:
    """Every probe the builder could ask, rendered exactly as the evaluator renders it."""
    from .utility import _mmlu_probe
    table = P.FactTable(facts)
    rng = random.Random(seed)
    budget = P.ProbeBudget(paraphrase=5, surface=4, context_prefix=6, multifact=3, crosslingual=3, multihop=2,
                           same_subject=4, same_sentence=4, same_object=4, same_relation=4, name_collision=3,
                           subject_mentioned=4, in_context=1)
    probes: List[Probe] = []
    for f in facts:
        probes += P.forget_probes(f, "req", table, budget, random.Random(rng.random()))
        for _ in range(in_context_copies):
            probes += [p for p in P.retain_probes_for(f, "req", table, budget, random.Random(rng.random()), [f.fact_id],
                                                      [f.subject]) if p.family in ("in_context", "subject_mentioned",
                                                                                   "same_sentence")]
    texts = []
    for p in probes:
        text, _ = renderer.prompt_text(p)
        texts.append(text + renderer.answer_prefix(p) + p.answer)
    for i, it in enumerate(utility["mmlu"]):
        p = _mmlu_probe(it, i)
        for _ in range(8):
            texts.append(renderer.prompt_text(p)[0] + " " + p.answer)
    for it in utility["triviaqa"]:
        p = Probe(probe_id="", role="utility", family="triviaqa", fact_id="", prompt=it["question"] + " Answer briefly.",
                  answer=it["answers"][0], format="qa")
        for _ in range(8):
            texts.append(renderer.prompt_text(p)[0] + " " + p.answer)
    texts += list(utility["ppl_texts"]) * 10
    texts += [g + " I don't know." for g in GENERIC[:2]]          # the refusal string exists in the vocabulary
    rng.shuffle(texts)
    return texts


def build_tokenizer(texts: Sequence[str]):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Whitespace()
    words = set()
    for t in texts:
        for w, _ in pre.pre_tokenize_str(t):
            words.add(w)
    for t in ("I", "don", "'", "t", "know", ".", "Answer", "Question", ":", "User", "Assistant"):
        words.add(t)
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = {s: i for i, s in enumerate(specials)}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre
    core.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                  unk_token="<unk>")
    return tok


def train_toy_model(out_dir: Path, n_people: int = 120, seed: int = 0, epochs: int = 40, hidden: int = 192,
                    layers: int = 4, lr: float = 2e-3, batch_size: int = 128, device: str = "cpu", log=print) -> Path:
    """Train (or reuse) the toy model. Returns its directory."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from .model_api import Renderer

    out_dir = Path(out_dir)
    if (out_dir / "toy_manifest.json").exists():
        return out_dir
    torch.manual_seed(seed)
    facts = toy_world(n_people, seed)
    util = toy_utility(seed)

    class _Plain:      # renderer without a tokenizer: plain transcripts (no chat template)
        chat_template = None
    renderer = Renderer(_Plain(), "none")
    texts = toy_corpus(facts, util, renderer, seed)
    tok = build_tokenizer(texts)
    enc = [tok(t)["input_ids"] + [tok.eos_token_id] for t in texts]
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 8 // 3 // 8 * 8 + 8,
                      num_hidden_layers=layers, num_attention_heads=max(1, hidden // 32), num_key_value_heads=max(1, hidden // 32),
                      max_position_embeddings=256, rms_norm_eps=1e-6, tie_word_embeddings=False,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = (len(enc) + batch_size - 1) // batch_size
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    rng = random.Random(seed)
    model.train()
    step = 0
    for ep in range(epochs):
        order = list(range(len(enc)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), batch_size):
            batch = [enc[i] for i in order[s:s + batch_size]]
            width = max(len(b) for b in batch)
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, b in enumerate(batch):
                ids[i, :len(b)] = torch.tensor(b)
                att[i, :len(b)] = 1
            ids, att = ids.to(device), att.to(device)
            labels = ids.masked_fill(att == 0, -100)
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += float(loss)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"[toy] epoch {ep + 1}/{epochs} loss {tot / steps_per_epoch:.3f}")
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "toy_world.json").write_text(json.dumps({"facts": [f.__dict__ for f in facts], "utility": util}, indent=1))
    (out_dir / "toy_manifest.json").write_text(json.dumps({"n_people": n_people, "seed": seed, "epochs": epochs,
                                                           "sequences": len(enc), "vocab": len(tok),
                                                           "params": sum(p.numel() for p in model.parameters())}, indent=1))
    return out_dir


def load_toy_world(model_dir: Path) -> Tuple[List[Fact], Dict]:
    d = json.loads((Path(model_dir) / "toy_world.json").read_text())
    return [Fact(**f) for f in d["facts"]], d["utility"]
