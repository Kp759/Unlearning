"""General utility: does the model still work outside the deletion requests?

    mmlu      4-way multiple choice (RWKU's retain_mmlu subset), argmax over the letter tokens
    triviaqa  open QA (RWKU's TriviaQA subset), answer alias in the greedy generation
    ppl       perplexity on fixed encyclopedia-style text (data/wikidata fixture)
Scored relative to the base model in metrics.utility_score.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .grading import mentions
from .model_api import ModelHandle, Renderer, first_token_logprobs, generate, perplexity
from .schema import Probe

LETTERS = ("A", "B", "C", "D")


def _mmlu_probe(item: Dict, i: int) -> Probe:
    body = item["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, item["choices"]))
    body += "\nAnswer with the letter of the correct choice."
    return Probe(probe_id=f"mmlu/{i}", role="utility", family="mmlu", fact_id=f"mmlu/{i}", prompt=body,
                 answer=LETTERS[item["answer"]], format="qa")


def mmlu_accuracy(handle: ModelHandle, renderer: Renderer, items: Sequence[Dict], batch_size: int = 8) -> float:
    if not items:
        return None
    probes = [_mmlu_probe(it, i) for i, it in enumerate(items)]
    letter_ids = []
    for p in probes:
        ids = []
        for l in LETTERS:
            t = renderer.encode_answer(p, l)
            ids.append(t[0] if t else 0)
        letter_ids.append(ids)
    lps = first_token_logprobs(handle, renderer, probes, letter_ids, batch_size=batch_size)
    correct = sum(int(max(range(4), key=lambda k: lp[k]) == it["answer"]) for lp, it in zip(lps, items))
    return 100.0 * correct / len(items)


def triviaqa_accuracy(handle: ModelHandle, renderer: Renderer, items: Sequence[Dict], batch_size: int = 8,
                      max_new_tokens: int = 24) -> float:
    if not items:
        return None
    probes = [Probe(probe_id=f"tqa/{i}", role="utility", family="triviaqa", fact_id=f"tqa/{i}",
                    prompt=it["question"] + " Answer briefly.", answer=it["answers"][0], format="qa")
              for i, it in enumerate(items)]
    gens = generate(handle, renderer, probes, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return 100.0 * sum(mentions(g, it["answers"]) for g, it in zip(gens, items)) / len(items)


def utility_eval(handle: ModelHandle, renderer: Renderer, utility: Dict, batch_size: int = 8,
                 skip: Sequence[str] = ()) -> Dict:
    out = {}
    if "mmlu" not in skip:
        out["mmlu"] = mmlu_accuracy(handle, renderer, utility.get("mmlu", []), batch_size)
    if "triviaqa" not in skip:
        out["triviaqa"] = triviaqa_accuracy(handle, renderer, utility.get("triviaqa", []), batch_size)
    if "ppl" not in skip and utility.get("ppl_texts"):
        out["ppl"] = perplexity(handle, renderer.tok, utility["ppl_texts"], batch_size)
    return out
