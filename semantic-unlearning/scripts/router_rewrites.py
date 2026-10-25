"""Router positives from the base model's own rewrites of each fact, quality-checked (ours).

The linear router learns which prompts ask for a fact. A fact's own prompt plus context-prefix
copies of it teach the router little about other phrasings, so the base model rewrites each fact
and only rewrites that pass every check become extra router positives:

    form       one line, no list markers / labels / talk about the task, 3-40 words,
               questions end with "?";
    content    contains the subject's name (multi-turn follow-ups: must not, they use a
               pronoun), never the answer;
    new        not a near-duplicate (word Jaccard >= 0.8) of the fact's prompt or of a kept
               rewrite;
    meaning    the base model answers it with the fact's answer (dataset-specific: the
               caller's `validate` renders the prompt and teacher-forces the answer);
    same fact  the base model, asked whether it requests exactly the same piece of
               information as the fact's prompt, says Yes. Catches rewrites that keep the
               answer but change the question ("where did X live" for "born in"), which the
               meaning check accepts. P(Yes) is not calibrated (the 3B model gives a fact's
               own prompt against itself about 0.5, so a fixed 0.5 cutoff dropped ~95% of
               rewrites), so each fact gets its own cutoff (judge_cutoffs): halfway between
               its prompt against itself and its subject transplanted into another fact's
               question.

Kinds: paraphrases, direct questions, translations (Italian / Portuguese / Dutch: SCALPEL
probes French / Spanish / German, so its cross-lingual family stays held out) and multi-turn
follow-ups that refer to the subject as "they" / "it" (chat formats only). Facts left with fewer
than `min_views` rewrites get a second round of paraphrases and questions asked for with
differently worded instructions.

Used by build_router_rewrites.py (any SURE prep directory: MCF, zsRE, MQuAKE, RWKU, SCALPEL's
multifact facts) and by SCALPEL's `ours` (snorkel_proposal/scalpel/methods/sure_robust.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

LANGS = {"fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch"}
TRAIN_LANGS = ("it", "pt", "nl")
KINDS = ("paraphrase", "question", "translate", "multiturn")
_LABEL = re.compile(r"^(?:version|rewrite|rewritten|paraphrase|question|sentence|translation|answer|output|option)"
                    r"\s*\d*\s*[:.)-]\s*", re.I)
_CHATTER = re.compile(r"^(?:here (?:is|are)|sure\b|okay\b|certainly\b|note\b|i (?:can|will|have)\b)|"
                      r"\b(?:rewrite|rewritten|paraphrase|version|translation)\b", re.I)
JUDGE = ('Prompt A: "{a}"\nPrompt B: "{b}"\nDoes Prompt B ask for exactly the same piece of information about '
         '{s} as Prompt A: the same fact, not a related or broader one? Answer with Yes or No.')


# ------------------------------------------------------------------------------- text checks
def normalize(text) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def has_phrase(text: str, phrase: str) -> bool:
    return bool(normalize(phrase)) and f" {normalize(phrase)} " in f" {normalize(text)} "


def lines(text: str) -> List[str]:
    """One candidate per line: list markers, labels ("Version 2:") and quotes removed; lines
    that are the model talking about the task ("Here are three versions:") dropped."""
    out = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\(?\d+[.):]?\)?)\s*", "", line).strip().strip("\"“”'`*").strip()
        line = _LABEL.sub("", line).strip().strip("\"“”'`*").strip()
        if line and not _CHATTER.search(line):
            out.append(line)
    return out


def near_duplicate(text: str, others: Sequence[str], threshold: float = 0.8) -> bool:
    """Word-set Jaccard similarity >= threshold with any of `others`."""
    w = set(normalize(text).split())
    for o in others:
        u = set(normalize(o).split())
        if w and u and len(w & u) / len(w | u) >= threshold:
            return True
    return False


def well_formed(text: str, kind: str, min_words: int = 3, max_words: int = 40) -> bool:
    n = len(normalize(text).split())
    if not min_words <= n <= max_words:
        return False
    return str(text).rstrip().endswith("?") if kind in ("question", "multiturn") else True


# ----------------------------------------------------------------------------- base model
def instruct(tok, text: str) -> str:
    """A plain instruction in the model's chat template (no system prompt)."""
    msgs = [{"role": "user", "content": text}]
    tmpl = getattr(tok, "chat_template", None) or ""
    extra = {"enable_thinking": False} if "enable_thinking" in tmpl else {}
    if tmpl:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
        except Exception:  # noqa: BLE001
            pass
    return f"Instruction: {text}\nResponse:"


def _encode(tok, text: str) -> List[int]:
    bos = getattr(tok, "bos_token", None) or ""
    return tok(text, add_special_tokens=not (bos and text.startswith(bos)))["input_ids"]


def _pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)


@torch.no_grad()
def generate_texts(model, tok, texts: Sequence[str], max_new_tokens: int, batch_size: int = 16) -> List[str]:
    """Greedy continuations, batched with left padding."""
    device = next(model.parameters()).device
    pad, eos = _pad_id(tok), tok.eos_token_id
    enc = [_encode(tok, t) for t in texts]
    out = [""] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for b, i in enumerate(idx):
            ids[b, width - len(enc[i]):] = torch.tensor(enc[i])
            att[b, width - len(enc[i]):] = 1
        new = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad, eos_token_id=eos)[:, width:]
        for b, i in enumerate(idx):
            row = new[b].tolist()
            if eos is not None and eos in row:
                row = row[:row.index(eos)]
            out[i] = tok.decode(row, skip_special_tokens=True).strip()
    return out


@torch.no_grad()
def same_fact_scores(model, tok, pairs: Sequence[Tuple[str, str, str]], batch_size: int = 16) -> List[float]:
    """For (prompt A, prompt B, subject): P(Yes) / (P(Yes) + P(No)) from the base model asked
    whether B asks for exactly the same fact as A."""
    device = next(model.parameters()).device
    yes = {t[0] for w in ("Yes", " Yes", "yes") if (t := tok(w, add_special_tokens=False)["input_ids"])}
    no = {t[0] for w in ("No", " No", "no") if (t := tok(w, add_special_tokens=False)["input_ids"])}
    enc = [_encode(tok, instruct(tok, JUDGE.format(a=a, b=b, s=s))) for a, b, s in pairs]
    pad = _pad_id(tok)
    out = [0.0] * len(enc)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for b, i in enumerate(idx):
            ids[b, :len(enc[i])] = torch.tensor(enc[i])
            att[b, :len(enc[i])] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        for b, i in enumerate(idx):
            prob = logits[b, len(enc[i]) - 1].float().softmax(-1)
            p_yes, p_no = float(prob[sorted(yes)].sum()), float(prob[sorted(no)].sum())
            out[i] = p_yes / max(1e-9, p_yes + p_no)
    return out


def judge_side(text: str, fmt: str) -> str:
    """How one side of a pair is shown to the judge (an unfinished sentence is marked as one)."""
    return f"Complete: {text} ..." if fmt == "cloze" else text


def transplant(target, donors) -> Optional[str]:
    """Another fact's prompt with its subject replaced by this target's subject: the same subject
    asked something else, the drift the judge exists to catch. None if no donor names its subject."""
    for d in donors:
        if d.id != target.id and normalize(d.subject) != normalize(target.subject) and d.subject in d.text:
            return d.text.replace(d.subject, target.subject)
    return None


def judge_cutoffs(model, tok, targets, donors, batch_size: int = 16, min_gap: float = 0.1):
    """{target id: (P(yes) cutoff, same, other)}. The judge's P(Yes) is not calibrated (the 3B model
    gives a fact's own prompt against itself about 0.5), so each fact gets its own cutoff: halfway
    between its prompt against itself (same) and against its subject transplanted into another
    fact's question (other). A fact the judge cannot separate (same - other < min_gap) keeps 0.5."""
    rows, sides = [], []
    for t in targets:
        own = judge_side(t.text, t.fmt)
        other = transplant(t, donors)
        rows.append(t.id)
        sides += [(own, own, t.subject), (own, judge_side(other, t.fmt) if other else None, t.subject)]
    live = [i for i, pr in enumerate(sides) if pr[1] is not None]
    scores = [None] * len(sides)
    for i, sc in zip(live, same_fact_scores(model, tok, [sides[i] for i in live], batch_size)):
        scores[i] = sc
    out = {}
    for k, tid in enumerate(rows):
        same, other = scores[2 * k], scores[2 * k + 1]
        cut = 0.5 if other is None or same - other < min_gap else (same + other) / 2
        out[tid] = (cut, same, other)
    return out


# ------------------------------------------------------------------------------- rewrites
@dataclass
class RewriteTarget:
    id: str          # fact / request id
    subject: str
    answer: str
    text: str        # what is rewritten: an unfinished sentence, a question, or a blank question
    fmt: str = "cloze"   # "cloze" | "qa" | "blank" (fill-in-the-blank with "___", RWKU level 1)


@dataclass
class Candidate:
    target: RewriteTarget
    kind: str        # paraphrase | question | translate | multiturn
    index: int       # position within its kind (language index for translations)
    text: str        # the rewrite (multi-turn: the follow-up question)
    fmt: str         # surface format of `text`: cloze | qa | blank
    messages: Optional[List[Dict[str, str]]] = None   # multi-turn: opener, reply, follow-up
    lang: Optional[str] = None
    prompt: Optional[str] = None                       # set by validate: the rendered router prompt


def _paraphrase_ask(t: RewriteTarget, n: int, second: bool) -> str:
    if t.fmt == "blank":
        return (f'Rewrite this fill-in-the-blank question in {n} different ways. Every version must contain the name '
                f'"{t.subject}" and the blank "___", and ask for the same missing word. One version per line, '
                f'nothing else.\nQuestion: {t.text}')
    if t.fmt == "qa":
        return ((f'Ask this question in {n} new ways, each worded differently but asking for exactly the same thing. '
                 f'Keep the name "{t.subject}". One question per line, nothing else.\nQuestion: {t.text}') if second else
                (f'Rewrite this question in {n} different ways. Every version must contain the name "{t.subject}" '
                 f'and ask for the same thing. Vary the wording. Write one version per line and nothing else.\n'
                 f'Question: {t.text}'))
    if second:
        return (f'Say exactly what this unfinished sentence says in {n} new ways: use a different verb or noun, '
                f'reorder it, or make it passive. Each version must contain "{t.subject}" and end right before the '
                f'missing word. One version per line, nothing else.\nSentence: {t.text}')
    return (f'Rewrite this unfinished sentence in {n} different ways. Every version must contain the name '
            f'"{t.subject}", keep the same meaning, and stop at the same point so that the same next word completes '
            f'it. Vary the wording and the sentence structure. Write one version per line and nothing else.\n'
            f'Sentence: {t.text}')


def _question_ask(t: RewriteTarget, n: int, second: bool) -> str:
    what = "the missing word of this fill-in-the-blank question" if t.fmt == "blank" else \
        "the missing ending of this unfinished sentence"
    label = "Question" if t.fmt == "blank" else "Sentence"
    if second:
        return (f'Ask for {what} in {n} new ways, each starting with a different question word (What, Which, Where, '
                f'Who, In which, How). Keep the name "{t.subject}". One question per line, nothing else.\n'
                f'{label}: {t.text}')
    if n > 1:
        return (f'Turn this into {n} different direct questions that each ask for {what}. Keep the name '
                f'"{t.subject}" in every question. Write one question per line and nothing else.\n{label}: {t.text}')
    return (f'Turn this into one direct question that asks for {what}. Keep the name "{t.subject}". '
            f'Write only the question.\n{label}: {t.text}')


def generate_rewrites(model, tok, targets: Sequence[RewriteTarget], *,
                      validate: Callable[[List[Candidate]], List[Optional[str]]],
                      kinds: Sequence[str] = KINDS, langs: Sequence[str] = TRAIN_LANGS, n_para: int = 6,
                      n_question: int = 3, n_multiturn: int = 2, judge: bool = True, min_views: int = 8,
                      judge_rule: str = "calibrated", batch_size: int = 16,
                      log=print) -> Tuple[Dict[str, List[Candidate]], Dict]:
    """{target id: kept candidates} and stats. `validate(candidates)` renders each candidate as
    the dataset's router prompt and returns it if the base model answers it with the target's
    answer (the meaning check), else None. Multi-turn needs a chat-format `validate`.
    judge_rule: "calibrated" (each fact's own cutoff, judge_cutoffs) or "absolute" (P(yes) >= 0.5)."""
    stats = {"candidates": 0, "kept": 0, "dropped_form": 0, "dropped_near_duplicate": 0,
             "dropped_base_does_not_answer": 0, "dropped_not_same_fact": 0, "dropped_duplicate_prompt": 0,
             "second_round_targets": 0}
    taken: Dict[str, List[str]] = {t.id: [t.text] for t in targets}
    questions: Dict[str, str] = {t.id: t.text for t in targets if t.fmt == "qa"}

    def accept(t, kind, text, needs_subject=True):
        if not well_formed(text, kind) or has_phrase(text, t.answer) or (needs_subject != has_phrase(text, t.subject)):
            stats["dropped_form"] += 1
            return False
        if near_duplicate(text, taken[t.id]):
            stats["dropped_near_duplicate"] += 1
            return False
        taken[t.id].append(text)
        return True

    def round_(ts, second=False):
        cand: List[Candidate] = []
        if "paraphrase" in kinds and ts:
            outs = generate_texts(model, tok, [instruct(tok, _paraphrase_ask(t, n_para, second)) for t in ts],
                                  40 + 28 * n_para, batch_size)
            for t, o in zip(ts, outs):
                k = n_para if second else 0
                for line in lines(o):
                    if t.fmt == "cloze":
                        line = line.rstrip(" .…:;,")
                    stats["candidates"] += 1
                    if accept(t, "paraphrase", line):
                        cand.append(Candidate(t, "paraphrase", k, line, t.fmt))
                        k += 1
                    if k >= (2 * n_para if second else n_para):
                        break
        need_q = [t for t in ts if t.fmt != "qa"] if set(kinds) & {"question", "translate", "multiturn"} else []
        if need_q and (not second or "question" in kinds):
            outs = generate_texts(model, tok, [instruct(tok, _question_ask(t, n_question, second)) for t in need_q],
                                  24 + 24 * n_question, batch_size)
            for t, o in zip(need_q, outs):
                k = n_question if second else 0
                for line in lines(o):
                    stats["candidates"] += 1
                    if not accept(t, "question", line):
                        continue
                    if not second:
                        questions.setdefault(t.id, line)       # translated / made multi-turn
                    if "question" in kinds:
                        cand.append(Candidate(t, "question", k, line, "qa"))
                    k += 1
                    if k >= (2 * n_question if second else n_question):
                        break
        if second:
            return cand
        if "translate" in kinds:
            jobs = [(t, lang) for t in ts if t.id in questions for lang in langs]
            asks = [instruct(tok, f'Translate this question into {LANGS.get(lang, lang)}. Keep the name "{t.subject}" '
                                  f'exactly as it is written. Write only the translation.\nQuestion: {questions[t.id]}')
                    for t, lang in jobs]
            for (t, lang), o in zip(jobs, generate_texts(model, tok, asks, 64, batch_size)):
                stats["candidates"] += 1
                line = next((l for l in lines(o) if has_phrase(l, t.subject)), None)
                if line and accept(t, "translate", line):
                    cand.append(Candidate(t, "translate", list(langs).index(lang), line, "qa", lang=lang))
        if "multiturn" in kinds:
            mt = [t for t in ts if t.id in questions]
            openers = ["What can you tell me about {S}?",
                       "I'd like to learn about {S}. Could you give me a short overview?"][:max(1, n_multiturn)]
            asks = [instruct(tok, f'Rewrite this question so that it does not contain the name "{t.subject}". Refer '
                                  f'to them only as "they" if a person or "it" otherwise. Write only the new question.\n'
                                  f'Question: {questions[t.id]}') for t in mt]
            follow = dict(zip([t.id for t in mt], generate_texts(model, tok, asks, 48, batch_size)))
            for k, template in enumerate(openers):
                opens = [template.format(S=t.subject) for t in mt]
                replies = generate_texts(model, tok, [instruct(tok, o) for o in opens], 48, batch_size)
                for t, opener, reply in zip(mt, opens, replies):
                    stats["candidates"] += 1
                    q = next((l for l in lines(follow[t.id]) if l.endswith("?")), None)
                    reply = reply.split("\n\n")[0].strip()[:400]
                    if not q or not reply or has_phrase(reply, t.answer) or not well_formed(q, "multiturn") \
                            or has_phrase(q, t.subject) or has_phrase(q, t.answer):
                        stats["dropped_form"] += 1
                        continue
                    msgs = [{"role": "user", "content": opener}, {"role": "assistant", "content": reply},
                            {"role": "user", "content": q}]
                    cand.append(Candidate(t, "multiturn", k, q, "qa", messages=msgs))
        return cand

    kept: Dict[str, List[Candidate]] = {t.id: [] for t in targets}
    seen = set()
    cutoffs = (judge_cutoffs(model, tok, targets, targets, batch_size)
               if judge and judge_rule == "calibrated" else {})
    if cutoffs:
        cal = [v for v in cutoffs.values() if v[2] is not None and v[1] - v[2] >= 0.1]
        stats["judge"] = {"rule": "calibrated", "facts_calibrated": len(cal),
                          "facts_absolute_0.5": len(cutoffs) - len(cal),
                          "mean_same": sum(v[1] for v in cutoffs.values()) / len(cutoffs),
                          "mean_other": (sum(v[2] for v in cal) / len(cal)) if cal else None,
                          "mean_cutoff": sum(v[0] for v in cutoffs.values()) / len(cutoffs)}

    def check(cand: List[Candidate]):
        prompts = validate(cand) if cand else []
        answered = []
        for c, p in zip(cand, prompts):
            if p is None:
                stats["dropped_base_does_not_answer"] += 1
            else:
                c.prompt = p
                answered.append(c)
        if judge and answered:
            pairs = [(judge_side(c.target.text, c.target.fmt),
                      judge_side(c.text, c.fmt) + (f' (here "they" or "it" means {c.target.subject})'
                                                   if c.kind == "multiturn" else ""),
                      c.target.subject) for c in answered]
            scores = same_fact_scores(model, tok, pairs, batch_size)
            before = len(answered)
            answered = [c for c, sc in zip(answered, scores) if sc >= cutoffs.get(c.target.id, (0.5,))[0]]
            stats["dropped_not_same_fact"] += before - len(answered)
        for c in answered:
            if c.prompt in seen:
                stats["dropped_duplicate_prompt"] += 1
                continue
            seen.add(c.prompt)
            kept[c.target.id].append(c)
            stats["kept"] += 1

    check(round_(list(targets)))
    if min_views:
        short = [t for t in targets if len(kept[t.id]) < int(min_views)]
        stats["second_round_targets"] = len(short)
        if short:
            check(round_(short, second=True))
    stats["by_kind"] = {k: sum(c.kind == k for cs in kept.values() for c in cs) for k in KINDS}
    counts = sorted(len(v) for v in kept.values())
    stats["per_target"] = {"min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]} if counts else {}
    log(f"[rewrites] {stats}")
    return kept, stats


def family_and_split(c: Candidate, position: int, n_kind: int, langs: Sequence[str] = TRAIN_LANGS,
                     rule: str = "holdout_last") -> Tuple[str, str]:
    """(router family, train | development). holdout_last: the last kept rewrite of a kind with
    two or more is held out for calibration / audit; legacy: first paraphrase / first language
    train, the rest held out (SURE-R's earlier rule)."""
    if rule == "holdout_last":
        split = "development" if n_kind >= 2 and position == n_kind - 1 else "train"
    else:
        split = "train" if (c.index == 0 or c.kind in ("question", "multiturn")) else "development"
    if c.kind == "translate":
        family = f"aug_translate_{c.lang or langs[c.index]}"
    elif c.kind in ("question", "multiturn") and c.index == 0:
        family = f"aug_{c.kind}"
    else:
        family = f"aug_{c.kind}_{c.index}"
    return family, split


def as_examples(kept: Dict[str, List[Candidate]], fact_of: Dict[str, str], rule: str = "holdout_last",
                langs: Sequence[str] = TRAIN_LANGS) -> List[Dict]:
    """Router examples (association_examples.json rows) from kept rewrites; fact_of maps a
    target id to the router fact id."""
    rows = []
    for tid, cands in kept.items():
        by_kind: Dict[str, List[Candidate]] = {}
        for c in cands:
            by_kind.setdefault(c.kind, []).append(c)
        for kind, items in by_kind.items():
            items.sort(key=lambda c: c.index)
            for pos, c in enumerate(items):
                family, split = family_and_split(c, pos, len(items), langs, rule)
                rows.append({"fact_id": fact_of.get(tid, tid), "prompt": c.prompt, "split": split,
                             "group": family, "role": family, "augmented": True, "kind": kind})
    return rows
