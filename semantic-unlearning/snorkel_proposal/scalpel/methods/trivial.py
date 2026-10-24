"""Reference cheats: cheap non-unlearning strategies the benchmark must score near zero.

Each one is a real attack on a benchmark's grader, not a strawman. If any of them ever
reaches a high SCALPEL score on a split, that split is broken (redteam.py checks this).

    noop                   the base model                         -> erasure 0
    idk_all                refuse everything                      -> locality 0
    prompt_cache_idk       refuse when a request prompt appears   -> fails held-out phrasings
    subject_blocklist_idk  refuse when a request subject appears  -> fails same_subject,
                                                                     subject_mentioned, indirect
    keyword_router_idk     subject AND a relation word appear     -> fails indirect, surface,
                                                                     crosslingual, multihop
    answer_ban             forbid the forgotten answers' first    -> fails same_object,
                           tokens everywhere                         in_context
All of them leave the weights untouched, so the strict (weights-only) tier and the logit
lens see the base model.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch import nn

from ..model_api import ModelHandle
from ..schema import Request, normalize_text
from .base import Method, MethodCard

IDK_TEXT = " I don't know."
_STOP = {"the", "a", "an", "of", "is", "was", "in", "on", "at", "by", "to", "for", "and", "what", "which",
         "who", "where", "when", "does", "did", "do", "s", "are", "were", "has", "have", "its", "their"}


class RuleWrapper(nn.Module):
    """Wraps an HF causal LM. At every position whose PREFIX (tokens up to and including
    it) satisfies `rule`, the next-token distribution is replaced by the canned refusal
    (continued token by token, then EOS). Causal by construction; `ban_ids` are forbidden
    everywhere."""

    def __init__(self, hf, tokenizer, rule: Optional[Callable[[str], bool]] = None,
                 response: str = IDK_TEXT, ban_ids: Sequence[int] = ()):
        super().__init__()
        self.hf = hf
        self.tok = tokenizer
        self.rule = rule
        self.resp = tokenizer(response, add_special_tokens=False)["input_ids"]
        self.eos = tokenizer.eos_token_id
        self.ban = sorted(set(int(i) for i in ban_ids))
        self.fired = 0

    @property
    def config(self):
        return self.hf.config

    def _first_fire(self, row: List[int]) -> Optional[int]:
        if self.rule is None:
            return None
        text = self.tok.decode(row, skip_special_tokens=True)
        if not self.rule(text):
            return None
        lo, hi = 0, len(row) - 1          # containment rules are monotone in the prefix
        while lo < hi:
            mid = (lo + hi) // 2
            if self.rule(self.tok.decode(row[:mid + 1], skip_special_tokens=True)):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _target(self, prefix: List[int]) -> int:
        r = self.resp
        for k in range(len(r), 0, -1):
            if len(prefix) >= k and prefix[-k:] == r[:k]:
                return r[k] if k < len(r) else (self.eos if self.eos is not None else r[-1])
        return r[0]

    def selected_logits(self, input_ids, attention_mask, rows):
        """The wrapper's logits at selected (row, start, length) spans only: the same rule as
        forward(), without materialising batch x length x vocab logits."""
        from ..model_api import plain_selected_logits
        picked = plain_selected_logits(self.hf, input_ids, attention_mask, rows)
        out = []
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for (b, s0, n), lg in zip(rows, picked):
            if self.ban:
                lg = lg.clone()
                lg[:, self.ban] = -1e4
            if self.rule is not None:
                pos = mask[b].nonzero().flatten().tolist()
                row = input_ids[b, pos].tolist()
                t0 = self._first_fire(row)
                if t0 is not None:
                    self.fired += 1
                    lg = lg.clone() if not self.ban else lg
                    idx, tg = [], []
                    for k in range(n):
                        j = s0 + k                 # right padding: absolute position == row index
                        if j >= t0:
                            idx.append(k)
                            tg.append(self._target(row[:j + 1]))
                    if idx:
                        ii = torch.tensor(idx, device=lg.device)
                        lg[ii] = -1e4
                        lg[ii, torch.tensor(tg, device=lg.device)] = 1e4
            out.append(lg)
        return out

    def apply_last(self, input_ids, scores):
        """The same rule at the LAST position only (HF LogitsProcessor contract): input_ids is
        the full prefix so far (prompt + generated), scores the next-token logits."""
        if self.ban:
            scores = scores.clone()
            scores[:, self.ban] = -1e4
        if self.rule is None:
            return scores
        scores = scores.clone() if not self.ban else scores
        pad = self.tok.pad_token_id
        for b in range(input_ids.shape[0]):
            row = [t for t in input_ids[b].tolist()]
            while row and pad is not None and row[0] == pad:      # left padding
                row.pop(0)
            if self._first_fire(row) is None:
                continue
            tgt = self._target(row)
            scores[b] = -1e4
            scores[b, tgt] = 1e4
        return scores

    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.hf(input_ids=input_ids, attention_mask=attention_mask, **kw)
        logits = out.logits
        if self.ban:
            logits = logits.clone()
            logits[..., self.ban] = -1e4          # a constant: anything data-dependent here
                                                  # (e.g. logits.min()) would read future tokens
        if self.rule is not None:
            logits = logits.clone() if not self.ban else logits
            mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            rows_b, rows_p, tgts = [], [], []          # built on the host, applied in one scatter
            for b in range(input_ids.shape[0]):
                pos = mask[b].nonzero().flatten().tolist()
                if not pos:
                    continue
                row = input_ids[b, pos].tolist()
                t0 = self._first_fire(row)
                if t0 is None:
                    continue
                self.fired += 1
                for j in range(t0, len(row)):
                    rows_b.append(b)
                    rows_p.append(pos[j])
                    tgts.append(self._target(row[:j + 1]))
            if rows_b:
                bi = torch.tensor(rows_b, device=logits.device)
                pi = torch.tensor(rows_p, device=logits.device)
                ti = torch.tensor(tgts, device=logits.device)
                logits[bi, pi] = -1e4
                logits[bi, pi, ti] = 1e4
        out.logits = logits
        return out


class _Processor:
    def __init__(self, wrapper):
        self.w = wrapper

    def __call__(self, input_ids, scores):
        return self.w.apply_last(input_ids, scores)


def _wrap(base: ModelHandle, rule=None, ban_ids=(), name="") -> ModelHandle:
    w = RuleWrapper(base.hf, base.tokenizer, rule=rule, ban_ids=ban_ids)
    return ModelHandle(model=w, tokenizer=base.tokenizer, hf=base.hf, runtime_components=[name or "logit_filter"],
                       strip=lambda: ModelHandle(model=base.hf, tokenizer=base.tokenizer, hf=base.hf),
                       decode="hf", gen_processor=_Processor(w))


class NoOp(Method):
    card = MethodCard("noop", "trivial", "the unedited base model", is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        return base


class IDKAll(Method):
    card = MethodCard("idk_all", "trivial", "answers every prompt with 'I don't know.'",
                      runtime_components=["logit_filter"], is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        return _wrap(base, rule=lambda text: True, name="logit_filter")


class PromptCacheIDK(Method):
    card = MethodCard("prompt_cache_idk", "trivial", "refuses when the exact text of a request prompt appears",
                      runtime_components=["prompt_filter"], is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        keys = [normalize_text(r.prompt) for r in requests if r.prompt]
        return _wrap(base, rule=lambda text: any(k and k in normalize_text(text) for k in keys), name="prompt_filter")


class SubjectBlocklistIDK(Method):
    card = MethodCard("subject_blocklist_idk", "trivial",
                      "refuses whenever a request subject appears in the prompt (exact string)",
                      runtime_components=["subject_filter"], is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        subs = sorted({r.subject for r in requests if r.subject}, key=len, reverse=True)
        return _wrap(base, rule=lambda text: any(s in text for s in subs), name="subject_filter")


def relation_keywords(r: Request) -> List[str]:
    words = normalize_text(r.prompt.replace(r.subject, " ")).split()
    subj = set(normalize_text(r.subject).split())
    return [w for w in words if w not in _STOP and w not in subj and len(w) > 2]


class KeywordRouterIDK(Method):
    card = MethodCard("keyword_router_idk", "trivial",
                      "refuses when a request subject AND one of that request's relation words appear "
                      "(a string-matching detect-then-refuse router)",
                      runtime_components=["keyword_router"], is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        table = [(r.subject, set(relation_keywords(r)), r.request_id) for r in requests if r.subject]

        def rule(text):
            n = set(normalize_text(text).split())
            return any(s in text and (kw & n) for s, kw, _ in table)

        h = _wrap(base, rule=rule, name="keyword_router")

        def route(prompts: List[str]):
            out = []
            for t in prompts:
                n = set(normalize_text(t).split())
                hit = next((rid for s, kw, rid in table if s in t and (kw & n)), None)
                out.append(hit)
            return out
        h.route = route
        return h


class AnswerBan(Method):
    card = MethodCard("answer_ban", "trivial", "forbids the first token of every forgotten answer, everywhere",
                      runtime_components=["logit_bias"], is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        tok = base.tokenizer
        ban = set()
        for r in requests:
            if not r.answer:
                continue
            for text in (" " + r.answer, r.answer):
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if ids:
                    ban.add(ids[0])
        return _wrap(base, ban_ids=sorted(ban), name="logit_bias")
