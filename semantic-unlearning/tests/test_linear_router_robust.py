"""Opt-in robustness options of the linear router (SCALPEL-driven).

The claims these protect:
  * defaults are the shipped router: exact subject tokens, no subject-free firing;
  * fuzzy matching accepts case / accent / spacing changes and one edit per name word,
    and rejects a different name that only shares a token;
  * the bank's fuzzy eligibility equals the fit-time eligibility_matrix on the same prompt;
  * free-fire only acts when no subject is eligible, and only above its logit;
  * passthrough leaves the hidden state untouched;
  * the options survive an artifact round trip; old artifacts load as exact.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from linear_router import (  # noqa: E402
    LinearClassifierAssociationBank,
    eligibility_matrix,
    fuzzy_subject_in_text,
    load_linear_classifier_artifact,
    subject_words,
    within_one_edit,
)

HIDDEN, WIDTH = 8, 6


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


class _Tok:
    """Whitespace tokenizer with BOS=1 and a decode that skips it."""

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 10
            self.inv[self.vocab[w]] = w
        return self.vocab[w]

    def __call__(self, text, add_special_tokens=True, **_):
        ids = [self._id(w) for w in str(text).split()]
        return {"input_ids": ([1] + ids) if add_special_tokens else ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[i] for i in ids if i in self.inv)


SUBJECTS = ["Danielle Darrieux", "Katrin Coldbrook", "Li Wei"]


def test_edit_distance_and_word_normalisation():
    assert within_one_edit("darrieux", "darreiux")      # transposition
    assert within_one_edit("darrieux", "darrieu")       # deletion
    assert within_one_edit("darrieux", "darriepux")     # insertion
    assert within_one_edit("darrieux", "darrieax")      # substitution
    assert not within_one_edit("darrieux", "dorreiux")
    assert subject_words("  DANIÈLLE   Darrieux's ") == ["danielle", "darrieux", "s"]


def test_fuzzy_subject_matches_variants_but_not_other_names():
    w = subject_words("Danielle Darrieux")
    for text in ("The mother tongue of danielle darrieux is", "DANIELLE DARRIEUX spoke",
                 "Danielle  Darreiux was", "Daniëlle Darrieux", "about DanielleDarrieux today"):
        assert fuzzy_subject_in_text(text, w), text
    assert not fuzzy_subject_in_text("Isabelle Darrieux was born", w)
    assert not fuzzy_subject_in_text("Niklas Coldbrook", subject_words("Katrin Coldbrook"))
    short = subject_words("Li Wei")                     # short words must match exactly
    assert fuzzy_subject_in_text("li wei", short) and not fuzzy_subject_in_text("Lu Wei", short)


def _bank(tok, subject_match="exact", free_fire_logit=None, bias=0.0):
    facts = [{"id": f"f{i}", "subject": s, "relation": "r"} for i, s in enumerate(SUBJECTS)]
    patterns = [[tuple(tok(s, add_special_tokens=False)["input_ids"])] for s in SUBJECTS]
    torch.manual_seed(0)
    bank = LinearClassifierAssociationBank(
        _FakeModel(), 0, torch.randn(len(facts), HIDDEN), torch.full((len(facts),), float(bias)),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=patterns, facts=facts, rows=torch.randn(len(facts), HIDDEN),
        ambiguity_margin=0.0, subject_match=subject_match, free_fire_logit=free_fire_logit,
    )
    bank.tokenizer = tok
    return bank


def _prompts():
    return ["Danielle Darrieux grew up speaking", "Danielle Darreiux grew up speaking",
            "KATRIN COLDBROOK works as", "The capital of France is a city"]


def _eligible_at_runtime(bank, tok, prompts):
    rows = [tok(p)["input_ids"] for p in prompts]
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    att = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    bank.bind(ids, attention_mask=att)
    try:
        return bank._eligibility(bank._prefix_lengths_for(torch.zeros(len(rows), width, HIDDEN)))
    finally:
        bank.unbind()


def test_exact_is_the_default_and_fuzzy_adds_variants():
    tok = _Tok()
    exact = _eligible_at_runtime(_bank(tok), tok, _prompts())
    fuzzy = _eligible_at_runtime(_bank(tok, "fuzzy"), tok, _prompts())
    assert exact[:, 0].tolist() == [True, False, False, False]
    assert fuzzy[:, 0].tolist() == [True, True, False, False]
    assert fuzzy[:, 1].tolist() == [False, False, True, False]
    assert not bool(fuzzy[3].any())


def test_runtime_fuzzy_mask_equals_fit_time_matrix():
    tok = _Tok()
    bank = _bank(tok, "fuzzy")
    runtime = _eligible_at_runtime(bank, tok, _prompts())
    offline = eligibility_matrix(tok, _prompts(), bank.subject_patterns, subjects=SUBJECTS,
                                 subject_match="fuzzy")
    assert torch.equal(runtime.cpu(), offline)


def _run(bank, tok, prompts, hidden):
    rows = [tok(p)["input_ids"] for p in prompts]
    width = hidden.shape[1]
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    att = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    bank.bind(ids, attention_mask=att)
    try:
        return bank._hook(None, None, (hidden,))
    finally:
        bank.unbind()


def test_free_fire_only_without_an_eligible_subject():
    tok = _Tok()
    prompts = ["The capital of France is a city", "Danielle Darrieux grew up speaking"]
    hidden = torch.randn(2, 10, HIDDEN)
    off = _bank(tok, bias=100.0)
    out = _run(off, tok, prompts, hidden)[0]
    assert torch.equal(out[0], hidden[0])                      # no subject, no free fire
    on = _bank(tok, free_fire_logit=50.0, bias=100.0)
    out = _run(on, tok, prompts, hidden)[0]
    assert not torch.equal(out[0], hidden[0])                  # fires without a subject
    high = _bank(tok, free_fire_logit=1e6, bias=100.0)
    out = _run(high, tok, prompts, hidden)[0]
    assert torch.equal(out[0], hidden[0])                      # below the free-fire logit


def test_passthrough_and_round_trip():
    tok = _Tok()
    bank = _bank(tok, "fuzzy", free_fire_logit=3.0, bias=100.0)
    bank.passthrough = True
    assert bank._hook(None, None, (torch.randn(1, 4, HIDDEN),)) is None
    art = bank.artifact()
    assert art["subject_match"] == "fuzzy" and art["free_fire_logit"] == 3.0
    _, restored = load_linear_classifier_artifact(_FakeModel(), art)
    assert restored.subject_match == "fuzzy" and restored.free_fire_logit == 3.0
    old = {k: v for k, v in art.items() if k not in ("subject_match", "free_fire_logit")}
    _, legacy = load_linear_classifier_artifact(_FakeModel(), old)
    assert legacy.subject_match == "exact" and legacy.free_fire_logit is None
