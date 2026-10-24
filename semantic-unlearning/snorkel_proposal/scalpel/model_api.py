"""Model-facing primitives: prompt rendering, teacher-forced scoring, greedy generation,
logit-lens readout. Every metric in the benchmark is built from these four functions.

A submitted model is a ModelHandle. `model` is anything callable as
model(input_ids=..., attention_mask=...) -> object with `.logits` (an HF causal LM, or a
wrapper around one with hooks / filters). `hf` is the underlying HF model (weights), used
by white-box attacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .schema import Probe


# ------------------------------------------------------------------------- handle
@dataclass
class ModelHandle:
    model: Any
    tokenizer: Any
    hf: Any = None
    runtime_components: List[str] = field(default_factory=list)  # e.g. ["router_hooks"]
    strip: Optional[Callable[[], Any]] = None     # weights-only model (strict white-box tier)
    route: Optional[Callable[[List[str]], List[Optional[str]]]] = None  # detector introspection
    decode: str = "hf"     # "hf": model.generate | "full": prefix recompute | "cached_boundary": model.generate_cached
    # The model is told where the user prompt ends (the request boundary), as at inference
    # time. Used by detect-then-edit methods (e.g. SURE) that edit at the boundary only.
    needs_prompt_lengths: bool = False
    lens_module: Any = None      # module whose hidden states the lens reads (default: hf)
    # A wrapper that only edits next-token scores from the prefix can expose itself as an HF
    # LogitsProcessor; generation then runs through hf.generate (KV cache) with exactly the
    # same causal rule it applies under teacher forcing.
    gen_processor: Any = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.hf is None:
            self.hf = self.model

    @property
    def device(self):
        return next(self.hf.parameters()).device


# ---------------------------------------------------------------------- rendering
import os

CHAT_DATE = os.environ.get("SCALPEL_CHAT_DATE", "26 Jul 2024")
QA_SYSTEM = ("Answer with only the answer (a name, place, number or short phrase). "
             "Do not write a full sentence.")


class Renderer:
    """Turns a probe into the exact token ids the model sees.

    cloze: raw text (+ BOS). qa / chat: the tokenizer's chat template when the model has
    one (instruct models), else a plain "Question: ... Answer:" / "User: ... Assistant:"
    transcript. The answer is scored right after the rendered prompt; cloze answers get a
    leading space (ROME/MEMIT convention).
    """

    def __init__(self, tokenizer, use_chat_template: str = "auto"):
        self.tok = tokenizer
        tmpl = getattr(tokenizer, "chat_template", None)
        self.chat = (use_chat_template == "always") or (use_chat_template == "auto" and bool(tmpl))
        self._jinja = None

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        tok = self.tok
        with_system = [{"role": "system", "content": QA_SYSTEM}] + list(messages)
        tmpl = tok.chat_template or ""
        # templates with a reasoning switch (Qwen3): answer directly, no <think> block
        extra = {"enable_thinking": False} if "enable_thinking" in tmpl else {}
        # Llama-3 templates stamp today's date into the system header; pin it (as the repository's
        # RWKU evaluator does) so a prompt renders identically on every day a split is scored
        if "date_string" in tmpl or "strftime_now" in tmpl:
            extra["date_string"] = CHAT_DATE
        if hasattr(tok, "apply_chat_template"):
            for msgs in (with_system, None):
                if msgs is None:      # template without a system role: put the instruction in turn 1
                    msgs = [dict(m) for m in messages]
                    msgs[0]["content"] = QA_SYSTEM + "\n\n" + msgs[0]["content"]
                try:
                    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
                except Exception:  # noqa: BLE001
                    continue
        messages = with_system
        if self._jinja is None:
            import jinja2
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
            env.globals["strftime_now"] = lambda fmt: CHAT_DATE
            self._jinja = env.from_string(tok.chat_template)
        return self._jinja.render(messages=messages, add_generation_prompt=True,
                                  bos_token=tok.bos_token or "", eos_token=tok.eos_token or "", **extra)

    def messages(self, probe: Probe) -> List[Dict[str, str]]:
        if probe.format == "chat" and probe.messages:
            return list(probe.messages)
        return [{"role": "user", "content": probe.prompt}]

    def prompt_text(self, probe: Probe) -> Tuple[str, bool]:
        """(text, add_special_tokens)."""
        if probe.format == "cloze":
            return probe.prompt, True
        msgs = self.messages(probe)
        if self.chat:
            return self._apply_template(msgs), False
        lines = []
        for m in msgs:
            role = "Question" if (probe.format == "qa" and m["role"] == "user") else m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        lines.append("Answer:" if probe.format == "qa" else "Assistant:")
        return "\n".join(lines), True

    def answer_prefix(self, probe: Probe) -> str:
        return " " if (probe.format == "cloze" or not self.chat) else ""

    def encode_prompt(self, probe: Probe) -> List[int]:
        text, special = self.prompt_text(probe)
        return self.tok(text, add_special_tokens=special)["input_ids"]

    def encode_answer(self, probe: Probe, answer: str) -> List[int]:
        ids = self.tok(self.answer_prefix(probe) + answer, add_special_tokens=False)["input_ids"]
        return ids


BUCKET = 32   # pad widths to a multiple: backends that compile per shape (Apple MPS) reuse graphs


def _bucket(n: int, size: int = BUCKET) -> int:
    return ((n + size - 1) // size) * size


def _pad_right(seqs: Sequence[Sequence[int]], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    width = _bucket(max(len(s) for s in seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(list(s), dtype=torch.long)
        att[i, :len(s)] = 1
    return ids.to(device), att.to(device)


def _pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)


# Architectures whose logits are exactly lm_head(backbone(x).last_hidden_state) (no logit
# scaling or soft-capping), so the head can be applied to the scored positions only.
_FAST_HEAD = {"llama", "qwen2", "qwen3", "mistral", "gpt_neox", "olmo", "olmo2", "phi3"}


def _fast_head_ok(handle: "ModelHandle") -> bool:
    hf = handle.model
    if hf is not handle.hf or handle.runtime_components:
        return False
    mt = getattr(getattr(hf, "config", None), "model_type", "")
    return mt in _FAST_HEAD and hasattr(hf, getattr(hf, "base_model_prefix", "")) and hf.get_output_embeddings() is not None


def plain_selected_logits(hf, ids, att, rows: Sequence[Tuple[int, int, int]]):
    """lm_head on the selected (row, start, length) spans of a plain decoder."""
    if getattr(getattr(hf, "config", None), "model_type", "") in _FAST_HEAD and hasattr(hf, getattr(hf, "base_model_prefix", "")):
        backbone = getattr(hf, hf.base_model_prefix)
        h = backbone(input_ids=ids, attention_mask=att).last_hidden_state
        head = hf.get_output_embeddings()
        bi = torch.tensor([b for b, s, n in rows for _ in range(n)], device=h.device)
        pi = torch.tensor([s + k for b, s, n in rows for k in range(n)], device=h.device)
        sel = head(h[bi, pi])
    else:
        logits = hf(input_ids=ids, attention_mask=att).logits
        sel = torch.cat([logits[b, s:s + n] for b, s, n in rows])
    out, k = [], 0
    for b, s, n in rows:
        out.append(sel[k:k + n])
        k += n
    return out


def selected_logits(handle: "ModelHandle", ids, att, rows: Sequence[Tuple[int, int, int]], prompt_lengths=None):
    """Logits at selected (row, start, length) spans, as a list of [length, vocab] tensors.
    Plain decoders: the unembedding runs on those positions only (a 150k-vocab model would
    otherwise materialise B x T x V logits). Anything else: the full forward."""
    if hasattr(handle.model, "selected_logits"):          # wrappers that edit scores position-wise
        if handle.needs_prompt_lengths:
            return handle.model.selected_logits(ids, att, rows, prompt_lengths=prompt_lengths)
        return handle.model.selected_logits(ids, att, rows)
    if _fast_head_ok(handle):
        hf = handle.model
        backbone = getattr(hf, hf.base_model_prefix)
        h = backbone(input_ids=ids, attention_mask=att).last_hidden_state
        head = hf.get_output_embeddings()
        bi = [b for b, s, n in rows for _ in range(n)]
        pi = [s + k for b, s, n in rows for k in range(n)]
        total = len(bi)
        pad = _bucket(total, 64) - total
        bi, pi = bi + [0] * pad, pi + [0] * pad
        sel = head(h[torch.tensor(bi, device=h.device), torch.tensor(pi, device=h.device)])[:total]
        out, k = [], 0
        for b, s, n in rows:
            out.append(sel[k:k + n])
            k += n
        return out
    logits = call(handle, ids, att, prompt_lengths).logits
    return [logits[b, s:s + n] for b, s, n in rows]


def call(handle: "ModelHandle", ids, att, prompt_lengths=None, module=None, **kw):
    """Forward pass; passes the prompt boundary to methods that declare they need it."""
    m = module if module is not None else handle.model
    if handle.needs_prompt_lengths and prompt_lengths is not None:
        kw["prompt_lengths"] = torch.as_tensor(list(prompt_lengths), dtype=torch.long, device=ids.device)
    return m(input_ids=ids, attention_mask=att, **kw)


# ------------------------------------------------------------------ teacher forcing
@torch.no_grad()
def teacher_forced(handle: ModelHandle, renderer: Renderer, probes: Sequence[Probe],
                   batch_size: int = 16, topk: int = 0) -> List[Dict[str, Any]]:
    """Per probe, over the answer and its aliases:
        correct   every answer token is the argmax under teacher forcing (any alias)
        logprob   sum log p(answer tokens) of the canonical answer
        prob      exp(logprob)
        first_topk  (optional) top-k (token, logprob) at the first answer position, for KL
    One device->host transfer per batch (per-item syncs are slow on some backends)."""
    import math
    tok = renderer.tok
    items = []  # (probe_index, alias_index, prompt_ids, answer_ids)
    for i, p in enumerate(probes):
        pids = renderer.encode_prompt(p)
        for j, a in enumerate(p.answers()):
            aids = renderer.encode_answer(p, a)
            if aids:
                items.append((i, j, pids, aids))
    res = [{"correct": False, "logprob": float("-inf"), "prob": 0.0, "n_answer_tokens": 0} for _ in probes]
    device = handle.device
    order = sorted(range(len(items)), key=lambda k: len(items[k][2]) + len(items[k][3]))
    for s in range(0, len(order), batch_size):
        chunk = [items[k] for k in order[s:s + batch_size]]
        seqs = [pids + aids for (_, _, pids, aids) in chunk]
        ids, att = _pad_right(seqs, _pad_id(tok), device)
        spans = [(b, len(c[2]) - 1, len(c[3])) for b, c in enumerate(chunk)]
        picked = selected_logits(handle, ids, att, spans, [len(c[2]) for c in chunk])
        cat = torch.cat(picked).float()                                   # [sum answer tokens, V]
        tgt = torch.tensor([t for c in chunk for t in c[3]], device=cat.device)
        hit = (cat.argmax(-1) == tgt)
        lp = torch.log_softmax(cat, -1)
        tok_lp = lp.gather(-1, tgt[:, None])[:, 0]
        firsts = [sum(len(c[3]) for c in chunk[:b]) for b in range(len(chunk))]
        top = None
        if topk:
            v, ix = lp[torch.tensor(firsts, device=lp.device)].topk(topk, dim=-1)
            top = (ix.cpu().tolist(), v.cpu().tolist())
        hit, tok_lp = hit.cpu().tolist(), tok_lp.cpu().tolist()
        for b, (i, j, pids, aids) in enumerate(chunk):
            k0, n = firsts[b], len(aids)
            if all(hit[k0:k0 + n]):
                res[i]["correct"] = True
            if j == 0:
                total = float(sum(tok_lp[k0:k0 + n]))
                res[i]["logprob"] = total
                res[i]["prob"] = math.exp(total) if total > -700 else 0.0
                res[i]["n_answer_tokens"] = n
                if top is not None:
                    res[i]["first_topk"] = [[int(a), float(c)] for a, c in zip(top[0][b], top[1][b])]
    return res


@torch.no_grad()
def first_token_logprobs(handle: ModelHandle, renderer: Renderer, probes: Sequence[Probe],
                         token_ids: Sequence[Sequence[int]], batch_size: int = 16) -> List[List[float]]:
    """log p(token) at the first answer position for given token ids per probe (KL, MMLU)."""
    out_rows: List[List[float]] = []
    tok = renderer.tok
    device = handle.device
    for s in range(0, len(probes), batch_size):
        chunk = probes[s:s + batch_size]
        seqs = [renderer.encode_prompt(p) for p in chunk]
        ids, att = _pad_right(seqs, _pad_id(tok), device)
        picked = selected_logits(handle, ids, att, [(b, len(x) - 1, 1) for b, x in enumerate(seqs)],
                                 [len(x) for x in seqs])
        lp = torch.log_softmax(torch.cat(picked).float(), -1)               # [B, V]
        want = [list(token_ids[s + b]) for b in range(len(chunk))]
        width = max(1, max(len(w) for w in want))
        idx = torch.tensor([w + [0] * (width - len(w)) for w in want], device=lp.device)
        vals = lp.gather(-1, idx).cpu().tolist()
        out_rows += [v[:len(w)] for v, w in zip(vals, want)]
    return out_rows


# ----------------------------------------------------------------------- generation
@torch.no_grad()
def generate(handle: ModelHandle, renderer: Renderer, probes: Sequence[Probe], max_new_tokens: int = 24,
             batch_size: int = 16) -> List[str]:
    """Greedy continuation of each rendered prompt (the open-ended track)."""
    tok = renderer.tok
    device = handle.device
    pad = _pad_id(tok)
    eos = tok.eos_token_id
    texts: List[str] = [""] * len(probes)
    prompts = [renderer.encode_prompt(p) for p in probes]
    order = sorted(range(len(probes)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [prompts[i] for i in idx]
        width = _bucket(max(len(x) for x in seqs))
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for b, x in enumerate(seqs):                      # left padding
            ids[b, width - len(x):] = torch.tensor(x)
            att[b, width - len(x):] = 1
        ids, att = ids.to(device), att.to(device)
        if handle.decode == "cached_boundary" and hasattr(handle.model, "generate_cached"):
            new = handle.model.generate_cached(ids, att, max_new_tokens, eos)
        elif handle.decode == "hf" and not handle.needs_prompt_lengths:
            new = _hf_generate(handle, ids, att, max_new_tokens, pad, eos)[:, width:]
        else:
            new = _greedy_full(handle, ids, att, max_new_tokens, eos)
        for b, i in enumerate(idx):
            row = new[b].tolist()
            if eos is not None and eos in row:
                row = row[:row.index(eos)]
            texts[i] = tok.decode(row, skip_special_tokens=True)
    return texts


def _hf_generate(handle, ids, att, max_new_tokens, pad, eos):
    from transformers import LogitsProcessorList
    kw = dict(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=False,
              pad_token_id=pad, eos_token_id=eos)
    if handle.gen_processor is not None:
        kw["logits_processor"] = LogitsProcessorList([handle.gen_processor])
    if ids.device.type == "mps":
        # Apple MPS compiles a graph per new shape; a static KV cache keeps shapes fixed
        try:
            return handle.hf.generate(cache_implementation="static", **kw)
        except Exception:  # noqa: BLE001 - architectures without static-cache support
            pass
    return handle.hf.generate(**kw)


def decoder_key(handle: "ModelHandle") -> str:
    """Which greedy decoder a handle uses; the base reference is generated with the same one.
    "cached_boundary" (KV-cached, boundary-aware) is the HF-style cached decoder for prompts
    that do not route, so it compares against the "hf" base generations."""
    if handle.decode == "cached_boundary":
        return "hf"
    return "full" if (handle.decode != "hf" or handle.needs_prompt_lengths) else "hf"


def _greedy_full(handle, ids, att, max_new_tokens, eos):
    """Greedy decoding without a KV cache: every step re-reads the whole prefix, so a wrapper
    that decides from the prefix text (filters, routers) sees exactly what it would see in a
    teacher-forced pass. Sequences are re-packed right-padded so positions stay exact."""
    device = ids.device
    rows = [ids[b][att[b].bool()].tolist() for b in range(ids.shape[0])]
    lengths = torch.tensor([len(r) for r in rows], device=device)
    width = _bucket(int(lengths.max()) + max_new_tokens)
    buf = torch.full((len(rows), width), int(ids[0, 0]), dtype=torch.long, device=device)
    mask = torch.zeros_like(buf)
    for b, r in enumerate(rows):
        buf[b, :len(r)] = torch.tensor(r, device=device)
        mask[b, :len(r)] = 1
    done = torch.zeros(len(rows), dtype=torch.bool, device=device)
    new = []
    ar = torch.arange(len(rows), device=device)
    prompt_lengths = lengths.tolist()          # the request boundary never moves
    for _ in range(max_new_tokens):
        cur = min(width, _bucket(int(lengths.max())))   # masked tail positions never affect earlier ones
        last = selected_logits(handle, buf[:, :cur], mask[:, :cur], [(b, int(lengths[b]) - 1, 1) for b in range(len(rows))],
                               prompt_lengths)
        nxt = torch.cat(last).float().argmax(-1)
        if eos is not None:
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            done |= nxt == eos
        new.append(nxt)
        buf[ar, lengths] = nxt
        mask[ar, lengths] = 1
        lengths = lengths + 1
        if bool(done.all()):
            break
    return torch.stack(new, 1)


# ------------------------------------------------------------------------ logit lens
def lens_modules(hf) -> Tuple[Any, Any]:
    """(final norm, unembedding) of common decoder families."""
    head = hf.get_output_embeddings()
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.final_layernorm",
                 "model.decoder.final_layer_norm"):
        obj = hf
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, head
        except AttributeError:
            continue
    raise ValueError("cannot find the final norm for the logit lens on this architecture")


@torch.no_grad()
def lens_ranks(handle: ModelHandle, renderer: Renderer, probes: Sequence[Probe],
               candidates: Sequence[Sequence[int]], batch_size: int = 8) -> List[List[int]]:
    """For each probe and layer: rank (0 = top) of the true answer's first token among its
    candidate first tokens (index 0 = true answer), reading the last prompt position through
    the final norm + unembedding. Uses the UNDERLYING weights (handle.hf) with every hook the
    method installed, i.e. the residual stream the shipped model actually computes."""
    norm, head = lens_modules(handle.hf)
    tok = renderer.tok
    device = handle.device
    out: List[List[int]] = []
    for s in range(0, len(probes), batch_size):
        chunk = probes[s:s + batch_size]
        seqs = [renderer.encode_prompt(p) for p in chunk]
        ids, att = _pad_right(seqs, _pad_id(tok), device)
        module = handle.lens_module or handle.hf
        if handle.lens_module is None and _fast_head_ok(handle):
            module = getattr(handle.hf, handle.hf.base_model_prefix)    # hidden states without full logits
        res = call(handle, ids, att, [len(x) for x in seqs], module=module, output_hidden_states=True)
        hs = res.hidden_states  # (embeddings, layer 1, ..., layer L)
        ar = torch.arange(len(chunk), device=device)
        last = torch.tensor([len(x) - 1 for x in seqs], device=device)
        H = torch.stack([hs[l][ar, last] for l in range(1, len(hs))])          # [L, B, d]
        H = H.to(head.weight.dtype)
        # HF appends the last layer AFTER the final norm; the others need it
        normed = torch.cat([norm(H[:-1]), H[-1:]], 0)
        logits = head(normed).float()                                        # [L, B, V]
        width = max(len(candidates[s + b]) for b in range(len(chunk)))
        cidx = torch.tensor([list(candidates[s + b]) + [candidates[s + b][0]] * (width - len(candidates[s + b]))
                             for b in range(len(chunk))], device=device)       # [B, C]
        sc = logits.gather(-1, cidx[None].expand(logits.shape[0], -1, -1))   # [L, B, C]
        ranks = (sc > sc[..., :1]).sum(-1).cpu().T.tolist()                 # [B, L]
        out += [[int(x) for x in r] for r in ranks]
    return out


# --------------------------------------------------------------------- perplexity
@torch.no_grad()
def perplexity(handle: ModelHandle, tokenizer, texts: Sequence[str], batch_size: int = 8) -> float:
    device = handle.device
    nll, count = 0.0, 0
    for s in range(0, len(texts), batch_size):
        seqs = [tokenizer(t)["input_ids"] for t in texts[s:s + batch_size]]
        ids, att = _pad_right(seqs, _pad_id(tokenizer), device)
        logits = call(handle, ids, att, [len(x) for x in seqs]).logits[:, :-1]
        tgt = ids[:, 1:]
        m = att[:, 1:].bool()
        lp = (logits.gather(-1, tgt[..., None])[..., 0].float() - torch.logsumexp(logits.float(), -1))
        nll -= float(lp[m].sum())
        count += int(m.sum())
    return float(torch.exp(torch.tensor(nll / max(1, count))))


# ------------------------------------------------------------------------ causality
@torch.no_grad()
def causality_violation(handle: ModelHandle, renderer: Renderer, probes: Sequence[Probe], n: int = 16) -> float:
    """Max |Δ logit| at answer positions when the tokens AFTER each position are changed.

    A causal LM gives 0. A wrapper that peeks at the teacher-forced answer (e.g. lowers the
    answer's score when it sees the answer later in the input) is a grader exploit; the
    runner refuses to score a model whose violation exceeds a small tolerance."""
    tok = renderer.tok
    device = handle.device
    worst = 0.0
    for p in list(probes)[:n]:
        pids = renderer.encode_prompt(p)
        aids = renderer.encode_answer(p, p.answer)
        if not aids:
            continue
        seq = pids + aids
        filler = (renderer.encode_answer(p, "the") or [aids[0]])[0]
        ids, att = _pad_right([seq], _pad_id(tok), device)
        ref = call(handle, ids, att, [len(pids)]).logits.float()[0]
        for t in range(len(pids) - 1, len(seq) - 1):
            alt = seq[:t + 1] + [filler] * (len(seq) - t - 1)
            if alt == seq:
                continue
            ids2, att2 = _pad_right([alt], _pad_id(tok), device)
            l2 = call(handle, ids2, att2, [len(pids)]).logits.float()[0, t]
            worst = max(worst, float((ref[t] - l2).abs().max()))
    return worst
