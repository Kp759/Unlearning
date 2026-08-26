#!/usr/bin/env python3
"""Can an LM-head row edit forget one fact without breaking its neighbors?

MCF's Eff and Spe read the SAME vocabulary row in opposite directions. For
"The native language of Gautier de Coincy is" -> French, Eff wants the French
row suppressed; the record's neighborhood prompts ("Henri Barbusse is a native
speaker of") are other French speakers, and Spe scores them as
NLL(target_true) < NLL(target_new), so Spe wants that row intact.

Feasibility is NOT the question. A delta d changes a logit by d.h, so any d in
the nullspace of the protected states leaves them EXACTLY unchanged at any
magnitude. The question is generalization: whether a delta fitted to spare the
protected states we can see also spares the neighborhood prompts we cannot
(they are evaluation-only).

So the measurement compares two residuals against the SAME protection basis:

    forget_residual    energy of a forget prompt outside the protected span
    heldout_residual   same, for protected states held out of that span

    forget >> heldout  -> the forget context is atypical; a protected-subspace
                          delta can suppress it while sparing the rest
    forget ~= heldout  -> it is indistinguishable from legitimate uses of the
                          same answer, at this protection size

That last qualifier matters. With a rank-10 basis in 3072 dimensions nearly
everything looks orthogonal, so a single small measurement cannot tell
"inseparable" from "protection set too small". --sweep-sizes therefore
reports the pair across protection sizes: if heldout falls while forget stays
put, the separation is real and worth spending protected-rank on.

Protection is pooled BY ANSWER TOKEN, not per record. One French row serves
every French fact, so the states it must keep serving are every context where
French is correct -- pooled across all records sharing that answer.

Three sources are reported:

  oracle   protect and test on real neighborhood prompts. Evaluation data,
           diagnostic only; nothing is fitted or selected from it. Upper bound
           on what any protection set could achieve.
  corpus   protect and test on naturally occurring Wikipedia positions where
           the answer is already the correct next token. No evaluation data,
           so this basis is deployable in training.
  cross    protect on corpus, test on real neighborhood prompts. THE decision
           number: whether a legitimately-built protection set actually spares
           the prompts Spe scores.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import gagd_compare as gagd
from mcf_sampling import sample_official_mcf_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument(
        "--sweep-sizes",
        default="8,32,128,512,2048",
        help=(
            "Protection-basis sizes to evaluate. The point of the sweep: a "
            "single size cannot separate 'inseparable' from 'too few "
            "protection vectors'."
        ),
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--corpus-docs",
        type=int,
        default=50000,
        help=(
            "Upper bound on documents scanned for contexts where an answer is "
            "already the correct next token. Scanning stops once every answer "
            "reaches its cap, so common answers cost little. 0 to skip."
        ),
    )
    p.add_argument(
        "--corpus-doc-start",
        type=int,
        default=20,
        help="Must stay >= 20: official PPL is scored on documents [:20].",
    )
    p.add_argument("--corpus-max-tokens", type=int, default=256)
    p.add_argument(
        "--corpus-contexts-per-answer",
        type=int,
        default=4096,
        help="Cap on mined contexts per answer. Must exceed the largest swept size.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if a.forget_num <= 0 or a.batch_size <= 0:
        p.error("forget-num and batch-size must be positive")
    if a.corpus_doc_start < 20:
        p.error("corpus-doc-start must be >= 20 to stay disjoint from official PPL text")
    a.sweep_sizes = sorted({int(x) for x in str(a.sweep_sizes).split(",") if x.strip()})
    if not a.sweep_sizes or a.sweep_sizes[0] < 2:
        p.error("sweep-sizes must contain integers >= 2")
    return a


def orthonormal_basis(states: torch.Tensor) -> torch.Tensor:
    """Rank-revealing orthonormal basis for the row space of ``states``."""
    if states.numel() == 0:
        return torch.empty((0, 0))
    m = states.to(torch.float32)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    tol = s.max() * max(m.shape) * torch.finfo(s.dtype).eps
    return vh[: int((s > tol).sum().item())]


def residual_fraction(vector: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ``vector``'s energy outside an orthonormal ``basis``.

    1.0 = orthogonal to the protected span, so a delta along it acts on this
    vector while leaving the protected states exactly unchanged.
    0.0 = fully inside it, so any delta sparing the protected states also
    leaves this vector unchanged.
    """
    if basis.numel() == 0 or vector.numel() == 0:
        return float("nan")
    v = vector.to(torch.float32)
    norm = torch.linalg.vector_norm(v)
    if float(norm) == 0.0:
        return float("nan")
    projected = basis.T @ (basis @ v)
    return float((torch.linalg.vector_norm(v - projected) / norm) ** 2)


def measure(
    forget_states: torch.Tensor,
    protect_states: torch.Tensor,
    test_states: torch.Tensor,
    sizes: Sequence[int],
    generator: torch.Generator,
) -> List[Dict[str, Any]]:
    """Residual pair across protection sizes.

    ``test_states`` are held out of the basis so they measure generalization:
    how much of a state the protection FAILS to cover when that state was not
    itself used to build it. The forget prompt is separable only if it sits
    further outside than those do.
    """
    rows: List[Dict[str, Any]] = []
    available = int(protect_states.shape[0]) if protect_states.numel() else 0
    for size in sizes:
        if available < size + 2 or forget_states.numel() == 0:
            continue
        perm = torch.randperm(available, generator=generator)
        basis = orthonormal_basis(protect_states[perm[:size]])
        if basis.numel() == 0:
            continue
        held = protect_states[perm[size:]] if test_states is None else test_states
        if held.numel() == 0:
            continue
        forget = [residual_fraction(v, basis) for v in forget_states]
        heldout = [residual_fraction(v, basis) for v in held]
        forget = [x for x in forget if x == x]
        heldout = [x for x in heldout if x == x]
        if not forget or not heldout:
            continue
        f_mean = sum(forget) / len(forget)
        h_mean = sum(heldout) / len(heldout)
        rows.append(
            {
                "protection_size": int(size),
                "basis_rank": int(basis.shape[0]),
                "forget_residual": f_mean,
                "heldout_residual": h_mean,
                "separability_gap": f_mean - h_mean,
                "forget_n": len(forget),
                "heldout_n": len(heldout),
            }
        )
    return rows


@torch.no_grad()
def last_token_hidden(model, tok, prompts, device, batch_size: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = [str(x) for x in prompts[start : start + batch_size]]
            enc = tok(chunk, padding=True, return_tensors="pt").to(device)
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
            for row, length in enumerate(enc["attention_mask"].sum(dim=1).tolist()):
                rows.append(hidden[row, int(length) - 1, :].detach().float().cpu())
    finally:
        tok.padding_side = old_side
    return torch.stack(rows) if rows else torch.empty((0, 0))


@torch.no_grad()
def corpus_answer_contexts(
    model, tok, documents, answer_first_token, device,
    max_tokens: int, per_answer_cap: int, batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Hidden states in real text where each answer is already the next token.

    These are precisely the contexts the row must keep serving: if the delta
    is orthogonal to them, suppressing the row cannot stop the model saying
    "French" wherever French was already right. No evaluation data is read.
    """
    wanted = {int(v): k for k, v in answer_first_token.items()}
    vocab = int(model.get_input_embeddings().weight.shape[0])
    wanted_lookup = torch.zeros(vocab, dtype=torch.bool)
    for token_id in wanted:
        wanted_lookup[int(token_id)] = True
    collected: Dict[str, List[torch.Tensor]] = defaultdict(list)
    remaining = {k: int(per_answer_cap) for k in answer_first_token}
    scanned = 0
    for start in range(0, len(documents), batch_size):
        if not any(v > 0 for v in remaining.values()):
            break
        chunk = [str(d) for d in documents[start : start + batch_size]]
        enc = tok(
            chunk, padding=True, truncation=True,
            max_length=int(max_tokens), return_tensors="pt",
        ).to(device)
        hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
        ids, mask = enc["input_ids"], enc["attention_mask"]
        scanned += len(chunk)
        # Vectorized: the per-position form calls .item() on a CUDA tensor for
        # every token, one host-device sync each, which dominates the run at
        # 50k documents. One masked comparison finds the hits; only those are
        # walked to apply caps, and they are gathered in a single transfer.
        next_ids = ids[:, 1:].cpu()
        next_valid = mask[:, 1:].bool().cpu()
        hits = next_valid & wanted_lookup[next_ids]
        hit_rows, hit_cols = hits.nonzero(as_tuple=True)
        if hit_rows.numel():
            hit_tokens = next_ids[hit_rows, hit_cols].tolist()
            keep_rows, keep_cols, keep_answers = [], [], []
            for r, c, t in zip(hit_rows.tolist(), hit_cols.tolist(), hit_tokens):
                answer = wanted.get(t)
                if answer is None or remaining[answer] <= 0:
                    continue
                remaining[answer] -= 1
                keep_rows.append(r)
                keep_cols.append(c)
                keep_answers.append(answer)
            if keep_rows:
                gathered = hidden[
                    torch.tensor(keep_rows, device=hidden.device),
                    torch.tensor(keep_cols, device=hidden.device),
                    :,
                ].detach().float().cpu()
                for slot, answer in enumerate(keep_answers):
                    collected[answer].append(gathered[slot])
        if scanned % 2000 < batch_size:
            filled = sum(1 for v in remaining.values() if v <= 0)
            print(f"  scanned {scanned} docs; {filled}/{len(remaining)} answers at cap")
    return {k: torch.stack(v) for k, v in collected.items() if v}


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    generator = torch.Generator().manual_seed(int(a.seed))

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, _ = sample_official_mcf_records(
        data, int(a.forget_num), int(a.retain_num), int(a.seed)
    )

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype,
        device_map=a.device_map, gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    device = gagd.first_device(model)

    # Pool by answer token: one row serves every fact sharing that answer, so
    # the protected population is the union across those records.
    by_answer: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"forget": [], "neighborhood": []}
    )
    for record in forget:
        rewrite = record["requested_rewrite"]
        if isinstance(rewrite, list):
            rewrite = rewrite[0]
        answer = str(rewrite["target_true"]["str"])
        by_answer[answer]["forget"].append(
            str(rewrite["prompt"]).format(str(rewrite["subject"]))
        )
        by_answer[answer]["neighborhood"].extend(
            str(x) for x in record.get("neighborhood_prompts", [])
        )
    print(
        f"{len(forget)} records -> {len(by_answer)} distinct answers "
        f"(one shared row each)"
    )

    corpus_states: Dict[str, torch.Tensor] = {}
    if int(a.corpus_docs) > 0:
        documents = gagd.load_wiki_frequency_documents(
            a.wikidata_dir, int(a.corpus_doc_start), int(a.corpus_docs)
        )
        if not documents:
            print(f"WARNING: no documents in {a.wikidata_dir!r}; skipping corpus source.")
        else:
            first_token = {}
            for answer in by_answer:
                ids = gagd.token_ids_for_text(tok, gagd.normalize_answer(answer))
                if ids:
                    first_token[answer] = ids[0]
            print(f"Scanning up to {len(documents)} documents for {len(first_token)} answers...")
            corpus_states = corpus_answer_contexts(
                model, tok, documents, first_token, device,
                int(a.corpus_max_tokens), int(a.corpus_contexts_per_answer),
                int(a.batch_size),
            )

    per_answer: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[int, List[float]]] = {
        s: defaultdict(list) for s in ("oracle", "corpus", "cross")
    }
    aggregate_forget: Dict[str, Dict[int, List[float]]] = {
        s: defaultdict(list) for s in ("oracle", "corpus", "cross")
    }

    for answer, prompts in by_answer.items():
        forget_states = last_token_hidden(
            model, tok, prompts["forget"], device, int(a.batch_size)
        )
        neigh_states = (
            last_token_hidden(model, tok, prompts["neighborhood"], device, int(a.batch_size))
            if prompts["neighborhood"] else torch.empty((0, 0))
        )
        corpus = corpus_states.get(answer, torch.empty((0, 0)))

        entry: Dict[str, Any] = {
            "answer": answer,
            "forget_prompts": len(prompts["forget"]),
            "neighborhood_prompts": int(neigh_states.shape[0]) if neigh_states.numel() else 0,
            "corpus_contexts": int(corpus.shape[0]) if corpus.numel() else 0,
        }
        sources = {
            "oracle": (neigh_states, None),
            "corpus": (corpus, None),
            "cross": (corpus, neigh_states if neigh_states.numel() else None),
        }
        for name, (protect, test) in sources.items():
            if protect.numel() == 0 or (name == "cross" and test is None):
                continue
            rows = measure(forget_states, protect, test, a.sweep_sizes, generator)
            if rows:
                entry[name] = rows
                for row in rows:
                    aggregate[name][row["protection_size"]].append(row["heldout_residual"])
                    aggregate_forget[name][row["protection_size"]].append(row["forget_residual"])
        per_answer.append(entry)

    summary: Dict[str, Any] = {
        "schema_version": 2,
        "protocol": "mcf_lm_head_row_separability_v2_answer_pooled_sweep",
        "model_path": str(Path(a.model_path).resolve()),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "distinct_answers": len(by_answer),
        "sweep_sizes": a.sweep_sizes,
        "sources": {
            "oracle": "protect and test on real neighborhood prompts (eval data, diagnostic only)",
            "corpus": "protect and test on mined Wikipedia contexts (deployable)",
            "cross": "protect on corpus, test on real neighborhood prompts (decision number)",
        },
        "interpretation": (
            "Compare forget_residual against heldout_residual at each "
            "protection size. If heldout falls with size while forget stays "
            "put, the forget context is genuinely atypical and a "
            "protected-subspace delta can suppress it while sparing "
            "neighbors -- raise --protected-rank against this basis. If both "
            "fall together, the forget prompt is indistinguishable from "
            "legitimate uses of the same answer and no row-level edit "
            "separates Eff from Spe."
        ),
        "per_answer": per_answer,
    }

    for name in ("oracle", "corpus", "cross"):
        curve = []
        for size in a.sweep_sizes:
            f_vals = aggregate_forget[name].get(size, [])
            h_vals = aggregate[name].get(size, [])
            if not f_vals or not h_vals:
                continue
            f_mean = sum(f_vals) / len(f_vals)
            h_mean = sum(h_vals) / len(h_vals)
            curve.append(
                {
                    "protection_size": size,
                    "forget_residual": f_mean,
                    "heldout_residual": h_mean,
                    "separability_gap": f_mean - h_mean,
                    "answers": len(f_vals),
                }
            )
        if curve:
            summary[f"{name}_curve"] = curve

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nSeparability report: {out}")
    for name in ("oracle", "corpus", "cross"):
        curve = summary.get(f"{name}_curve")
        if not curve:
            continue
        label = {
            "oracle": "protect+test on neighborhoods (eval data, diagnostic)",
            "corpus": "protect+test on mined contexts (deployable)",
            "cross": "protect on corpus, TEST ON NEIGHBORHOODS (decision number)",
        }[name]
        print(f"\n[{name}]  {label}")
        print(f"  {'size':>6} {'rank':>6} {'forget':>9} {'heldout':>9} {'gap':>9}")
        for row in curve:
            print(
                f"  {row['protection_size']:>6} {'':>6} "
                f"{row['forget_residual']:>9.4f} {row['heldout_residual']:>9.4f} "
                f"{row['separability_gap']:>+9.4f}"
            )
        first, last = curve[0], curve[-1]
        print(
            f"  gap {first['separability_gap']:+.4f} @ size {first['protection_size']} "
            f"-> {last['separability_gap']:+.4f} @ size {last['protection_size']}"
        )


if __name__ == "__main__":
    main()
