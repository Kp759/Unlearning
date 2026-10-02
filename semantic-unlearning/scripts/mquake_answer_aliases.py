"""Answer aliases for MQuAKE atomic facts (forget-side data only).

MQuAKE-CF-3k-v2 ships, for every hop of every chain, the hop's cloze, its
answer and the answer's Wikidata aliases (`single_hops`, `new_single_hops`).
An atomic forget fact (requested_rewrite: subject, prompt, target_true) is
looked up by (cloze, answer) = (prompt.format(subject), target_true.str); about
two thirds of the forget rewrites have aliases ("United States of America" ->
"U.S.", "America", ...).

Used for two things:
  * evaluate_mquake_alias_leak.py: is a forgotten answer still recoverable
    through an alias? (the official metrics score the exact answer only)
  * train_direct_linear_router_rows.py --alias-targets: add the first token of
    each alias as an extra row-training target, so the hinge takes the worst
    case over the answer and its aliases.

Only the forget fact's own answer and aliases are read; no retain fact.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

# First tokens that are function words or fragments: suppressing them at a
# prompt would hit unrelated continuations, so such aliases are not used as
# training targets (they are still reported by the evaluator).
GENERIC_FIRST_TOKENS = frozenset({
    "the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "de", "la",
    "le", "el", "les", "los", "del", "der", "die", "das", "von", "van", "st",
    "saint", "mr", "mrs", "dr", "sir", "king", "queen", "prince", "lord",
})


def load_raw(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("MQuAKE JSON must be a list")
    return raw


def alias_table(raw):
    """(cloze, answer) -> sorted aliases, pooled over all hops of all chains."""
    table = defaultdict(set)
    for record in raw:
        for key in ("single_hops", "new_single_hops"):
            for hop in record.get(key, []) or []:
                cloze, answer = str(hop.get("cloze", "")).strip(), str(hop.get("answer", ""))
                if cloze and answer:
                    table[(cloze, answer)].update(str(a) for a in hop.get("answer_alias", []) or [])
    return {k: sorted(v) for k, v in table.items()}


def record_aliases(record, table):
    """Aliases of an atomic record's original answer (answer itself removed)."""
    rr = record["requested_rewrite"]
    cloze = str(rr["prompt"]).format(str(rr["subject"])).strip()
    answer = str(rr["target_true"]["str"])
    seen, out = {answer.strip().casefold()}, []
    for alias in table.get((cloze, answer), []):
        key = alias.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(alias.strip())
    return out


def is_generic_token(text):
    stripped = str(text).strip()
    word = stripped.casefold().rstrip(".")
    return len(word) < 2 or not any(ch.isalnum() for ch in word) or word in GENERIC_FIRST_TOKENS


def classify_aliases(tokenizer, answer, aliases, *, llama_like):
    """[(alias, token_ids, kind)], kind: alias_same_first | alias_diff_first.

    token_ids follow the official MQuAKE target convention (" " + text, BOS
    dropped for Llama-style tokenizers). Aliases with no tokens are skipped.
    """
    from mquake_zero_unlearn_official_eval import original_answer_token_ids

    answer_first = original_answer_token_ids(tokenizer, answer, llama_like=llama_like)[0]
    out = []
    for alias in aliases:
        try:
            ids = original_answer_token_ids(tokenizer, alias, llama_like=llama_like)
        except ValueError:
            continue
        kind = "alias_same_first" if ids[0] == answer_first else "alias_diff_first"
        out.append((alias, ids, kind))
    return out


def alias_training_targets(tokenizer, answer, aliases, *, llama_like):
    """First tokens to add as row-training targets: aliases whose first token
    differs from the answer's, is not generic, and round-trips through the
    official target encoding. Returns [(alias, token_id)], one per distinct token."""
    import torch
    from mquake_zero_unlearn_official_eval import official_target_ids

    out, used = [], set()
    for alias, ids, kind in classify_aliases(tokenizer, answer, aliases, llama_like=llama_like):
        token = int(ids[0])
        if kind != "alias_diff_first" or token in used:
            continue
        text = tokenizer.decode([token])
        if is_generic_token(text):
            continue
        back = official_target_ids(tokenizer, [text], llama_like=llama_like, device=torch.device("cpu"))
        if int(back[0]) != token:
            continue
        used.add(token)
        out.append((alias, token))
    return out


def alias_token_cases(records, facts, tokenizer, *, llama_like, table):
    """Extra row-training cases: (direct cloze, first token of an alias).

    One case per distinct (cloze, token); the same boundary and fact as the
    answer's own cases, so routing and the genie map are unchanged. The direct
    trainer's hinge then acts on the worst token over answer and aliases.
    """
    from mquake_fact_association_embeddings import DirectTokenTrainingCase, association_key_from_record

    fact_by_key = {str(f["association_key"]): f for f in facts}
    cases, seen, per_fact = [], set(), {}
    for record in records:
        fact = fact_by_key[association_key_from_record(record)]
        rr = record["requested_rewrite"]
        boundary = str(rr["prompt"]).format(str(rr["subject"]))
        answer = str(rr["target_true"]["str"])
        for alias, token in alias_training_targets(
                tokenizer, answer, record_aliases(record, table), llama_like=llama_like):
            if (boundary, token) in seen:
                continue
            seen.add((boundary, token))
            cases.append(DirectTokenTrainingCase(
                id=f"{fact['id']}:case_{int(record['case_id'])}:alias_token_{token}",
                fact_id=fact["id"], case_id=int(record["case_id"]), token_index=0,
                prompt=boundary, boundary_prompt=boundary, target_text=tokenizer.decode([token])))
            per_fact.setdefault(fact["id"], [])
            if alias not in per_fact[fact["id"]]:
                per_fact[fact["id"]].append(alias)
    info = {"alias_cases": len(cases), "facts_with_alias_targets": len(per_fact),
            "facts_total": len(facts), "aliases_by_fact": per_fact,
            "filter": "first token differs from the answer's, not generic, round-trips the official target encoding"}
    return cases, info
