"""Genie routing for the official evaluators (`--genie`): perfect routing, same rows.

Every official FORGET prompt (rewrite / paraphrase / atomic / same-50 probe) is routed to
its own fact's row; every other prompt (retain, neighbourhood, held-out probes with no row,
PPL text) gets no row, so it stays exact base. The router is not run at all. The gap
between `ours` and `genie` on the same trained rows is therefore the cost of routing, and
genie Eff/Gen is the best these rows can do.

Keys are prompt-prefix token tuples (the request boundary the bank sees); leading BOS is
ignored by EarlyWriteCausalLM.set_oracle_routes, so tokenising with special tokens is fine.
"""
from __future__ import annotations


def _rr(record):
    rr = record["requested_rewrite"]
    return rr[0] if isinstance(rr, list) else rr


def install(bank, tok, prompt_fact_pairs):
    """prompt_fact_pairs: iterable of (prompt text, fact row index). Returns a report."""
    if not hasattr(bank, "set_oracle_routes"):
        raise ValueError("--genie needs a bank with set_oracle_routes (early-write or linear router)")
    table, collisions, n = {}, 0, 0
    for prompt, fact in prompt_fact_pairs:
        n += 1
        key = tuple(int(t) for t in tok(str(prompt))["input_ids"])
        if key in table and table[key] != int(fact):
            collisions += 1          # same prompt text owned by two facts: keep the first
            continue
        table.setdefault(key, int(fact))
    bank.set_oracle_routes(table)
    return {"genie": True, "prompts": n, "table_size": len(table), "collisions": collisions,
            "definition": "official forget prompts -> own fact row; all other prompts -> no row; "
                          "router not run"}


def fact_rows(artifact):
    return {str(f["id"]): i for i, f in enumerate(artifact["facts"])}


def mcf_pairs(artifact, forget_records):
    rows = fact_rows(artifact)
    for record in forget_records:
        row = rows[f"mcf_forget_{int(record['case_id'])}"]
        rr = _rr(record)
        yield str(rr["prompt"]).format(rr["subject"]), row
        for p in record.get("paraphrase_prompts", []):
            yield str(p), row


def zsre_pairs(artifact, forget_records):
    by_case = {int(f["case_id"]): i for i, f in enumerate(artifact["facts"])}
    for record in forget_records:
        row = by_case[int(record["case_id"])]
        rr = _rr(record)
        yield str(rr["prompt"]).format(str(rr["subject"])), row
        for p in record.get("paraphrase_prompts", []):
            yield str(p), row


def rwku_pairs(tok, forget_rows, expected_source_to_row):
    """Same-50 probes, formatted exactly as the evaluator formats them; held-out probes ask
    other facts about the same people, have no row, and so get none (genie = base there)."""
    import rwku_eval as rwku

    for row in forget_rows:
        target = expected_source_to_row.get(str(row.get("source_record_sha256", "")))
        if target is not None:
            yield rwku.format_qa_prompt(tok, row), int(target)


def rwku_entity_pairs(model, tok, facts, forget_rows, expected_source_to_row, person_rows, fmt=None):
    """Entity-level genie for RWKU (`--genie-entity`): perfect PERSON routing on the same rows.

    same-50 probes        -> their own row (as rwku_pairs)
    person_rows           held-out probes about a target person (Level 1, Level 2, Level-2
                          paraphrase, Level 3) -> a row of THAT person: the one the run's own
                          router picks when it fires one of that person's rows, else the
                          person's first row
    everything else       (neighbours = other people, PPL text) -> no row
    The router decides only WHICH of the right person's rows; whether to fire is ground truth.
    Yields (prompt, row); `stats` (a dict, filled while iterating) counts the row choices."""
    import torch

    if fmt is None:
        import rwku_eval as rwku
        fmt = lambda row: rwku.format_qa_prompt(tok, row)   # noqa: E731

    def person(s):
        return " ".join(str(s).casefold().split())

    rows_of = {}
    for i, f in enumerate(facts):
        rows_of.setdefault(person(f.get("subject", "")), []).append(i)
    stats = rwku_entity_pairs.stats = {"own_row": 0, "router_choice": 0, "fallback_first_row": 0,
                                       "router_fired_other_person": 0, "unknown_person": 0}
    for row in forget_rows:
        target = expected_source_to_row.get(str(row.get("source_record_sha256", "")))
        if target is not None:
            stats["own_row"] += 1
            yield fmt(row), int(target)
    route = getattr(model, "_route", None)
    for row in person_rows:
        candidates = rows_of.get(person(row.get("subject", "")))
        if not candidates:
            stats["unknown_person"] += 1
            continue
        prompt = fmt(row)
        chosen = -1
        if route is not None:
            ids = torch.tensor([tok(prompt)["input_ids"]], device=next(model.parameters()).device)
            chosen = int(route(ids, torch.ones_like(ids), [ids.shape[1]])[0])
        if chosen in candidates:
            stats["router_choice"] += 1
        else:
            stats["router_fired_other_person"] += int(chosen >= 0)
            stats["fallback_first_row"] += 1
            chosen = candidates[0]
        yield prompt, chosen


def mquake_pairs(tok, forget_records, expected_case_to_row, llama_like, boundary_text):
    """Every forget case (rewrite + atomic_gen) at its request boundary -> its gold row."""
    import mquake_zero_unlearn_official_eval as mquake

    for record in forget_records:
        row = expected_case_to_row.get(int(record["case_id"]))
        if row is None:
            continue
        for case in mquake.expand_prediction_cases(record, tok, llama_like=llama_like,
                                                   prompt_types=("rewrite", "atomic_gen")):
            yield boundary_text(record, case), int(row)
