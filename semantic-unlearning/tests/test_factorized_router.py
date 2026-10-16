"""Factorized subject x relation router (scripts/factorized_router.py)."""
from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from factorized_router import (  # noqa: E402
    ARCHITECTURE,
    FactorizedRouter,
    build_router_data,
    calibrate,
    decide_from_probabilities,
    fit_softmax,
    misspell,
    norm,
)

transformers = pytest.importorskip("transformers")
HIDDEN = 8


def test_misspell_is_deterministic_and_changes_one_word():
    a = misspell("Ignazio Silone", random.Random(3))
    assert a == misspell("Ignazio Silone", random.Random(3))
    changed = [x != y for x, y in zip(a.split(), "Ignazio Silone".split())]
    assert a != "Ignazio Silone" and sum(changed) <= 1 and len(a.split()) == 2


def test_fire_needs_both_heads_and_picks_the_best_fact():
    fact_subject, fact_relation = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])
    ps = torch.tensor([[0.9, 0.05, 0.05],       # subject 0
                       [0.9, 0.05, 0.05],
                       [0.05, 0.05, 0.9]])      # none
    pr = torch.tensor([[0.8, 0.2],              # relation 0 -> fact 0
                       [0.3, 0.7],              # relation 1 -> fact 1
                       [0.9, 0.1]])             # relation 0, but subject none
    d = decide_from_probabilities(ps, pr, fact_subject, fact_relation, 0.5, 0.5)
    assert d.tolist() == [0, 1, -1]
    assert decide_from_probabilities(ps, pr, fact_subject, fact_relation, 0.5, 0.95).tolist() == [-1, -1, -1]


def test_softmax_fit_separates_and_calibration_meets_recall():
    torch.manual_seed(0)
    centres = torch.randn(3, 16) * 4
    y = torch.arange(3).repeat_interleave(40)
    X = centres[y] + torch.randn(len(y), 16)
    W, b = fit_softmax(X, y, 3, 1e-3, device="cpu")
    assert float(((X @ W.T + b).argmax(-1) == y).float().mean()) == 1.0
    # one fact (subject 0, relation 0); negatives: same subject other relation, other subject
    ps = torch.tensor([[0.9, 0.1]] * 10 + [[0.9, 0.1]] * 10 + [[0.1, 0.9]] * 10)
    pr = torch.tensor([[0.9, 0.1]] * 10 + [[0.2, 0.8]] * 10 + [[0.9, 0.1]] * 10)
    target = [0] * 10 + [-1] * 20
    kinds = ["own"] * 10 + ["subject_other_relation"] * 10 + ["other_subject"] * 10
    cal = calibrate(ps, pr, target, torch.tensor([0]), torch.tensor([0]), kinds, min_recall=0.95)
    assert cal["recall"] == 1.0 and cal["false_fire_same_subject"] == 0.0 and cal["false_fire_other"] == 0.0


def _records():
    def rec(cid, subject, rel, prompt, paras=(), neigh=(), gen=()):
        return {"case_id": cid, "requested_rewrite": {"prompt": prompt, "relation_id": rel, "subject": subject,
                                                      "target_true": {"str": "X"}, "target_new": {"str": "Y"}},
                "paraphrase_prompts": list(paras), "neighborhood_prompts": list(neigh),
                "attribute_prompts": [], "generation_prompts": list(gen)}
    rs = [rec(0, "Alice Smith", "P19", "{} was born in", paras=["Intro text here. Alice Smith was born in"],
              neigh=["Bob Jones was born in"], gen=["Alice Smith grew up in"])]
    rels = {"P27": "{} is a citizen of", "P106": "{} works as a", "P20": "{} died in",
            "P1412": "{} speaks", "P101": "{} works in the field of", "P19": "{} was born in"}
    k = 1
    for name in ("Carl Moe", "Dana Ray", "Ed Fox", "Alice Smithson Bridge", "Fay Lee"):
        for rel, t in rels.items():
            rs.append(rec(k, name, rel, t, paras=[f"Some lead in sentence. {t.format(name)}"]))
            k += 1
    rs.append(rec(k, "Gus Hall", "P27", "{} is a citizen of", neigh=["Alice Smith is a citizen of"]))
    return rs


def test_data_never_trains_on_heldout_strings_and_none_never_names_a_forget_subject():
    records = _records()
    forget = [records[0]]
    facts = [{"id": "f0", "subject": "Alice Smith", "relation": "P19"}]
    examples = [{"fact_id": "f0", "prompt": "Alice Smith was born in", "split": "train", "group": "canonical"},
                {"fact_id": "f0", "prompt": "Recall: Alice Smith was born in", "split": "development",
                 "group": "authored_0"}]
    data = build_router_data(records, forget, facts, examples, seed=1)
    prompts = [i["prompt"] for i in data["items"]]
    assert "Intro text here. Alice Smith was born in" not in prompts        # official paraphrase
    assert "Bob Jones was born in" not in prompts                           # official neighbourhood
    none = len(data["subjects"])
    assert not any(i["subject"] == none and "alice smith" in norm(i["prompt"]) for i in data["items"])
    own = [i for i in data["items"] if i["target"] == 0]
    assert {i["split"] for i in own} == {"train", "val"}
    assert any("+typo" in i["kind"] for i in own) and any("+lower" in i["kind"] for i in own)


def _tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=16, num_hidden_layers=3,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                   bos_token_id=1, eos_token_id=2)
    return transformers.LlamaForCausalLM(cfg).eval()


def _artifact(subject_bias, relation_bias, t=0.5, rows=None):
    state = {"layer": 2, "feature_mean": torch.zeros(HIDDEN),
             "subject_weight": torch.zeros(2, HIDDEN), "subject_bias": torch.tensor(subject_bias),
             "relation_weight": torch.zeros(2, HIDDEN), "relation_bias": torch.tensor(relation_bias),
             "fact_subject": torch.tensor([0]), "fact_relation": torch.tensor([0]),
             "t_subject": t, "t_relation": t}
    art = {"architecture": ARCHITECTURE, "layer": 2, "facts": [{"id": "f0", "subject": "s", "relation": "P19"}],
           "subject_patterns": [[(20,)]], "rows": torch.zeros(1, HIDDEN) if rows is None else rows,
           "factorized_router": state}
    return art


IDS = torch.tensor([[1, 5, 6, 7, 20, 8, 9]])


@pytest.mark.parametrize("fires", [True, False])
def test_runtime_fires_only_when_both_heads_agree_and_unrouted_is_exact_base(fires):
    from early_write import EarlyWriteCausalLM

    base = _tiny_llama()
    with torch.no_grad():
        reference = base(input_ids=IDS).logits
    sb = [5.0, -5.0] if fires else [-5.0, 5.0]               # subject 0 vs none
    art = _artifact(sb, [5.0, -5.0])
    model = EarlyWriteCausalLM(base, art, write_layer=0, write_mode="query", rows=torch.full((1, HIDDEN), 2.0))
    model.set_association_prefix_lengths([IDS.shape[1]])
    with torch.no_grad():
        logits = model(input_ids=IDS, use_cache=False).logits
    assert model.last_active_fact_indices == ([[0]] if fires else [[]])
    assert torch.equal(logits, reference) != fires
    model.close()


def test_router_only_artifact_loads_and_never_edits():
    from linear_router import load_router_artifact

    base = _tiny_llama()
    with torch.no_grad():
        reference = base(input_ids=IDS).logits
    lm, proxy = load_router_artifact(base, _artifact([5.0, -5.0], [5.0, -5.0]))
    lm.set_association_prefix_lengths([IDS.shape[1]])
    with torch.no_grad():
        logits = lm(input_ids=IDS, use_cache=False).logits
    assert proxy.last_active_fact_indices == [[0]] and torch.equal(logits, reference)
    lm.close()


def test_router_decision_matches_offline_decide():
    base = _tiny_llama()
    torch.manual_seed(1)
    state = _artifact([0.0, 0.0], [0.0, 0.0])["factorized_router"]
    state["subject_weight"] = torch.randn(2, HIDDEN)
    state["relation_weight"] = torch.randn(2, HIDDEN)
    state["t_subject"] = state["t_relation"] = 0.3
    router = FactorizedRouter(state)
    ids = torch.tensor([[1, 5, 6, 7, 20, 8, 9, 0], [1, 3, 4, 20, 0, 0, 0, 0]])
    att = (torch.arange(8)[None] < torch.tensor([[7], [4]])).long()
    online = router.route(base, ids, att)
    store = {}
    h = base.model.layers[2].register_forward_hook(lambda m, a, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        base(input_ids=ids, attention_mask=att)
    h.remove()
    offline = router.decide(store["h"][torch.arange(2), att.sum(1) - 1]).tolist()
    assert online == offline
