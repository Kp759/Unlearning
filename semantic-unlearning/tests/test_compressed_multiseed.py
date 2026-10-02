import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_compressed_multiseed as summ  # noqa: E402
from train_direct_compressed_bank import checkpoint_key, first_answer_tokens  # noqa: E402
from train_direct_linear_router_rows import dataset_adapter  # noqa: E402


def test_checkpoint_key_prefers_fewer_failing_then_lower_prob():
    a = checkpoint_key({"facts_total": 50, "facts_passing_probability_constraint": 49,
                        "maximum_sensitive_token_probability": 1e-9})
    b = checkpoint_key({"facts_total": 50, "facts_passing_probability_constraint": 50,
                        "maximum_sensitive_token_probability": 5e-7})
    c = checkpoint_key({"facts_total": 50, "facts_passing_probability_constraint": 50,
                        "maximum_sensitive_token_probability": 1e-8})
    assert c < b < a


def test_first_answer_tokens_uses_token_index_zero():
    cases = [SimpleNamespace(fact_id="f1", token_index=1, target_text="b"),
             SimpleNamespace(fact_id="f1", token_index=0, target_text="a"),
             SimpleNamespace(fact_id="f2", token_index=0, target_text="c")]
    seen = {}

    def official_target_ids(tok, texts, *, llama_like, device):
        seen["texts"] = list(texts)
        return torch.tensor([ord(t) for t in texts])

    ids = first_answer_tokens(SimpleNamespace(official_target_ids=official_target_ids), None,
                              cases, [{"id": "f1"}, {"id": "f2"}], True, "cpu")
    assert seen["texts"] == ["a", "c"] and ids == [ord("a"), ord("c")]


def test_direct_adapters_expose_the_differentiable_objective():
    for name in ("zsre", "mquake"):
        module = dataset_adapter(name)["module"]
        assert callable(module.sensitive_token_state) and callable(module.direct_training_metrics)


def _run(path, eff, gen, mode=None, stop="global_train_and_development_target_met",
         shared=None):
    path.mkdir(parents=True)
    (path / "association_manifest.json").write_text(json.dumps({"plan": {"layer": 19}}))
    (path / "official_mcf_eval.json").write_text(json.dumps({
        "forget": {"Eff": eff, "Gen": gen, "Spe": 20.4},
        "retain": {"Eff": 12.0, "Gen": 12.0}, "forget_PPL": 11.3,
        "static_branch_display_zero_check": {"passed": True}}))
    if mode:
        (path / "training_report.json").write_text(json.dumps({
            "stop_reason": stop, "value_storage": {
                "per_fact_floats": 3072 if mode == "full" else 1,
                "shared_vectors": shared if shared is not None else 0}}))


def test_summary_table(tmp_path, capsys):
    for s in (1, 2):
        _run(tmp_path / "mcf_multiseed_regular_v1" / f"seed{s}" / "L19" / "linear_global", 1e-4, 3e-3)
        base = tmp_path / "compressed_multiseed_v1" / "mcf" / f"seed{s}" / "L19"
        _run(base / "full", 1e-10, 2e-3, "full")
        _run(base / "tied_answer", 1e-12, 2e-3, "tied_answer", shared=39 + s)
    _run(tmp_path / "compressed_multiseed_v1" / "mcf" / "seed1" / "L19" / "answer_fixed",
         1e-6, 2e-3, "answer_fixed", stop="wall_time_budget")
    summ.main(["--root", str(tmp_path), "--datasets", "mcf", "--seeds", "1", "2"])
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("| ")]
    assert lines[1].startswith("| shipped (row-wise, per fact) | 2 |")
    assert lines[2].startswith("| full | 2 |") and lines[2].endswith("| 3072 | 0 | 2/2 |")
    assert lines[3].startswith("| tied_answer | 2 |") and "| 40/41 | 2/2 |" in lines[3]
    assert lines[4].startswith("| answer_fixed | 1 |") and lines[4].endswith("| 0/1 |")


class _Tok:
    """Characters -> ids; BOS=1 added unless add_special_tokens=False; pad=0."""
    pad_token_id, eos_token_id = 0, 2

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + (ord(c) % 20) for c in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(32, 8)
        self.out = torch.nn.Linear(8, 32)
        self.prefix = None

    def set_association_prefix_lengths(self, lengths):
        self.prefix = list(lengths)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


def test_abstain_nll_scores_only_the_completion_after_each_boundary():
    from train_direct_compressed_bank import abstain_batch, abstain_nll

    tok, model = _Tok(), _Model()
    batch = abstain_batch(tok, ["ab?", "abcd?"], " ok", "cpu")
    assert batch["prefix"] == [4, 6] and batch["k"] == 3
    nll = abstain_nll(model, batch)
    assert model.prefix == [4, 6]                      # edit bound to each request end
    logits = model(batch["ids"]).logits
    manual = sum(torch.nn.functional.cross_entropy(logits[i, p - 1:p + 2], batch["ids"][i, p:p + 3])
                 for i, p in enumerate([4, 6])) / 2
    assert torch.allclose(nll, manual)
    nll.backward()                                     # differentiable
    assert model.out.weight.grad is not None


def test_checkpoint_key_with_abstention_prefers_lower_nll_once_feasible():
    ok = {"facts_total": 50, "facts_passing_probability_constraint": 50,
          "maximum_sensitive_token_probability": 5e-7}
    deeper = dict(ok, maximum_sensitive_token_probability=1e-9)
    failing = dict(ok, facts_passing_probability_constraint=49)
    assert checkpoint_key(ok, 0.5) < checkpoint_key(deeper, 2.0)   # abstention wins once feasible
    assert checkpoint_key(deeper, 2.0) < checkpoint_key(failing, 0.1)  # feasibility first
    assert checkpoint_key(ok) == (0, 5e-7)                         # unchanged without abstention
