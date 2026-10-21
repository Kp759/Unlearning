"""scripts/build_edit_covariance.py writes the covariance caches the vendored ROME / MEMIT /
ZeroUnlearn loaders read, with the statistics they would compute; paper_baselines.configure_mom2
points the baselines at them and fails fast when one is missing."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("yaml")

import build_edit_covariance as bec  # noqa: E402

TOKENIZER = Path("/home/claude/kp759/unlearning/semantic-unlearning/outputs/mcf_base_llama32_3b_policy1000_soft_overlap")


def _tok():
    if not TOKENIZER.is_dir():
        pytest.skip("local Llama tokenizer not available")
    return transformers.AutoTokenizer.from_pretrained(str(TOKENIZER))


def _tiny(tok, path):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=24,
                                   num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=128, bos_token_id=tok.bos_token_id,
                                   eos_token_id=tok.eos_token_id)
    transformers.LlamaForCausalLM(cfg).save_pretrained(path)
    return transformers.LlamaForCausalLM.from_pretrained(path).eval()


TEXTS = [f"Document {i}: the river {['Nile', 'Rhine', 'Volga'][i % 3]} flows past town number {i} "
         f"and {'many ' * (i % 5)}people live near it." for i in range(30)]


def test_caches_match_direct_statistics_and_load_through_the_vendored_loaders(tmp_path):
    tok = _tok()
    model = _tiny(tok, tmp_path / "tiny-model")
    stats = tmp_path / "stats"
    rep = bec.build(model, tok, TEXTS, [2, 3], stats_dir=stats, dataset_name="wikitest",
                    sample_size=20, max_len=64, batch_tokens=200, batch_size=4, device="cpu",
                    num_workers=0, log_every=0)
    assert rep["computed"] == [2, 3] and rep["documents"] == 20 and rep["model_name"] == "tiny-model"

    # direct reference on the same 20 documents tally(sample_size=20, random_sample=1) draws
    rs, TokenizedDataset, *_ = bec.vendored()
    ds = TokenizedDataset([{"text": t} for t in TEXTS], tokenizer=tok, maxlen=64)
    idx = rs.FixedRandomSubsetSampler(ds, seed=1, end=20).samples
    ks, vs = [], []
    store = {}
    h = model.model.layers[3].mlp.down_proj.register_forward_hook(
        lambda m, i, o: store.__setitem__("kv", (i[0][0], o[0])))
    with torch.no_grad():
        for i in idx:
            model(input_ids=ds[i]["input_ids"][None])
            ks.append(store["kv"][0]); vs.append(store["kv"][1])
    h.remove()
    K, Vv = torch.cat(ks).float(), torch.cat(vs).float()

    k_path, v_path = bec.cache_paths(stats, "tiny-model", "wikitest", 3, 20)
    with bec._cwd(bec.ZERO_ROOT):
        from rome.layer_stats import layer_stats
    with bec._cwd(bec.ZERO_ROOT):
        stat = layer_stats(model, tok, "model.layers.3.mlp.down_proj", stats, "wikitest",
                           to_collect=["mom2"], sample_size=20, precision="float32")
    assert stat.mom2.count == K.shape[0]
    assert torch.allclose(stat.mom2.moment(), K.T @ K / K.shape[0], atol=1e-5, rtol=1e-4)
    vkt = rs.VKT()
    vkt.load_state_dict(rs.load_cached_state(str(v_path), {"sample_size": 20}))
    centred = (Vv - Vv.mean(0)).T @ (K - K.mean(0)) / (K.shape[0] - 1)
    assert torch.allclose(vkt.moment(unbiased=True).float(), centred, atol=1e-5, rtol=1e-4)
    assert k_path.is_file() and v_path.is_file()

    again = bec.build(model, tok, TEXTS, [2, 3], stats_dir=stats, dataset_name="wikitest",
                      sample_size=20, max_len=64, device="cpu", num_workers=0)
    assert again["computed"] == [] and again["skipped_existing"] == [2, 3]


def test_configure_mom2_points_hparams_at_the_cache_and_fails_fast_when_missing(tmp_path, monkeypatch):
    import paper_baselines as pb

    monkeypatch.setattr(pb, "SEMANTIC_ROOT", tmp_path)
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="/x/snapshots/abc123"))
    hp = SimpleNamespace(layers=[16, 17], mom2_n_samples=100, mom2_dataset="wikipedia",
                         rewrite_module_tmp="model.layers.{}.mlp.down_proj")
    args = SimpleNamespace(mom2_dataset="wiki100k", mom2_text="/corpus", allow_mom2_compute=False)
    monkeypatch.delenv("ZU_MOM2_TEXT_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        pb.configure_mom2(args, hp, model, need_vkt=True)
    for L in hp.layers:
        for p in bec.cache_paths(tmp_path / "data" / "stats", "abc123", "wiki100k", L, 100):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
    rep = pb.configure_mom2(args, hp, model, need_vkt=True)
    assert hp.mom2_dataset == "wiki100k" and rep["mom2_cache_missing"] == []
    assert os.environ["ZU_MOM2_TEXT_DIR"] == "/corpus"
    plain = SimpleNamespace(layers=[16], mom2_n_samples=100, mom2_dataset="wikipedia")
    assert pb.configure_mom2(SimpleNamespace(), plain, model)["mom2_dataset"] == "wikipedia"
