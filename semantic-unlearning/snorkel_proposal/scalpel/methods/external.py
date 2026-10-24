"""Bringing your own method.

    --method checkpoint --method-arg path=/models/my_unlearned
        a model unlearned elsewhere (ROME / MEMIT / ZeroUnlearn runs, or a submission).
        The checkpoint must have been produced from the split's `requests` only.
    --method plugin --method-arg target=my_package.my_module:MyMethod
        any class implementing scalpel.methods.base.Method.
    --method sure --method-arg root=<semantic-unlearning dir> [layer=19 device=cuda]
        the repository's SURE pipeline (router + Δe rows) trained on the split's requests
        through its MultiFactPerson entry points, loaded with its own runtime.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

import torch
from torch import nn

from ..model_api import ModelHandle
from .base import Method, MethodCard, arg


class Checkpoint(Method):
    card = MethodCard("checkpoint", "external", "an HF checkpoint unlearned outside this harness")

    def unlearn(self, base, requests, allowed_retain, ctx):
        from transformers import AutoModelForCausalLM
        path = self.args["path"]
        dtype = next(base.hf.parameters()).dtype
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(base.device).eval()
        self.card = MethodCard(self.args.get("name", "checkpoint"), self.args.get("family", "external"),
                               f"checkpoint {path}")
        return ModelHandle(model=m, tokenizer=base.tokenizer, hf=m)


class Plugin(Method):
    card = MethodCard("plugin", "external", "user-supplied Method class")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mod, cls = str(kwargs["target"]).split(":")
        impl = getattr(importlib.import_module(mod), cls)
        rest = {k: v for k, v in kwargs.items() if k != "target"}
        self.impl = impl(**rest)
        self.card = self.impl.card

    def unlearn(self, base, requests, allowed_retain, ctx):
        return self.impl.unlearn(base, requests, allowed_retain, ctx)

    def unlearn_sequential(self, current, new_requests, history, allowed_retain, ctx):
        return self.impl.unlearn_sequential(current, new_requests, history, allowed_retain, ctx)

    def report(self):
        return self.impl.report()


# ----------------------------------------------------------------------------- SURE
class _SureRuntime(nn.Module):
    """Adapts SURE's AssociationCausalLM to the harness: the request boundary arrives as
    `prompt_lengths` (the evaluator knows where the user prompt ends, as at inference)."""

    def __init__(self, assoc, bank):
        super().__init__()
        self.assoc = assoc
        self.bank = bank

    @property
    def config(self):
        return self.assoc.config

    def forward(self, input_ids=None, attention_mask=None, prompt_lengths=None, **kw):
        if prompt_lengths is not None:
            self.assoc.set_association_prefix_lengths(prompt_lengths)
        return self.assoc(input_ids=input_ids, attention_mask=attention_mask, **kw)

    def selected_logits(self, input_ids, attention_mask, rows, prompt_lengths=None):
        """Same computation as forward() (bank bound to the batch and its request boundary,
        then the base model), with the unembedding applied at the scored positions only."""
        base = self.assoc.base_model
        pl = None if prompt_lengths is None else torch.as_tensor(list(prompt_lengths), device=input_ids.device)
        self.bank.bind(input_ids, attention_mask=attention_mask, prefix_lengths=pl)
        try:
            h = getattr(base, base.base_model_prefix)(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        finally:
            self.bank.unbind()
        head = base.get_output_embeddings()
        out = []
        for b, s0, n in rows:
            out.append(head(h[b, s0:s0 + n]))
        return out

    @torch.no_grad()
    def generate_cached(self, ids, att, max_new_tokens, eos):
        """Batched greedy decoding with a KV cache. Valid for write_mode "last": the bank edits
        only the request-boundary position, so the prefill (prompt, bank bound, boundary = the
        last prompt token of every left-padded row) carries the whole edit and the decode
        steps run the base model with the hook in passthrough. Same tokens as the
        prefix-recompute decoder (tests: test_sure_cached_decoder_*)."""
        if self.bank.write_mode != "last":
            raise ValueError("cached decoding needs write_mode 'last'")
        base = self.assoc.base_model
        width = ids.shape[1]
        pos = (att.long().cumsum(-1) - 1).clamp_min(0)
        self.bank.bind(ids, attention_mask=att,
                       prefix_lengths=torch.full((ids.shape[0],), width, device=ids.device, dtype=torch.long))
        try:
            out = base(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        finally:
            self.bank.unbind()
        past = out.past_key_values
        nxt = out.logits[:, -1].float().argmax(-1)
        done = torch.zeros_like(nxt, dtype=torch.bool)
        new, last_pos = [], pos[:, -1]
        self.bank.passthrough = True
        try:
            for step in range(max_new_tokens):
                if eos is not None:
                    nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
                    done |= nxt == eos
                new.append(nxt)
                if step == max_new_tokens - 1 or bool(done.all()):
                    break
                att = torch.cat([att, torch.ones_like(nxt)[:, None]], 1)
                last_pos = last_pos + 1
                out = base(input_ids=nxt[:, None], attention_mask=att, position_ids=last_pos[:, None],
                           past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1].float().argmax(-1)
        finally:
            self.bank.passthrough = False
        return torch.stack(new, 1)


def export_multifact(requests, out_dir: Path, seed: int, model_path: str) -> Path:
    """Write the split's requests in the MultiFactPerson locked format the SURE entry points
    read (training_visible_forget.json + split_manifest.json). Only the requests are
    written; no probe ever reaches SURE's training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, r in enumerate(requests):
        if r.kind != "fact" or r.subject not in r.prompt:
            continue
        records.append({"case_id": i + 1, "mquake_case_id": i + 1, "source_index": i, "rewrite_index": 0,
                        "requested_rewrite": {"prompt": r.prompt.replace("{", "{{").replace("}", "}}").replace(r.subject, "{}", 1),
                                              "subject": r.subject,
                                              "relation_id": r.relation, "target_true": {"str": r.answer, "id": None}},
                        "source": f"scalpel:{r.request_id}", "scalpel_request_id": r.request_id})
    visible = out_dir / "training_visible_forget.json"
    visible.write_text(json.dumps(records, indent=2) + "\n")
    manifest = {"dataset": "MultiFactPerson-v1", "seed": int(max(1, seed)),
                "sampling": {"forget_num_instances": len(records), "forget_atomic_fact_count": len(records),
                             "retain_person_count": 0,
                             "forget_atomic_case_ids": [r["case_id"] for r in records]},
                "model_path": str(model_path), "training_visible_path": str(visible),
                "protocol": {"training_visible": "SCALPEL requests only"}}
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


class Sure(Method):
    card = MethodCard("sure", "detect_then_edit",
                      "SURE: per-fact linear router on a boundary state + one residual row per fact "
                      "(the repository's method), trained on the split's requests",
                      runtime_components=["router_hooks"], reference="this repository", decode="full")

    def unlearn(self, base, requests, allowed_retain, ctx):
        a = self.args
        root = Path(a.get("root", Path(__file__).resolve().parents[3])).resolve()
        work = Path(a.get("work_dir", "outputs/scalpel_sure")).resolve()
        layer = arg(a, "layer", 19, int)
        device = a.get("device", "cuda")
        fit_device = a.get("fit_device", "cpu" if device == "mps" else device)   # float64 L-BFGS: no MPS
        model_path = a["model_path"]
        data = export_multifact(requests, work / "data", ctx.seed, model_path)
        prep, router, final = work / f"L{layer:02d}" / "prep", work / f"L{layer:02d}" / "router", work / f"L{layer:02d}" / "rows"
        py = [sys.executable, "-u"]
        env = {**os.environ, "PYTHONPATH": str(root / "scripts") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        steps = [
            (prep, py + [str(root / "scripts/prepare_multifact_association_source.py"), "--model-path", model_path,
                         "--training-visible", str(data / "training_visible_forget.json"),
                         "--split-manifest", str(data / "split_manifest.json"), "--output-dir", str(prep),
                         "--layer", str(layer), "--device", device, "--local-files-only"]),
            (router, py + [str(root / "scripts/fit_linear_router.py"), "--run-dir", str(prep), "--output-dir", str(router),
                           "--device", device, "--fit-device", fit_device, "--local-files-only",
                           "--threshold-policy", "global",
                           "--decision-rule", "calibrated_bias", "--min-recall", "0.98",
                           "--threshold-placement-fraction", "0.1"]),
            (final, py + [str(root / "scripts/train_direct_linear_router_rows.py"), "--dataset", "multifact",
                          "--router-dir", str(router), "--output-dir", str(final), "--training-route", "router",
                          "--norm-scale", "1", "--max-training-seconds", str(int(ctx.max_seconds)),
                          "--device", device, "--local-files-only"]),
        ]
        for out, cmd in steps:
            if (out / "fact_association_embeddings.pt").exists():
                continue
            ctx.log("[sure] " + " ".join(cmd))
            subprocess.run(cmd, cwd=str(root), env=env, check=True)
        sys.path.insert(0, str(root / "scripts"))
        from linear_router import load_router_artifact  # noqa: E402  (repository code)
        artifact = torch.load(final / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        assoc, bank = load_router_artifact(base.hf, artifact)
        runtime = _SureRuntime(assoc, bank).eval()
        facts = artifact["facts"]
        visible = json.loads((data / "training_visible_forget.json").read_text())
        subj_to_req = {r["requested_rewrite"]["subject"]: r["scalpel_request_id"] for r in visible}

        def route(prompts: List[str]) -> List[Optional[str]]:
            out = []
            tok = base.tokenizer
            for t in prompts:
                special = not (tok.bos_token and t.startswith(tok.bos_token))   # chat text carries its BOS
                ids = tok(t, return_tensors="pt", add_special_tokens=special)["input_ids"].to(base.device)
                runtime(input_ids=ids, attention_mask=torch.ones_like(ids), prompt_lengths=[ids.shape[1]])
                hit = (bank.last_active_fact_indices or [[]])[0]
                out.append(subj_to_req.get(facts[hit[0]].get("subject")) if hit else None)
            return out

        self._report = {"work_dir": str(work), "layer": layer, "facts": len(facts), "counters": bank.counters()}

        def strip():
            # weights-only tier: the router / row hooks live on the base layers; detach them
            bank.close()
            return ModelHandle(model=base.hf, tokenizer=base.tokenizer, hf=base.hf)
        return ModelHandle(model=runtime, tokenizer=base.tokenizer, hf=base.hf, runtime_components=["router_hooks"],
                           strip=strip,
                           route=route, decode="full", needs_prompt_lengths=True, lens_module=runtime)
