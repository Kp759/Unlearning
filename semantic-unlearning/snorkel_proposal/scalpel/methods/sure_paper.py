"""The paper's method ("ours") in SCALPEL: SURE's linear router + one Δe row per fact, the rows
trained by the paper's row trainer (scripts/train_mcf_jlens_feedback_rows.py), on the split's
requests only (export_multifact, as `sure`).

    sure_paper      ONE forward pass. The classifier reads layer L (SURE's layer: 19 on
                    Llama-3.2-3B) and Δe is added at the same layer L, at the last prompt
                    token and at every generated token (linear_router write_mode "query").
    sure_paper_l16  ablation: the earlier two-pass design. Pass 1 routes at layer L, pass 2
                    adds Δe at layer 16 (scripts/early_write.py, query mode).

Rows: on-policy feedback on generated continuations of each request's views, output loss +
KL keep + teacher-forced gold views; no logit lens (--lens-type none) and no PCA in the router
(fit_linear_router's default). Facts the router never routes are trained with their row
forced on their own views (training only: the saved bank routes by its heads alone).
    --method sure_paper --method-arg model_path=<weights> [layer=19 device=cuda work_dir=...
                         train_args="--rounds 3"]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import List, Optional

import torch
from torch import nn

from ..model_api import ModelHandle
from .base import Method, MethodCard, arg
from .external import _SureRuntime, export_multifact, gpu_released

SEMANTIC_ROOT = Path(__file__).resolve().parents[3]


class _EarlyRuntime(nn.Module):
    """scripts/early_write.EarlyWriteCausalLM behind the harness interface (two passes; full
    forward only: no position-selected scoring, no cached decoding)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    @property
    def config(self):
        return self.lm.config

    def forward(self, input_ids=None, attention_mask=None, prompt_lengths=None, **kw):
        if prompt_lengths is not None:
            self.lm.set_association_prefix_lengths(torch.as_tensor(prompt_lengths))
        return self.lm(input_ids=input_ids, attention_mask=attention_mask, **kw)


def trainer_command(root: Path, router: Path, out: Path, *, router_layer: int, write_layer: int,
                    device: str, extra: str = "") -> List[str]:
    """The paper trainer's command: one pass (write layer = router layer) or an early write."""
    cmd = [sys.executable, "-u", str(root / "scripts/train_mcf_jlens_feedback_rows.py"),
           "--router-dir", str(router), "--output-dir", str(out), "--write-mode", "query",
           "--lens-type", "none", "--teacher-forced-views", "--oracle-unrouted",
           "--device", device, "--local-files-only"]
    if write_layer != router_layer:
        cmd += ["--write-layer", str(write_layer)]
    return cmd + shlex.split(extra)


class SurePaper(Method):
    card = MethodCard("sure_paper", "detect_then_edit",
                      "the paper's method: linear router at layer L + Δe at the same layer L, at the last "
                      "prompt token and every generated token, in one forward pass; rows by the paper's "
                      "feedback trainer (no logit lens, no PCA)",
                      runtime_components=["router_hooks"], reference="this repository", decode="hf")
    default_write_layer: Optional[int] = None          # None = the router layer: one pass

    def unlearn(self, base, requests, allowed_retain, ctx):
        a = self.args
        root = Path(a.get("root", SEMANTIC_ROOT)).resolve()
        work = Path(a.get("work_dir", f"outputs/scalpel_{self.card.name}")).resolve()
        layer = arg(a, "layer", 19, int)
        write_layer = arg(a, "write_layer", self.default_write_layer if self.default_write_layer is not None else layer, int)
        one_pass = write_layer == layer
        device = a.get("device", "cuda")
        fit_device = a.get("fit_device", "cpu" if device == "mps" else device)   # float64 L-BFGS: no MPS
        model_path = a["model_path"]
        data = export_multifact(requests, work / "data", ctx.seed, model_path)
        L = f"L{layer:02d}"
        prep, router = work / L / "prep", work / L / "router"
        rows = work / L / ("rows_query" if one_pass else f"rows_query_w{write_layer:02d}")
        py = [sys.executable, "-u"]
        env = {**os.environ, "PYTHONPATH": str(root / "scripts") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        steps = [
            (prep, py + [str(root / "scripts/prepare_multifact_association_source.py"), "--model-path", model_path,
                         "--training-visible", str(data / "training_visible_forget.json"),
                         "--split-manifest", str(data / "split_manifest.json"), "--output-dir", str(prep),
                         "--layer", str(layer), "--device", device, "--local-files-only"]),
            (router, py + [str(root / "scripts/fit_linear_router.py"), "--run-dir", str(prep), "--output-dir", str(router),
                           "--device", device, "--fit-device", fit_device, "--local-files-only",
                           "--threshold-policy", "global", "--decision-rule", "calibrated_bias", "--min-recall", "0.98",
                           "--threshold-placement-fraction", "0.1"]),
            (rows, trainer_command(root, router, rows, router_layer=layer, write_layer=write_layer, device=device,
                                   extra=str(a.get("train_args", "")))),
        ]
        for out, cmd in steps:
            if (out / "fact_association_embeddings.pt").exists():
                continue
            if out == rows and out.exists():      # the trainer refuses an existing output dir
                out.rename(out.with_name(f"{out.name}.incomplete_{time.strftime('%Y%m%d_%H%M%S')}"))
            ctx.log(f"[{self.card.name}] " + " ".join(cmd))
            with gpu_released(base.hf):
                subprocess.run(cmd, cwd=str(root), env=env, check=True)

        sys.path.insert(0, str(root / "scripts"))
        from linear_router import load_router_artifact  # noqa: E402  (repository code)
        artifact = torch.load(rows / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        assoc, bank = load_router_artifact(base.hf, artifact)
        if one_pass:
            if getattr(bank, "write_mode", None) != "query":
                raise ValueError(f"expected a one-pass query artifact, got write_mode {getattr(bank, 'write_mode', None)}")
            bank.tokenizer = base.tokenizer
            runtime = _SureRuntime(assoc, bank).eval()
        else:
            runtime = _EarlyRuntime(assoc).eval()
        facts = artifact["facts"]
        visible = json.loads((data / "training_visible_forget.json").read_text())
        subj_to_req = {r["requested_rewrite"]["subject"]: r["scalpel_request_id"] for r in visible}

        def route(prompts):
            out = []
            tok = base.tokenizer
            for t in prompts:
                special = not (tok.bos_token and t.startswith(tok.bos_token))   # chat text carries its BOS
                ids = tok(t, return_tensors="pt", add_special_tokens=special)["input_ids"].to(base.device)
                runtime(input_ids=ids, attention_mask=torch.ones_like(ids), prompt_lengths=[ids.shape[1]])
                hit = (bank.last_active_fact_indices or [[]])[0]
                out.append(subj_to_req.get(facts[hit[0]].get("subject")) if hit else None)
            return out

        report = json.loads((rows / "training_report.json").read_text()) if (rows / "training_report.json").exists() else {}
        self._report = {"work_dir": str(work), "router_layer": layer, "write_layer": write_layer,
                        "forward_passes": 1 if one_pass else 2, "facts": len(facts),
                        "facts_routed_by_router": report.get("facts_routed_by_router"),
                        "facts_trained_with_oracle_routing": report.get("facts_trained_with_oracle_routing"),
                        "onpolicy_leak_by_round": [r.get("onpolicy_leak") for r in report.get("rounds", [])]}

        def strip():
            bank.close()
            return ModelHandle(model=base.hf, tokenizer=base.tokenizer, hf=base.hf)
        return ModelHandle(model=runtime, tokenizer=base.tokenizer, hf=base.hf, runtime_components=["router_hooks"],
                           strip=strip, route=route, decode="cached_boundary" if one_pass else "full",
                           needs_prompt_lengths=True, lens_module=runtime)


class SurePaperL16(SurePaper):
    card = MethodCard("sure_paper_l16", "detect_then_edit",
                      "ablation of sure_paper: route at layer L in pass 1, add Δe at layer 16 in pass 2 "
                      "(the earlier two-pass early write)",
                      runtime_components=["router_hooks"], reference="this repository", decode="full")
    default_write_layer = 16
