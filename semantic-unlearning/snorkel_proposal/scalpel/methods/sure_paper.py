"""The paper's method ("ours") in SCALPEL: SURE's linear router + one Δe row per fact, the rows
trained by the paper's row trainer (scripts/train_mcf_jlens_feedback_rows.py), on the split's
requests only (export_multifact, as `sure`).

    sure_paper      ONE forward pass. The classifier reads layer L (SURE's layer: 19 on
                    Llama-3.2-3B) and Δe is added at the same layer L, at the last prompt
                    token and at every generated token (linear_router write_mode "query").
    sure_paper_l16  ablation: the earlier two-pass design. Pass 1 routes at layer L, pass 2
                    adds Δe at layer 16 (scripts/early_write.py, query mode).
    ours            sure_paper's edit on SURE-R's router: fuzzy subject matching and 20
                    quality-checked positives per fact (the request, validated base-model
                    rewrites, context prefixes) and 20 quality-checked negatives (base-model
                    answer-collision and plausibility checks). See write_quality_report.
    ours_l16        ablation of ours: Δe at layer 16, two passes.
    ours_early      ablation of ours: classifier AND Δe at the early layer (16 on Llama-3.2-3B),
                    one pass (the paper pipeline's ONEPASS=1 ROUTER_LAYER=16).

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


def request_of_fact(prep: Path, data: Path) -> dict:
    """SURE fact id -> SCALPEL request id, through the prep manifest's case mapping."""
    manifest = json.loads((prep / "association_manifest.json").read_text())
    case_to_fact = {str(k): v for k, v in manifest["atomic_case_to_association_id"].items()}
    visible = json.loads((data / "training_visible_forget.json").read_text())
    return {case_to_fact[str(rec["case_id"])]: rec["scalpel_request_id"] for rec in visible}


class SurePaper(Method):
    card = MethodCard("sure_paper", "detect_then_edit",
                      "the paper's method: linear router at layer L + Δe at the same layer L, at the last "
                      "prompt token and every generated token, in one forward pass; rows by the paper's "
                      "feedback trainer (no logit lens, no PCA)",
                      runtime_components=["router_hooks"], reference="this repository", decode="hf")
    default_layer: int = 19                            # router layer (SURE's, Llama-3.2-3B)
    default_write_layer: Optional[int] = None          # None = the router layer: one pass
    fit_args: tuple = ()                               # extra fit_linear_router.py flags
    train_args: tuple = ()                             # extra row-trainer flags

    def router_inputs(self, base, requests, prep: Path, req_of_fact: dict, ctx) -> dict:
        """Write extra router inputs into SURE's prep directory before the fit (none here)."""
        return {}

    def fit_extra(self, prep: Path) -> list:
        """Per-run fit flags (ours: --views <prep>/router_rewrites.json)."""
        return []

    def unlearn(self, base, requests, allowed_retain, ctx):
        a = self.args
        root = Path(a.get("root", SEMANTIC_ROOT)).resolve()
        work = Path(a.get("work_dir", f"outputs/scalpel_{self.card.name}")).resolve()
        layer = arg(a, "layer", self.default_layer, int)
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
                           "--threshold-placement-fraction", "0.1", *self.fit_args, *self.fit_extra(prep)]),
            (rows, trainer_command(root, router, rows, router_layer=layer, write_layer=write_layer, device=device,
                                   extra=" ".join([*self.train_args, str(a.get("train_args", ""))]))),
        ]
        inputs_report = {}
        for out, cmd in steps:
            if (out / "fact_association_embeddings.pt").exists():
                continue
            if out == router:                     # this method's own router inputs, then the fit
                inputs_report = self.router_inputs(base, requests, prep, request_of_fact(prep, data), ctx)
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
            if getattr(assoc, "bank", None) is not None:
                assoc.bank.tokenizer = base.tokenizer     # the routing pass's bank (fuzzy matching)
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
        quality = write_quality_report(router, work / L / "router_quality.md", inputs_report)
        self._report = {"work_dir": str(work), "router_layer": layer, "write_layer": write_layer,
                        "router_quality": quality,
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


# ------------------------------------------------------------------ ours: SURE-R's router
OURS_FIT_ARGS = ("--subject-match", "fuzzy", "--positives-per-fact", "20", "--negatives-per-fact", "20",
                 "--max-prefix-positives", "8", "--pca-dims", "0", "--negative-answer-check",
                 "--negative-plausibility-delta", "1.5")


class Ours(SurePaper):
    card = MethodCard("ours", "detect_then_edit",
                      "SURE-R's router (fuzzy subject matching; 20 quality-checked positives per fact: the request, "
                      "validated base-model rewrites, context prefixes; 20 quality-checked negatives incl. model-written "
                      "same-subject hard negatives) + Δe at the "
                      "router layer in one forward pass, at the last prompt token and every generated token; rows by "
                      "the paper's feedback trainer (no logit lens, no PCA)",
                      runtime_components=["router_hooks"], reference="this repository", decode="hf")
    fit_args = OURS_FIT_ARGS
    train_args = ("--views-per-fact", "8")

    def router_inputs(self, base, requests, prep, req_of_fact, ctx):
        """The same rewrites the paper pipeline builds (scripts/router_rewrites.py, every check),
        rendered as SCALPEL's deployment renders prompts; handed to the fit with --views, as
        build_router_rewrites.py hands them to the paper pipeline's fit."""
        from .sure_robust import TRAIN_LANGS, VIEW_KINDS, make_views
        views, stats = make_views(base, ctx.renderer, requests, VIEW_KINDS, langs=TRAIN_LANGS, n_para=6, n_question=3,
                                  n_multiturn=2, split_rule="holdout_third", judge=True, min_views=8, log=ctx.log)
        fact_of = {rid: fid for fid, rid in req_of_fact.items()}
        rows = [{"fact_id": fact_of[v.request_id], "prompt": v.sure_text, "split": v.split, "group": v.family,
                 "role": v.family, "augmented": True, "kind": v.kind}
                for vs in views.values() for v in vs if v.request_id in fact_of]
        (prep / "router_rewrites.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
        # hard negatives: other questions about the subject, as in build_router_rewrites.py
        from .sure_robust import RR, sure_text
        from ..schema import Probe
        reqs = [r for r in requests if r.kind == "fact" and r.subject and r.prompt and r.request_id in fact_of]
        targets = [RR.RewriteTarget(r.request_id, r.subject, r.answer, r.prompt, "cloze" if r.format == "cloze" else "qa")
                   for r in reqs]
        positives = {rid: [v.probe.prompt for v in vs if v.probe is not None] for rid, vs in views.items()}
        negs, neg_stats = RR.generate_hard_negatives(base.hf, ctx.renderer.tok, targets, positives=positives, log=ctx.log)
        by_req = {r.request_id: r for r in reqs}
        hard = {}
        for rid, xs in negs.items():
            r = by_req[rid]
            rendered = [sure_text(ctx.renderer, Probe(probe_id=f"hardneg/{rid}/{k}", role="retain", family="hardneg",
                                                      fact_id=r.fact_id, prompt=x, answer=r.answer,
                                                      format="cloze" if r.format == "cloze" else "qa"))
                        for k, x in enumerate(xs)]
            if any(rendered):
                hard[fact_of[rid]] = [{"prompt": x} for x in rendered if x]
        (prep / "router_hard_negatives.json").write_text(json.dumps(hard, indent=1, ensure_ascii=False) + "\n")
        return {"views": stats, "rewrites": len(rows), "hard_negatives": neg_stats}

    def fit_extra(self, prep):
        return ["--views", str(prep / "router_rewrites.json"),
                "--extra-negatives", str(prep / "router_hard_negatives.json")]


class OursL16(Ours):
    card = MethodCard("ours_l16", "detect_then_edit",
                      "ablation of ours: the same router and rows trainer, Δe at layer 16 in a second pass",
                      runtime_components=["router_hooks"], reference="this repository", decode="full")
    default_write_layer = 16


class OursEarly(Ours):
    card = MethodCard("ours_early", "detect_then_edit",
                      "ablation of ours: the same router recipe and rows trainer, with the classifier reading and Δe "
                      "added at the early layer (16) in one forward pass",
                      runtime_components=["router_hooks"], reference="this repository", decode="hf")
    default_layer = 16


def write_quality_report(router: Path, out: Path, inputs_report: dict, sample_facts: int = 2) -> dict:
    """router_quality.md: per-fact positive / negative counts, the rewrite and negative checks,
    and every positive and negative of a few facts, to read the training data itself."""
    rep_path, ds_path = router / "linear_router_report.json", router / "linear_router_dataset.json"
    if not rep_path.exists():
        return {}
    ds = json.loads(rep_path.read_text()).get("dataset", {})
    counts = ds.get("per_fact_counts") or {}
    quality = ds.get("negative_quality") or {}
    lines = ["# Router training data", "",
             f"Positives per fact: {counts.get('positives')}", f"Negatives per fact: {counts.get('negatives')}", ""]
    views = (inputs_report or {}).get("views")
    if views:
        lines += [f"Rewrites (views): {json.dumps({k: v for k, v in views.items()})}", ""]
    if quality:
        lines += [f"Negative checks: {json.dumps({k: v for k, v in quality.items() if k != 'examples'})}", ""]
        for reason, rows in (quality.get("examples") or {}).items():
            if rows:
                lines += [f"Dropped ({reason}):", ""] + [f"- `{json.dumps(r, ensure_ascii=False)}`" for r in rows[:6]] + [""]
    records = json.loads(ds_path.read_text()) if ds_path.exists() else []
    facts = sorted({r["owner_fact_id"] for r in records if r.get("kind") == "positive"})[:sample_facts]
    for fid in facts:
        pos = [r for r in records if r.get("owner_fact_id") == fid and r.get("kind") == "positive"]
        neg = [r for r in records if fid in (r.get("negative_for") or [])]
        lines += [f"## {fid}: {len(pos)} positives, {len(neg)} negatives", "", "| split | family | positive |", "|---|---|---|"]
        lines += [f"| {r['split']} | {r['group']} | {r['prompt'][-140:].replace('|', '/').replace(chr(10), ' ')} |" for r in pos]
        lines += ["", "| split | kind | negative |", "|---|---|---|"]
        lines += [f"| {r['split']} | {r['kind']} | {r['prompt'][-140:].replace('|', '/').replace(chr(10), ' ')} |" for r in neg]
        lines.append("")
    out.write_text("\n".join(lines) + "\n")
    return {"positives": counts.get("positives"), "negatives": counts.get("negatives"),
            "negative_checks": {k: v for k, v in quality.items() if k != "examples"}, "report": str(out)}
