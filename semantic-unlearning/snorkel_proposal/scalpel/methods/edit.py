"""Locate-and-edit baselines through the repository's own implementation.

    rome         rank-one edit per request (sequential)                (Meng et al., 2022)
    memit        one batched edit for all requests                     (Meng et al., 2023)
    zerounlearn  original ZeroUnlearn                                  (arXiv 2605.18879)

These call scripts/paper_baselines.py (apply_rome_memit / apply_zerounlearn) unchanged: the vendored
code in ../ZeroUnlearn, its hparams/<METHOD>/<model>.json, target_true -> EOS (ZeroUnlearn's safe
state), no retain requests. The only input is the split's requests, in the fact format those
functions read (canonical_prompt / subject / object). Covariance statistics come from
data/stats/<model>/<mom2_dataset>_stats (scripts/build_edit_covariance.py); without --method-arg
mom2_dataset=... the vendored 200-sentence fixture is used, on which ROME / MEMIT collapsed.

    --method rome --method-arg hparams=../ZeroUnlearn/hparams/ROME/Llama-3.2-3B-Instruct.json \
                  --method-arg mom2_dataset=wikipedia_20231101_100k --method-arg mom2_text=<corpus dir>
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from ..model_api import ModelHandle
from .base import Method, MethodCard, arg

SEMANTIC_ROOT = Path(__file__).resolve().parents[3]
ZERO_ROOT = SEMANTIC_ROOT.parent / "ZeroUnlearn"
HPARAM_DIR = {"rome": "ROME", "memit": "MEMIT", "zerounlearn": "ZeroUnlearn"}


def requests_as_facts(requests):
    """SCALPEL requests in the fact format of paper_baselines.zero_requests."""
    return [{"id": r.request_id, "canonical_prompt": r.prompt, "subject": r.subject, "object": r.answer}
            for r in requests if r.kind == "fact" and r.prompt and r.subject]


class _LocateEdit(Method):
    kind = "rome"

    def unlearn(self, base, requests, allowed_retain, ctx):
        a = self.args
        scripts = str(SEMANTIC_ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import paper_baselines as pb  # noqa: E402  (repository code)

        model_name = a.get("model_name") or Path(str(base.hf.config._name_or_path)).name
        hp = a.get("hparams") or str(ZERO_ROOT / "hparams" / HPARAM_DIR[self.kind] / f"{model_name}.json")
        if not Path(hp).is_file():
            raise FileNotFoundError(f"{self.kind} hparams not found: {hp} (pass --method-arg hparams=...)")
        args = SimpleNamespace(
            zero_root=str(a.get("zero_root", ZERO_ROOT)), zero_hparams=hp, rome_hparams=hp, memit_hparams=hp,
            mom2_dataset=a.get("mom2_dataset"), mom2_text=a.get("mom2_text"),
            allow_mom2_compute=arg(a, "allow_mom2_compute", False, bool),
            edit_target=a.get("edit_target", "eos"), baseline=self.kind)
        artifact = {"facts": requests_as_facts(requests)}
        model, tok = base.hf, base.tokenizer
        dtype = next(model.parameters()).dtype
        use_cache = getattr(model.config, "use_cache", True)
        torch.manual_seed(ctx.seed)
        model.float()                      # as paper_baselines.apply: edits are computed in fp32
        model.config.use_cache = False
        try:
            if self.kind == "zerounlearn":
                rep = pb.apply_zerounlearn(args, model, tok, artifact, ctx.seed)
            else:
                rep = pb.apply_rome_memit(args, model, tok, artifact, ctx.seed, self.kind)
        finally:
            model.requires_grad_(False)
            model.to(dtype=dtype).eval()
            model.config.use_cache = use_cache
        self._report = {**rep, "hparams": hp, "requests_given": len(artifact["facts"])}
        return ModelHandle(model=model, tokenizer=tok, hf=model)


class ROME(_LocateEdit):
    kind = "rome"
    card = MethodCard("rome", "locate_edit", "rank-one MLP edit per request, target_true -> EOS (vendored ROME)",
                      reference="Meng et al., 2022")


class MEMIT(_LocateEdit):
    kind = "memit"
    card = MethodCard("memit", "locate_edit", "batched multi-layer MLP edit, target_true -> EOS (vendored MEMIT)",
                      reference="Meng et al., 2023")


class ZeroUnlearn(_LocateEdit):
    kind = "zerounlearn"
    card = MethodCard("zerounlearn", "locate_edit", "original ZeroUnlearn, target_true -> EOS, no retain requests",
                      reference="arXiv 2605.18879")
