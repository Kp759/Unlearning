"""Method registry."""
from __future__ import annotations

from typing import Dict, Type

from .base import Method, MethodCard, UnlearnContext
from .edit import MEMIT, ROME, ZeroUnlearn
from .external import Checkpoint, Plugin, Sure
from .gradient import GA, GD, IDKFT, NPO, RMU, SimNPO, TaskVector
from .sure_paper import Ours, OursEarly, OursL16, SurePaper, SurePaperL16
from .sure_robust import PRESET_CLASSES as SURE_ROBUST_PRESETS
from .trivial import AnswerBan, IDKAll, KeywordRouterIDK, NoOp, PromptCacheIDK, SubjectBlocklistIDK

REGISTRY: Dict[str, Type[Method]] = {
    "noop": NoOp, "idk_all": IDKAll, "prompt_cache_idk": PromptCacheIDK,
    "subject_blocklist_idk": SubjectBlocklistIDK, "keyword_router_idk": KeywordRouterIDK, "answer_ban": AnswerBan,
    "ga": GA, "gd": GD, "npo": NPO, "simnpo": SimNPO, "idk_ft": IDKFT, "tv": TaskVector, "rmu": RMU,
    "rome": ROME, "memit": MEMIT, "zerounlearn": ZeroUnlearn,
    "checkpoint": Checkpoint, "plugin": Plugin, "sure": Sure,
    "sure_paper": SurePaper, "sure_paper_l16": SurePaperL16,      # the paper's edit on the plain router
    "ours": Ours, "ours_l16": OursL16, "ours_early": OursEarly,    # + SURE-R's router, 20/20 checked prompts
}
REGISTRY.update(SURE_ROBUST_PRESETS)   # sure_r, sure_r_noviews, sure_r_exact, sure_r_free, sure_r_idk, sure_r_bake
SURE_VARIANTS = ("sure", "ours", "ours_l16", "ours_early", "sure_paper", "sure_paper_l16") + tuple(SURE_ROBUST_PRESETS)
REFERENCE_CHEATS = ("noop", "idk_all", "prompt_cache_idk", "subject_blocklist_idk", "keyword_router_idk", "answer_ban")


def make(name: str, **kwargs) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
