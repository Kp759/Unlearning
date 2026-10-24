"""Method registry."""
from __future__ import annotations

from typing import Dict, Type

from .base import Method, MethodCard, UnlearnContext
from .external import Checkpoint, Plugin, Sure
from .gradient import GA, GD, NPO, RMU
from .trivial import AnswerBan, IDKAll, KeywordRouterIDK, NoOp, PromptCacheIDK, SubjectBlocklistIDK

REGISTRY: Dict[str, Type[Method]] = {
    "noop": NoOp, "idk_all": IDKAll, "prompt_cache_idk": PromptCacheIDK,
    "subject_blocklist_idk": SubjectBlocklistIDK, "keyword_router_idk": KeywordRouterIDK, "answer_ban": AnswerBan,
    "ga": GA, "gd": GD, "npo": NPO, "rmu": RMU,
    "checkpoint": Checkpoint, "plugin": Plugin, "sure": Sure,
}
REFERENCE_CHEATS = ("noop", "idk_all", "prompt_cache_idk", "subject_blocklist_idk", "keyword_router_idk", "answer_ban")


def make(name: str, **kwargs) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
