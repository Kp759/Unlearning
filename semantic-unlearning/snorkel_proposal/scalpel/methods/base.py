"""The contract between the benchmark and an unlearning method.

A method receives ONLY:
    requests        the deletion requests (direct cloze prompt + answer, or an entity name)
    allowed_retain  generic retain text from subjects that never appear in evaluation
    ctx             seed, device, renderer (to format text the way the evaluator will),
                    generic prompts for self-generated retain data, a wall-clock budget
and returns a ModelHandle. The evaluator never gives a method any probe, paraphrase,
neighbour, or base-model score.

A method declares a MethodCard. `runtime_components` lists anything that is not a weight
change (hooks, routers, logit filters, retrieval). Such methods are scored in the
behavioural and shipped-weights tiers and, separately, in the strict tier where only the
weights are kept (handle.strip()).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..model_api import ModelHandle, Renderer
from ..schema import Probe, Request, RetainExample


@dataclass
class MethodCard:
    name: str
    family: str                 # trivial | gradient | representation | locate_edit | detect_then_edit | external
    description: str
    runtime_components: List[str] = field(default_factory=list)
    uses_allowed_retain: bool = False
    reference: str = ""
    is_reference_cheat: bool = False   # a red-team baseline the benchmark must defeat
    decode: str = "hf"          # "full": the method needs the prefix-recompute decoder (e.g. a fixed
                                # request boundary); the base reference is then generated the same way

    def to_dict(self):
        return asdict(self)


@dataclass
class UnlearnContext:
    seed: int
    renderer: Renderer
    generic_prompts: Sequence[str] = ()
    max_seconds: float = 4 * 3600
    log: Callable[[str], None] = print
    started: float = field(default_factory=time.time)

    def over_budget(self) -> bool:
        return time.time() - self.started > self.max_seconds


class Method:
    card: MethodCard

    def __init__(self, **kwargs: Any):
        self.args = dict(kwargs)

    def unlearn(self, base: ModelHandle, requests: Sequence[Request], allowed_retain: Sequence[RetainExample],
                ctx: UnlearnContext) -> ModelHandle:
        raise NotImplementedError

    def unlearn_sequential(self, current: ModelHandle, new_requests: Sequence[Request],
                           history: Sequence[Request], allowed_retain: Sequence[RetainExample],
                           ctx: UnlearnContext) -> ModelHandle:
        """Deletion requests arriving over time. Default: apply unlearn() to the current model
        with the new batch only. Methods that keep state (a bank, a router) override this."""
        return self.unlearn(current, new_requests, allowed_retain, ctx)

    def report(self) -> Dict[str, Any]:
        return getattr(self, "_report", {})


def request_probe(r: Request) -> Probe:
    """A request in probe form, so methods tokenize it exactly like the evaluator."""
    return Probe(probe_id=f"request/{r.request_id}", role="forget", family="direct", fact_id=r.fact_id,
                 prompt=r.prompt, answer=r.answer, format=r.format, request_id=r.request_id,
                 subject=r.subject, relation=r.relation)


def arg(d: Dict[str, Any], key: str, default, cast=None):
    v = d.get(key, default)
    if cast is not None and v is not None:
        if cast is bool and isinstance(v, str):
            return v.lower() in ("1", "true", "yes")
        return cast(v)
    return v
