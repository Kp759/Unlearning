from dataclasses import dataclass
from typing import Any

from util.hparams import HyperParams


@dataclass
class MENDHyperParams(HyperParams):
    model_name: str = ""


class MendRewriteExecutor:
    def apply_to_model(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("MEND is not available in this checkout.")
