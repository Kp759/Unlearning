from dataclasses import dataclass

from util.hparams import HyperParams


@dataclass
class NSEHyperParams(HyperParams):
    model_name: str = ""
