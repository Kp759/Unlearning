from dataclasses import dataclass
from typing import Literal

from util.hparams import HyperParams


@dataclass
class ZeroUnlearnEmbHeadHyperParams(HyperParams):
    model_name: str
    num_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    max_grad_norm: float
    forget_loss_weight: float
    retain_loss_weight: float
    l2_weight: float
    forget_target: str = "target_new"
    retain_target: str = "target_true"
    update_scope: Literal["all", "touched_rows"] = "all"
