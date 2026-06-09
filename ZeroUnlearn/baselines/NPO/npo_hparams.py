from dataclasses import dataclass
from typing import List, Optional

from util.hparams import HyperParams


@dataclass
class NPOHyperParams(HyperParams):
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    batch_size: int = 64
    wd_power_law: tuple = None

    beta: float = 0.1
    npo_coeff: float = 1.0
    grad_diff_coeff: float = 1.0
    forget_loss: str = "npo_grad_diff"

    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
