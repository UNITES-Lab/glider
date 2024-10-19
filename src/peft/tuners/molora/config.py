# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/12/11
from dataclasses import dataclass
from typing import Optional, Union, List

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MoLoRAConfig(PeftConfig):
    """
    Configuration class for LoRA and MoLE.

    Parameters
    ----------
    lora_dim: int
        The dimension of the LoRA layer.
    lora_alpha: int
        The scaling factor for the LoRA layer.
    lora_dropout: float
        The dropout rate for the LoRA layer.
    moe_num_experts: int
        The number of experts in the MoLE layer.
    """

    bias: Optional[str] = "none"
    target_modules: Optional[Union[List[str], str]] = None
    layers_to_transform: Optional[Union[List, int]] = None
    layers_pattern: Optional[str] = None
    lora_dim: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.0
    moe_num_experts: Optional[int] = 8
    moe_router_aux_loss_coef: Optional[float] = 0.0
    modules_to_save: Optional[List[str]] = None
    top_k: Optional[int] = 2
    top_p: Optional[float] = 1.0
    reweight_output: Optional[bool] = True

    def __post_init__(self):
        self.peft_type = PeftType.MOLORA
        assert self.bias == "none", "Only support lora_bias == none now."
