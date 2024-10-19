# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/3

from dataclasses import dataclass
from enum import Enum

from torch import nn

from peft import MoLoRAModel


class TuningStrategy(Enum):
    LORA = "lora"
    ROUTER = "router"
    LAST_EXPERT_LORA = (
        "last_expert_lora"  # only use last expert, without using its gate
    )
    LAST_EXPERT_GATE = "last_expert_gate"  # only use last expert, using its gate and only train the gate


def make_model_trainable_parameters(model: MoLoRAModel, strategy) -> MoLoRAModel:
    """
    Make the model trainable based on the strategy.
        - LORA:
            Only the MoLORA parameters are trainable. NOT just use the last expert.

        - ROUTER:
            Only the router parameters are trainable. NOT just use the last expert.

        - LAST_EXPERT_LORA:
            All parameters are trainable. Just use the last expert. NOT using its gate.

        - LAST_EXPERT_GATE:
            Only the last expert's gate parameters are trainable.
            Just use the last expert. Using its gate and only train the gate.
    """
    if strategy == TuningStrategy.LORA:
        model.disable_only_last_expert()
        model = set_only_lora_trainable(model)
    elif strategy == TuningStrategy.ROUTER:
        model.disable_only_last_expert()
        model = set_only_gate_trainable(model)
    elif strategy == TuningStrategy.LAST_EXPERT_LORA:
        model.enable_only_last_expert(gate_only_last_expert=False)
        model = set_only_lora_trainable(model)
    elif strategy == TuningStrategy.LAST_EXPERT_GATE:
        model.enable_only_last_expert(gate_only_last_expert=True)
        model = set_only_gate_trainable(model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Model trainable parameters are set to {strategy}.")
    model.print_trainable_parameters()
    return model


def set_only_lora_trainable(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def set_only_gate_trainable(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if "lora_gate" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


@dataclass
class TrainingParameterManager:
    """
    Functionalities:

    1. Decide how to change the trainable parameters of the model, based on {strategy, results, model arch}.

    2. Be able to read/write the results. todo: need to discuss

    """

    strategy: str
    model: nn.Module
