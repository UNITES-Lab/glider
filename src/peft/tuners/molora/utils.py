# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/1/3
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union, List

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from peft.tuners.molora import MoLoRAModel


@dataclass
class BaseMoEModelOutputWithPast(BaseModelOutputWithPast):
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@torch.no_grad()
def compute_average_lora_weights(
        lora_a_weights: torch.Tensor,
        lora_b_weights: torch.Tensor,
        lora_router_weights: torch.Tensor,
) -> Tuple:
    """
    Compute the average LoRA weights from the LoRA weights of all experts.

    Parameters
    ----------
    lora_a_weights: torch.Tensor
        The stacked LoRA A weights of all selected experts, of shape (k, rank, in_features)
    lora_b_weights: torch.Tensor
        The stacked LoRA B weights of all selected experts, of shape (k, out_features, rank)
    lora_router_weights: torch.Tensor
        The LoRA router weights of all selected experts, of shape (k, num_experts)

    Returns
    -------
    lora_a_weight: torch.Tensor
        The average LoRA A weight, of shape (rank, in_features)
    lora_b_weight: torch.Tensor
        The average LoRA B weight, of shape (out_features, rank)
    lora_router_weight: torch.Tensor
        The average LoRA router weight, of shape (num_experts,)

    Examples
    --------
    >>> lora_a_weights = torch.randn(2, 4, 32)
    >>> lora_b_weights = torch.randn(2, 64, 4)
    >>> lora_router_weights = torch.randn(2, 4)
    >>> lora_a_weight, lora_b_weight, lora_router_weight = compute_average_lora_weights(
    ...     lora_a_weights, lora_b_weights, lora_router_weights
    ... )
    >>> lora_a_weight.shape
    torch.Size([4, 32])
    >>> lora_b_weight.shape
    torch.Size([64, 4])
    >>> lora_router_weight.shape
    torch.Size([4])
    """
    average_weights = torch.mean(
        torch.einsum("kor,kri->koi", lora_b_weights, lora_a_weights), dim=0
    )
    lora_b_weight, s, lora_a_weight = torch.pca_lowrank(
        average_weights,
        q=lora_a_weights.shape[1],
        center=False,
    )
    s = torch.sqrt(s)
    lora_b_weight = lora_b_weight @ torch.diag(s)
    lora_a_weight = torch.diag(s) @ lora_a_weight.t()

    return lora_a_weight, lora_b_weight, lora_router_weights.mean(dim=0)


@torch.no_grad()
def extend_molora_experts(
        model: MoLoRAModel,
        new_expert_task_names: Union[str, List[str]],
        num_new_experts: Optional[int] = 1,
        lora_init_method: Optional[str] = "random",
        gate_init_method: Optional[str] = "random",
        usage_statistics: Optional[Dict[str, torch.Tensor]] = None,
) -> MoLoRAModel:
    """
    Examples
    --------
    >>> from peft import get_peft_model, MoLoRAConfig
    >>> from transformers import OPTModel
    >>> config = MoLoRAConfig(moe_num_experts=4)
    >>> model = get_peft_model(OPTModel.from_pretrained("facebook/opt-125m"), config)
    >>> model = extend_molora_experts(model, num_new_experts=2)
    >>> model(input_ids=torch.tensor([[1, 2, 3]])).last_hidden_state.shape
    torch.Size([1, 3, 768])
    """
    assert num_new_experts > 0, "num_new_experts must be a positive integer."
    if lora_init_method in ["usage-based", "top-2-used", "top-1-used"]:
        assert (
                usage_statistics is not None
        ), "usage_statistics must be provided when lora_init_method is not random."

    if isinstance(new_expert_task_names, str):
        new_expert_task_names = [new_expert_task_names]
    assert len(
        new_expert_task_names
    ) == num_new_experts, "The length of new_expert_task_names must be equal to num_new_experts."

    for name, module in model.molora_linear_named_modules():
        experts = module.lora_experts[module.adapter_name]
        device = experts[0].lora_A.weight.device
        dtype = experts[0].lora_A.weight.dtype
        if usage_statistics:
            expert_usage = usage_statistics[name].to(device=device, dtype=dtype)

        # Inititalize lora parameters for new expert
        if lora_init_method == "random":
            init_lora_A_weight = None
            init_lora_B_weight = None
        elif lora_init_method == "usage-based":
            assert (
                    num_new_experts == 1
            ), "num_new_experts must be 1 when lora_init_method is usage-based."
            if name not in usage_statistics:
                raise ValueError(f"Usage statistics for {name} is not provided.")

            lora_A_weights = torch.stack(
                [exp.lora_A.weight.data.clone() for exp in experts], dim=0
            )
            lora_B_weights = torch.stack(
                [exp.lora_B.weight.data.clone() for exp in experts], dim=0
            )
            # init by usaged-based-weighted average
            init_lora_A_weight = (
                    torch.einsum("k, kor->or", expert_usage, lora_A_weights)
                    / expert_usage.sum()
            )
            init_lora_B_weight = (
                    torch.einsum("k, kri->ri", expert_usage, lora_B_weights)
                    / expert_usage.sum()
            )
        elif lora_init_method == "top-2-used":
            assert (
                    num_new_experts == 1
            ), "num_new_experts must be 1 when lora_init_method is top-2-used."
            if name not in usage_statistics:
                raise ValueError(f"Usage statistics for {name} is not provided.")

            top_2_index = torch.topk(expert_usage, k=2, dim=-1).indices
            init_lora_A_weight = torch.stack(
                [experts[i].lora_A.weight.data.clone() for i in top_2_index], dim=0
            ).mean(dim=0)
            init_lora_B_weight = torch.stack(
                [experts[i].lora_B.weight.data.clone() for i in top_2_index], dim=0
            ).mean(dim=0)
        elif lora_init_method == "top-1-used":
            assert (
                    num_new_experts == 1
            ), "num_new_experts must be 1 when lora_init_method is top-1-used."
            if name not in usage_statistics:
                raise ValueError(f"Usage statistics for {name} is not provided.")

            top_1_index = torch.argmax(expert_usage, dim=-1)
            init_lora_A_weight = experts[top_1_index].lora_A.weight.data.clone()
            init_lora_B_weight = experts[top_1_index].lora_B.weight.data.clone()
        else:
            raise NotImplementedError(
                f"lora_init_method {lora_init_method} not implemented yet."
            )

        # Inititalize gate parameters for new expert
        if gate_init_method == "random":
            init_lora_router_weight = None
        elif gate_init_method == "zero":
            init_lora_router_weight = torch.zeros(
                (
                    num_new_experts,
                    module.lora_gate[module.adapter_name].weight.shape[1],
                ),
            ).to(device=module.lora_gate[module.adapter_name].weight.device, dtype=dtype)
        elif gate_init_method == "usage-based":
            lora_router_weights = module.lora_gate[
                module.adapter_name
            ].weight.data.clone()  # of shape (num_experts, d_model)
            init_lora_router_weight = (
                    torch.einsum("k, ki->i", expert_usage, lora_router_weights) / expert_usage.sum()
            )
        elif gate_init_method == "top-2-used":
            top_2_index = torch.topk(expert_usage, k=2, dim=-1).indices
            lora_router_weights = (
                module.lora_gate[module.adapter_name].weight.data[top_2_index, :].clone()
            )
            init_lora_router_weight = lora_router_weights.mean(dim=0).repeat(num_new_experts, 1)
        elif gate_init_method == "top-1-used":
            top_1_index = torch.argmax(expert_usage, dim=-1)
            init_lora_router_weight = module.lora_gate[module.adapter_name].weight.data[top_1_index, :].clone()
        else:
            raise NotImplementedError(
                f"gate_init_method {gate_init_method} not implemented yet."
            )

        module.add_new_expert_(
            num_new_experts=num_new_experts,
            init_weight_lora_A=init_lora_A_weight,
            init_weight_lora_B=init_lora_B_weight,
            init_weight_router_new_rows=init_lora_router_weight,
        )

    model.expert_name_list += new_expert_task_names

    # update config
    model.active_peft_config.moe_num_experts += num_new_experts

    return model


@torch.no_grad()
def free_least_used_experts(
        model: MoLoRAModel, expert_usage_dict: Dict[str, torch.Tensor]
) -> MoLoRAModel:
    """
    Make the least used expert parameters trainable, this does not make sure other experts are frozen.
    """
    for name, module in model.molora_linear_named_modules():
        previous_expert_usage = expert_usage_dict[
            f"base_model.model.{name}.lora_router.{module.adapter_name}"
        ]
        least_used_index = torch.argmin(previous_expert_usage, dim=-1).item()
        for param in module.lora_experts[module.adapter_name][
            least_used_index
        ].parameters():
            param.requires_grad = True
    return model


@torch.no_grad()
def freeze_first_k_experts_(
        model: MoLoRAModel,
        k: int,
) -> MoLoRAModel:
    assert k > 0, "k must be a positive integer."
    assert (
            k <= model.peft_config["default"].moe_num_experts
    ), "k must be no larger than the number of experts."
    for name, module in model.molora_linear_named_modules():
        module.freeze_first_k_experts_(k=k)
    return model
