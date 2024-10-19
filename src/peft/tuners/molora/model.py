# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/12/11
import re
from copy import deepcopy
from typing import Union, Optional, Dict, List

import torch
from torch import nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)
from .config import MoLoRAConfig
from .layers import MoLoRALinearLayer


def load_balancing_loss_func(
        gate_logits: torch.Tensor,
        num_experts: torch.Tensor = None,
        top_k=2,
        attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, float]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )
    else:
        compute_device = gate_logits.device
        concatenated_gate_logits = gate_logits.reshape(-1, gate_logits.shape[-1])

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
                batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MoLoRAModel(BaseTuner):
    prefix: str = "lora_"

    def __init__(
            self,
            model,
            config,
            adapter_name: str = "default",
    ):
        super().__init__(model, config, adapter_name)
        self.peft_config = config
        self.model = model
        self.init_adapter_name = adapter_name

        self.expert_name_list = []

        self.router_aux_loss_coef = config[adapter_name].moe_router_aux_loss_coef

        self.task_embedding_dict = None
        self.task_input_embedding_dict = None
        self.bias_router = None
        self.routing_bias = None
        self.bias_routing_scale = 0.

        # self._init_lora_model(adapter_name, self.peft_config[adapter_name])
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def get_expert_name_list(self):
        return deepcopy(self.expert_name_list)

    def _init_linear_layer_lora(
            self, adapter_name: str, layer: nn.Linear
    ) -> MoLoRALinearLayer:
        config = self.peft_config[adapter_name]
        layer = MoLoRALinearLayer(
            adapter_name,
            layer.in_features,
            layer.out_features,
            bias=(layer.bias is not None),
            lora_dim=config.lora_dim,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            moe_num_experts=config.moe_num_experts,
            top_k=config.top_k,
            top_p=config.top_p,
            reweight_output=config.reweight_output,
        )
        return layer

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                    model_config["model_type"]
                    not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    @staticmethod
    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        if hasattr(old_module, "bias") and old_module.bias is not None:
            new_module.bias = old_module.bias
        else:
            new_module.bias = None

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    @staticmethod
    def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
        for n, p in model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        if bias == "none":
            return
        elif bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in model.modules():
                if (
                        isinstance(m, LoraLayer)
                        and hasattr(m, "bias")
                        and m.bias is not None
                ):
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError

    @staticmethod
    def _check_target_module_exists(lora_config: MoLoRAConfig, key: str):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(
                key.endswith(target_key) for target_key in lora_config.target_modules
            )
            is_using_layer_indexes = (
                    getattr(lora_config, "layers_to_transform", None) is not None
            )
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = (
                    COMMON_LAYERS_PATTERN
                    if layer_indexing_pattern is None
                    else layer_indexing_pattern
                )
                layers_pattern = (
                    [layers_pattern]
                    if isinstance(layers_pattern, str)
                    else layers_pattern
                )

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = (
                                    layer_index == lora_config.layers_to_transform
                            )
                        else:
                            target_module_found = (
                                    layer_index in lora_config.layers_to_transform
                            )

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def molora_linear_named_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, MoLoRALinearLayer):
                yield name, module

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        original_loss = outputs.loss if hasattr(outputs, "loss") else None

        attention_mask = kwargs.get("attention_mask", None)

        if original_loss is not None:
            adapter_name = self.init_adapter_name
            # gather z_loss and aux_loss
            all_router_logits = []

            for name, module in self.molora_linear_named_modules():
                all_router_logits.append(
                    module.last_router_logits.reshape(
                        -1, module.last_router_logits.shape[-1]
                    )
                )

            all_router_logits = torch.cat(all_router_logits, dim=0)

            aux_loss = load_balancing_loss_func(
                all_router_logits,
                num_experts=self.peft_config[adapter_name].moe_num_experts,
                top_k=self.peft_config[adapter_name].top_k,
                attention_mask=attention_mask,
            )

            loss = original_loss + aux_loss * self.router_aux_loss_coef

            outputs.loss = loss
            outputs.aux_loss = aux_loss
            outputs.lm_loss = original_loss

        return outputs

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                    model_config["model_type"]
                    not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _create_and_replace(
            self,
            molora_config,
            adapter_name,
            target,
            target_name,
            parent,
            **optional_kwargs,
    ):
        assert isinstance(
            target, nn.Linear
        ), f"Target module must be a nn.Linear, but got {target}"
        new_module = self._init_linear_layer_lora(adapter_name, target)
        self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if (
                            isinstance(m, MoLoRALinearLayer)
                            and hasattr(m, "bias")
                            and m.bias is not None
                    ):
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(
                    f"Requested bias: {bias}, is not implemented."
                )

    def enable_only_last_expert(self, gate_only_last_expert: bool):
        for name, module in self.molora_linear_named_modules():
            module.only_last_expert = True
            module.gate_only_last_expert = gate_only_last_expert

    def disable_only_last_expert(self):
        for name, module in self.molora_linear_named_modules():
            module.only_last_expert = False
            module.gate_only_last_expert = False

    @torch.no_grad()
    def reset_routing_counter(self):
        for name, module in self.molora_linear_named_modules():
            module.reset_routing_counter()

    @torch.no_grad()
    def reset_gates_to_zero(self):
        for name, module in self.molora_linear_named_modules():
            module.reset_gates_to_zero()

    @torch.no_grad()
    def get_routing_counter_state_dict(
            self, normalize: bool = False
    ) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, module in self.molora_linear_named_modules():
            routing_stat = module.routing_counter.clone()
            if normalize:
                routing_stat = routing_stat / routing_stat.sum()
            state_dict[name] = routing_stat

        return state_dict

    @torch.no_grad()
    def apply_layer_norm_to_gate(self):
        for name, module in self.molora_linear_named_modules():
            module.apply_layer_norm_to_gate()

    @torch.no_grad()
    def fake_freeze_first_k_gate_rows(self, k: int):
        for name, module in self.molora_linear_named_modules():
            module.fake_freeze_first_k_gate_rows(k)

    @torch.no_grad()
    def fake_unfreeze_gate_rows(self):
        for name, module in self.molora_linear_named_modules():
            module.fake_unfreeze_gate_rows()

    @torch.no_grad()
    def freeze_first_k_experts(self, k: int):
        for name, module in self.molora_linear_named_modules():
            module.freeze_first_k_experts(k)

    @torch.no_grad()
    def unfreeze_experts(self):
        for name, module in self.molora_linear_named_modules():
            module.unfreeze_experts()

    @torch.no_grad()
    def filter_experts(self, kept_experts_ids: List[int]):
        kept_experts_ids = sorted(kept_experts_ids)
        for name, module in self.molora_linear_named_modules():
            module.filter_experts(kept_experts_ids)

    @torch.no_grad()
    def gate_vector_similarity_dict(self):
        similarity_dict = dict()
        for name, module in self.molora_linear_named_modules():
            lora_gate_weight = module.lora_gate[module.adapter_name].weight.data
            similarity_dict[name] = torch.matmul(lora_gate_weight, lora_gate_weight.t()
                                                 ) / (torch.norm(lora_gate_weight, dim=-1).unsqueeze(-1) * torch.norm(
                lora_gate_weight, dim=-1).unsqueeze(-1).t())

        return similarity_dict

    def set_routing_counter_per_sequence(self, enable: bool):
        for name, module in self.molora_linear_named_modules():
            module.enable_routing_counter_per_sequence = enable
            module.sample_routing_count_cache = torch.zeros(0, module.moe_num_experts, dtype=torch.long, device="cpu")

    @torch.no_grad()
    def get_routing_counter_state_dict_by_sequence(
            self, sequence_indices: List[int], normalize: bool = False
    ) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, module in self.molora_linear_named_modules():
            sequence_routing_count_cache = module.sample_routing_count_cache[sequence_indices]
            routing_stat = sequence_routing_count_cache.sum(dim=0)
            if normalize:
                routing_stat = routing_stat / routing_stat.sum()
            state_dict[name] = routing_stat

        return state_dict

    def router_weight_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, module in self.molora_linear_named_modules():
            lora_gate_weight = module.lora_gate[module.adapter_name].weight.data
            state_dict[name] = lora_gate_weight

        return state_dict

    @property
    def is_bias_routing(self):
        return self.bias_router is not None

    def load_bias_router_embedding(self, bias_router_embed: torch.Tensor):
        self.bias_router = bias_router_embed

    def load_task_embedding_dict(self, embedding_dict: Dict[str, torch.Tensor]):
        self.task_embedding_dict = deepcopy(embedding_dict)

    def set_bias_routing_scale(self, scale: float):
        self.bias_routing_scale = scale
        for _, module in self.molora_linear_named_modules():
            module.bias_routing_scale = scale

    def update_routing_bias(
            self, bias_token_embed: torch.Tensor = None, bias_token_task: str = None
    ):
        # todo: handle issue when expand expert number
        if bias_token_embed is None and bias_token_task is None:
            raise KeyError("Either provide token embedding or token task name")

        if bias_token_embed is None:
            bias_token_embed = self.task_embedding_dict[bias_token_task]

        with torch.no_grad():
            self.routing_bias = nn.functional.cosine_similarity(self.bias_router, bias_token_embed)

            for _, module in self.molora_linear_named_modules():
                module.routing_bias = self.routing_bias

    @torch.no_grad()
    def lora_weight_similarity_dict(self):
        similarity_dict = dict()
        for name, module in tqdm(self.molora_linear_named_modules(), desc="Computing weight similarity"):
            lora_module_list = module.lora_experts[module.adapter_name]
            num_experts = len(lora_module_list)
            lora_A_similarity_matrix = torch.zeros(num_experts, num_experts)
            lora_B_similarity_matrix = torch.zeros(num_experts, num_experts)
            for i in range(num_experts):
                for j in range(num_experts):
                    lora_A_similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
                        lora_module_list[i].lora_A.weight.data.view(-1),
                        lora_module_list[j].lora_A.weight.data.view(-1),
                        dim=0
                    )
                    lora_B_similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
                        lora_module_list[i].lora_B.weight.data.view(-1),
                        lora_module_list[j].lora_B.weight.data.view(-1),
                        dim=0
                    )

            similarity_dict[name] = (lora_A_similarity_matrix, lora_B_similarity_matrix)

        return similarity_dict
