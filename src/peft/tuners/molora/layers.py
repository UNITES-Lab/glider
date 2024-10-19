# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/12/11
import math
from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F


def _auto_encoder(in_features, intermediate_dim):
    return nn.Sequential(
        nn.Linear(in_features, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, in_features)
    )


class AutoEncodersGate(nn.Module):
    def __init__(
            self, in_features: int, num_experts: int, intermediate_dim: int = 128, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.num_experts = num_experts

        self.auto_encoders = nn.ModuleList([
            _auto_encoder(in_features, intermediate_dim) for _ in range(num_experts)
        ])
        self.auto_encode_loss_fn = nn.MSELoss(reduction="none")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.reshape(-1, hidden_dim)  # (batch_size * sequence_length, hidden_dim)
        auto_encoder_loss = torch.stack([
            self.auto_encode_loss_fn(hidden_states, ae(hidden_states)).mean(dim=-1)
            for ae in self.auto_encoders
        ])  # (num_experts, batch_size * sequence_length)
        auto_encoder_loss = auto_encoder_loss.transpose(0, 1)  # (batch_size * sequence_length, num_experts)

        routing_logits = - auto_encoder_loss.reshape(batch_size, sequence_length, self.num_experts)
        # (batch_size, sequence_length, num_experts)

        return routing_logits


class HierarchicalGate(nn.Module):
    def __init__(
            self, in_features: int, cluster_expert_id_list: List[List[int]], cluster_token_routing: bool
    ):
        """
        A hierarchical gate that routes hidden states to different experts based on the cluster gate and expert gates.
            - cluster gate does first level routing to clusters with sentence-level top-1 routing
            - expert gates do second level routing to experts within the clusters with token-level top-k routing

        Parameters
        ----------
        in_features: int
            The number of input features
        cluster_expert_id_list: List[List[int]]
            A list of expert indices for each cluster
        cluster_token_routing: bool
            Whether to use token-level routing or sentence-level routing for the first level routing between clusters

        Examples
        --------
        >>> gate = HierarchicalGate(8, [[0, 1], [2, 3], [3, 4]], cluster_token_routing=False)
        >>> hidden_states = torch.randn(2, 3, 8)
        >>> routing_logits = gate(hidden_states)
        >>> routing_weights = F.softmax(routing_logits, dim=-1)
        >>> (routing_weights < 1e-5).sum().item() # Each token has only two non-zero routing weights for the clustering
        18
        """
        super().__init__()
        self.in_features = in_features
        self.num_clusters = len(cluster_expert_id_list)
        self.num_total_experts = len(set(
            expert_id for expert_list in cluster_expert_id_list for expert_id in expert_list
        ))
        self.cluster_token_routing = cluster_token_routing

        self.register_buffer("clusters", torch.arange(self.num_clusters, dtype=torch.long))
        for cluster, expert_list in enumerate(cluster_expert_id_list):
            self.register_buffer(f"cluster_{cluster}_expert_id", torch.tensor(expert_list, dtype=torch.long))

        self.cluster_gate = nn.Linear(in_features, self.num_clusters, bias=False)
        # todo: compute cluster gate similarity
        self.expert_gates = nn.ModuleList([
            nn.Linear(in_features, len(expert_list), bias=False) for expert_list in cluster_expert_id_list
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.cluster_token_routing:
            hidden_states = hidden_states.reshape(-1, hidden_dim)  # (batch_size * sequence_length, hidden_dim)
            cluster_logits = self.cluster_gate(hidden_states)  # (batch_size * sequence_length, num_clusters)
            cluster_weights = F.softmax(cluster_logits, dim=-1)  # (batch_size * sequence_length, num_clusters)

            cluster_index = cluster_weights.argmax(dim=-1)  # (batch_size * sequence_length,)
            cluster_index = F.one_hot(cluster_index, num_classes=self.num_clusters)
            # (batch_size * sequence_length, num_clusters)
        else:
            sentence_embedding = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            hidden_states = hidden_states.reshape(-1, hidden_dim)  # (batch_size * sequence_length, hidden_dim)
            cluster_logits = self.cluster_gate(sentence_embedding)  # (batch_size, num_clusters)
            cluster_weights = F.softmax(cluster_logits, dim=-1)  # (batch_size, num_clusters)

            cluster_index = cluster_weights.argmax(dim=-1)  # (batch_size,)
            cluster_index = F.one_hot(cluster_index, num_classes=self.num_clusters)  # (batch_size, num_clusters)
            cluster_index = cluster_index.unsqueeze(1).expand(
                batch_size, sequence_length, -1).reshape(-1, self.num_clusters)
            # (batch_size * sequence_length, num_clusters)

        routing_logits = torch.ones(
            batch_size * sequence_length, self.num_total_experts, device=hidden_states.device, dtype=hidden_states.dtype
        ) * torch.finfo(hidden_states.dtype).min  # (batch_size * sequence_length, num_total_experts)
        for idx, gate in enumerate(self.expert_gates):
            # Calculate local expert logits for each token
            token_indices = cluster_index[:, idx].bool()  # (batch_size * sequence_length,)
            if token_indices.sum() == 0:
                continue
            local_routing_logits = gate(hidden_states[token_indices])
            # (batch_size * sequence_length, num_cluster_experts)

            # Map local routing logits to global routing logits
            expert_indices = getattr(self, f"cluster_{idx}_expert_id")  # (num_cluster_experts,)
            token_routing_indices = torch.nonzero(token_indices).squeeze(-1)
            token_expert_indices = (
                token_routing_indices.repeat_interleave(len(expert_indices)),
                expert_indices.repeat(len(token_routing_indices))
            )
            routing_logits[token_expert_indices] = local_routing_logits.flatten()

        routing_logits = routing_logits.reshape(batch_size, sequence_length, self.num_total_experts)

        return routing_logits

    @classmethod
    def from_linear_gate(
            cls, linear_gate: nn.Linear, expert_cluster_labels: List[int], *args, **kwargs
    ):
        """
        Examples
        --------
        >>> linear_gate = nn.Linear(8, 6)
        >>> expert_labels = [0, 0, 0, 1, 1, 1]
        >>> router = HierarchicalGate.from_linear_gate(linear_gate, expert_cluster_labels)
        >>> hidden_states = torch.randn(2, 3, 8)
        >>> routing_weights = F.softmax(router(hidden_states), dim=-1)
        >>> (routing_weights < 1e-5).sum().item() # Each token has only three non-zero routing weights for the clustering
        18
        """
        in_features = linear_gate.in_features
        cluster_expert_id_list = []
        cluster_expert_weights_list = []
        expert_cluster_labels = torch.tensor(expert_cluster_labels)

        for label in expert_cluster_labels.unique():
            expert_indices = torch.where(expert_cluster_labels == label)[0]
            cluster_expert_weights_list.append(linear_gate.weight.data[expert_indices])
            cluster_expert_id_list.append(expert_indices.tolist())

        gate = cls(in_features, cluster_expert_id_list, *args, **kwargs)

        # Copy the weights
        for cluster, expert_weights in enumerate(cluster_expert_weights_list):
            gate.cluster_gate.weight.data[cluster] = expert_weights.mean(dim=0)
            gate.expert_gates[cluster].weight.data.copy_(expert_weights)

        return gate


class MoLoRALinearDense(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dim: int,
            lora_alpha: int,
            lora_dropout: float,
            reset_parameters: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_alpha = lora_alpha
        self.lora_dim = lora_dim
        self.scaling = (
                self.lora_alpha / self.lora_dim
        )  # py: phatgoose use 1/r as scaling here so set alpha=1.
        self.lora_dropout = lora_dropout

        self.lora_A = nn.Linear(in_features, lora_dim, bias=False)
        self.lora_B = nn.Linear(lora_dim, out_features, bias=False)
        self.lora_dropout = nn.Dropout(lora_dropout)

        if reset_parameters:
            self._reset_lora_parameters()

    def _reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, hidden_states: torch.Tensor):
        return self.lora_B(self.lora_A(self.lora_dropout(hidden_states))) * self.scaling

    @property
    def dtype(self):
        return self.lora_A.weight.dtype

    def get_embedding(self):
        return self.lora_B.weight.data @ self.lora_A.weight.data * self.scaling


class MoLoRALinearLayer(nn.Linear):
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            top_k: int,
            top_p: float,
            reweight_output: bool,
            lora_dim: int,
            lora_alpha: int,
            lora_dropout: float,
            moe_num_experts: int,
            same_init: Optional[bool] = True,
            only_last_expert: Optional[bool] = False,
            arrow_routing: Optional[bool] = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        adapter_name: str
            The name of the adapter, always "default" by design
        in_features: int
            The input dimension of the linear layer
        out_features: int
            The output dimension of the linear layer
        top_k: int
            The number of experts to route to
        top_p: float
            If set to a value between 0 and 1,only the smallest set of experts whose cumulative probability mass exceeds
                top_p will be routed to
        lora_dim: int
            The dimension of the Lora layer
        lora_alpha: int
            The alpha parameter of the Lora layer
        lora_dropout: float
            The dropout rate of the Lora layer
        moe_num_experts: int
            The number of experts in the Mixture of Experts
        same_init: bool, optional
            Whether to initialize the weights of all experts the same, by default True
        only_last_expert: bool, optional
            Whether to use only the last expert, by default False
        arrow_routing: bool, optional
            Whether to use Arrow routing, by default False
        kwargs: dict
            Additional keyword arguments for the parent class
        """
        super().__init__(in_features, out_features, **kwargs)
        self.top_k = top_k
        self.top_p = top_p
        self.reweight_output = reweight_output
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.moe_num_experts = moe_num_experts
        self.adapter_name = adapter_name

        self.last_router_logits = None
        self.routing_counter = torch.zeros(moe_num_experts, dtype=torch.long, device="cpu")

        self.sample_routing_count_cache = torch.zeros(0, moe_num_experts, dtype=torch.long, device="cpu")
        self.enable_routing_counter_per_sequence = False

        self.routing_bias = None
        self.bias_routing_scale = 0.

        self.lora_experts = nn.ModuleDict(
            {
                adapter_name: nn.ModuleList(
                    MoLoRALinearDense(
                        in_features,
                        out_features,
                        self.lora_dim,
                        self.lora_alpha,
                        self.lora_dropout,
                        reset_parameters=True,
                    )
                    for _ in range(self.moe_num_experts)
                )
            }
        )
        self.lora_gate = nn.ModuleDict({adapter_name: nn.Linear(in_features, self.moe_num_experts, bias=False)})
        self.router_input_layer_norm = nn.LayerNorm(in_features, elementwise_affine=False)

        if same_init:
            self.sync_expert_weights()

        nn.Linear.reset_parameters(self)

        self.only_last_expert = only_last_expert
        self.gate_only_last_expert = False
        self.freeze_first_gate_rows = 0
        self.first_gate_rows_cache = None
        self.min_experts_to_keep = 1

        self.arrow_routing = arrow_routing

    @torch.no_grad()
    def reset_routing_counter(self):
        self.sample_routing_count_cache = torch.zeros(0, self.moe_num_experts, dtype=torch.long, device="cpu")
        self.routing_counter = torch.zeros(self.moe_num_experts, dtype=torch.long, device="cpu")

    def sync_expert_weights(self):
        for i in range(1, self.moe_num_experts):
            self.lora_experts[self.adapter_name][i].lora_A.weight.data.copy_(
                self.lora_experts[self.adapter_name][0].lora_A.weight.data
            )
            self.lora_experts[self.adapter_name][i].lora_B.weight.data.copy_(
                self.lora_experts[self.adapter_name][0].lora_B.weight.data
            )

    def forward(self, hidden_states: torch.Tensor):
        # pzli: I basically copied them from PHATGOOSE, though I did some renaming
        adapter_name = self.adapter_name
        ret = F.linear(hidden_states, self.weight, bias=self.bias)  # (bsz, seq, d_out)

        if self.freeze_first_gate_rows > 0:
            with torch.no_grad():
                self.lora_gate[adapter_name].weight.data[:self.freeze_first_gate_rows].copy_(self.first_gate_rows_cache)

        if self.only_last_expert:
            lora_output = self.lora_experts[adapter_name][-1](hidden_states)
            if self.gate_only_last_expert:
                gate_scores = torch.sum(hidden_states * self.lora_gate[adapter_name].weight[-1], dim=-1)
                # (batch_size, sequence_length)
                gate_scores = torch.sigmoid(gate_scores)
                lora_output = torch.einsum("bse,bs->bse", lora_output, gate_scores)
            ret += lora_output
            return ret

        batch_size, sequence_length, hidden_dim = hidden_states.shape

        is_auto_encoder_routing = isinstance(self.lora_gate[adapter_name], AutoEncodersGate)

        if is_auto_encoder_routing:
            router_logits = self.lora_gate[adapter_name](hidden_states)  # (bsz, seq, n_exp)
        elif self.arrow_routing:
            router_logits = self.lora_gate[adapter_name](hidden_states)
            temperature = 10
            router_logits = torch.abs(router_logits / temperature)
        else:
            router_logits = self.lora_gate[adapter_name](self.router_input_layer_norm(hidden_states))
            router_logits = router_logits * math.sqrt(1 / self.in_features)  # (bsz, seq, n_exp)

        hidden_states = hidden_states.reshape(-1, hidden_dim)  # (bsz * seq, d_in)

        if self.routing_bias is not None:
            self.routing_bias = self.routing_bias.to(router_logits.device)
            router_logits = router_logits + self.routing_bias.expand(*router_logits.shape) * self.bias_routing_scale
            # print("DEBUG", router_logits.norm(), self.routing_bias.norm())

        if self.top_p < 1.0:
            assert self.top_p > 0.0, "top_p must be in (0, 1)"
            assert not is_auto_encoder_routing, "AutoEncoderGate does not support top-p routing."

            sorted_logits, sorted_indices = torch.sort(router_logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            sorted_indices_to_remove[..., -self.min_experts_to_keep:] = False

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits_processed = router_logits.masked_fill(indices_to_remove, -float("inf"))

            padded_weights = F.softmax(logits_processed, dim=-1, dtype=torch.float)

            # why ain't they masked? duno, let's just be safe
            padded_weights = padded_weights.masked_fill(indices_to_remove, 0).reshape(-1, self.moe_num_experts)
            # (bsz * seq, n_exp) (sparse)

        else:
            if is_auto_encoder_routing:
                routing_weights = router_logits
            else:
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)  # (bsz, seq, n_exp)
            routing_weights = routing_weights.reshape(-1, self.moe_num_experts)  # (bsz * seq, n_exp)


            assert (routing_weights.shape[0] == hidden_states.shape[
                0]), f"{routing_weights.shape} & {hidden_states.shape}"
            topk_weights, topk_indices = torch.topk(routing_weights, k=self.top_k, dim=-1)  # both (bsz * seq, top_k)

            if is_auto_encoder_routing:
                topk_weights = torch.ones_like(topk_weights) / self.top_k
            else:
                topk_weights = topk_weights / (torch.sum(topk_weights, dim=-1, keepdim=True) + 1e-6)

            padded_weights = torch.zeros_like(routing_weights).scatter(1, topk_indices, topk_weights)
            # (bsz * seq, n_exp) (sparse)

        sorted_experts, index_sorted_experts = torch.nonzero(padded_weights).sort(0)  # (bsz * seq * top_k, 2)

        _, expert_index = sorted_experts.split(1, dim=1)  # (bsz * seq * top_k, 1)
        batch_seq_index = torch.nonzero(padded_weights)[index_sorted_experts[:, 1], 0]  # (bsz * seq * top_k,)
        part_sizes = (padded_weights > 0).sum(0).tolist()  # List[int] of length n_exp

        padded_weights_exp = padded_weights[batch_seq_index.flatten()]  # (bsz * seq * top_k, n_exp)
        nonzero_weights_exp = torch.gather(padded_weights_exp, 1, expert_index)  # (bsz * seq * top_k, 1)
        hidden_states_exp = hidden_states[batch_seq_index].squeeze(1)  # (bsz * seq * top_k, d_in)
        expert_inputs_list = torch.split(hidden_states_exp, part_sizes, dim=0)  # List[torch.Tensor], of length n_exp

        expert_outputs_list = []
        for expert_idx in range(self.moe_num_experts):
            expert_layer = self.lora_experts[adapter_name][expert_idx]
            expert_input = expert_inputs_list[expert_idx]
            expert_output = expert_layer(expert_input)
            expert_outputs_list.append(expert_output)

        output_hidden_states_exp = torch.cat(expert_outputs_list, 0)  # (bsz * seq * top_k, d_out)
        if self.reweight_output:
            output_hidden_states_exp = output_hidden_states_exp.mul(nonzero_weights_exp)  # (bsz * seq * top_k, d_out)

        output_hidden_states = torch.zeros(
            padded_weights.size(0),
            expert_outputs_list[-1].size(1),
            device=output_hidden_states_exp.device,
            dtype=output_hidden_states_exp.dtype,
        )  # (bsz * seq, d_out)
        output_hidden_states = output_hidden_states.index_add(0, batch_seq_index, output_hidden_states_exp)
        output_hidden_states = output_hidden_states.reshape(batch_size, sequence_length, self.out_features)
        # (bsz, seq, d_out)

        ret += output_hidden_states

        self.last_router_logits = router_logits

        with torch.no_grad():
            routing_indexes = expert_index.squeeze()
            self.routing_counter += (
                F.one_hot(routing_indexes, num_classes=self.moe_num_experts).sum(dim=0).cpu()
            )
            if self.enable_routing_counter_per_sequence:
                routing_counter_per_sequence = (padded_weights.reshape(batch_size, sequence_length, -1) > 0).sum(dim=1)
                self.sample_routing_count_cache = torch.cat([
                    self.sample_routing_count_cache, routing_counter_per_sequence.cpu()
                ])

        return ret

    @torch.no_grad()
    def add_new_expert_(
            self,
            num_new_experts: int,
            init_weight_lora_A: Optional[torch.Tensor] = None,
            init_weight_lora_B: Optional[torch.Tensor] = None,
            init_weight_router_new_rows: Optional[torch.Tensor] = None,
    ):
        molora_experts = self.lora_experts[self.adapter_name]
        random_init = (init_weight_lora_A is None) or (init_weight_lora_B is None)

        # update router
        self.moe_num_experts += num_new_experts
        self.routing_counter = torch.cat([
            self.routing_counter, torch.zeros(num_new_experts, dtype=torch.long, device="cpu")
        ])
        old_lora_gate_weight = self.lora_gate[self.adapter_name].weight.data
        device = old_lora_gate_weight.device
        dtype = old_lora_gate_weight.dtype
        self.lora_gate[self.adapter_name] = nn.Linear(
            self.in_features, self.moe_num_experts, bias=False,
            dtype=dtype, device=device
        )
        self.lora_gate[self.adapter_name].weight.data[:-num_new_experts, :].copy_(
            old_lora_gate_weight
        )
        if init_weight_router_new_rows is not None:
            init_weight_router_new_rows = init_weight_router_new_rows.reshape(
                -1, self.in_features
            )
            assert init_weight_router_new_rows.shape == (
                num_new_experts,
                self.in_features,
            ), f"Shape mismatch of shape {init_weight_router_new_rows.shape} and {num_new_experts, self.in_features}."
            self.lora_gate[self.adapter_name].weight.data[-num_new_experts:, :].copy_(
                init_weight_router_new_rows
            )
        # update experts
        for i in range(len(molora_experts), len(molora_experts) + num_new_experts):
            molora_experts.append(
                MoLoRALinearDense(
                    self.in_features,
                    self.out_features,
                    self.lora_dim,
                    self.lora_alpha,
                    self.lora_dropout,
                    reset_parameters=random_init,
                ).to(dtype=dtype, device=device)
            )
            if init_weight_lora_A is not None:
                molora_experts[i].lora_A.weight.data.copy_(init_weight_lora_A)
            if init_weight_lora_B is not None:
                molora_experts[i].lora_B.weight.data.copy_(init_weight_lora_B)

        self.reset_routing_counter()

    @torch.no_grad()
    def freeze_first_k_experts_(self, k: int):
        for i in range(k):
            self.lora_experts[self.adapter_name][i].lora_A.weight.requires_grad = False
            self.lora_experts[self.adapter_name][i].lora_B.weight.requires_grad = False

    @torch.no_grad()
    def reset_gates_to_zero(self):
        self.lora_gate[self.adapter_name].weight.zero_()

    @torch.no_grad()
    def apply_layer_norm_to_gate(self):
        ln = nn.LayerNorm(self.in_features, elementwise_affine=False)
        self.lora_gate[self.adapter_name].weight.data.copy_(
            ln(self.lora_gate[self.adapter_name].weight.data)
        )

    @torch.no_grad()
    def fake_freeze_first_k_gate_rows(self, k: int):
        self.freeze_first_gate_rows = k
        self.first_gate_rows_cache = self.lora_gate[self.adapter_name].weight.data[:k].detach().clone()

    @torch.no_grad()
    def fake_unfreeze_gate_rows(self):
        self.freeze_first_gate_rows = 0
        self.first_gate_rows_cache = None

    @torch.no_grad()
    def freeze_first_k_experts(self, k: int):
        for i in range(k):
            self.lora_experts[self.adapter_name][i].lora_A.weight.requires_grad = False
            self.lora_experts[self.adapter_name][i].lora_B.weight.requires_grad = False

    @torch.no_grad()
    def unfreeze_experts(self):
        for expert in self.lora_experts[self.adapter_name]:
            expert.lora_A.weight.requires_grad = True
            expert.lora_B.weight.requires_grad = True

    @torch.no_grad()
    def filter_experts(self, kept_experts_ids: List[int]):
        molora_experts = self.lora_experts[self.adapter_name]

        # update router
        self.moe_num_experts = len(kept_experts_ids)
        self.routing_counter = torch.zeros(self.moe_num_experts, dtype=torch.long, device="cpu")
        old_lora_gate_weight = self.lora_gate[self.adapter_name].weight.data
        device = old_lora_gate_weight.device
        dtype = old_lora_gate_weight.dtype
        self.lora_gate[self.adapter_name] = nn.Linear(
            self.in_features, self.moe_num_experts, bias=False, dtype=dtype, device=device
        )
        self.lora_gate[self.adapter_name].weight.data.copy_(old_lora_gate_weight[kept_experts_ids, :])

        # update experts
        self.lora_experts[self.adapter_name] = nn.ModuleList([molora_experts[i] for i in kept_experts_ids])

        self.reset_routing_counter()
