# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/12
import torch
from sklearn.cluster import SpectralClustering
from torch import nn
from tqdm import tqdm

from peft.tuners.molora import MoLoRAModel, MoLoRALinearDense, HierarchicalGate

EPSILON = 1e-5


def compute_global_expert_clustering(
        model: MoLoRAModel, num_clusters: int
):
    global_pair_similarity = []
    for name, module in tqdm(model.molora_linear_named_modules(), desc="Global Clustering"):
        # Group!
        gate_weight = module.lora_gate[module.adapter_name].weight.data
        pair_similarity = torch.matmul(
            gate_weight, gate_weight.t()) / (torch.norm(
            gate_weight, dim=-1).unsqueeze(-1) * torch.norm(gate_weight, dim=-1).unsqueeze(-1).t() + EPSILON)
        pair_similarity = (pair_similarity + 1) / 2
        global_pair_similarity.append(pair_similarity)
    global_pair_similarity = torch.stack(global_pair_similarity, dim=0).mean(dim=0)
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0
    ).fit(global_pair_similarity.cpu())
    global_cluster_labels = torch.from_numpy(clustering.labels_)
    return global_cluster_labels


@torch.no_grad()
def cluster_and_merge_molora_experts(
        model: MoLoRAModel,
        num_clusters: int,
        global_clustering: bool
) -> MoLoRAModel:
    if global_clustering:
        global_cluster_labels = compute_global_expert_clustering(model, num_clusters)
    else:
        global_cluster_labels = None

    for name, module in tqdm(model.molora_linear_named_modules(), desc="Clustering and Merging Experts"):
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype

        experts = module.lora_experts[module.adapter_name]
        router = module.lora_gate[module.adapter_name]
        new_experts = nn.ModuleList()
        new_router = nn.Linear(module.in_features, num_clusters)
        gate_weight = router.weight.data

        # Group!
        if global_cluster_labels is None:
            pair_similarity = torch.matmul(
                gate_weight, gate_weight.t()) / (torch.norm(
                gate_weight, dim=-1).unsqueeze(-1) * torch.norm(gate_weight, dim=-1).unsqueeze(-1).t() + EPSILON)
            pair_similarity = (pair_similarity + 1) / 2
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0
            ).fit(pair_similarity.cpu())
            cluster_labels = torch.from_numpy(clustering.labels_)
        else:
            cluster_labels = global_cluster_labels

        # Merge!
        for i, label in enumerate(cluster_labels.unique()):
            expert_indices = torch.where(cluster_labels == label)[0]
            lora_A_weight_list = torch.stack(
                [experts[expert_idx].lora_A.weight for expert_idx in expert_indices], dim=0
            )
            lora_B_weight_list = torch.stack(
                [experts[expert_idx].lora_B.weight for expert_idx in expert_indices], dim=0
            )
            lora_A_weight = lora_A_weight_list.mean(dim=0).to(device=device, dtype=dtype)
            lora_B_weight = lora_B_weight_list.mean(dim=0).to(device=device, dtype=dtype)
            router_weight = gate_weight[expert_indices].mean(dim=0).to(device=device, dtype=dtype)

            merged_expert = MoLoRALinearDense(
                module.in_features,
                module.out_features,
                module.lora_dim,
                module.lora_alpha,
                module.lora_dropout,
                reset_parameters=False
            ).to(device=device, dtype=dtype)
            merged_expert.lora_A.weight.data.copy_(lora_A_weight)
            merged_expert.lora_B.weight.data.copy_(lora_B_weight)

            new_experts.append(merged_expert)
            new_router.weight.data[i] = router_weight

        module.lora_experts[module.adapter_name] = new_experts
        module.lora_gate[module.adapter_name] = new_router.to(device=device, dtype=dtype)

        module.moe_num_experts = num_clusters
        module.reset_routing_counter()

    return model


@torch.no_grad()
def cluster_and_replace_hierarchical_routing(
        model: MoLoRAModel,
        num_clusters: int,
        cluster_token_routing: bool,
        global_clustering: bool
) -> MoLoRAModel:
    if global_clustering:
        global_cluster_labels = compute_global_expert_clustering(model, num_clusters)
    else:
        global_cluster_labels = None

    for name, module in tqdm(model.molora_linear_named_modules(),
                             desc="Clustering and Replacing w/ Hierarchical Gates"):
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype

        router = module.lora_gate[module.adapter_name]

        # Group!
        if global_cluster_labels is None:
            gate_weight = router.weight.data
            pair_similarity = torch.matmul(
                gate_weight, gate_weight.t()) / (torch.norm(
                gate_weight, dim=-1).unsqueeze(-1) * torch.norm(gate_weight, dim=-1).unsqueeze(-1).t() + EPSILON)
            pair_similarity = (pair_similarity + 1) / 2
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0
            ).fit(pair_similarity.cpu())
            cluster_labels = torch.from_numpy(clustering.labels_)
        else:
            cluster_labels = global_cluster_labels

        new_gate = HierarchicalGate.from_linear_gate(
            router, cluster_labels.tolist(), cluster_token_routing=cluster_token_routing
        )
        module.lora_gate[module.adapter_name] = new_gate.to(device=device, dtype=dtype)

    return model
