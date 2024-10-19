import os
import re
from typing import List

import torch

from dataloaders.constants import TAG2TASK_LIST
from peft.tuners.molora import MoLoRAModel, AutoEncodersGate
from .clustering import cluster_and_merge_molora_experts, cluster_and_replace_hierarchical_routing
from .config import Config
from .manipulation import gather_lora_state_dicts_to_molora, gather_auto_encoder_state_dicts_to_molora


def hf_model(model_name_or_path, config, peft_type=None, model_class="", **kwargs):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    model_class = os.path.expandvars(model_class)

    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    model_class = {
        "": AutoModel,
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "seq_cls": AutoModelForSequenceClassification,
        "token_cls": AutoModelForTokenClassification,
        "qa": AutoModelForQuestionAnswering,
    }[model_class]
    load_torch_dtype = getattr(torch, config.load_model_dtype)
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": load_torch_dtype,
    }
    if config.auto_device_map:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cuda:0"
    model = model_class.from_pretrained(
        model_name_or_path,
        **load_kwargs
    )

    # gradient checkpointing should be enabled here before PEFT
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if peft_type is None:
        return model

    if peft_type.lower() == "lora":
        from peft import get_peft_model, LoraConfig

        peft_config = LoraConfig(
            target_modules=config.target_modules,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            init_lora_weights=config.init_lora_weights,
            use_rslora=config.use_rslora,
            inference_mode=False,
            bias=config.lora_bias,
        )
        model = get_peft_model(model, peft_config).to(dtype=load_torch_dtype)
        model.print_trainable_parameters()

    elif peft_type.lower() == "molora":
        from peft import MoLoRAConfig, get_peft_model

        if "moe_num_experts" not in kwargs:
            raise ValueError("moe_num_experts must be provided for MoLoRA model.")

        moe_num_experts = kwargs["moe_num_experts"]
        peft_config = MoLoRAConfig(
            target_modules=config.target_modules,
            lora_dim=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            moe_num_experts=moe_num_experts,
            top_k=config.moe_top_k,
            top_p=config.moe_top_p,
            reweight_output=config.moe_reweight_output,
        )
        model = get_peft_model(model, peft_config).to(dtype=load_torch_dtype)
        model.print_trainable_parameters()

    return model


def hf_tokenizer(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if model_name_or_path.startswith("EleutherAI/pythia"):
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    elif "mistral" in model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None

    test_tokens = tokenizer.build_inputs_with_special_tokens([-100])
    if test_tokens[0] != -100:
        tokenizer.bos_token_id = test_tokens[0]
    else:
        tokenizer.bos_token_id = None
    if test_tokens[-1] != -100:
        tokenizer.eos_token_id = test_tokens[-1]
    else:
        tokenizer.eos_token_id = None

    return tokenizer


def load_molora_from_distributed_lora_checkpoints(
        origin_model_name_or_path: str,
        config: Config,
        lora_checkpoint_path_list: List[str],
        task_name_list: List[str] = None,
) -> MoLoRAModel:
    """
    Load MoLoRA model from distributed LoRA checkpoints.

    Parameters
    ----------
    origin_model_name_or_path: str
        The original model name or path.
    config: Config
        The config object.
    lora_checkpoint_path_list: List[str]
        The list of LoRA checkpoint paths.
    task_name_list: List[str]
        The list of task names. Default is None.

    Returns
    -------
        The MoLoRA model.

    """
    if task_name_list is None:
        task_name_list = [path.split("/")[-1].split("_")[0] for path in lora_checkpoint_path_list]

    lora_state_dict_list = [
        torch.load(path, map_location="cpu") for path in lora_checkpoint_path_list
    ]
    num_experts = len(lora_state_dict_list)

    if config.cl_checkpoint_path:
        print(f"Loading CL statedict from {config.cl_checkpoint_path}")
        cl_state_dict = torch.load(config.cl_checkpoint_path, map_location="cpu")
        expert_idx = []
        for k in cl_state_dict.keys():
            if ("lora_A" in k) or ("lora_B" in k):
                expert_idx.append(
                    int(re.search(r"(?<=default\.)(\d+)(?=\.lora_(A|B))", k).group(1))
                )
        num_experts = max(expert_idx) + 1
    else:
        cl_state_dict = None

    molora_model = hf_model(
        origin_model_name_or_path,
        config,
        model_class=config.model_class,
        peft_type="molora",
        moe_num_experts=num_experts,
    )
    load_ret = molora_model.load_state_dict(
        gather_lora_state_dicts_to_molora(
            lora_state_dict_list, cl_state_dict, norm_router_weight=not config.arrow_routing
        ),
        strict=False,
    )

    if len(load_ret.unexpected_keys) > 0:
        raise ValueError(
            f"Some of the unexpected keys in the MoLoRA model are: {list(load_ret.unexpected_keys)[:16]}"
            f"\nWhile we expected them to be like: {list(molora_model.state_dict().keys())[:16]}"
        )

    if config.selected_expert_ids is not None:
        molora_model.filter_experts(kept_experts_ids=[int(idx) for idx in config.selected_expert_ids])

    molora_model.expert_name_list = task_name_list

    return molora_model


def load_and_apply_auto_encoder_checkpoints(
        model: MoLoRAModel,
        lora_checkpoint_path_list: List[str],
) -> MoLoRAModel:
    # Load
    lora_state_dict_list = [torch.load(path, map_location="cpu") for path in lora_checkpoint_path_list]
    ae_state_dict = gather_auto_encoder_state_dicts_to_molora(lora_state_dict_list)
    num_experts = len(lora_state_dict_list)

    # Apply
    for name, module in model.molora_linear_named_modules():
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype

        new_gate = AutoEncodersGate(in_features=module.in_features, num_experts=num_experts)
        module.lora_gate[module.adapter_name] = new_gate.to(device=device, dtype=dtype)

    load_ret = model.load_state_dict(ae_state_dict, strict=False)
    assert len(load_ret.unexpected_keys) == 0, (f"Unexpected keys: {load_ret.unexpected_keys[:16]}......"
                                                f"\nWe expected them to be like: {list(model.state_dict().keys())[:16]}")

    return model


def load_continual_learning_model(config):
    ae_path_list = None
    path_list = None
    if config.moe_inference or config.lora_hub_eval:
        # Load MoLoRA model from distributed LoRA checkpoints.
        assert os.path.isdir(config.checkpoint_dir_or_path), f"{config.checkpoint_dir_or_path} is not a directory."
        if config.init_datasets[0] in TAG2TASK_LIST.keys():
            config.init_datasets = (TAG2TASK_LIST[config.init_datasets[0]] + config.init_datasets[1:])

        path_list = []
        loaded_task_list = []
        file_list = sorted(os.listdir(config.checkpoint_dir_or_path))
        for file in file_list:
            dname = file.split("/")[-1].split("_")[0]
            if file.endswith("_best.pt") and dname in config.init_datasets:
                path_list.append(os.path.join(config.checkpoint_dir_or_path, file))
                loaded_task_list.append(dname)

    if config.moe_inference:

        print(
            f"Loading MoLoRA from distributed LoRA checkpoints in {config.checkpoint_dir_or_path} witn init datasets {config.init_datasets}."
        )

        model = load_molora_from_distributed_lora_checkpoints(
            origin_model_name_or_path=config.origin_model,
            config=config,
            lora_checkpoint_path_list=path_list,
        )
        # ).cuda()

        if config.merge_num_clusters is not None:
            before_merge_num_params = model.num_parameters()
            model = cluster_and_merge_molora_experts(
                model, config.merge_num_clusters, global_clustering=config.global_clustering
            )
            after_merge_num_params = model.num_parameters()
            print(f"Number of parameters before & after merging: {before_merge_num_params} -> {after_merge_num_params}")
        elif config.hierarchical_num_clusters is not None:
            before_merge_num_params = model.num_parameters()
            model = cluster_and_replace_hierarchical_routing(
                model, config.hierarchical_num_clusters,
                cluster_token_routing=config.hierarchical_cluster_token_routing,
                global_clustering=config.global_clustering
            )
            after_merge_num_params = model.num_parameters()
            print(f"Number of parameters before & after hierarchical routing: "
                  f"{before_merge_num_params} -> {after_merge_num_params}")
        elif config.ae_checkpoint_dir is not None:
            before_merge_num_params = model.num_parameters()
            ae_path_list = []
            ae_file_list = sorted(os.listdir(config.ae_checkpoint_dir))
            for file in ae_file_list:
                dname = file.split("/")[-1].split("_")[0]
                if file.endswith("_auto_encoder_last.pt") and dname in config.init_datasets:
                    ae_path_list.append(os.path.join(config.ae_checkpoint_dir, file))
            model = load_and_apply_auto_encoder_checkpoints(model, ae_path_list)
            after_merge_num_params = model.num_parameters()
            print(f"Number of parameters before & after applying auto-encoder: "
                  f"{before_merge_num_params} -> {after_merge_num_params}")

        if config.bias_router_embedding_path is not None:
            print("Loading the bias router embedding......")
            task_embedding_dict = torch.load(config.bias_router_embedding_path)
            task_embedding_dict = {
                name: embed.cuda()[:config.bias_routing_dim] for name, embed in task_embedding_dict.items()
            }
            model.load_bias_router_embedding(
                torch.stack([task_embedding_dict[dname] for dname in loaded_task_list])
            )

            if config.bias_input_embedding_path is not None:
                print("Using different router & input embedding for instruction-auxiliary routing.")
                task_input_embedding_dict = torch.load(config.bias_input_embedding_path)
                task_input_embedding_dict = {
                    name: embed.cuda()[:config.bias_routing_dim] for name, embed in task_input_embedding_dict.items()
                }
                model.load_task_embedding_dict(task_input_embedding_dict)
            else:
                model.load_task_embedding_dict(task_embedding_dict)

            model.set_bias_routing_scale(config.bias_routing_scale)
            print("The bias router embedding has been loaded but not enabled yet.")

    else:
        # Load the model from the checkpoint.
        print(f"Loading LoRA from {config.checkpoint_dir_or_path}.")

        # model = hf_model(config.origin_model, config, config.peft_type, config.model_class).cuda()
        model = hf_model(config.origin_model, config, config.peft_type, config.model_class).cuda()

        if config.checkpoint_dir_or_path is not None and os.path.isfile(config.checkpoint_dir_or_path):
            load_results = model.load_state_dict(torch.load(config.checkpoint_dir_or_path), strict=False)
            assert len(load_results.unexpected_keys) == 0, f"Unexpected keys: {load_results.unexpected_keys}"
        else:
            print("No checkpoint is loaded.")

    return {
        "model": model,
        "path_list": path_list,
        "ae_path_list": ae_path_list,
    }
