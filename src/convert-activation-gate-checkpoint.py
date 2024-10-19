# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/15
import argparse
import os
import subprocess
import sys
from functools import partial

import datasets
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloaders import P3CLDataModule
from dataloaders.constants import TAG2TASK_LIST
from peft.tuners.lora import LoraLayer
from trainers.evaluator import Evaluator
from trainers.interface_mixin import InterfaceMixin
from utils.config import Config, ParseKwargs
from utils.get_model import hf_model
from utils.get_model import hf_tokenizer
from utils.util import get_logger, setup_wandb_logger

sys.path.insert(1, os.getcwd())
datasets.disable_progress_bar()


def evaluate(config, task_list, model, tokenizer, datamodule, loggers, split):
    evaluator = Evaluator(
        config=config,
        eval_tasks=task_list,
        tokenizer=tokenizer,
        datamodule=datamodule,
        loggers=loggers,
    )
    eval_results = evaluator.eval_all(model, split=split)
    return eval_results


def main(config, loggers):
    assert config.peft_type == "lora", "This script is only for LoRA."

    model = hf_model(
        config.origin_model, config, config.peft_type, config.model_class
    ).cuda()

    tokenizer = hf_tokenizer(config.origin_model)
    model.interface = InterfaceMixin(model_type=config.model_type)

    if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
        config.dataset = TAG2TASK_LIST[config.dataset[0]]
    all_tasks = config.dataset

    datamodule = P3CLDataModule(config, tokenizer, loggers, is_moe=False, max_examples_per_dataset=1000)

    for task_idx, train_task in enumerate(all_tasks):

        loggers["logger"].info(
            f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dumping {train_task} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n"
        )

        state_dict = torch.load(os.path.join(config.checkpoint_dir_or_path, f"{train_task}_best.pt"),
                                map_location="cpu")
        load_res = model.load_state_dict(state_dict, strict=False)
        assert len(load_res.unexpected_keys) == 0, f"Unexpected keys: {load_res.unexpected_keys}"

        average_activations = {}
        activations_counter = {}

        def act_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            sum_tensor = tensor.sum(0).cpu()
            if name not in average_activations:
                average_activations[name] = sum_tensor
                activations_counter[name] = tensor.shape[0]
            else:
                average_activations[name] += sum_tensor
                activations_counter[name] += tensor.shape[0]

        def act_input_hook(module, input, output, name):
            if isinstance(input, tuple):
                input = input[0]
            act_tensor(name, input)

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, LoraLayer):
                hooks.append(
                    m.register_forward_hook(partial(act_input_hook, name=name))
                )

        task_datamodule = datamodule(train_task)
        dataloader = task_datamodule.get_train_dataloader()

        model.eval()
        for batch_idx, batch_inputs in enumerate(
                tqdm(dataloader, desc=f"Inference on {task_datamodule.data_tag}...")
        ):
            with torch.no_grad():
                model.interface.__call__(
                    batch_inputs,
                    task_datamodule.dataset["train"].interface_info,
                    model,
                    tokenizer,
                )

        for hook in hooks:
            hook.remove()

        for name in average_activations:
            average_act = average_activations[name] / activations_counter[name]
            state_dict[f"{name}.lora_gate.default"] = average_act

        torch.save(state_dict, os.path.join(config.checkpoint_dir, f"{train_task}_best.pt"))


def main_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)

    log_config = os.path.join(config.project_dir, "utils/")
    logger = get_logger("log.txt", f"{config.log_dir}/", log_config)

    logger.info(f"Start experiment {config.project_name}/{config.name}")
    logger.info(config.to_json())

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loggers = {"logger": logger}
    if not config.debug:
        loggers["tb"] = SummaryWriter(config.run_output_dir)
        loggers["wandb"], _, _ = setup_wandb_logger(config.__dict__)
        loggers["wandb"].log(
            {"command": subprocess.list2cmdline(["python"] + sys.argv)}
        )

    return config, loggers


#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.lora_B.default.weight'
#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.base_layer.weight'

if __name__ == "__main__":
    config, loggers = main_setup()
    main(config, loggers)

# free up gpu memory.
# ps aux |grep "main.py" |awk '{print $2}' |xargs kill
