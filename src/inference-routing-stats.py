import argparse
import datetime
import json
import os
import subprocess
import sys
from collections import OrderedDict

sys.path.insert(1, os.getcwd())
import datasets
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders import P3CLDataModule
from dataloaders.constants import P3_DATASET_CONFIGS, TAG2TASK_LIST
from trainers.evaluator import Evaluator
from trainers.interface_mixin import InterfaceMixin
from utils.config import Config, ParseKwargs
from utils.get_model import hf_tokenizer, load_continual_learning_model
from utils.util import get_logger, setup_wandb_logger

datasets.disable_progress_bar()


def main(config, loggers):
    start = datetime.datetime.now()

    model = load_continual_learning_model(config)["model"]

    print(model.expert_name_list)

    lora_weight_similarity = model.lora_weight_similarity_dict()
    torch.save(lora_weight_similarity, os.path.join(config.run_output_dir, "lora_weight_similarity.pt"))
    print("lora_weight_similarity saved")
    gate_similarity = model.gate_vector_similarity_dict()
    torch.save(gate_similarity, os.path.join(config.run_output_dir, "gate_similarity.pt"))

    tokenizer = hf_tokenizer(config.origin_model)
    model.interface = InterfaceMixin(model_type=config.model_type)

    if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
        config.dataset = TAG2TASK_LIST[config.dataset[0]]

    all_tasks = config.dataset
    dump_dict = OrderedDict()
    dump_dict["tasks"] = all_tasks

    datamodule = P3CLDataModule(
        config, tokenizer, loggers, is_moe=config.moe_inference, stage=config.eval_split
    )

    final_evaluator = Evaluator(
        config=config,
        eval_tasks=all_tasks,
        tokenizer=tokenizer,
        datamodule=datamodule,
        loggers=loggers,
    )
    results = final_evaluator.eval_all(
        model, split=config.eval_split,
        return_routing_stats=True, verbose=False, return_routing_state_dict=True,
        return_correct_incorrect_routing_state_dict=False
    )

    task_routing_state_dict = {}
    task_correct_routing_state_dict = {}
    task_incorrect_routing_state_dict = {}

    for task_key in results:
        if "routing_state_dict" in results[task_key]:
            task_routing_state_dict[task_key] = results[task_key].pop("routing_state_dict")
        if "correct_routing_state_dict" in results[task_key]:
            task_correct_routing_state_dict[task_key] = results[task_key].pop("correct_routing_state_dict")
        if "incorrect_routing_state_dict" in results[task_key]:
            task_incorrect_routing_state_dict[task_key] = results[task_key].pop("incorrect_routing_state_dict")

    if len(task_routing_state_dict) > 0:
        for task_key in task_routing_state_dict:
            torch.save(
                task_routing_state_dict[task_key],
                os.path.join(config.run_output_dir, f"{task_key}_routing_state_dict.pt"),
            )
    if len(task_correct_routing_state_dict) > 0:
        for task_key in task_correct_routing_state_dict:
            torch.save(
                task_correct_routing_state_dict[task_key],
                os.path.join(config.run_output_dir, f"{task_key}_correct_routing_state_dict.pt"),
            )
    if len(task_incorrect_routing_state_dict) > 0:
        for task_key in task_incorrect_routing_state_dict:
            torch.save(
                task_incorrect_routing_state_dict[task_key],
                os.path.join(config.run_output_dir, f"{task_key}_incorrect_routing_state_dict.pt"),
            )

    dump_dict[f"{config.eval_split}_results"] = results

    config_dict = vars(config)
    del config_dict["device"]
    dump_dict["config"] = config_dict
    dataset_configs = OrderedDict()
    for data_tag in all_tasks:
        dataset_configs[data_tag] = P3_DATASET_CONFIGS[data_tag]
    dump_dict["dataset_configs"] = dataset_configs

    with open(os.path.join(config.run_output_dir, "metrics.json"), "w") as file:
        json.dump(dump_dict, file, indent=4)

    end = datetime.datetime.now()
    loggers["logger"].info(
        f"\nTook {(end - start) // datetime.timedelta(hours=1)} hours {(end - start) // datetime.timedelta(minutes=1)} minutes."
    )


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


if __name__ == "__main__":
    config, loggers = main_setup()
    main(config, loggers)

# free up gpu memory.
# ps aux |grep "main.py" |awk '{print $2}' |xargs kill
