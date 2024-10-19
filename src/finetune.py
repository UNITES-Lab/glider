import argparse
import datetime
import json
import os
import subprocess
import sys
from collections import OrderedDict

import datasets
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders import P3CLDataModule
from dataloaders.constants import P3_DATASET_CONFIGS, TAG2TASK_LIST
from trainers.evaluator import Evaluator
from trainers.interface_mixin import InterfaceMixin
from trainers.trainer import Trainer
from utils.config import Config, ParseKwargs
from utils.get_model import hf_model, hf_tokenizer
from utils.manipulation import load_molora_from_distributed_lora_checkpoints
from utils.util import get_logger, setup_wandb_logger
from trainers import training_state
sys.path.insert(1, os.getcwd())
datasets.disable_progress_bar()


def main(config, loggers):
    start = datetime.datetime.now()

    is_molora = config.peft_type == "molora"

    if is_molora:
        assert os.path.isdir(
            config.checkpoint_dir_or_path
        ), f"{config.checkpoint_dir_or_path} is not a directory."
        print(
            f"Loading MoLoRA from distributed LoRA checkpoints in {config.checkpoint_dir_or_path}."
        )
        path_list = [
            os.path.join(config.checkpoint_dir_or_path, file)
            for file in os.listdir(config.checkpoint_dir_or_path)
            if file.endswith("_best.pt")
        ]
        model = load_molora_from_distributed_lora_checkpoints(
            origin_model_name_or_path=config.origin_model,
            config=config,
            lora_checkpoint_path_list=path_list,
        ).cuda()
    else:
        model = hf_model(
            config.origin_model, config, config.peft_type, config.model_class
        ).cuda()

    tokenizer = hf_tokenizer(config.origin_model)
    model.interface = InterfaceMixin(model_type=config.model_type)

    if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
        config.dataset = TAG2TASK_LIST[config.dataset[0]]
    all_tasks = config.dataset
    dump_dict = OrderedDict()
    dump_dict["tasks"] = all_tasks

    datamodule = P3CLDataModule(config, tokenizer, loggers)

    val_results_dict = OrderedDict()
    for task_idx, train_task in enumerate(all_tasks):
        trainer = Trainer(
            config=config,
            task=train_task,
            model=model,
            tokenizer=tokenizer,
            datamodule=datamodule,
            loggers=loggers,
        )
        trainer.train_task(train_task)

        evaluator = Evaluator(
            config=config,
            eval_tasks=all_tasks[: min(task_idx + 2, len(all_tasks) + 1)],
            tokenizer=tokenizer,
            datamodule=datamodule,
            loggers=loggers,
        )
        val_results = evaluator.eval_all(model, split="val")
        val_results_dict[train_task] = val_results

        loggers["logger"].info(val_results)
        if "wandb" in loggers.keys():
            loggers["wandb"].log(
                {
                    f"{task}_val_{m}": f"{v:.4f}"
                    for task, metrics in val_results.items()
                    for m, v in metrics.items()
                }, step=training_state.global_training_step
            )

    final_evaluator = Evaluator(
        config=config,
        eval_tasks=all_tasks,
        tokenizer=tokenizer,
        datamodule=datamodule,
        loggers=loggers,
    )
    test_results = final_evaluator.eval_all(model, split="test")
    dump_dict["val_results"] = val_results_dict
    dump_dict["test_results"] = test_results

    loggers["logger"].info(test_results)
    if "wandb" in loggers.keys():
        loggers["wandb"].log(
            {
                f"{task}_test_{m}": f"{v:.4f}"
                for task, metrics in test_results.items()
                for m, v in metrics.items()
            }, step=training_state.global_training_step
        )

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
        # loggers["wandb"].log(
        #     {"command": subprocess.list2cmdline(["python"] + sys.argv)}
        # )
        loggers["wandb"].summary["command"] = subprocess.list2cmdline(["python"] + sys.argv)

    training_state.global_training_step = 0

    return config, loggers


#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.lora_B.default.weight'
#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.base_layer.weight'

if __name__ == "__main__":
    config, loggers = main_setup()
    main(config, loggers)

# free up gpu memory.
# ps aux |grep "main.py" |awk '{print $2}' |xargs kill
