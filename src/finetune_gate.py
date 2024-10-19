import argparse
import datetime
import os
import subprocess
import sys

import datasets
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders import P3CLDataModule, FlatCLDataModule
from dataloaders.constants import TAG2TASK_LIST
from trainers.evaluator import Evaluator
from trainers.interface_mixin import InterfaceMixin
from trainers.parameter_manager import TuningStrategy, make_model_trainable_parameters
from trainers.trainer import Trainer
from utils.config import Config, ParseKwargs
from utils.get_model import hf_tokenizer, load_continual_learning_model
from utils.util import get_logger, setup_wandb_logger

sys.path.insert(1, os.getcwd())
datasets.disable_progress_bar()


def main(config, loggers):
    start = datetime.datetime.now()

    assert config.peft_type == "molora", "This script is only for MoLoRA."

    assert os.path.isdir(
        config.checkpoint_dir_or_path
    ), f"{config.checkpoint_dir_or_path} is not a directory."
    print(
        f"Loading MoLoRA from distributed LoRA checkpoints in {config.checkpoint_dir_or_path} with init pool {config.init_datasets}."
    )

    load_model_dict = load_continual_learning_model(config)
    model = load_model_dict["model"]

    tokenizer = hf_tokenizer(config.origin_model)
    model.interface = InterfaceMixin(model_type=config.model_type)

    if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
        config.dataset = TAG2TASK_LIST[config.dataset[0]]
    train_task = config.dataset[0]

    if "c4" in train_task:
        datamodule = FlatCLDataModule(
            config, tokenizer, loggers, stage="train", is_moe=True
        )
    else:
        datamodule = P3CLDataModule(
            config, tokenizer, loggers, stage="train", is_moe=True
        )

    loggers["logger"].info(
        f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training grate for {train_task} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n"
    )

    # logic to train the last expert's router.
    tuning_strategy = TuningStrategy.LAST_EXPERT_GATE
    model = make_model_trainable_parameters(model, tuning_strategy)
    model.reset_gates_to_zero()
    trainer = Trainer(
        config=config,
        task=train_task,
        model=model,
        tokenizer=tokenizer,
        datamodule=datamodule,
        loggers=loggers,
        num_steps=config.single_lora_gate_train_steps,
    )
    trainer.train_task(train_task)

    # Apply layer norm to the gate after training.
    if config.layer_norm_after_train_single_lora:
        model.apply_layer_norm_to_gate()

    # Re-evaluate after training the last expert.
    config.dataset = config.init_datasets
    eval_datamodule = P3CLDataModule(
        config, tokenizer, loggers, stage="val", is_moe=True
    )
    evaluator = Evaluator(
        config=config,
        eval_tasks=config.init_datasets,
        tokenizer=tokenizer,
        datamodule=eval_datamodule,
        loggers=loggers,
    )
    val_results = evaluator.eval_all(model, split="val")
    loggers["logger"].info(val_results)

    loggers["logger"].info(
        f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished {train_task} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n"
    )

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

    training_state.global_training_step = 0

    return config, loggers


#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.lora_B.default.weight'
#  'base_model.model.decoder.block.23.layer.2.DenseReluDense.wo.base_layer.weight'

if __name__ == "__main__":
    config, loggers = main_setup()
    main(config, loggers)

# free up gpu memory.
# ps aux |grep "main.py" |awk '{print $2}' |xargs kill
