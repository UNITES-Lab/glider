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

from dataloaders import P3CLDataModule, BBCLDataModule
from dataloaders.constants import P3_DATASET_CONFIGS, TAG2TASK_LIST, BB_DATASET_CONFIGS
from trainers.evaluator import Evaluator
from trainers.interface_mixin import InterfaceMixin
from utils.config import Config, ParseKwargs
from utils.get_model import hf_tokenizer, load_continual_learning_model
from utils.util import get_logger, setup_wandb_logger

datasets.disable_progress_bar()


def main(config, loggers):
    start = datetime.datetime.now()
    if config.lora_hub_eval and config.moe_inference:
        raise (ValueError("LoRA Hub evaluation is not supported for MOE inference."))

    model_load_results = load_continual_learning_model(config)
    model = model_load_results["model"]
    lora_module_path_list = model_load_results["path_list"]

    if "bigbench" in config.dataset[0] or "bb" in config.dataset[0]:
        data_type = "bigbench"
    else:
        data_type = "p3"

    if config.save_router_state_dict:
        torch.save(model.router_weight_state_dict(), os.path.join(config.checkpoint_dir, "router_state_dict.pt"))
        print(f"Router weight state dict saved.")

    tokenizer = hf_tokenizer(config.origin_model)
    model.interface = InterfaceMixin(model_type=config.model_type)

    if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
        config.dataset = TAG2TASK_LIST[config.dataset[0]]

    all_tasks = config.dataset
    dump_dict = OrderedDict()
    dump_dict["tasks"] = all_tasks

    if data_type == "bigbench":
        datamodule = BBCLDataModule(
            config, tokenizer, loggers, is_moe=config.moe_inference, stage=config.eval_split
        )
    elif data_type == "p3":
        datamodule = P3CLDataModule(
            config, tokenizer, loggers, is_moe=config.moe_inference, stage=config.eval_split
        )
    else:
        raise KeyError(f"Unknown data type {data_type}")

    final_evaluator = Evaluator(
        config=config,
        eval_tasks=all_tasks,
        tokenizer=tokenizer,
        datamodule=datamodule,
        loggers=loggers,
    )

    results = final_evaluator.eval_all(
        model, split=config.eval_split,
        lora_hub_eval=config.lora_hub_eval, lora_hub_module_path_list=lora_module_path_list,
    )

    dump_dict[f"{config.eval_split}_results"] = results

    loggers["logger"].info(results)
    if "wandb" in loggers.keys():
        loggers["wandb"].log(
            {
                f"{task}_test_{m}": f"{v:.4f}"
                for task, metrics in results.items()
                for m, v in metrics.items() if isinstance(v, float)
            }
        )

    config_dict = vars(config)
    del config_dict["device"]
    dump_dict["config"] = config_dict
    dataset_configs = OrderedDict()
    for data_tag in all_tasks:
        if data_type == "bigbench":
            dataset_configs[data_tag] = BB_DATASET_CONFIGS[data_tag]
        elif data_type == "p3":
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


# config = Config()
# # config.dataset = ["t0"]
# config.dataset = ["full"]
# config.eval_split = "test"
# loggers = {"logger": get_logger("log.txt", f"{config.log_dir}/", os.path.join(config.project_dir, "utils/"))}
# tokenizer = hf_tokenizer(config.origin_model)
# datamodule = P3CLDataModule(
#     config, tokenizer, loggers, is_moe=config.moe_inference, stage=config.eval_split
# )
# # datamodule = FlatCLDataModule(
# #     config, tokenizer, loggers, is_moe=config.moe_inference, stage=config.eval_split
# # )
#
#
# def get_pairs(dataset_name):
#     dataloader = datamodule.task_modules[dataset_name].get_test_dataloader()
#     budget = 8
#     pairs = []
#     for batch in dataloader:
#         for input_str, target_str in zip(batch['input_str'], batch['target_str']):
#             pairs.append((input_str, target_str))
#         budget -= len(batch['input_str'])
#         if budget <= 0:
#             break
#     return pairs

#
# task_2_data = {}
# for task in datamodule.task_modules.keys():
#     task_2_data[task] = get_pairs(task)
#
# with open("flan_samples.json", "w") as f:
#     json.dump(task_2_data, f)
#
# dict_keys(['p3socialiqa', 'p3wiqa', 'p3cosmosqa', 'p3quail', 'p3quartz', 'p3qasc',
#            'p3commonsenseqa', 'p3quarel', 'p3dream', 'p3sciq', 'p3wikihop', 'p3ropes', 'p3adversarialqa',
#            'p3duorc', 'p3quoref', 'p3hotpotqa', 'p3wikiqa', 'p3amazonpolarity', 'p3appreviews', 'p3rottentomatoes',
#            'p3imdb', 'p3yelp', 'p3agnews', 'p3dbpedia14', 'p3trec', 'p3wikibio', 'p3commongen', 'p3cnndailymail',
#            'p3multinews', 'p3gigaword', 'p3samsum', 'p3xsum', 'p3paws', 'p3qqp', 'p3mrpc', 'p3hswag', 'p3copa',
#            'p3storycloze', 'p3cb', 'p3rte', 'p3anlir1', 'p3anlir2', 'p3anlir3', 'p3winogrande', 'p3wscfixed', 'p3wic'])
#
# dict_keys(['bbbooleanexpressions', 'bbcausaljudgement', 'bbdateunderstanding', 'bbdisambiguationqa', 'bbdycklanguages',
#            'bbformalfallacies', 'bbgeometricshapes', 'bbhyperbaton', 'bblogicaldeduction', 'bbmovierecommendation',
#            'bbmultisteparithmetictwo', 'bbnavigate', 'bbobjectcounting', 'bbpenguinsinatable',
#            'bbreasoningaboutcoloredobjects', 'bbruinnames', 'bbsalienttranslationerrordetection', 'bbsnarks',
#            'bbsportsunderstanding', 'bbtemporalsequences', 'bbtrackingshuffledobjects', 'bbweboflies', 'bbwordsorting',
#            'bbautodebugging', 'bbbbqlitejson', 'bbcodelinedescription', 'bbconceptualcombinations',
#            'bbconlangtranslation',
#            'bbemojimovie', 'bbhinduknowledge', 'bbknownunknowns', 'bblanguageidentification', 'bblinguisticspuzzles',
#            'bblogicgridpuzzle', 'bbmisconceptionsrussian', 'bbnovelconcepts', 'bboperators',
#            'bbparsinlureadingcomprehension', 'bbplaydialogsameordifferent', 'bbrepeatcopylogic', 'bbstrangestories',
#            'bbstrategyqa', 'bbsymbolinterpretation', 'bbvitamincfactverification', 'bbwinowhy'])
