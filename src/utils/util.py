import torch
import datetime
import os
import sys
import json
import psutil
import wandb
from shutil import copytree, ignore_patterns
import random
import numpy as np
import logging, logging.config
import re
import sys
from pathlib import Path
import torch.backends.cudnn as cudnn


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_gpu(model, config):
    if config.multigpu is None:
        config.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {config.multigpu} gpus")
        torch.cuda.set_device(config.multigpu[0])
        config.gpu = config.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=config.multigpu).cuda(
            config.multigpu[0]
        )
        config.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model


def setup_wandb_logger(config):
    wandb_writer = wandb.init(
        project=config["project_name"],
        save_code=False,
        name=config["name"],
        config=config,
        dir=config["output_dir"],
    )  # , group=config.group)

    src_dir = Path(__file__).resolve().parent
    base_path = str(src_dir.parent)
    src_dir = str(src_dir)
    return wandb_writer, src_dir, base_path


class PrintAndLog(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def my_collate_fn(batch):

    dict_batch = {}
    dict_batch["input"] = {}
    dict_batch["output"] = {}

    for datapoint in batch:
        for k, v in datapoint["input"].items():
            if k in dict_batch["input"]:
                dict_batch["input"][k].append(v)
                # dict_batch["input"][k].append(v[0])
            else:
                # dict_batch["input"][k] = [v[0]]
                dict_batch["input"][k] = [v]

        for k, v in datapoint["output"].items():
            if k in dict_batch["output"]:
                # dict_batch["output"][k].append(v[0])
                dict_batch["output"][k].append(v)

            else:
                # dict_batch["output"][k] = [v[0]]
                dict_batch["output"][k] = [v]

    for k, list_v in dict_batch["input"].items():
        if isinstance(list_v[0], int):
            dict_batch["input"][k] = torch.tensor(list_v)
    for k, list_v in dict_batch["output"].items():
        if isinstance(list_v[0], int):
            dict_batch["output"][k] = torch.tensor(list_v)

    return dict_batch


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object
    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + "log_config.json"))
    config_dict["handlers"]["file_handler"]["filename"] = log_dir + name.replace(
        "/", "-"
    )
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = "%(asctime)s - [%(levelname)s] - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def make_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)


def check_dir(dir_name):
    return os.path.exists(dir_name)


def make_exp_dir(base_exp_dir):
    """
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
    Returns:
        exp_dir_name: experiment directory name
    """
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, "src")

    copytree(
        os.path.join(os.environ["NICL_ROOT"], "src"),
        src_file,
        ignore=ignore_patterns("*.pyc", "tmp*"),
    )

    return exp_dir_name


def print_mem_usage(loc):
    """
    Print memory usage in GB
    :return:
    """
    print(
        "%s gpu mem allocated: %.2f GB; reserved: %.2f GB; max: %.2f GB; cpu mem %d"
        % (
            loc,
            float(torch.cuda.memory_allocated() / 1e9),
            float(torch.cuda.memory_reserved() / 1e9),
            float(torch.cuda.max_memory_allocated() / 1e9),
            psutil.virtual_memory().percent,
        )
    )
    sys.stdout.flush()


def update_dict_val_store(dict_val_store, dict_update_val, grad_accum_factor):
    """
    Update dict_val_store with dict_update_val

    :param dict_val_store:
    :param dict_update_val:
    :return:
    """
    if dict_val_store is None:
        dict_val_store = dict_update_val
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k] / grad_accum_factor

    return dict_val_store


def get_avg_dict_val_store(dict_val_store, num_batches, grad_accumulation_factor):
    """
    Get average dictionary val

    :param dict_val_store:
    :param eval_every:
    :return:
    """
    dict_avg_val = {}

    for k in dict_val_store.keys():
        dict_avg_val[k] = float(
            "%.3f" % (dict_val_store[k] / num_batches / grad_accumulation_factor)
        )

    return dict_avg_val
