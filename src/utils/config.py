import argparse
import ast
import json
import multiprocessing
import os

import numpy as np

from .util import seed_everything


class Config(object):
    def __init__(self, filenames=None, kwargs=None):

        ## Basic Setup and Directories.
        self.debug = False
        self.project_name = "phatgoose"
        self.name = "test"
        self.project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.data_dir = "/nas-hdd/prateek/data"
        self.output_dir = "saved_runs"
        self.config_dir = "configs"
        self.seed = 42
        self.hf_write_token = os.environ.get("HF_TOKEN")

        # Model Configs
        self.origin_model = "google/t5-xl-lm-adapt"
        self.model_class = "seq2seq_lm"
        self.model_type = "encdec"
        self.peft_type = "lora"
        self.load_model_dtype = "float32"
        self.auto_device_map = False

        # Dataset Configs
        self.val_fraction = 0.2  # py: set this to zero for heldout tasks to evaluate on the full test set, we don't need to split for them.
        self.dataset = ["sst2"]
        self.eval_dataset = None  # extra dataset for evaluation
        self.eval_split = "test"

        # Trainer configs
        self.num_steps = 1500
        self.effective_train_batch_size = (
            128  # py:  use to calculate gradient accumulation factor
        )
        self.patience = 3
        self.verbose = False
        self.do_test = False
        self.eval_steps = 100
        self.save_last = True
        self.save_best = True
        self.logging_steps = 5
        self.gradient_checkpointing = False

        # Inference configs
        self.moe_inference = False
        self.arrow_routing = False
        self.inference_batch_size_scale = 1  # used to scale the batch size for inference
        self.checkpoint_dir_or_path = None  # when moe_inference is True, this should be the directory of all individual LoRAs
        self.cl_checkpoint_path = None
        self.load_checkpoint_dataset = None
        self.ae_checkpoint_dir = None
        self.init_datasets = ["t0-cl-init1"]
        self.selected_expert_ids = None
        self.merge_num_clusters = None
        self.global_clustering = False
        self.hierarchical_num_clusters = None
        self.hierarchical_cluster_token_routing = False
        self.save_router_state_dict = False
        self.bias_router_embedding_path = None
        self.bias_input_embedding_path = None
        self.lora_hub_eval = False

        # Optimization configs
        self.optimizer = "adamw"
        self.lr = 3e-3
        self.trainable_param_names = ".*lora.*"  # todo: remove this from codebase
        self.scheduler = "linear_decay_with_warmup"
        self.warmup_steps = None
        self.warmup_ratio = 0.02
        self.weight_decay = 0
        self.scale_parameter = True
        self.mix_precision = "bf16"
        self.gradient_clipping = 1.0

        # LoRA configs
        self.target_modules = "all-linear"
        self.lora_rank = 16
        self.lora_alpha = (
            1  # py: lora_alpha needs to be 1 to mimic phatgoose lora implementation.
        )
        self.lora_dropout = 0.0
        self.use_rslora = (
            False  # Py: this needs to be false to mimic phatgoose lora implementation.
        )
        self.init_lora_weights = True
        self.lora_bias = "none"

        # MoLoRA configs
        self.moe_router_aux_loss_coef = 0.0
        self.moe_top_k = 2
        self.moe_top_p = 1.0
        self.moe_reweight_output = True
        self.bias_routing_scale = 0
        self.bias_routing_dim = -1

        # Composition Configs
        self.lora_init_method = (
            "usage-based"  # "random" or "usage-based" or "top-2-used"
        )
        self.gate_init_method = "zero"
        self.zeroshot_tolerance = 0.05
        self.upper_bound_tolerance = 0.05
        self.single_lora_gate_train_steps = 200
        self.molora_gate_train_samples = 1000
        self.molora_gate_train_steps = 100
        self.layer_norm_after_train_single_lora = True

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(self.project_dir, self.config_dir, filename)
                self.update_kwargs(json.load(open(filename)), eval=False)

        if kwargs:
            self.update_kwargs(kwargs, eval=True)

        self.set_exp_dirs()

        if self.seed is None:
            self.seed = np.random.randint(10000)
            print(f"SETTING SEED TO {self.seed}")
        seed_everything(self.seed)

    def update_kwargs(self, kwargs, eval=True):
        for k, v in kwargs.items():
            if eval:
                try:
                    if k != "name" and k != "load_model_path":
                        v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def set_exp_dirs(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        self.cpu_cont = multiprocessing.cpu_count()
        self.output_dir = os.path.join(self.project_dir, self.output_dir)
        self.config_dir = os.path.join(self.project_dir, self.config_dir)

        self.run_output_dir = f"{self.output_dir}/{self.project_name}/{self.name}"
        os.makedirs(self.run_output_dir, exist_ok=True)
        self.log_dir = os.path.join(self.run_output_dir, "logs")
        self.prediction_dir = os.path.join(self.run_output_dir, "prediction")
        self.checkpoint_dir = os.path.join(self.run_output_dir, "checkpoints")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.save_config(
            os.path.join(self.log_dir, os.path.join("initial_config.json"))
        )
        self.finish_flag_file = os.path.join(self.run_output_dir, "exp_completed.txt")

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        # Hack to make it work with vs code debugger. Works for command line without this.
        if len(values) == 1:
            values = values[0].split(" ")

        for value in values:
            key, value = value.split("=")
            if key in [
                "dataset",
                "eval_tasks",
                "init_datasets",
                "eval_dataset",
                "load_checkpoint_dataset",
                "selected_expert_ids"
            ]:
                value = value.split(",")
            getattr(namespace, self.dest)[key] = value
