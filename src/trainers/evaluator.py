from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
from tqdm.auto import tqdm

from lorahub import lorahub_learning
from .trainer_utils import redirect_to_tqdm
import random


class Evaluator(object):
    def __init__(
            self,
            config,
            eval_tasks,
            tokenizer,
            datamodule,
            loggers=None,
    ):
        self.config = config
        self.eval_tasks = eval_tasks
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        self.logger = loggers["logger"] if "logger" in loggers.keys() else None
        self.wandb = loggers["wandb"] if "wandb" in loggers.keys() else None

    def log_scalar_dict(self, msg: Dict[str, float], step: int):
        # pzli: I did not use logger.info here, cos I cannot find a compatible way between logger & tqdm progress bar
        with redirect_to_tqdm():
            if self.wandb is not None:
                self.wandb.log(msg, step=step)
            print(f" Step: {step} | {msg}")

    def log_string_info(self, msg: str, step: int = None):
        with redirect_to_tqdm():
            log_msg = f"{msg}" if step is None else f"Step: {step} | {msg}"
            print(log_msg)

    def eval_all(self, model, split="val", flat_names=False, verbose=True,
                 return_routing_stats=False, return_routing_state_dict=False,
                 return_correct_incorrect_routing_state_dict=False,
                 lora_hub_eval=False, lora_hub_module_path_list=None):
        eval_all_results = OrderedDict()
        if return_routing_state_dict:
            self.log_string_info(f"[WARNING] Return all routing state dict will lead to more memory consumption.")
        if verbose:
            self.log_string_info("***** Running Evaluation *****")
            if return_routing_stats:
                self.log_string_info("***** Return Routing Stats *****")
                self.log_string_info(f"[WARNING] Routing stats will be reset after each task evaluation.")
            # self.log_string_info(f"\tEval batch size = {self.config.eval_batch_size}")
            progress_bar = tqdm(range(len(self.eval_tasks)))
        else:
            progress_bar = None

        for idx, eval_task in enumerate(self.eval_tasks):

            task_datamodule = self.datamodule(eval_task)

            if return_routing_stats or return_routing_state_dict:
                model.reset_routing_counter()

            if hasattr(model, "is_bias_routing") and model.is_bias_routing:
                # print("DEBUG-B", model.task_embedding_dict.keys())
                print(f"Updating routing bias for task {eval_task}")
                model.update_routing_bias(bias_token_task=eval_task)

            if verbose:
                progress_bar.set_description(
                    f"[Eval Task {eval_task}:{idx + 1}/{len(self.eval_tasks)}]"
                )

            try:
                task_eval_metrics = self.run_task_eval(
                    model, task_datamodule, split,
                    return_correct_incorrect_routing_state_dict=return_correct_incorrect_routing_state_dict,
                    lora_hub_eval=lora_hub_eval, lora_hub_module_path_list=lora_hub_module_path_list
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise e
                print(f"[WARNING] Skipping due to RuntimeError: {e}")
                eval_all_results[eval_task] = {"error": f"{e}", "score": "-100"}
                continue

            if return_routing_stats or return_routing_state_dict:
                routing_stats_dict = model.get_routing_counter_state_dict(normalize=True)
                if return_routing_state_dict:
                    routing_stats = torch.stack([v for v in routing_stats_dict.values()], dim=0)
                    routing_mean = routing_stats.mean(dim=0)
                    routing_std = routing_stats.std(dim=0)
                    task_eval_metrics["routing_mean"] = routing_mean.tolist()
                    task_eval_metrics["routing_std"] = routing_std.tolist()

                if return_routing_state_dict:
                    task_eval_metrics["routing_state_dict"] = routing_stats_dict

            eval_all_results[eval_task] = task_eval_metrics

            if verbose:
                progress_bar.update(1)
                metric_str = " ".join(
                    f"{k}:{v:.4f}" for k, v in task_eval_metrics.items()
                )
                progress_bar.set_postfix(metrics=f"{metric_str}")

            self.log_string_info(f"All current results: {eval_all_results}")

        if flat_names:
            flat_metrics = {
                f"{task}_{mname}": round(mval, 6)
                for task, tms in eval_all_results.items()
                for mname, mval in tms.items()
            }
            return dict(sorted(flat_metrics.items()))

        return dict(sorted(eval_all_results.items()))

    def run_task_eval(
            self, model, task_datamodule, split,
            return_correct_incorrect_routing_state_dict=False,
            lora_hub_eval=False, lora_hub_module_path_list=None,
    ):

        if split == "val":
            dataloader = task_datamodule.get_val_dataloader()
        elif split == "test":
            dataloader = task_datamodule.get_test_dataloader()
        elif split == "train":
            dataloader = task_datamodule.get_train_dataloader()
        else:
            raise ValueError(f"Split {split} not implemented!")

        if lora_hub_eval is True:
            budget = 3
            example_inp_out_pairs = []
            for batch in dataloader:
                for input_str, target_str in zip(batch['input_str'], batch['target_str']):
                    example_inp_out_pairs.append((input_str, target_str))
                budget -= len(batch['input_str'])
                if budget <= 0:
                    break
            _, model, _ = lorahub_learning(
                random.sample(lora_hub_module_path_list, 20),
                config=self.config,
                example_inputs=[pair[0] for pair in example_inp_out_pairs],
                example_outputs=[pair[1] for pair in example_inp_out_pairs],
                model=model,
                max_inference_step=10,
            )

        scorer = task_datamodule.scorer
        if return_correct_incorrect_routing_state_dict:
            model.set_routing_counter_per_sequence(enable=True)
            is_correct = []
            assert "accuracy" in task_datamodule.data_config["val"]["metrics"], (
                "Accuracy is required for correct/incorrect routing state dict.")

        model.eval()
        with torch.no_grad():
            for batch_idx, batch_inputs in enumerate(
                    tqdm(
                        dataloader, desc=f"Evaluating {task_datamodule.data_tag}-{split}..."
                    )
            ):
                batch_outputs = model.interface.__call__(
                    batch_inputs,
                    task_datamodule.dataset[split].interface_info,
                    model,
                    self.tokenizer,
                )
                scorer.add_batch(batch_inputs, batch_outputs)

                if return_correct_incorrect_routing_state_dict:
                    batch_label = batch_inputs["label"]
                    batch_prediction = batch_outputs["prediction"]
                    is_correct.extend((batch_label.cpu() == batch_prediction.cpu()).tolist())

            _current_results = scorer.get_score()
            _current_results["score"] = sum(_current_results.values()) / len(
                _current_results
            )

            if return_correct_incorrect_routing_state_dict:
                correct_routing_state_dict = model.get_routing_counter_state_dict_by_sequence(
                    [i for i, val in enumerate(is_correct) if val], normalize=True
                )
                incorrect_routing_state_dict = model.get_routing_counter_state_dict_by_sequence(
                    [i for i, val in enumerate(is_correct) if not val], normalize=True
                )

            print(f"\t{task_datamodule.data_tag} results: {_current_results}")

        ret_results = deepcopy(_current_results)

        if return_correct_incorrect_routing_state_dict:
            ret_results["correct_routing_state_dict"] = correct_routing_state_dict
            ret_results["incorrect_routing_state_dict"] = incorrect_routing_state_dict

        return ret_results
