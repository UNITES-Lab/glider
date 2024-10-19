import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
from tqdm.auto import tqdm

from utils.earlystop import EarlyStopping
from utils.optimizer_and_scheduler import get_optimizer, get_scheduler
from . import training_state
from .evaluator import Evaluator
from .trainer_utils import Tracker, redirect_to_tqdm


def get_named_trainable_parameters(model):
    return OrderedDict(
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    )


class Trainer(object):
    def __init__(
            self,
            config,
            task,  # todo: remove it as it is not used
            model,
            tokenizer,
            datamodule,
            loggers,
            checkpoint_dir=None,
            **kwargs,
    ):
        config = deepcopy(config)  # in case we modify the config
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                raise ValueError(f"{k} is not in the config")
        self._named_trainable_parameters = None
        self.checkpoint_dir = (
            config.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        )
        self.num_steps = config.num_steps
        self.warmup_steps = (
            config.warmup_steps
            if config.warmup_steps is not None
            else int(config.num_steps * config.warmup_ratio)
        )
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        self.logger_dict = loggers
        self.logger = loggers["logger"] if "logger" in loggers.keys() else None
        self.wandb = loggers["wandb"] if "wandb" in loggers.keys() else None
        self.report_tracker = Tracker()

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

    @staticmethod
    def _get_train_batches(dataloader):
        while True:
            for batch_inputs in dataloader:
                yield batch_inputs

    def get_named_trainable_parameters(self):
        if getattr(self, "_named_trainable_parameters", None) is None:
            self._named_trainable_parameters = OrderedDict(
                (name, param)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            )
        return self._named_trainable_parameters

    def train_task(self, task_name):

        self.training_task_name = task_name
        config = self.config

        task_datamodule = self.datamodule[task_name]
        grad_accum_factor = (
                config.effective_train_batch_size
                // task_datamodule.dataset["train"].batch_size
        )
        self.num_steps = config.num_steps

        optimizer, self.trainable_param_names = get_optimizer(self.model, config)
        scheduler = get_scheduler(
            scheduler_class=config.scheduler,
            optimizer=optimizer,
            num_steps=self.num_steps,
            num_warmup_steps=self.warmup_steps,
        )
        if config.mix_precision in ["bf16", "fp16"]:
            loss_scaler = torch.cuda.amp.GradScaler()
            dtype = torch.float16 if config.mix_precision == "fp16" else torch.bfloat16
        else:
            loss_scaler = None
            dtype = torch.float32

        self.early_stop = EarlyStopping(
            save_checkpoint=self.save_checkpoint,
            patience=config.patience,
            checkpoint_dir=self.checkpoint_dir,
            verbose=True,
            log_fn=self.log_string_info,
        )

        current_step = 0
        best_metrics = None
        self.model.train()

        train_dataloader = task_datamodule.get_train_dataloader()
        data_iter = self._get_train_batches(train_dataloader)

        self.log_string_info("***** Running training *****")
        self.log_string_info(f" Num Steps = {self.num_steps}")
        self.log_string_info(
            f" Train batch size (w. accumulation) = {task_datamodule.dataset['train'].batch_size * grad_accum_factor}"
        )
        self.log_string_info(f" Gradient Accumulation steps = {grad_accum_factor}")
        self.log_string_info(f" Num. Warmup steps: {self.warmup_steps}")
        self.log_string_info(f" Total optimization steps = {self.num_steps}")
        self.log_string_info(f" Train Task: {task_name}")

        progress_bar = tqdm(range(self.num_steps), desc=f"Training task: {task_name}")

        while current_step < self.num_steps:
            optimizer.zero_grad()
            for _ in range(grad_accum_factor):
                batch_inputs = next(data_iter)
                with torch.autocast(dtype=dtype, device_type="cuda", enabled=config.mix_precision in ["bf16", "fp16"]):
                    batch_outputs = self.model.interface.__call__(
                        batch_inputs,
                        task_datamodule.dataset["train"].interface_info,
                        self.model,
                        self.tokenizer,
                    )
                loss = batch_outputs["loss"]
                scaled_loss = loss / grad_accum_factor
                if loss_scaler is not None:
                    scaled_loss = loss_scaler.scale(scaled_loss)
                scaled_loss.backward()
                self.report_tracker.add(
                    loss=loss,
                    global_hidden_dict=self.model.interface.global_hidden_dict,
                )

            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                get_named_trainable_parameters(self.model).values(),
                config.gradient_clipping,
                error_if_nonfinite=False,
            )
            self.report_tracker.add(
                grad_norm=grad_norm, lr=optimizer.param_groups[0]["lr"]
            )

            if loss_scaler is not None:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            current_step += 1
            training_state.global_training_step += 1
            progress_bar.update()
            if current_step % config.logging_steps == 0:
                report = self.report_tracker.get_summary(clear=True)
                self.log_scalar_dict(report, step=training_state.global_training_step)

            if current_step % config.eval_steps == 0:
                evaluator = Evaluator(
                    config=config,
                    eval_tasks=[task_name],
                    tokenizer=self.tokenizer,
                    datamodule=self.datamodule,
                    loggers=self.logger_dict,
                )
                val_results = evaluator.run_task_eval(
                    self.model, task_datamodule, split="val"
                )
                best_metrics = self.early_stop(
                    metrics=val_results,
                    model=self.model,
                    step=training_state.global_training_step,
                    best_file_name=f"{task_name}_best.pt" if config.save_best else None,
                )
                self.log_scalar_dict(best_metrics, step=training_state.global_training_step)

                if self.early_stop.stop:
                    self.log_string_info(
                        f"\n\nEarly stopping at step {current_step}.\n\n"
                    )
                    break

        progress_bar.close()

        self.log_string_info(f"Best result: {best_metrics}")
        self.log_string_info(f"\n\nFinished {task_name}\n\n")

        self.on_train_end()

    def train_task_auto_encoder(self, task_name):

        self.training_task_name = task_name
        config = deepcopy(self.config)
        config.trainable_param_names = ".*input_auto_encoder.*"

        self.model.enable_input_auto_encoder_training()  # Note: this will reset previous AE modules if exists

        task_datamodule = self.datamodule[task_name]
        grad_accum_factor = (
                config.effective_train_batch_size // task_datamodule.dataset["train"].batch_size
        )
        self.num_steps = config.num_steps

        optimizer, self.trainable_param_names = get_optimizer(self.model, config)
        scheduler = get_scheduler(
            scheduler_class=config.scheduler,
            optimizer=optimizer,
            num_steps=self.num_steps,
            num_warmup_steps=self.warmup_steps,
        )
        if config.mix_precision in ["bf16", "fp16"]:
            loss_scaler = torch.cuda.amp.GradScaler()
            dtype = torch.float16 if config.mix_precision == "fp16" else torch.bfloat16
        else:
            loss_scaler = None
            dtype = torch.float32

        self.early_stop = EarlyStopping(
            save_checkpoint=self.save_checkpoint,
            patience=config.patience,
            checkpoint_dir=self.checkpoint_dir,
            verbose=True,
            log_fn=self.log_string_info,
        )

        current_step = 0
        best_metrics = None
        self.model.train()

        train_dataloader = task_datamodule.get_train_dataloader()
        data_iter = self._get_train_batches(train_dataloader)

        self.log_string_info("***** Running training Auto-Encoders *****")
        self.log_string_info(f" Num Steps = {self.num_steps}")
        self.log_string_info(
            f" Train batch size (w. accumulation) = {task_datamodule.dataset['train'].batch_size * grad_accum_factor}"
        )
        self.log_string_info(f" Gradient Accumulation steps = {grad_accum_factor}")
        self.log_string_info(f" Num. Warmup steps: {self.warmup_steps}")
        self.log_string_info(f" Total optimization steps = {self.num_steps}")
        self.log_string_info(f" Train Task: {task_name}")

        progress_bar = tqdm(range(self.num_steps), desc=f"Training task: {task_name}")

        while current_step < self.num_steps:
            optimizer.zero_grad()
            for _ in range(grad_accum_factor):
                batch_inputs = next(data_iter)
                with torch.autocast(dtype=dtype, device_type="cuda", enabled=config.mix_precision in ["bf16", "fp16"]):
                    _ = self.model.interface.__call__(
                        batch_inputs,
                        task_datamodule.dataset["train"].interface_info,
                        self.model,
                        self.tokenizer,
                    )
                loss = self.model.gather_input_auto_encoder_loss()
                scaled_loss = loss / grad_accum_factor
                if loss_scaler is not None:
                    scaled_loss = loss_scaler.scale(scaled_loss)
                scaled_loss.backward()
                self.report_tracker.add(
                    loss=loss,
                    global_hidden_dict=self.model.interface.global_hidden_dict,
                )

            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                get_named_trainable_parameters(self.model).values(),
                config.gradient_clipping,
                error_if_nonfinite=False,
            )
            self.report_tracker.add(grad_norm=grad_norm, lr=optimizer.param_groups[0]["lr"])

            if loss_scaler is not None:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            current_step += 1
            training_state.global_training_step += 1
            progress_bar.update()
            if current_step % config.logging_steps == 0:
                report = self.report_tracker.get_summary(clear=True)
                self.log_scalar_dict(report, step=training_state.global_training_step)

            if current_step % config.eval_steps == 0:
                evaluator = Evaluator(
                    config=config,
                    eval_tasks=[task_name],
                    tokenizer=self.tokenizer,
                    datamodule=self.datamodule,
                    loggers=self.logger_dict,
                )
                val_results = evaluator.run_task_eval(self.model, task_datamodule, split="val")
                best_metrics = self.early_stop(
                    metrics=val_results,
                    model=self.model,
                    step=training_state.global_training_step,
                    best_file_name=f"{task_name}_auto_encoder_best.pt" if config.save_best else None,
                )
                self.log_scalar_dict(best_metrics, step=training_state.global_training_step)

                if self.early_stop.stop:
                    self.log_string_info(f"\n\nEarly stopping at step {current_step}.\n\n")
                    break

        progress_bar.close()

        self.log_string_info(f"Best result: {best_metrics}")
        self.log_string_info(f"\n\nFinished {task_name} Auto Encoder Training\n\n")

        self.on_train_end(file_name=f"{self.training_task_name}_auto_encoder_last.pt")

        self.model.disable_input_auto_encoder_training()

    def on_train_end(self, file_name=None):
        if self.config.save_last:
            self.save_checkpoint(
                self.model,
                filename=f"{self.training_task_name}_last.pt" if file_name is None else file_name,
            )

    def save_checkpoint(self, model, filename, save_func=torch.save):
        """Saves model's trainable parameters."""
        # TODO:: dump only parameters that have require_grad true
        trainable_states = {
            param_name: param_weight.cpu()
            for param_name, param_weight in model.state_dict().items()
            if param_name in self.trainable_param_names
        }
        output_model_file = os.path.join(self.checkpoint_dir, filename)
        self.log_string_info(f"Saved model state at {output_model_file}")
        save_func(trainable_states, output_model_file)

    def load_model(self):
        if self.config.load_weight != "":
            trainable_states = torch.load(
                self.config.load_weight, map_location=torch.device("cpu")
            )
            load_result = self.model.load_state_dict(trainable_states, strict=False)
            assert (
                    len(load_result.unexpected_keys) == 0
            ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
