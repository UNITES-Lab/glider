from typing import Callable

import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        save_checkpoint,
        patience=7,
        checkpoint_dir="./",
        verbose=True,
        delta=1e-6,
        log_fn: Callable = print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            log_fn (function): trace print function.
                            Default: print
        """
        self.save_checkpoint = save_checkpoint
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_metrics = None
        self.stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        self.log_fn = log_fn

    def __call__(self, metrics, model, step, best_file_name=None):
        # TODO: Fix this function this should not work.
        score = metrics["score"]

        if self.best_score is None:
            self.log_fn(
                f" Step: {step} | Best metric score {score:.6f}.  Saving model ..."
            )
            self.best_score = score
            self.best_metrics = metrics
            if best_file_name:
                self.save_checkpoint(model, best_file_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.log_fn(
                f" EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.stop = True
        else:
            if self.verbose:
                self.log_fn(
                    f" Step: {step} | Best metric changed from ({self.best_score:.6f} --> {score:.6f})."
                )
            self.best_score = score
            self.best_metrics = metrics
            if best_file_name:
                self.save_checkpoint(model, best_file_name)
            self.counter = 0
        return self.best_metrics
