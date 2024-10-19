# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/29

import contextlib
import inspect
from collections import defaultdict
from typing import List

import numpy as np
import tqdm


@contextlib.contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


class Tracker:
    def __init__(self, global_hidden_to_keep: List[str] = None):
        if global_hidden_to_keep is None:
            global_hidden_to_keep = ["loss", "scale"]
        self.global_hidden_to_keep = global_hidden_to_keep
        self._results = defaultdict(list)

    def add(self, loss=None, global_hidden_dict=None, grad_norm=None, lr=None):
        if loss is not None:
            self._results["loss"].append(loss.detach().cpu().item())
        if grad_norm is not None:
            self._results["grad_norm"].append(grad_norm.detach().cpu().item())
        if lr is not None:
            self._results["lr"].append(lr)
        if global_hidden_dict is not None:
            for key, value in global_hidden_dict.items():
                if key[0] in self.global_hidden_to_keep:
                    report_key = f"{key[0]}/{'.'.join(key[1:])}"
                    self._results[report_key].append(value.detach().cpu().item())

    def get_summary(self, clear=True):
        summary = {}
        for key, value in self._results.items():
            value = np.mean(value).item()
            if value > 1e-4:
                summary[key] = round(value, 4)
            else:
                summary[key] = value
        if clear:
            self._results.clear()
        return summary
