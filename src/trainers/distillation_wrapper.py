# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/18
from typing import Union

import torch
from torch.nn import functional as F
from transformers import PreTrainedModel

from peft import PeftModel
from peft.tuners.tuners_utils import BaseTuner

ModelType = Union[PreTrainedModel, PeftModel, BaseTuner]


class KDModelWrapper(torch.nn.Module):
    def __init__(
            self, student_model: ModelType, teacher_model: ModelType, temperature: float = 1.0, alpha: float = 0.1
    ):
        super(KDModelWrapper, self).__init__()
        self.student_model = student_model

        assert torch.cuda.device_count() > 1, "KDModelWrapper requires at least 2 GPUs to keep both student&teacher model in memory."
        self.teacher_device_id = 1
        self.teacher_model = teacher_model.cuda(self.teacher_device_id)

        self.temperature = temperature
        self.alpha = alpha

    def forward(self, *args, **kwargs):
        student_outputs = self.student_model(*args, **kwargs)
        self.teacher_model.eval()

        args = [arg.cuda(self.teacher_device_id) for arg in args]
        kwargs = {k: v.cuda(self.teacher_device_id) for k, v in kwargs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(*args, **kwargs)

        # KD!
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"].to(student_logits.device)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        )

        student_outputs["loss"] = kd_loss * self.alpha + student_outputs["loss"]

        return student_outputs

    def __getattr__(self, name: str):
        try:
            return super(KDModelWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(self.student_model, name)
