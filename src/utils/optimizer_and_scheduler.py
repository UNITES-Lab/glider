import re

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import Adafactor


def get_optimizer(model, config):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config.optimizer

    param_groups = [{"params": []}]
    trainable_param_names = set()
    for param_name, param in model.named_parameters():
        if re.fullmatch(config.trainable_param_names, param_name):
            param_groups[0]["params"].append(param)
            trainable_param_names.add(param_name)
        else:
            param.requires_grad = False

    if optim_name.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=config.lr)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups, lr=config.lr, weight_decay=config.weight_decay
        )
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(
            param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-8
        )
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scale_parameter=config.scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return optimizer, trainable_param_names


def get_scheduler(
    optimizer,
    num_steps,
    scheduler_class,
    gamma=None,
    warmup_ratio=None,
    num_warmup_steps=None,
):
    """
    Args:
        optimizer: an optimizer
        scheduler_class: scheduler class, one of "constant_with_warmup", "polynomial_decay_with_warmup", "exponential_decay", "linear_decay_with_warmup", "cosine_annealing", "adafactor"
        num_steps: total number of steps
        warmup_ratio: ratio of warmup steps
        num_warmup_steps: number of warmup steps
    """
    if num_warmup_steps is None and warmup_ratio is not None:
        num_warmup_steps = int(num_steps * warmup_ratio)

    if num_warmup_steps is None:
        num_warmup_steps = 0

    if scheduler_class == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif scheduler_class == "defrost":
        return get_defrost_schedule(optimizer, num_warmup_steps)
    elif scheduler_class == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_steps
        )
    elif scheduler_class == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_class == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_steps)
    elif scheduler_class == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    elif scheduler_class == "adafactor":
        return AdafactorSchedule(optimizer, initial_lr=optimizer.defaults["lr"])
    else:
        raise ValueError("Invalid scheduler class %s" % scheduler_class)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which
    the learning rate increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_defrost_schedule(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Like get_constant_schedule_with_warmup, but with an addition num_warmup_steps of zero learning rate in the beginning.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return 0.0
        elif current_step < 2 * num_warmup_steps:
            return float(current_step - num_warmup_steps) / float(
                max(1, num_warmup_steps)
            )
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the optimizer to end lr defined by `lr_end`,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is
    based on the original BERT implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    assert (
        lr_init > lr_end
    ), f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
