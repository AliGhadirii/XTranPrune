import random
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import warnings


def set_seeds(seed):
    """Fixes random seeds for the experiment replicability"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class LinearWarmup(_LRScheduler):
    __name__ = "LinearWarmup"

    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_iters=0,
        warmup_epochs=0,
        eta_min=1e-8,
        last_epoch=-1,
        verbose=False,
        steps_per_epoch=None,
    ):
        if warmup_iters and warmup_epochs:
            print(
                "\033[93m Found nonzero arguments for warmup_iters and warmup_epochs \033[0m"
            )
            print("\033[93m Using warmup_epochs instead of warmup_iters \033[0m")
            warmup_iters = steps_per_epoch * warmup_epochs
        if not warmup_iters and not warmup_epochs:
            print("\033[93m No warmup period found but LinearWarmup is used \033[0m")
            warmup_iters = 1
        else:
            if warmup_epochs and steps_per_epoch is None:
                raise TypeError(
                    "LinearWarmup with warmup_epochs settings must include steps_per_epoch"
                )
            elif warmup_epochs and steps_per_epoch is not None:
                warmup_iters = steps_per_epoch * warmup_epochs

        self.warmup_iters = warmup_iters
        self.eta_min = eta_min
        self.max_lr = max_lr
        for group in optimizer.param_groups:
            group["lr"] = self.eta_min
        super(LinearWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == -1:
            return [self.eta_min for group in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + (1 / self.warmup_iters) * (self.max_lr - self.eta_min)
                for group in self.optimizer.param_groups
            ]
