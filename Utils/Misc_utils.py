import random
import sys
import torch
import numpy as np
import torch
import torch.nn.functional as F
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


class Logger(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


# For Debugging


def get_stat(tensor):
    return [
        torch.min(tensor).item(),
        torch.quantile(tensor, 0.25).item(),
        torch.quantile(tensor, 0.5).item(),
        torch.quantile(tensor, 0.75).item(),
        torch.max(tensor).item(),
        torch.mean(tensor).item(),
        torch.std(tensor).item(),
    ]


def get_mask_idx(tensor, rate):
    tmp1 = tensor.flatten()

    threshold_tmp = torch.quantile(tmp1, 1 - rate)

    mask = (tensor < threshold_tmp).float()

    return mask


def js_divergence(P, Q, eps=1e-10):
    # Normalize the tensors to get probability distributions
    P = P / (torch.sum(P) + eps)
    Q = Q / (torch.sum(Q) + eps)

    # Calculate the midpoint distribution
    M = 0.5 * (P + Q)

    # Calculate the KL divergence between P and M and between Q and M
    def kl_divergence(A, B):
        A = A + eps  # Avoid log(0) by adding a small constant
        B = B + eps  # Avoid division by zero
        return torch.sum(A * torch.log(A / B))

    Dkl_PM = kl_divergence(P, M)
    Dkl_QM = kl_divergence(Q, M)

    # Calculate the Jensen-Shannon Divergence
    JSD = 0.5 * (Dkl_PM + Dkl_QM)

    # Normalize JSD by dividing by log(2), limiting the the output in the range [0, 1]
    JSD /= torch.log(torch.tensor(2.0))

    return JSD
