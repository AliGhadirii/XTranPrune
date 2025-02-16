import random
import sys
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Sampler
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
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return {
        "min": torch.min(tensor).item(),
        "Q1": torch.quantile(tensor, 0.25).item(),
        "Q2": torch.quantile(tensor, 0.5).item(),
        "Q3": torch.quantile(tensor, 0.75).item(),
        "max": torch.max(tensor).item(),
        "mean": torch.mean(tensor).item(),
        "std": torch.std(tensor).item(),
    }


def preprocess_matrix(
    matrix,
    clip_threshold=None,
    log_transform=False,
    normalize=False,
    min_max_scale=False,
    sum_normalize=False,
    global_normalize=False,
):
    # Initialize a placeholder for the preprocessed matrix with the same shape
    preprocessed_matrix = torch.zeros_like(
        matrix,
        dtype=matrix.dtype,
        device=matrix.device,
    )

    # Iterate through each block and each head
    for b in range(matrix.shape[0]):  # For each block
        for h in range(matrix.shape[1]):  # For each head
            # Extract the current matrix (197x197) for this block and head
            current_matrix = matrix[b, h]

            # 1. Clipping
            if clip_threshold is not None:
                current_matrix = torch.clamp(current_matrix, min=clip_threshold)

            # 2. Log Transformation
            if log_transform:
                current_matrix = torch.log1p(
                    current_matrix
                )  # Use log1p to handle log(0) issues

            # 3. Normalize the weights so that each row sums up to 1
            if normalize:
                row_sums = current_matrix.sum(dim=1, keepdim=True)
                # Protect against division by zero
                row_sums[row_sums == 0] = 1
                current_matrix = current_matrix / row_sums

            # 4. Min-Max Scaling to [0, 1] range
            if min_max_scale:
                min_val, max_val = current_matrix.min(), current_matrix.max()
                if max_val - min_val != 0:  # Avoid division by zero
                    current_matrix = (current_matrix - min_val) / (max_val - min_val)
                else:
                    current_matrix = torch.zeros_like(current_matrix)

            if sum_normalize:
                total_sum = current_matrix.sum()
                if total_sum != 0:
                    current_matrix /= total_sum

            # Store the preprocessed matrix back into the placeholder
            preprocessed_matrix[b, h] = current_matrix

    # Global normalization across all heads and blocks
    if global_normalize:
        total_sum = preprocessed_matrix.sum()
        if total_sum != 0:
            preprocessed_matrix /= total_sum

    return preprocessed_matrix


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


class StratifiedSampler(Sampler):
    def __init__(self, df):
        self.df = df
        self.indices = self._get_balanced_indices()

    def _get_balanced_indices(self):
        indices = []
        min_samples = min(self.df.groupby(["high", "fitzpatrick_binary"]).size())

        for _, group in self.df.groupby(["high", "fitzpatrick_binary"]):
            group_indices = group.index.tolist()
            sampled_indices = np.random.choice(group_indices, min_samples, replace=True)
            indices.extend(sampled_indices)

        return indices

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class CustomStratifiedSampler(Sampler):
    def __init__(self, df, label_col, sensitive_attr_col, batch_size):
        self.df = df
        self.label_col = label_col
        self.sensitive_attr_col = sensitive_attr_col
        self.batch_size = batch_size

        # Create a dictionary to store indices for each combination of label and sensitive attribute
        self.combination_indices = {}
        labels = df[label_col].values
        sensitive_attributes = df[sensitive_attr_col].values
        for label in np.unique(labels):
            for attr in np.unique(sensitive_attributes):
                key = (label, attr)
                self.combination_indices[key] = np.where(
                    (labels == label) & (sensitive_attributes == attr)
                )[0].tolist()

        self.unused_samples = {
            key: set(indices) for key, indices in self.combination_indices.items()
        }
        self.keys = list(self.combination_indices.keys())

    def __iter__(self):
        batch = []
        while True:
            for key in self.keys:
                if len(self.unused_samples[key]) > 0:
                    sample_idx = self.unused_samples[key].pop()
                else:
                    sample_idx = np.random.choice(self.combination_indices[key])
                batch.append(sample_idx)
                if len(batch) == self.batch_size:
                    for idx in batch:
                        yield idx
                    batch = []
            if all(len(self.unused_samples[key]) == 0 for key in self.unused_samples):
                break
        if batch:
            for idx in batch:
                yield idx

    def __len__(self):
        return len(self.df)
