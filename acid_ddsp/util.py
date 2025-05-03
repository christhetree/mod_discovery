import importlib
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Union, Any, Dict, Optional

import torch as tr
import torch.nn.functional as F
import yaml
from scipy.stats import loguniform
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def get_activation(act_name: str) -> nn.Module:
    act_name = act_name.lower()
    if not act_name or act_name == "none":
        return nn.Identity()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "softmax":
        return nn.Softmax(dim=-1)
    elif act_name == "prelu":
        return nn.PReLU()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {act_name}")


def interpolate_dim(
    x: T,
    n: int,
    dim: int = -1,
    mode: str = "linear",
    align_corners: Optional[bool] = True,
) -> T:
    n_dim = x.ndim
    assert 0 < n_dim <= 3
    if dim < 0:
        dim = n_dim + dim
    assert 0 <= dim < n_dim
    if x.size(dim) == n:
        return x

    swapped_dims = False
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        assert dim != 0  # TODO(cm)
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        assert dim != 0  # TODO(cm)
        if dim == 1:
            x = x.swapaxes(1, 2)
            swapped_dims = True
    x = F.interpolate(x, n, mode=mode, align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    elif swapped_dims:
        x = x.swapaxes(1, 2)
    return x


def sample_uniform(
    low: float, high: float, n: int = 1, rand_gen: Optional[tr.Generator] = None
) -> Union[float, T]:
    x = (tr.rand(n, generator=rand_gen) * (high - low)) + low
    if n == 1:
        return x.item()
    return x


def sample_log_uniform(
    low: float, high: float, n: int = 1, seed: Optional[int] = None
) -> Union[float, T]:
    # TODO(cm): replace with torch
    if low == high:
        if n == 1:
            return low
        else:
            return tr.full(size=(n,), fill_value=low)
    x = loguniform.rvs(low, high, size=n, random_state=seed)
    if n == 1:
        return float(x)
    return tr.from_numpy(x).float()


def calc_h(a: T, b: T, n_frames: int = 50, n_fft: int = 1024) -> T:
    assert a.ndim == 3
    assert a.shape == b.shape
    a = interpolate_dim(a, n_frames, dim=1)
    b = interpolate_dim(b, n_frames, dim=1)
    A = tr.fft.rfft(a, n_fft)
    B = tr.fft.rfft(b, n_fft)
    H = B / A  # TODO(cm): Make more stable
    return H


def load_class_from_config(config_path: str) -> (Any, Dict[str, Any]):
    assert os.path.isfile(config_path)
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    class_path = config["class_path"]
    tokens = class_path.split(".")
    class_name = tokens[-1]
    module_path = ".".join(tokens[:-1])
    class_ = getattr(importlib.import_module(module_path), class_name)
    init_args = config["init_args"]
    return class_, init_args


def extract_model_and_synth_from_config(
    config_path: str, ckpt_path: Optional[str] = None
) -> (nn.Module, nn.Module):
    assert os.path.isfile(config_path)
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    del config["ckpt_path"]

    tmp_config_file = NamedTemporaryFile()
    with open(tmp_config_file.name, "w") as out_f:
        yaml.dump(config, out_f)
        from cli import CustomLightningCLI

        cli = CustomLightningCLI(
            args=["-c", tmp_config_file.name],
            trainer_defaults=CustomLightningCLI.trainer_defaults,
            run=False,
        )
    lm = cli.model

    if ckpt_path is not None:
        log.info(f"Loading checkpoint from {ckpt_path}")
        assert os.path.isfile(ckpt_path)
        ckpt_data = tr.load(ckpt_path, map_location=tr.device("cpu"))
        lm.load_state_dict(ckpt_data["state_dict"])

    return lm.model, lm.synth_hat


def stable_softmax(logits: T, tau: float = 1.0) -> T:
    assert tau > 0, f"Invalid temperature: {tau}, must be > 0"
    # Subtract the max logit for numerical stability
    max_logit = tr.max(logits, dim=-1, keepdim=True).values
    logits = logits - max_logit
    # Apply temperature scaling
    scaled_logits = logits / tau
    # Compute the exponential
    exp_logits = tr.exp(scaled_logits)
    # Normalize the probabilities
    sum_exp_logits = tr.sum(exp_logits, dim=-1, keepdim=True)
    softmax_probs = exp_logits / sum_exp_logits
    return softmax_probs


def load_class_from_yaml(config_path: str) -> Any:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    class_path = config["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    cls_instantiated = cls(**config["init_args"])
    return cls_instantiated


def scale_function(x: T, eps: float = 1e-7) -> T:
    """
    Scale function as per DDSP paper Section B.5, equation 5
    """
    return 2 * (tr.sigmoid(x) ** tr.log(tr.tensor(10).to(x.device))) + eps


def compute_lstsq_with_bias(x_hat: T, x: T) -> T:
    """
    Given x and x_hat of shape (bs, n_signals, n_samples), compute the best
    linear combination matrix W and bias vector b (per batch) such that:
        x ≈ W @ x_hat + b
    using a batched least-squares approach.

    Args:
        x (Tensor): Target tensor of shape (bs, n_signals, n_samples).
        x_hat (Tensor): Basis tensor of shape (bs, n_signals, n_samples).

    Returns:

    """
    bs, n_signals, n_samples = x_hat.shape
    assert x.ndim == 3
    assert x.size(0) == bs
    assert x.size(2) == n_samples

    # Augment x_hat with a row of ones to account for the bias term.
    # ones shape: (bs, 1, n_samples)
    ones = tr.ones(bs, 1, n_samples, device=x_hat.device, dtype=x_hat.dtype)
    x_hat_aug = tr.cat([x_hat, ones], dim=1)  # shape: (bs, n_signals+1, n_samples)

    # Transpose the last two dimensions so that we set up the least-squares problem as:
    # A @ (solution) ≈ B
    A = x_hat_aug.transpose(1, 2)  # shape: (bs, n_samples, n_signals+1)
    B = x.transpose(1, 2)  # shape: (bs, n_samples, n_signals)

    # Solve the least-squares problem for the augmented system.
    lstsq_result = tr.linalg.lstsq(A, B)
    solution = lstsq_result.solution  # shape: (bs, n_signals+1, n_signals)

    # The solution consists of weights and bias:
    # The first n_signals rows correspond to the weight matrix (transposed), and the last row is the bias.
    W_t = solution[:, :-1, :]  # shape: (bs, n_signals, n_signals)
    bias_t = solution[:, -1:, :]  # shape: (bs, 1, n_signals)

    # Transpose to obtain the weight matrix in the right orientation.
    W = W_t.transpose(1, 2)  # shape: (bs, n_signals, n_signals)
    bias = bias_t.transpose(1, 2)  # shape: (bs, n_signals, 1)

    # Compute the predicted x using the estimated weights and bias.
    # Note: bias is added to each sample in the time dimension.
    x_pred = tr.bmm(W, x_hat) + bias  # shape: (bs, n_signals, n_samples)
    return x_pred
