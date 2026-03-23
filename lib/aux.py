import os
import os.path as osp
import hashlib
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ── Image conversion ──────────────────────────────────────────────────────────

def tensor2image(tensor, adaptive=False):
    """Convert a CHW float tensor in [-1, 1] (or [0, 1]) to a PIL RGB Image.

    Args:
        tensor (torch.Tensor): image tensor of shape (C, H, W) or (1, C, H, W).
        adaptive (bool): if True, normalise to [0, 255] using the tensor's own
                         min/max instead of assuming [-1, 1] input range.

    Returns:
        PIL.Image.Image: RGB image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if adaptive:
        t_min = tensor.min()
        t_max = tensor.max()
        tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
    else:
        tensor = (tensor + 1.0) / 2.0  # [-1, 1] → [0, 1]

    tensor = tensor.clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Experiment directory ──────────────────────────────────────────────────────

def anon_exp_dir(args_dict):
    """Return (and create) a reproducible experiment output directory.

    The directory is placed under ``experiments/`` and named by a short SHA-1
    of the JSON-serialised arguments, so re-running with the same config
    resumes from the same folder.

    Args:
        args_dict (dict): parsed argument dictionary (``vars(args)``).

    Returns:
        str: path to the experiment directory.
    """
    # Exclude keys that are irrelevant to the experiment identity.
    skip = {'verbose', 'cuda', 'gpu_id'}
    stable = {k: v for k, v in sorted(args_dict.items()) if k not in skip}
    digest = hashlib.sha1(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:10]
    exp_dir = osp.join('experiments', digest)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# ── Multi-GPU passthrough ─────────────────────────────────────────────────────

class DataParallelPassthrough(nn.DataParallel):
    """nn.DataParallel wrapper that transparently proxies attribute access to
    the wrapped module, so code like ``model.n_latent`` keeps working after
    wrapping with DataParallel.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
