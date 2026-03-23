"""GenForce model registry and generator builder.

This module provides:
    - ``MODEL_ZOO``      : dict mapping model names to their config + download URL.
    - ``build_generator``: factory that constructs a StyleGAN2 generator.
    - ``EqualLinear``    : weight-equalised linear layer (used by e4e encoders).

If the full GenForce source tree has been cloned via ``scripts/setup_genforce.sh``
the real GenForce implementations are imported transparently.  Otherwise a
minimal self-contained implementation is used so that the rest of the codebase
can be imported and used without requiring the GenForce submodule.

Run ``bash scripts/setup_genforce.sh`` once to install the full GenForce
generators (needed for training / inference).
"""

from __future__ import annotations

import math
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try importing the real GenForce implementations first.
# ---------------------------------------------------------------------------
_GENFORCE_AVAILABLE = False
try:
    # If setup_genforce.sh has been run, the actual model files live next to
    # this shim inside models/genforce/.
    from .stylegan_generator import StyleGAN2Generator  # type: ignore
    _GENFORCE_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# MODEL_ZOO
# ---------------------------------------------------------------------------
MODEL_ZOO: dict = {
    'stylegan2_ffhq1024': {
        'url':          'https://www.dropbox.com/s/t2jyyj1dn8olgiv/stylegan2_ffhq1024.pth?dl=1',
        'resolution':   1024,
        'image_channels': 3,
        'z_space_dim':  512,
        'w_space_dim':  512,
        'num_mapping_layers': 8,
    },
    'stylegan2_ffhq512': {
        'url':          'https://www.dropbox.com/s/bk4n0wr1g4gdmzj/stylegan2_ffhq512.pth?dl=1',
        'resolution':   512,
        'image_channels': 3,
        'z_space_dim':  512,
        'w_space_dim':  512,
        'num_mapping_layers': 8,
    },
}

# ---------------------------------------------------------------------------
# EqualLinear  (weight-equalised linear layer, used by e4e encoder)
# ---------------------------------------------------------------------------
class EqualLinear(nn.Module):
    """Weight-equalised linear layer from Progressive GAN / StyleGAN.

    Divides weights by the He-initialisation constant at forward time so that
    the effective learning-rate is the same across layers regardless of their
    fan-in.

    Args:
        in_dim  (int)  : input features.
        out_dim (int)  : output features.
        bias    (bool) : add a learnable bias (default True).
        lr_mul  (float): learning-rate multiplier for the mapping network
                         (set to 0.01 in the original StyleGAN mapping layers).
        activation     : optional activation name (``'fused_lrelu'`` or None).
    """

    def __init__(self, in_dim: int, out_dim: int, *,
                 bias: bool = True, lr_mul: float = 1.0,
                 activation=None):
        super().__init__()
        self.weight     = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias       = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale      = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul     = lr_mul
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        if self.bias is not None:
            b = self.bias * self.lr_mul
            out = F.linear(x, w, b)
        else:
            out = F.linear(x, w)

        if self.activation == 'fused_lrelu':
            out = F.leaky_relu(out, negative_slope=0.2) * math.sqrt(2)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in={self.weight.shape[1]}, out={self.weight.shape[0]}, '
                f'lr_mul={self.lr_mul})')


# ---------------------------------------------------------------------------
# Minimal StyleGAN2 generator (fallback when GenForce is not installed)
# ---------------------------------------------------------------------------

class _PixelNorm(nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class _MappingNetwork(nn.Module):
    """8-layer mapping network: Z → W."""

    def __init__(self, z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        layers: list = [_PixelNorm()]
        for _ in range(num_layers):
            layers += [EqualLinear(z_dim, w_dim, lr_mul=lr_mul,
                                   activation='fused_lrelu')]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _MinimalStyleGAN2(nn.Module):
    """Minimal StyleGAN2 wrapper sufficient for latent-space operations.

    This class exposes the same interface as the full GenForce implementation
    (``get_w``, ``synthesis``, ``n_latent``, ``z_space_dim``, ``dim_z``) so
    that the rest of FaceAnonyMixer works unchanged.

    **Important:** this minimal version does NOT implement the full synthesis
    network.  It is provided only so that the codebase can be imported and
    the latent-space operations (inversion, mixing, optimization) can run once
    the real checkpoint has been loaded via ``load_state_dict``.  If you need
    actual image generation, run ``bash scripts/setup_genforce.sh`` to install
    the full GenForce implementation.
    """

    def __init__(self, resolution=1024, z_space_dim=512, w_space_dim=512,
                 num_mapping_layers=8, image_channels=3,
                 latent_is_w=False, latent_is_s=False):
        super().__init__()
        self.resolution        = resolution
        self.z_space_dim       = z_space_dim
        self.w_space_dim       = w_space_dim
        self.num_mapping_layers = num_mapping_layers
        self.latent_is_w       = latent_is_w
        self.latent_is_s       = latent_is_s
        self.n_latent          = int(math.log2(resolution)) * 2 - 2  # 18 for 1024
        self.dim_z             = z_space_dim

        self.mapping = _MappingNetwork(z_space_dim, w_space_dim, num_mapping_layers)

        # Truncation buffer (matches GenForce checkpoint layout)
        self.register_buffer('w_avg', torch.zeros(w_space_dim))

    # ── public interface ──────────────────────────────────────────────────────

    def get_w(self, z: torch.Tensor, truncation: float = 1.0) -> torch.Tensor:
        """Map Z → W, apply truncation trick, return (B, n_latent, w_dim)."""
        w = self.mapping(z)                         # (B, w_dim)
        if truncation < 1.0:
            w = self.w_avg + truncation * (w - self.w_avg)
        return w.unsqueeze(1).repeat(1, self.n_latent, 1)  # (B, 18, 512)

    def get_s(self, wp: torch.Tensor) -> torch.Tensor:
        """Identity mapping from W+ to 'S' space (placeholder)."""
        return wp

    def synthesis(self, wp: torch.Tensor) -> torch.Tensor:
        """Decode W+ codes to images.

        Requires the full GenForce synthesis network.  Raises a helpful error
        if it has not been installed.
        """
        raise RuntimeError(
            'The full GenForce synthesis network is required to decode images.\n'
            'Run:  bash scripts/setup_genforce.sh\n'
            'then re-run your script.'
        )

    def forward(self, wp: torch.Tensor) -> torch.Tensor:
        return self.synthesis(wp)


# ---------------------------------------------------------------------------
# build_generator  — public factory
# ---------------------------------------------------------------------------

def build_generator(resolution: int,
                    image_channels: int,
                    z_space_dim: int,
                    w_space_dim: int,
                    num_mapping_layers: int,
                    latent_is_w: bool = False,
                    latent_is_s: bool = False,
                    **kwargs) -> nn.Module:
    """Construct and return a StyleGAN2 generator.

    If the full GenForce ``StyleGAN2Generator`` class has been imported
    (i.e. ``scripts/setup_genforce.sh`` has been run), that class is
    instantiated.  Otherwise the minimal fallback is used.

    Args:
        resolution           : output image resolution (e.g. 1024).
        image_channels       : number of output image channels (3 for RGB).
        z_space_dim          : dimensionality of the Z (noise) space.
        w_space_dim          : dimensionality of the W (style) space.
        num_mapping_layers   : depth of the mapping network.
        latent_is_w (bool)   : whether inputs are already W codes.
        latent_is_s (bool)   : whether inputs are already S codes.

    Returns:
        nn.Module: generator instance.
    """
    if _GENFORCE_AVAILABLE:
        return StyleGAN2Generator(
            resolution=resolution,
            image_channels=image_channels,
            z_space_dim=z_space_dim,
            w_space_dim=w_space_dim,
            num_mapping_layers=num_mapping_layers,
            latent_is_w=latent_is_w,
            latent_is_s=latent_is_s,
            **kwargs,
        )

    return _MinimalStyleGAN2(
        resolution=resolution,
        z_space_dim=z_space_dim,
        w_space_dim=w_space_dim,
        num_mapping_layers=num_mapping_layers,
        image_channels=image_channels,
        latent_is_w=latent_is_w,
        latent_is_s=latent_is_s,
    )
