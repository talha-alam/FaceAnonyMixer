import torch
import torch.nn as nn
from models.encoders import psp_encoders


def get_keys(d, name):
    """Extract sub-module weights from a checkpoint dict."""
    if 'state_dict' in d:
        d = d['state_dict']
    return {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}


class pSp(nn.Module):
    """Wrapper around the e4e / pSp encoder.

    Loads a pre-trained e4e checkpoint and exposes ``encoder`` (the
    ``Encoder4Editing`` backbone) together with the mean latent vector
    ``latent_avg`` used to offset predictions into W+.

    Args:
        opts (argparse.Namespace): must contain at least:
            * ``checkpoint_path`` – path to the ``.pt`` checkpoint.
            * ``device``          – ``'cuda'`` or ``'cpu'``.
            * ``encoder_type``    – ``'Encoder4Editing'`` (default for e4e).
            * ``start_from_latent_avg`` – bool.
            * ``stylegan_size``   – generator output resolution (e.g. 1024).
    """

    def __init__(self, opts):
        super().__init__()
        self.opts      = opts
        self.encoder   = self._build_encoder()
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))
        self._load_weights()

    # ── encoder construction ─────────────────────────────────────────────────

    def _build_encoder(self):
        etype = getattr(self.opts, 'encoder_type', 'Encoder4Editing')
        if etype == 'GradualStyleEncoder':
            return psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif etype == 'Encoder4Editing':
            return psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise ValueError(f'Unknown encoder_type: {etype!r}')

    def _load_weights(self):
        ckpt_path = getattr(self.opts, 'checkpoint_path', None)
        if ckpt_path is None:
            return

        print(f'  \\__Loading e4e checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self._load_latent_avg(ckpt)

    def _load_latent_avg(self, ckpt, repeat=None):
        device = getattr(self.opts, 'device', 'cpu')
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(device)
        else:
            self.latent_avg = None

        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, resize=True, return_latents=False):
        codes = self.encoder(x)

        if getattr(self.opts, 'start_from_latent_avg', True) and self.latent_avg is not None:
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if return_latents:
            return codes
        return codes
