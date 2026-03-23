import os.path as osp
import torch
import torch.nn as nn
from torchvision import transforms
import clip
from lib.config import FARL_PRETRAIN_MODEL


class AttrLoss(nn.Module):
    """Identity-agnostic attribute preservation loss.

    Minimises the L1 distance between the feature representations of the
    original and the protected face in an attribute-encoding space (FaRL,
    CLIP, or DINO)::

        L_attr = || A(x_p) - A(x_r) ||_1

    Args:
        feat_ext (str): feature extractor to use — ``'farl'`` (default),
                        ``'clip'``, or ``'dino'``.
        use_cuda (bool): move model to GPU if ``True``.
    """

    _FARL_WEIGHTS = osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL)

    def __init__(self, feat_ext: str = 'farl', use_cuda: bool = True):
        super().__init__()
        if feat_ext not in ('clip', 'farl', 'dino'):
            raise NotImplementedError(
                f"feat_ext must be 'clip', 'farl' or 'dino', got '{feat_ext}'"
            )

        self.feat_ext = feat_ext
        device = 'cuda' if use_cuda else 'cpu'

        if feat_ext == 'clip':
            model, _ = clip.load('ViT-B/32', device=device, jit=False)
            model.float().eval()
            self.feat_ext_model = model.visual
            self.feat_ext_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])

        elif feat_ext == 'farl':
            model, _ = clip.load('ViT-B/16', device=device, jit=False)
            state = torch.load(self._FARL_WEIGHTS, map_location='cpu')
            model.load_state_dict(state['state_dict'], strict=False)
            model.float().eval()
            self.feat_ext_model = model.visual
            self.feat_ext_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
            ])

        else:  # dino
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            model.float().eval()
            self.feat_ext_model = model
            self.feat_ext_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])

        self.feat_ext_model.to(device)
        for p in self.feat_ext_model.parameters():
            p.requires_grad = False

        self.l1 = nn.L1Loss()

    # ── ViT token-level feature extraction ───────────────────────────────────

    def _extract_visual(self, x: torch.Tensor):
        """Extract intermediate token features from a ViT visual encoder."""
        m = self.feat_ext_model
        x = m.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = m.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1) + m.positional_embedding.to(x.dtype)
        x = m.ln_pre(x)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        tokens = x.clone()
        out = m.ln_post(x[:, 0, :])
        if m.proj is not None:
            out = out @ m.proj
        return out, tokens

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute L1 attribute loss between *y_hat* (protected) and *y* (original).

        Args:
            y_hat: protected face, shape (B, 3, H, W).
            y:     original face,  shape (B, 3, H, W).

        Returns:
            Scalar attribute loss.
        """
        _, y_fts     = self._extract_visual(self.feat_ext_transform(y))
        _, y_hat_fts = self._extract_visual(self.feat_ext_transform(y_hat))

        loss = sum(
            self.l1(y_fts[:, i].float(), y_hat_fts[:, i].float())
            for i in range(y_fts.shape[1])
        )
        return loss
