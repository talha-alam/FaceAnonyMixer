"""Shared building blocks for IR / IR-SE ResNet encoders.

These are re-exported here so that ``models/psp.py`` and
``models/encoders/psp_encoders.py`` can both import from one place without
circular dependencies.
"""

import torch.nn.functional as F
from lib.id_loss import (  # noqa: F401 — re-export
    Bottleneck,
    get_block,
    get_blocks,
    bottleneck_IR,
    bottleneck_IR_SE,
    Flatten,
    l2_norm,
)


def _upsample_add(x, y):
    """Upsample *x* to *y*'s spatial size and add element-wise (FPN merge)."""
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
