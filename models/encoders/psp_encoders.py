import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add

try:
    from models.genforce.models import EqualLinear
except ImportError:
    # Fallback minimal EqualLinear if genforce is not yet initialised
    class EqualLinear(nn.Module):
        def __init__(self, in_dim, out_dim, lr_mul=1):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.linear(x)


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c   = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules  = [Conv2d(in_c, out_c, 3, stride=2, padding=1), nn.LeakyReLU()]
        modules += [Conv2d(out_c, out_c, 3, stride=2, padding=1), nn.LeakyReLU()] * (num_pools - 1)
        self.convs  = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x).view(-1, self.out_c)
        return self.linear(x)


class GradualStyleEncoder(Module):
    """pSp-style encoder that outputs one style vector per StyleGAN layer."""

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in (50, 100, 152)
        assert mode in ('ir', 'ir_se')
        unit  = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        blocks = get_blocks(num_layers)

        self.input_layer = Sequential(
            Conv2d(3, 64, 3, 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.body = Sequential(*[unit(b.in_channel, b.depth, b.stride)
                                  for block in blocks for b in block])

        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind  = 3
        self.middle_ind  = 7

        self.styles = nn.ModuleList([
            GradualStyleBlock(512, 512, 16 if i < self.coarse_ind
                              else 32 if i < self.middle_ind else 64)
            for i in range(self.style_count)
        ])
        self.latlayer1 = nn.Conv2d(256, 512, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(128, 512, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        latents, modulelist = [], list(self.body._modules.values())
        for i, layer in enumerate(modulelist):
            x = layer(x)
            if i == 6:  c1 = x
            elif i == 20: c2 = x
            elif i == 23: c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        return torch.stack(latents, dim=1)


class Encoder4Editing(Module):
    """e4e encoder: produces a base W code plus per-layer deltas."""

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in (50, 100, 152)
        assert mode in ('ir', 'ir_se')
        unit  = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        blocks = get_blocks(num_layers)

        self.input_layer = Sequential(
            Conv2d(3, 64, 3, 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.body = Sequential(*[unit(b.in_channel, b.depth, b.stride)
                                  for block in blocks for b in block])

        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind  = 3
        self.middle_ind  = 7

        self.styles = nn.ModuleList([
            GradualStyleBlock(512, 512, 16 if i < self.coarse_ind
                              else 32 if i < self.middle_ind else 64)
            for i in range(self.style_count)
        ])
        self.latlayer1 = nn.Conv2d(256, 512, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(128, 512, 1, 1, 0)

        from enum import Enum
        class ProgressiveStage(Enum):
            Inference = 18
        self.progressive_stage = ProgressiveStage.Inference

    def forward(self, x):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, layer in enumerate(modulelist):
            x = layer(x)
            if i == 6:  c1 = x
            elif i == 20: c2 = x
            elif i == 23: c3 = x

        w0 = self.styles[0](c3)
        w  = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage    = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))
                features = p1
            w[:, i] += self.styles[i](features)
        return w
