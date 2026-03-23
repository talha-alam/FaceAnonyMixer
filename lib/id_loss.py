import torch
import torch.nn as nn
from torchvision import transforms
from collections import namedtuple
from torch.nn import (
    Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid,
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module,
    Linear, BatchNorm1d, Dropout,
)
import torch.nn.functional as F


# ── IDLoss ────────────────────────────────────────────────────────────────────

class IDLoss(nn.Module):
    """Anonymity loss based on ArcFace cosine similarity.

    Penalises protected faces that are too similar to their originals in the
    ArcFace embedding space::

        L_anon = max(0, cos(I(x_p), I(x_r)) - m)

    where ``m`` is a margin (default 0 → maximum anonymisation).

    Args:
        id_margin (float): cosine similarity margin.
    """

    WEIGHTS_PATH = 'models/pretrained/arcface/model_ir_se50.pth'

    def __init__(self, id_margin: float = 0.0):
        super().__init__()
        self.id_margin = id_margin

        self.facenet = Backbone(input_size=112, num_layers=50,
                                drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(
            torch.load(self.WEIGHTS_PATH, map_location='cpu')
        )
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        for module in (self.facenet, self.face_pool):
            for p in module.parameters():
                p.requires_grad = False

        self.id_transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
        ])

    def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        return self.facenet(x)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = y.shape[0]
        y_feats     = self.extract_feats(self.id_transform(y)).detach()
        y_hat_feats = self.extract_feats(self.id_transform(y_hat))

        loss = sum(
            torch.abs(y_hat_feats[i].dot(y_feats[i]) - self.id_margin)
            for i in range(n)
        )
        return loss / n


# ── ArcFace backbone (IR-SE ResNet) ──────────────────────────────────────────
# Adapted from TreB1eN/InsightFace_Pytorch

class Flatten(Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm)


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """Named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        return [
            get_block(64,  64,  3),
            get_block(64,  128, 4),
            get_block(128, 256, 14),
            get_block(256, 512, 3),
        ]
    elif num_layers == 100:
        return [
            get_block(64,  64,  3),
            get_block(64,  128, 13),
            get_block(128, 256, 30),
            get_block(256, 512, 3),
        ]
    elif num_layers == 152:
        return [
            get_block(64,  64,  3),
            get_block(64,  128, 8),
            get_block(128, 256, 36),
            get_block(256, 512, 3),
        ]
    raise ValueError(f'num_layers must be 50, 100 or 152, got {num_layers}')


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1  = Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2  = Conv2d(channels // reduction, channels, 1, bias=False)
        self.sig  = Sigmoid()

    def forward(self, x):
        w = self.sig(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * w


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        self.shortcut_layer = (
            MaxPool2d(1, stride) if in_channel == depth
            else Sequential(Conv2d(in_channel, depth, 1, stride, bias=False),
                            BatchNorm2d(depth))
        )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, 3, 1, 1, bias=False), PReLU(depth),
            Conv2d(depth,      depth, 3, stride, 1, bias=False), BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        self.shortcut_layer = (
            MaxPool2d(1, stride) if in_channel == depth
            else Sequential(Conv2d(in_channel, depth, 1, stride, bias=False),
                            BatchNorm2d(depth))
        )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, 3, 1, 1, bias=False), PReLU(depth),
            Conv2d(depth,      depth, 3, stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir',
                 drop_ratio=0.4, affine=True):
        super().__init__()
        assert input_size in (112, 224)
        assert num_layers in (50, 100, 152)
        assert mode in ('ir', 'ir_se')

        unit = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        blocks = get_blocks(num_layers)

        self.input_layer = Sequential(
            Conv2d(3, 64, 3, 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        flat_size = 7 * 7 if input_size == 112 else 14 * 14
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(drop_ratio),
            Flatten(),
            Linear(512 * flat_size, 512),
            BatchNorm1d(512, affine=affine),
        )
        self.body = Sequential(*[
            unit(b.in_channel, b.depth, b.stride)
            for block in blocks for b in block
        ])

    def forward(self, x):
        return l2_norm(self.output_layer(self.body(self.input_layer(x))))


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
