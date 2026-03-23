import torch
import torch.nn as nn
from torchvision import transforms
from lib.id_loss import Backbone


class ArcFace(nn.Module):
    """Pre-trained ArcFace feature extractor (IR-SE50).

    Loads weights from ``models/pretrained/arcface/model_ir_se50.pth``.
    Returns L2-normalised 512-d facial identity embeddings.
    """

    WEIGHTS_PATH = 'models/pretrained/arcface/model_ir_se50.pth'

    def __init__(self):
        super().__init__()
        self.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.net.load_state_dict(torch.load(self.WEIGHTS_PATH, map_location='cpu'))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))

    def forward(self, x):
        """Extract ArcFace embeddings.

        Args:
            x (torch.Tensor): face images, shape (B, 3, H, W), values in
                              [0, 1] or normalised — crop/pool is applied
                              internally.

        Returns:
            torch.Tensor: L2-normalised embeddings of shape (B, 512).
        """
        # Central crop used by ArcFace
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        return self.net(x)
