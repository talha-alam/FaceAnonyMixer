import torch
import torch.nn as nn
from torchvision import transforms
from lib.id_loss import Backbone


class ConsistencyLoss(nn.Module):
    """Identity preservation loss (L_idp).

    Enforces that protected faces of the **same person** (produced from
    multiple augmented views under the same key) converge to a common
    pseudo-identity in ArcFace embedding space::

        L_idp = 1/(N(N-1)) * sum_{i!=j} || I(x_p^i) - I(x_p^j) ||_2

    Args:
        None — weights are loaded from the canonical ArcFace checkpoint.
    """

    WEIGHTS_PATH = 'models/pretrained/arcface/model_ir_se50.pth'

    def __init__(self):
        super().__init__()
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

    def forward(self, images_batch: torch.Tensor) -> torch.Tensor:
        """Compute mean pairwise L2 distance over embeddings.

        Args:
            images_batch (torch.Tensor): (N, 3, H, W) batch of *N* protected
                                         face images of the same individual.

        Returns:
            Scalar consistency loss (0 if N < 2).
        """
        n = images_batch.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=images_batch.device,
                                requires_grad=True)

        feats = [
            self.extract_feats(self.id_transform(images_batch[i:i+1]))
            for i in range(n)
        ]

        total = torch.tensor(0.0, device=images_batch.device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total = total + torch.norm(feats[i] - feats[j], p=2)
                count += 1

        return total / count
