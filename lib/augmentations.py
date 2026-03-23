import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class ImageAugmenter:
    """Generate N augmented views of a face image for the identity preservation loss.

    Each view applies a random combination of:
        - Horizontal flip
        - Small rotation (±10°)
        - Brightness / contrast / saturation jitter
        - Small random crop + resize back to original size

    The augmentations are intentionally mild so that the face identity is
    preserved across views (needed for the consistency loss to be meaningful).

    Args:
        n_augmentations (int): number of augmented views to produce.
    """

    def __init__(self, n_augmentations: int = 5):
        self.n_augmentations = n_augmentations

        self.color_jitter = T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
        )

    def _augment_one(self, img: torch.Tensor) -> torch.Tensor:
        """Apply a single random augmentation to *img* (C, H, W) in [-1, 1]."""
        # Work in [0, 1] for torchvision ops
        x = (img.clamp(-1, 1) + 1.0) / 2.0  # → [0, 1]

        # Random horizontal flip
        if random.random() < 0.5:
            x = TF.hflip(x)

        # Random small rotation
        angle = random.uniform(-10.0, 10.0)
        x = TF.rotate(x, angle)

        # Color jitter
        x = self.color_jitter(x)

        # Small random crop → resize back
        _, h, w = x.shape
        crop_frac = random.uniform(0.85, 1.0)
        crop_h, crop_w = int(h * crop_frac), int(w * crop_frac)
        top  = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        x = TF.crop(x, top, left, crop_h, crop_w)
        x = TF.resize(x, [h, w], antialias=True)

        # Back to [-1, 1]
        return x * 2.0 - 1.0

    def __call__(self, img: torch.Tensor) -> list:
        """Return a list of N augmented tensors.

        Args:
            img (torch.Tensor): single image tensor of shape (1, C, H, W) or
                                (C, H, W).

        Returns:
            list[torch.Tensor]: each element has shape (1, C, H, W).
        """
        if img.dim() == 4:
            img = img.squeeze(0)  # → (C, H, W)

        return [self._augment_one(img).unsqueeze(0) for _ in range(self.n_augmentations)]
