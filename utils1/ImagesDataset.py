from torch.utils.data import Dataset
from PIL import Image
from utils1.data_utils import make_dataset


class ImagesDataset(Dataset):
    """Simple flat image folder dataset.

    Walks *source_root* recursively and returns ``(filename_stem, image_tensor)``
    pairs.  Used by the standalone inversion entry-point.

    Args:
        source_root (str): directory containing images.
        source_transform: torchvision transform applied to each image.
    """

    def __init__(self, source_root: str, source_transform=None):
        self.source_paths     = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        img = Image.open(from_path).convert('RGB')
        if self.source_transform:
            img = self.source_transform(img)
        return fname, img
