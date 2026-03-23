import os
import os.path as osp
import json
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image


class VGGFace2(data.Dataset):
    """VGGFace2 dataset with optional inversion, fake NN, and anonymization support.

    Expects the dataset root to contain a ``train/`` sub-folder organised as
    one directory per identity::

        <root_dir>/train/<identity>/<img>.jpg

    Each ``__getitem__`` returns a list of 8 elements::

        [img_orig, img_path,
         img_nn, img_nn_code,
         img_recon, img_recon_code,
         img_anon, img_anon_code]

    Missing optional items are returned as zero tensors.

    Args:
        root_dir (str): dataset root directory.
        fake_nn_map (str | None): path to a NN map JSON file.
        inv (bool): load e4e+PTI reconstructions.
        anon (str | None): path to anonymized-dataset directory.
        transform: torchvision transform applied to every image.
    """

    def __init__(self,
                 root_dir: str,
                 fake_nn_map=None,
                 inv: bool = False,
                 anon=None,
                 transform=None):
        self.root_dir  = root_dir
        self.train_dir = osp.join(root_dir, 'train')
        self.inv       = inv
        self.anon      = anon

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Inversion paths
        self.inv_data_dir  = osp.join('datasets', 'inv_pivot', 'vggface2', 'data')
        self.inv_codes_dir = osp.join('datasets', 'inv_pivot', 'vggface2', 'latent_codes')

        # NN map
        self.fake_dataset_root = None
        self.nn_map_dict       = {}
        if fake_nn_map is not None:
            if not osp.isfile(fake_nn_map):
                raise FileNotFoundError(f'NN map file not found: {fake_nn_map}')
            with open(fake_nn_map) as f:
                self.nn_map_dict = json.load(f)
            self.fake_dataset_root = osp.dirname(fake_nn_map)
        self.fake_nn_map = fake_nn_map

        self._prepare_identity_based_data()

    def _prepare_identity_based_data(self):
        identity_folders = sorted(
            f for f in os.listdir(self.train_dir)
            if osp.isdir(osp.join(self.train_dir, f))
        )
        self.images, self.labels = [], []
        for label, folder in enumerate(identity_folders):
            folder_path = osp.join(self.train_dir, folder)
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(osp.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path      = self.images[idx]
        img_basename  = osp.basename(img_path)
        identity_folder = osp.basename(osp.dirname(img_path))
        stem          = img_basename.split('.')[0]

        img_orig = self.transform(Image.open(img_path).convert('RGB'))

        # ── Fake NN ──────────────────────────────────────────────────────────
        img_nn      = torch.zeros_like(img_orig)
        img_nn_code = torch.zeros(18, 512)
        if self.fake_nn_map and img_basename in self.nn_map_dict:
            nn_key       = self.nn_map_dict[img_basename]
            nn_img_path  = osp.join(self.fake_dataset_root, nn_key, 'image.jpg')
            nn_code_path = osp.join(self.fake_dataset_root, nn_key, 'latent_code_w+.pt')
            if osp.isfile(nn_img_path):
                img_nn = self.transform(Image.open(nn_img_path).convert('RGB'))
            if osp.isfile(nn_code_path):
                img_nn_code = torch.load(nn_code_path).squeeze(0)

        # ── Inversion ────────────────────────────────────────────────────────
        img_recon      = torch.zeros_like(img_orig)
        img_recon_code = torch.zeros(18, 512)
        if self.inv:
            recon_path = osp.join(self.inv_data_dir,  f'{stem}_recon.jpg')
            code_path  = osp.join(self.inv_codes_dir, f'{stem}.pt')
            if osp.isfile(recon_path):
                img_recon = self.transform(Image.open(recon_path).convert('RGB'))
            if osp.isfile(code_path):
                img_recon_code = torch.load(code_path)

        # ── Anonymization ────────────────────────────────────────────────────
        img_anon      = torch.zeros_like(img_orig)
        img_anon_code = torch.zeros(18, 512)
        if self.anon:
            anon_img_path  = osp.join(self.anon, 'data',         identity_folder, f'{stem}.jpg')
            anon_code_path = osp.join(self.anon, 'latent_codes', identity_folder, f'{stem}.pt')
            if osp.isfile(anon_img_path):
                img_anon = self.transform(Image.open(anon_img_path).convert('RGB'))
            if osp.isfile(anon_code_path):
                img_anon_code = torch.load(anon_code_path)

        return [img_orig, img_path,
                img_nn, img_nn_code,
                img_recon, img_recon_code,
                img_anon, img_anon_code]
