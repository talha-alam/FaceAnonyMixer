import os
import os.path as osp
import json
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image


class CelebAHQ(data.Dataset):
    """CelebA-HQ dataset with optional support for inversions, fake NNs, and anonymized images.

    Supports two loading modes, selected via ``use_facial_identity``:

    * **Identity mode** (``use_facial_identity=True``, default for anonymization):
      Walks ``<root_dir>/train/<identity_folder>/*.jpg`` and assigns integer
      class labels by folder.

    * **Partition mode** (``use_facial_identity=False``):
      Uses the official CelebA train/val/test split files stored under
      ``<root_dir>/annotations/``.

    Each ``__getitem__`` call returns a list of 9 elements::

        [img_orig, img_orig_attr, img_path,
         img_nn, img_nn_code,
         img_recon, img_recon_code,
         img_anon, img_anon_code]

    Missing optional items (NN map not provided, inversion not run, etc.) are
    returned as zero tensors of the appropriate shape so that downstream code
    can unconditionally unpack all 9 slots.

    Args:
        root_dir (str): dataset root.
        subset (str): one of ``'train'``, ``'val'``, ``'test'``,
                      ``'train+val'``, ``'train+val+test'``.
        fake_nn_map (str | None): path to a JSON nearest-neighbour map
                                  produced by ``pair_unique.py``.
        inv (bool): load e4e+PTI inversions from
                    ``datasets/inv_pivot/sample_IJB-C/``.
        anon (str | None): path to an anonymized-dataset directory produced
                           by ``anonymize.py``.
        use_facial_identity (bool): use identity-folder mode (default True).
        transform: image transform applied to every loaded image.
    """

    def __init__(self,
                 root_dir: str,
                 subset: str = 'train',
                 fake_nn_map=None,
                 inv: bool = False,
                 anon=None,
                 use_facial_identity: bool = True,
                 transform=transforms.Compose([transforms.ToTensor()])):

        self.root_dir           = root_dir
        self.data_dir           = osp.join(root_dir, 'data')
        self.anno_dir           = osp.join(root_dir, 'annotations')
        self.use_facial_identity = use_facial_identity
        self.transform          = transform
        self.inv                = inv
        self.anon               = anon

        if subset not in ('train', 'val', 'test', 'train+val', 'train+val+test'):
            raise ValueError(f'Invalid subset: {subset!r}')
        self.subset = subset

        # Inversion paths
        self.inv_data_dir  = osp.join('datasets', 'inv_pivot', 'sample_IJB-C', 'data')
        self.inv_codes_dir = osp.join('datasets', 'inv_pivot', 'sample_IJB-C', 'latent_codes')

        if anon and not osp.isdir(anon):
            raise NotADirectoryError(f'Anonymized dataset directory not found: {anon}')

        # Nearest-neighbour map
        self.fake_dataset_root = None
        self.nn_type           = None
        self.nn_map_dict       = {}
        if fake_nn_map is not None:
            if not osp.isfile(fake_nn_map):
                raise FileNotFoundError(f'NN map file not found: {fake_nn_map}')
            self.fake_nn_map       = fake_nn_map
            self.fake_dataset_root = osp.dirname(fake_nn_map)
            self.nn_type           = osp.basename(fake_nn_map).split('.')[0]
            with open(fake_nn_map) as f:
                self.nn_map_dict = json.load(f)
        else:
            self.fake_nn_map = None

        # Build image list
        if self.use_facial_identity:
            self._prepare_identity_based_data()
        else:
            self._prepare_partitioned_data()

        self._load_attributes()

    # ── data preparation ──────────────────────────────────────────────────────

    def _prepare_identity_based_data(self):
        identity_dir = osp.join(self.root_dir, 'train')
        self.images, self.labels = [], []
        for label, folder in enumerate(sorted(os.listdir(identity_dir))):
            folder_path = osp.join(identity_dir, folder)
            if not osp.isdir(folder_path):
                continue
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(osp.join(folder_path, img_file))
                    self.labels.append(label)

    def _prepare_partitioned_data(self):
        partition_file = osp.join(self.anno_dir, 'list_eval_partition.txt')
        with open(partition_file) as f:
            lines = [x.strip() for x in f.readlines()]

        mapping_file = osp.join(self.anno_dir, 'CelebA-HQ-to-CelebA-mapping.txt')
        celeba_to_celebahq = {}
        with open(mapping_file) as f:
            for line in f.readlines()[1:]:
                celebahq_idx, _, celeba_idx = line.split()
                celeba_to_celebahq[celeba_idx] = int(celebahq_idx)

        split_map = {'train': 0, 'val': 1, 'test': 2}
        wanted = {split_map[s] for s in self.subset.split('+')}

        self.images = []
        for item in lines:
            img_filename, img_label = item.split(' ')
            if img_filename in celeba_to_celebahq and int(img_label) in wanted:
                idx = celeba_to_celebahq[img_filename]
                self.images.append(osp.join(self.data_dir, f'{idx}.jpg'))

    def _load_attributes(self):
        attr_file = osp.join(self.anno_dir, 'CelebAMask-HQ-attribute-anno.txt')
        self.attributes = {}
        if not osp.isfile(attr_file):
            return
        with open(attr_file) as f:
            for line in f.readlines()[2:]:
                parts = line.strip().split()
                self.attributes[parts[0]] = [1 if int(a) == 1 else 0 for a in parts[1:]]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path    = self.images[idx]
        img_basename = osp.basename(img_path)
        img_orig    = self.transform(Image.open(img_path).convert('RGB'))

        # Attributes (zero tensor if not available)
        num_attrs = len(next(iter(self.attributes.values()))) if self.attributes else 40
        img_orig_attr = torch.tensor(
            self.attributes.get(img_basename, [0] * num_attrs),
            dtype=torch.int64,
        )

        # ── Fake NN ──────────────────────────────────────────────────────────
        img_nn      = torch.zeros_like(img_orig)
        img_nn_code = torch.zeros(18, 512)
        if self.fake_nn_map is not None:
            nn_key = self.nn_map_dict.get(img_basename)
            if nn_key:
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
            stem = img_basename.split('.')[0]
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
            stem = img_basename.split('.')[0]
            if self.use_facial_identity:
                id_folder = osp.basename(osp.dirname(img_path))
                anon_img_path  = osp.join(self.anon, 'data',         id_folder, f'{stem}.jpg')
                anon_code_path = osp.join(self.anon, 'latent_codes', id_folder, f'{stem}.pt')
            else:
                anon_img_path  = osp.join(self.anon, 'data',         f'{stem}.jpg')
                anon_code_path = osp.join(self.anon, 'latent_codes', f'{stem}.pt')
            if osp.isfile(anon_img_path):
                img_anon = self.transform(Image.open(anon_img_path).convert('RGB'))
            if osp.isfile(anon_code_path):
                img_anon_code = torch.load(anon_code_path)

        return [img_orig, img_orig_attr, img_path,
                img_nn, img_nn_code,
                img_recon, img_recon_code,
                img_anon, img_anon_code]
