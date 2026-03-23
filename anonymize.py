"""Core FaceAnonyMixer anonymization script.

For each identity group the script:
  1. Initialises a LatentCode for every image (naïve mixing of real + fake layers).
  2. Jointly optimises all latent codes under L_anon + L_idp + L_attr.
  3. Saves the final anonymized images and latent codes.

Identity groups with only one image use augmented views for the consistency
loss (L_idp) instead of multiple real samples.

Usage
-----
    python anonymize.py \\
        --dataset sample_IJB-C \\
        --fake-nn-map datasets/fake/.../random_nn_map_sample_IJB-C.json \\
        --latent-space W+ \\
        --epochs 50 --lr 0.01 \\
        --lambda-id 10.0 --lambda-attr 0.15 --lambda-consistency 10.0 \\
        --cuda --gpu-id 0 --verbose
"""

import argparse
import json
import os
import os.path as osp
import shutil
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from tqdm import tqdm

from lib import (
    DATASETS, CelebAHQ,
    DataParallelPassthrough,
    IDLoss, AttrLoss, ConsistencyLoss,
    ImageAugmenter, LatentCode,
    tensor2image, anon_exp_dir,
)
from models.load_generator import load_generator


# ── IdentityLatentCode ────────────────────────────────────────────────────────

class IdentityLatentCode(LatentCode):
    """LatentCode that saves into a per-identity sub-directory."""

    def __init__(self, latent_code_real, latent_code_fake_nn,
                 img_id, out_code_dir, identity_folder, latent_space='W+'):
        id_code_dir = osp.join(out_code_dir, identity_folder)
        os.makedirs(id_code_dir, exist_ok=True)
        super().__init__(latent_code_real, latent_code_fake_nn,
                         img_id, id_code_dir, latent_space=latent_space)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_identity_mapping(dataset_root: str) -> dict:
    """Return {image_id (int): identity_folder (str)} for every training image."""
    train_dir  = osp.join(dataset_root, 'train')
    id_mapping = {}
    for folder in os.listdir(train_dir):
        folder_path = osp.join(train_dir, folder)
        if not osp.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_id = int(osp.splitext(img_name)[0])
                    id_mapping[img_id] = folder
                except ValueError:
                    pass
    return id_mapping


def _make_output_dirs(out_dir: str, dataset_root: str):
    """Create identity-organised output directories."""
    train_dir    = osp.join(dataset_root, 'train')
    out_data_dir = osp.join(out_dir, 'data')
    out_code_dir = osp.join(out_dir, 'latent_codes')
    for folder in os.listdir(train_dir):
        if osp.isdir(osp.join(train_dir, folder)):
            os.makedirs(osp.join(out_data_dir, folder), exist_ok=True)
            os.makedirs(osp.join(out_code_dir, folder), exist_ok=True)
    return out_data_dir, out_code_dir


def _build_optimizer(params, optim_name: str, lr: float):
    if optim_name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── args ──────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser('FaceAnonyMixer anonymization')
    p.add_argument('-v', '--verbose',         action='store_true')
    p.add_argument('--dataset',               type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--dataset-root',          type=str, default=None)
    p.add_argument('--fake-nn-map',           type=str, required=True)
    p.add_argument('--latent-space',          type=str, default='W+',
                   choices=['W+'])
    p.add_argument('-m', '--id-margin',       type=float, default=0.0)
    p.add_argument('--epochs',                type=int,   default=50)
    p.add_argument('--optim',                 type=str,   default='adam',
                   choices=['sgd', 'adam'])
    p.add_argument('--lr',                    type=float, default=0.01)
    p.add_argument('--lr-milestones',         nargs='+',  type=float,
                   default=[0.75, 0.9])
    p.add_argument('--lr-gamma',              type=float, default=0.8)
    p.add_argument('--lambda-id',             type=float, default=10.0,
                   help='Weight for anonymity loss L_anon')
    p.add_argument('--lambda-attr',           type=float, default=0.15,
                   help='Weight for attribute preservation loss L_attr')
    p.add_argument('--lambda-consistency',    type=float, default=10.0,
                   help='Weight for identity preservation loss L_idp')
    p.add_argument('--cuda',                  dest='cuda', action='store_true')
    p.add_argument('--no-cuda',               dest='cuda', action='store_false')
    p.add_argument('--gpu-id',                type=int,   default=0)
    p.set_defaults(cuda=True)
    args = p.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    use_cuda  = args.cuda and torch.cuda.is_available()
    multi_gpu = use_cuda and torch.cuda.device_count() > 1
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
    device = f'cuda:{args.gpu_id}' if use_cuda else 'cpu'

    # ── experiment directory ──────────────────────────────────────────────────
    args_dict    = vars(args).copy()
    out_dir      = anon_exp_dir(args_dict)
    dataset_root = args.dataset_root or DATASETS[args.dataset]

    if args.verbose:
        print(f'#. Experiment directory: {out_dir}')

    # Save config
    cfg = {k: v for k, v in args_dict.items() if k != 'lr_milestones'}
    with open(osp.join(out_dir, 'args.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    id_mapping                 = _build_identity_mapping(dataset_root)
    out_data_dir, out_code_dir = _make_output_dirs(out_dir, dataset_root)

    # ── models ────────────────────────────────────────────────────────────────
    G = load_generator('stylegan2_ffhq1024', latent_is_w=True,
                       verbose=args.verbose).eval().to(device)
    if multi_gpu:
        G = DataParallelPassthrough(G)

    id_criterion  = IDLoss(id_margin=args.id_margin).eval().to(device)
    attr_loss_fn  = AttrLoss(feat_ext='farl').eval().to(device)
    cons_loss_fn  = ConsistencyLoss().eval().to(device)
    augmenter     = ImageAugmenter(n_augmentations=5)

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = CelebAHQ(
        root_dir=dataset_root,
        subset='test',
        fake_nn_map=args.fake_nn_map,
        inv=True,
        use_facial_identity=True,
    )
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    # ── group images by identity ──────────────────────────────────────────────
    identity_groups = defaultdict(list)
    for data_ in loader:
        img_id  = int(osp.basename(data_[2][0]).split('.')[0])
        id_fold = id_mapping.get(img_id)
        if id_fold:
            identity_groups[id_fold].append(data_)

    # ── per-identity optimisation ─────────────────────────────────────────────
    desc = f'Anonymizing {args.dataset}' if args.verbose else ''
    for identity_folder, identity_data in tqdm(identity_groups.items(), desc=desc):

        # -- Handle single-image identities via augmentation --
        if len(identity_data) == 1:
            base_data   = identity_data[0]
            img_orig    = base_data[0]
            aug_images  = augmenter(img_orig)
            # Build synthetic data entries for augmented views
            extra = []
            for aug_img in aug_images:
                aug_entry       = list(base_data)
                aug_entry[0]    = aug_img
                extra.append(aug_entry)
            identity_data = [base_data] + extra

        # -- Initialise latent codes and optimisers --
        latent_codes = []
        optimizers   = []
        schedulers   = []
        orig_images  = []
        data_refs    = []

        for data_ in identity_data:
            img_orig     = data_[0].to(device)
            img_id       = int(osp.basename(data_[2][0]).split('.')[0])
            img_nn_code  = data_[4].to(device)
            img_recon_code = data_[6].to(device)

            orig_images.append(img_orig)

            lc = IdentityLatentCode(
                latent_code_real=img_recon_code,
                latent_code_fake_nn=img_nn_code,
                img_id=img_id,
                out_code_dir=out_code_dir,
                identity_folder=identity_folder,
                latent_space='W+',
            ).to(device)

            if not lc.do_optim():
                continue

            latent_codes.append(lc)
            data_refs.append(data_)

            opt = _build_optimizer(lc.parameters(), args.optim, args.lr)
            optimizers.append(opt)
            schedulers.append(MultiStepLR(
                optimizer=opt,
                milestones=[int(m * args.epochs) for m in args.lr_milestones],
                gamma=args.lr_gamma,
            ))

        if not latent_codes:
            continue

        G.zero_grad()

        # -- Optimisation loop --
        for _ in range(args.epochs):
            anon_images = []
            for lc, opt in zip(latent_codes, optimizers):
                opt.zero_grad()
                anon_images.append(G(lc()))

            anon_batch   = torch.cat(anon_images)          # (N, 3, H, W)
            orig_batch   = torch.cat(orig_images[:len(anon_images)])

            total_id   = sum(id_criterion(ai, oi)
                             for ai, oi in zip(anon_images, orig_images))
            total_attr = sum(attr_loss_fn(oi, ai)
                             for ai, oi in zip(anon_images, orig_images))
            cons       = cons_loss_fn(anon_batch)

            loss = (args.lambda_id          * total_id   +
                    args.lambda_attr        * total_attr +
                    args.lambda_consistency * cons)
            loss.backward()

            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

        # -- Save results --
        for lc, data_ in zip(latent_codes, data_refs):
            lc.save()
            img_id = int(osp.basename(data_[2][0]).split('.')[0])
            with torch.no_grad():
                anon_img = G(lc())
            out_path = osp.join(out_data_dir, identity_folder, f'{img_id}.jpg')
            tensor2image(anon_img.cpu(), adaptive=True).save(
                out_path, 'JPEG', quality=75, subsampling=0, progressive=True,
            )


if __name__ == '__main__':
    main()
