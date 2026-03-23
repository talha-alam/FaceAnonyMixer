"""Extract feature embeddings for every image in a real dataset.

Supported feature spaces: CLIP, FaRL, DINO, ArcFace.
Results are saved under ``datasets/features/<dataset>/``.

Usage
-----
    python extract_features.py \\
        --dataset sample_IJB-C \\
        --batch-size 128 \\
        --cuda --verbose
"""

import argparse
import os
import os.path as osp

import clip
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from lib import DATASETS, FARL_PRETRAIN_MODEL, ArcFace, VGGFace2


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract CLIP/FaRL/DINO/ArcFace features for a real dataset.'
    )
    p.add_argument('-v', '--verbose',  action='store_true')
    p.add_argument('--dataset',        type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--dataset-root',   type=str, default=None)
    p.add_argument('--batch-size',     type=int, default=128)
    p.add_argument('--no-clip',        action='store_true')
    p.add_argument('--no-farl',        action='store_true')
    p.add_argument('--no-dino',        action='store_true')
    p.add_argument('--no-arcface',     action='store_true')
    p.add_argument('--cuda',           dest='cuda', action='store_true')
    p.add_argument('--no-cuda',        dest='cuda', action='store_false')
    p.set_defaults(cuda=True)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    out_dir = osp.join('datasets', 'features', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Skip already-computed features
    if osp.exists(osp.join(out_dir, 'clip_features.pt')):    args.no_clip    = True
    if osp.exists(osp.join(out_dir, 'farl_features.pt')):    args.no_farl    = True
    if osp.exists(osp.join(out_dir, 'dino_features.pt')):    args.no_dino    = True
    if osp.exists(osp.join(out_dir, 'arcface_features.pt')): args.no_arcface = True

    if all([args.no_clip, args.no_farl, args.no_dino, args.no_arcface]):
        print(f'All features already computed under {out_dir}. Nothing to do.')
        return

    # ── build models ──────────────────────────────────────────────────────────
    clip_model   = farl_model   = dino_model   = arcface_model   = None
    clip_tf      = farl_tf      = dino_tf      = arcface_tf      = None

    if not args.no_clip:
        clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
        clip_model.float().eval()
        clip_tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711)),
        ])

    if not args.no_farl:
        farl_model, _ = clip.load('ViT-B/16', device=device, jit=False)
        state = torch.load(
            osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL),
            map_location='cpu',
        )
        farl_model.load_state_dict(state['state_dict'], strict=False)
        farl_model.float().eval()
        farl_tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711)),
        ])

    if not args.no_dino:
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        dino_model.float().eval().to(device)
        dino_tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if not args.no_arcface:
        arcface_model = ArcFace().eval().to(device)
        arcface_tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(256),
        ])

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset_root = args.dataset_root or DATASETS[args.dataset]
    dataset      = VGGFace2(root_dir=dataset_root)
    loader       = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── extraction loop ───────────────────────────────────────────────────────
    filenames     = []
    clip_feats    = []
    farl_feats    = []
    dino_feats    = []
    arcface_feats = []

    desc = f'Extracting features [{args.dataset}]' if args.verbose else ''
    for batch in tqdm(loader, desc=desc):
        imgs       = batch[0].to(device)
        batch_fns  = [osp.basename(p) for p in batch[1]]
        filenames.extend(batch_fns)

        with torch.no_grad():
            if clip_model:
                clip_feats.append(clip_model.encode_image(clip_tf(imgs)).cpu())
            if farl_model:
                farl_feats.append(farl_model.encode_image(farl_tf(imgs)).cpu())
            if dino_model:
                dino_feats.append(dino_model(dino_tf(imgs)).cpu())
            if arcface_model:
                arcface_feats.append(arcface_model(arcface_tf(imgs)).cpu())

    # ── save ──────────────────────────────────────────────────────────────────
    with open(osp.join(out_dir, 'image_filenames.txt'), 'w') as f:
        f.writelines(fn + '\n' for fn in filenames)

    def _save(feats, name):
        if feats:
            mat = torch.cat(feats)
            torch.save(mat, osp.join(out_dir, name))
            if args.verbose:
                print(f'  \\__ {name}: {mat.shape}')

    _save(clip_feats,    'clip_features.pt')
    _save(farl_feats,    'farl_features.pt')
    _save(dino_feats,    'dino_features.pt')
    _save(arcface_feats, 'arcface_features.pt')

    if args.verbose:
        print(f'#. Features saved to {out_dir}')


if __name__ == '__main__':
    main()
