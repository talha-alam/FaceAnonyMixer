"""Generate a pool of fake face images from a pretrained StyleGAN2 generator.

For each sampled latent code the script saves:
    - the generated face image (image.jpg)
    - the W+ latent code (latent_code_w+.pt)
    - optional feature embeddings: CLIP, FaRL, DINO, ArcFace

The per-image directories are named by the SHA-1 hash of the Z code so that
the pool can be extended incrementally without collisions.

Usage
-----
    python create_fake_dataset.py \\
        --gan stylegan2_ffhq1024 \\
        --num-samples 60000 \\
        --truncation 0.7 \\
        --cuda --verbose
"""

import argparse
import json
import os
import os.path as osp
import shutil
from hashlib import sha1

import clip
import torch
from torchvision import transforms
from tqdm import tqdm

from lib import (
    GENFORCE_MODELS, FARL_PRETRAIN_MODEL, ArcFace, tensor2image,
)
from models.load_generator import load_generator


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate a fake face dataset using a pretrained StyleGAN2 generator.'
    )
    p.add_argument('-v', '--verbose',    action='store_true')
    p.add_argument('--gan',              type=str, default='stylegan2_ffhq1024',
                   choices=list(GENFORCE_MODELS.keys()))
    p.add_argument('--truncation',       type=float, default=0.7)
    p.add_argument('--num-samples',      type=int,   default=60000)
    p.add_argument('--no-clip',          action='store_true')
    p.add_argument('--no-farl',          action='store_true')
    p.add_argument('--no-dino',          action='store_true')
    p.add_argument('--no-arcface',       action='store_true')
    p.add_argument('--cuda',             dest='cuda', action='store_true')
    p.add_argument('--no-cuda',          dest='cuda', action='store_false')
    p.set_defaults(cuda=True)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # Output directory name encodes the config
    feat_tag = ''
    if not args.no_clip:   feat_tag += '-CLIP'
    if not args.no_farl:   feat_tag += '-FaRL'
    if not args.no_dino:   feat_tag += '-DINO'
    if not args.no_arcface:feat_tag += '-ArcFace'

    out_dir = osp.join(
        'datasets', 'fake',
        f'fake_dataset_{args.gan}-{args.truncation}-{args.num_samples}{feat_tag}',
    )
    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(osp.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.verbose:
        print(f'#. Output directory: {out_dir}')

    # ── generator ─────────────────────────────────────────────────────────────
    G = load_generator(args.gan, latent_is_s=True, verbose=args.verbose).eval().to(device)

    # ── feature extractors ────────────────────────────────────────────────────
    clip_model = farl_model = dino_model = arcface_model = None
    clip_tf    = farl_tf    = dino_tf    = arcface_tf    = None

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

    # ── sampling loop ─────────────────────────────────────────────────────────
    if args.verbose:
        print(f'#. Sampling {args.num_samples} latent codes ...')

    zs = torch.randn(args.num_samples, G.dim_z, device=device)

    hashes       = []
    clip_feats   = []
    farl_feats   = []
    dino_feats   = []
    arcface_feats = []

    for i in tqdm(range(args.num_samples), desc='Generating'):
        z   = zs[i].unsqueeze(0)
        h   = sha1(z.cpu().numpy().tobytes()).hexdigest()
        hashes.append(h)

        sample_dir = osp.join(out_dir, h)
        os.makedirs(sample_dir, exist_ok=True)

        wp         = G.get_w(z, truncation=args.truncation)
        styles_dict = G.get_s(wp)

        with torch.no_grad():
            img = G(styles_dict)

        # Feature extraction
        with torch.no_grad():
            if clip_model:
                f = clip_model.encode_image(clip_tf(img))
                torch.save(f.cpu(), osp.join(sample_dir, 'clip_features.pt'))
                clip_feats.append(f.cpu())
            if farl_model:
                f = farl_model.encode_image(farl_tf(img))
                torch.save(f.cpu(), osp.join(sample_dir, 'farl_features.pt'))
                farl_feats.append(f.cpu())
            if dino_model:
                f = dino_model(dino_tf(img))
                torch.save(f.cpu(), osp.join(sample_dir, 'dino_features.pt'))
                dino_feats.append(f.cpu())
            if arcface_model:
                f = arcface_model(arcface_tf(img))
                torch.save(f.cpu(), osp.join(sample_dir, 'arcface_features.pt'))
                arcface_feats.append(f.cpu())

        tensor2image(img.cpu(), adaptive=True).save(
            osp.join(sample_dir, 'image.jpg'),
            'JPEG', quality=95, subsampling=0, progressive=True,
        )
        torch.save(wp.cpu(),          osp.join(sample_dir, 'latent_code_w+.pt'))
        torch.save(styles_dict,       osp.join(sample_dir, 'latent_code_s.pt'))

    # Write hash list
    with open(osp.join(out_dir, 'latent_code_hashes.txt'), 'w') as f:
        f.writelines(h + '\n' for h in hashes)

    # Save stacked feature matrices
    def _save_feats(feats, name):
        if feats:
            mat = torch.cat(feats)
            torch.save(mat, osp.join(out_dir, name))
            if args.verbose:
                print(f'  \\__ {name}: {mat.shape}')

    _save_feats(clip_feats,    'clip_features.pt')
    _save_feats(farl_feats,    'farl_features.pt')
    _save_feats(dino_feats,    'dino_features.pt')
    _save_feats(arcface_feats, 'arcface_features.pt')

    if args.verbose:
        print(f'#. Done — {args.num_samples} images saved to {out_dir}')


if __name__ == '__main__':
    main()
