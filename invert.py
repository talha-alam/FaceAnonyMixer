"""GAN inversion: project real face images into StyleGAN2's W+ space.

Pipeline
--------
1. Align and crop each face using ``FaceAligner``.
2. Encode the aligned crop with the pre-trained e4e encoder to get an initial
   W+ code.
3. Refine the latent code via Pivot Tuning Inversion (PTI): optimise only the
   latent (not the generator) to minimise perceptual + L2 loss.
4. Save the optimised latent code (.pt) and the reconstructed image (.jpg).

Usage
-----
    python invert.py \\
        --dataset sample_IJB-C \\
        --batch-size 1 \\
        --num-steps 150 \\
        --learning-rate 0.0005 \\
        --cuda --verbose
"""

import argparse
import math
import os
import os.path as osp

import cv2
import lpips
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from lib import DATASETS, FaceAligner, tensor2image, CelebAHQ, VGGFace2
from models.load_generator import load_generator
from models.psp import pSp


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_e4e(device: str) -> pSp:
    ckpt_path = osp.join('models', 'pretrained', 'e4e', 'e4e_ffhq_encode.pt')
    ckpt      = torch.load(ckpt_path, map_location='cpu')
    opts      = argparse.Namespace(**ckpt['opts'])
    opts.checkpoint_path = ckpt_path
    opts.device          = device
    if not hasattr(opts, 'stylegan_size'):
        opts.stylegan_size = 1024
    if not hasattr(opts, 'start_from_latent_avg'):
        opts.start_from_latent_avg = True
    e4e = pSp(opts).eval().to(device)
    return e4e


def _get_latents(e4e: pSp, x: torch.Tensor) -> torch.Tensor:
    """Encode *x* and return (B, 18, 512) W+ codes."""
    codes = e4e.encoder(x)
    if getattr(e4e.opts, 'start_from_latent_avg', True) and e4e.latent_avg is not None:
        codes = codes + e4e.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.ndim == 2:
        codes = codes.unsqueeze(1).repeat(1, 18, 1)
    return codes


def _generate_noise(device: str, resolution: int = 1024):
    noise = []
    for i in range(int(math.log2(resolution)) - 1):
        size = 2 ** (i + 2)
        noise += [torch.randn(1, 1, size, size, device=device),
                  torch.randn(1, 1, size, size, device=device)]
    return noise


def _pivot_tune(G, e4e, w_init, real_img, num_steps, lr, percept, device):
    """Optimise the latent code (not the generator) to fit *real_img*."""
    if w_init.dim() == 2:
        w_init = w_init.unsqueeze(1).repeat(1, G.n_latent, 1)

    latent = w_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([latent], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        gen_out = G.synthesis(latent)
        gen_img = gen_out['image'] if isinstance(gen_out, dict) else gen_out
        gen_img_small  = F.interpolate(gen_img,   size=256)
        real_img_small = F.interpolate(real_img,  size=256)
        loss = percept(gen_img_small, real_img_small).sum() + \
               0.1 * F.mse_loss(gen_img_small, real_img_small)
        loss.backward()
        optimizer.step()

    return latent.detach()


# ── main class ────────────────────────────────────────────────────────────────

class PivotTuningInversion:
    def __init__(self, opts):
        self.opts   = opts
        self.device = 'cuda' if opts.cuda else 'cpu'

        self.percept      = lpips.LPIPS(net='vgg').to(self.device)
        self.face_aligner = FaceAligner(device=self.device)
        self.e4e          = _build_e4e(self.device)
        self.G            = load_generator(
            model_name='stylegan2_ffhq1024',
            latent_is_w=True,
            verbose=opts.verbose,
        ).eval().to(self.device)

        if hasattr(self.G, 'truncation'):
            self.G.truncation.truncation = 1.0

        # Output directories
        self.out_dir   = osp.join('datasets', 'inv_pivot', opts.dataset)
        self.align_dir = osp.join(self.out_dir, 'aligned')
        self.recon_dir = osp.join(self.out_dir, 'reconstructed')
        self.lat_dir   = osp.join(self.out_dir, 'latents')
        for d in (self.align_dir, self.recon_dir, self.lat_dir):
            os.makedirs(d, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # ── per-batch processing ──────────────────────────────────────────────────

    def process_batch(self, batch):
        image_paths = batch[1]

        aligned = []
        for img_path in image_paths:
            aligned_np = self.face_aligner.align_face(
                image_file=img_path,
                alignment_errors_file=osp.join(self.out_dir, 'alignment_errors.txt'),
                face_detection_errors_file=osp.join(self.out_dir, 'face_detection_errors.txt'),
            )
            stem = osp.splitext(osp.basename(img_path))[0]
            cv2.imwrite(
                osp.join(self.align_dir, f'{stem}_aligned.jpg'),
                cv2.cvtColor(aligned_np, cv2.COLOR_RGB2BGR),
            )
            aligned.append(self.transform(Image.fromarray(aligned_np)).unsqueeze(0))

        aligned_batch = torch.cat(aligned).to(self.device)

        # Encode
        latents = _get_latents(self.e4e, aligned_batch)

        # PTI refinement
        for i, img_path in enumerate(image_paths):
            stem = osp.splitext(osp.basename(img_path))[0]
            lat_path = osp.join(self.lat_dir, f'{stem}.pt')
            if osp.exists(lat_path):
                continue  # already processed

            w_opt = _pivot_tune(
                self.G, self.e4e,
                latents[i:i+1],
                aligned_batch[i:i+1],
                self.opts.num_steps,
                self.opts.learning_rate,
                self.percept,
                self.device,
            )
            torch.save(w_opt.squeeze(0), lat_path)

            with torch.no_grad():
                gen_out = self.G.synthesis(w_opt)
                recon   = gen_out['image'] if isinstance(gen_out, dict) else gen_out
            tensor2image(recon.squeeze(0).cpu(), adaptive=True).save(
                osp.join(self.recon_dir, f'{stem}_recon.jpg'),
                'JPEG', quality=90, subsampling=0, progressive=True,
            )

    # ── full run ──────────────────────────────────────────────────────────────

    def run(self):
        dataset_root = self.opts.dataset_root or DATASETS[self.opts.dataset]
        dataset      = VGGFace2(root_dir=dataset_root)
        loader       = data.DataLoader(dataset, batch_size=self.opts.batch_size,
                                       shuffle=False)

        desc = f'Inverting {self.opts.dataset}' if self.opts.verbose else ''
        for batch in tqdm(loader, desc=desc):
            self.process_batch(batch)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='GAN inversion via e4e + Pivot Tuning')
    p.add_argument('--dataset',      type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--dataset-root', type=str, default=None)
    p.add_argument('--batch-size',   type=int, default=1)
    p.add_argument('--num-steps',    type=int, default=150,
                   help='PTI optimisation steps per image')
    p.add_argument('--learning-rate',type=float, default=5e-4)
    p.add_argument('--cuda',         action='store_true', default=True)
    p.add_argument('--no-cuda',      dest='cuda', action='store_false')
    p.add_argument('-v', '--verbose',action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inverter = PivotTuningInversion(args)
    inverter.run()
