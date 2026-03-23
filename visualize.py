"""Visualize original / reconstructed / fake-NN / anonymized face images.

Renders a side-by-side grid and either displays it interactively or saves it
under ``viz/``.

Usage
-----
    python visualize.py \\
        --dataset sample_IJB-C \\
        --subset test \\
        --fake-nn-map datasets/fake/.../random_nn_map_sample_IJB-C.json \\
        --inv \\
        --anon experiments/<hash> \\
        --batch-size 4 --save --verbose
"""

import argparse
import os
import os.path as osp

from PIL import Image, ImageDraw, ImageFont
from torch.utils import data
from torchvision import transforms

from lib import DATASETS, CelebAHQ


# ── grid rendering ────────────────────────────────────────────────────────────

def _make_grid(data_batch, nn_type, show_inv, anon_dir,
               img_size=256, header_h=30, font_path='lib/fonts/RobotoCondensed-Bold.ttf'):

    img_orig      = data_batch[0]
    img_orig_path = data_batch[2]
    img_nn        = data_batch[3]
    img_recon     = data_batch[5]
    img_anon      = data_batch[7]

    to_pil = transforms.ToPILImage()
    B      = img_orig.shape[0]

    # Determine columns
    columns = ['Original']
    if show_inv:    columns.append('Recon (e4e)')
    if nn_type:     columns.append('Fake NN')
    if anon_dir:    columns.append('Anonymized')

    grid_w = img_size * len(columns)
    grid_h = header_h + B * img_size
    grid   = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(grid)

    # Header font
    try:
        font = ImageFont.truetype(font_path, 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for col_idx, col_name in enumerate(columns):
        draw.text((col_idx * img_size + 4, 4), col_name, font=font, fill=(0, 0, 0))

    # Image rows
    for row in range(B):
        y = header_h + row * img_size

        tensors = [img_orig[row]]
        if show_inv:  tensors.append(img_recon[row])
        if nn_type:   tensors.append(img_nn[row])
        if anon_dir:  tensors.append(img_anon[row])

        for col_idx, t in enumerate(tensors):
            pil_img = to_pil(t.clamp(0, 1)).resize((img_size, img_size))
            grid.paste(pil_img, (col_idx * img_size, y))

        # Filename label on original column
        fn = osp.basename(img_orig_path[row])
        draw.rectangle([(0, y), (len(fn) * 8, y + 18)], fill=(255, 208, 0))
        draw.text((2, y), fn, font=font, fill=(0, 0, 0))

    return grid


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('FaceAnonyMixer visualization')
    p.add_argument('-v', '--verbose',    action='store_true')
    p.add_argument('--dataset',          type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--dataset-root',     type=str, default=None)
    p.add_argument('--subset',           type=str, default='test',
                   choices=('train', 'val', 'test', 'train+val', 'train+val+test'))
    p.add_argument('--fake-nn-map',      type=str, default=None)
    p.add_argument('--inv',              action='store_true',
                   help='Show e4e reconstructions')
    p.add_argument('--anon',             type=str, default=None,
                   help='Path to anonymized dataset directory')
    p.add_argument('--batch-size',       type=int, default=4)
    p.add_argument('--shuffle',          action='store_true')
    p.add_argument('--save',             action='store_true',
                   help='Save figures to viz/<dataset>/')
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = args.dataset_root or DATASETS[args.dataset]
    dataset      = CelebAHQ(
        root_dir=dataset_root,
        subset=args.subset,
        fake_nn_map=args.fake_nn_map,
        inv=args.inv,
        anon=args.anon,
        use_facial_identity=True,
    )
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=args.shuffle)

    # Save directory
    save_dir = None
    if args.save:
        tag = args.dataset
        if args.fake_nn_map: tag += '+nn'
        if args.inv:         tag += '+inv'
        if args.anon:        tag += '+anon'
        save_dir = osp.join('viz', tag)
        os.makedirs(save_dir, exist_ok=True)
        if args.verbose:
            print(f'#. Saving figures to {save_dir}')

    for batch in loader:
        grid = _make_grid(
            data_batch=batch,
            nn_type=dataset.nn_type,
            show_inv=args.inv,
            anon_dir=args.anon,
        )

        if save_dir:
            stems = '_'.join(
                osp.splitext(osp.basename(f))[0] for f in batch[2]
            )
            grid.save(osp.join(save_dir, f'{stems}.jpg'),
                      'JPEG', quality=95, subsampling=0, progressive=True)
        else:
            grid.show()
            input('__> Press ENTER to continue...\n')


if __name__ == '__main__':
    main()
