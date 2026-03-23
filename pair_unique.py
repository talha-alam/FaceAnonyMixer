"""Pair each real identity to a unique fake identity.

Every image belonging to the same real person is mapped to the *same* fake
image (the first image found in that fake identity's folder).  This ensures
that the key (= fake latent code) is shared across all protected images of
the same individual, which is required for the identity-preservation loss to
work correctly.

The resulting JSON map has the form::

    { "n000001/0001.jpg": "abc123def/image.jpg", ... }

Usage
-----
    python pair_unique.py \\
        --real-dataset sample_IJB-C \\
        --fake-dataset-root datasets/fake/fake_dataset_... \\
        --verbose
"""

import argparse
import json
import os
import os.path as osp
import random

from lib import DATASETS


def parse_args():
    p = argparse.ArgumentParser(
        description='Pair each real identity to a unique fake identity.'
    )
    p.add_argument('-v', '--verbose',        action='store_true')
    p.add_argument('--real-dataset',         type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--real-dataset-root',    type=str, default=None,
                   help='Override the dataset root from lib/config.py')
    p.add_argument('--fake-dataset-root',    type=str, required=True)
    p.add_argument('--seed',                 type=int, default=42,
                   help='Random seed for reproducible fake-identity shuffling')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Real dataset root (use config default if not overridden)
    real_root = args.real_dataset_root or DATASETS[args.real_dataset]
    real_train_dir = osp.join(real_root, 'train')

    if not osp.isdir(real_train_dir):
        raise NotADirectoryError(
            f'Real dataset train directory not found: {real_train_dir}\n'
            f'Update DATASETS in lib/config.py or pass --real-dataset-root.'
        )

    # Fake dataset root
    if not osp.isdir(args.fake_dataset_root):
        raise NotADirectoryError(
            f'Fake dataset directory not found: {args.fake_dataset_root}'
        )

    # Collect real identities (sub-folder names)
    real_ids = sorted(
        d for d in os.listdir(real_train_dir)
        if osp.isdir(osp.join(real_train_dir, d))
    )

    # Collect fake identities (sub-folder names that contain an image.jpg)
    fake_ids = sorted(
        d for d in os.listdir(args.fake_dataset_root)
        if osp.isfile(osp.join(args.fake_dataset_root, d, 'image.jpg'))
    )

    if args.verbose:
        print(f'#. Real identities : {len(real_ids)}')
        print(f'#. Fake identities : {len(fake_ids)}')

    if len(fake_ids) < len(real_ids):
        raise ValueError(
            f'Not enough fake identities ({len(fake_ids)}) to uniquely pair '
            f'with {len(real_ids)} real identities. '
            f'Re-run create_fake_dataset.py with --num-samples >= {len(real_ids)}.'
        )

    # Shuffle fake identities for random (but reproducible) assignment
    random.shuffle(fake_ids)
    id_map = dict(zip(real_ids, fake_ids))

    # Build per-image NN map
    nn_map = {}
    for real_id, fake_id in id_map.items():
        id_folder = osp.join(real_train_dir, real_id)
        fake_img  = osp.join(fake_id, 'image.jpg')   # relative to fake_dataset_root

        for img_file in sorted(os.listdir(id_folder)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = osp.join(real_id, img_file)    # relative path used as dict key
                nn_map[key] = fake_img

    # Save
    out_path = osp.join(
        args.fake_dataset_root,
        f'random_nn_map_{args.real_dataset}.json',
    )
    with open(out_path, 'w') as f:
        json.dump(nn_map, f, indent=2)

    if args.verbose:
        print(f'#. NN map ({len(nn_map)} entries) saved to {out_path}')


if __name__ == '__main__':
    main()
