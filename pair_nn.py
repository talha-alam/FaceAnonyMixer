"""Randomly pair each real image to a fake image (per-image, not per-identity).

Note: This script performs *random* assignment, not nearest-neighbour search,
despite the filename.  For the full FaceAnonyMixer pipeline use
``pair_unique.py``, which enforces per-identity consistency.  This script is
kept for ablation experiments.

Usage
-----
    python pair_nn.py \\
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
        description='Randomly pair each real image to a fake image.'
    )
    p.add_argument('-v', '--verbose',      action='store_true')
    p.add_argument('--real-dataset',       type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument('--real-dataset-root',  type=str, default=None)
    p.add_argument('--fake-dataset-root',  type=str, required=True)
    p.add_argument('--seed',               type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Read real image filenames
    feat_dir = osp.join('datasets', 'features', args.real_dataset)
    fn_file  = osp.join(feat_dir, 'image_filenames.txt')
    if not osp.isfile(fn_file):
        raise FileNotFoundError(
            f'Image filename list not found: {fn_file}\n'
            f'Run extract_features.py first.'
        )
    with open(fn_file) as f:
        real_filenames = [x.strip() for x in f.readlines()]

    # Read fake image hashes
    hash_file = osp.join(args.fake_dataset_root, 'latent_code_hashes.txt')
    if not osp.isfile(hash_file):
        raise FileNotFoundError(
            f'Latent code hash list not found: {hash_file}\n'
            f'Run create_fake_dataset.py first.'
        )
    with open(hash_file) as f:
        fake_hashes = [x.strip() for x in f.readlines()]

    if args.verbose:
        print(f'#. Real images : {len(real_filenames)}')
        print(f'#. Fake images : {len(fake_hashes)}')

    # Random assignment
    nn_map = {fn: random.choice(fake_hashes) for fn in real_filenames}

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
