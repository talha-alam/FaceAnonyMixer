import os
import os.path as osp
import subprocess
import torch

from models.genforce.models import MODEL_ZOO, build_generator


def load_generator(model_name: str,
                   latent_is_w: bool = False,
                   latent_is_s: bool = False,
                   verbose: bool = False,
                   checkpoint_dir: str = 'models/pretrained/genforce') -> torch.nn.Module:
    """Build a GenForce generator and load pretrained weights.

    Downloads the checkpoint automatically if it is not already present.

    Args:
        model_name (str): key in ``models.genforce.models.MODEL_ZOO``
                          (e.g. ``'stylegan2_ffhq1024'``).
        latent_is_w (bool): tell the generator that inputs are already in W space.
        latent_is_s (bool): tell the generator that inputs are already in S space.
        verbose (bool): print progress messages.
        checkpoint_dir (str): local directory for cached checkpoints.

    Returns:
        torch.nn.Module: generator in eval mode with ``dim_z`` attribute set.
    """
    if verbose:
        print(f'  \\__Building generator: {model_name} ...', end='')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')

    if 'stylegan' in model_name:
        model_config['latent_is_w'] = latent_is_w
        model_config['latent_is_s'] = latent_is_s

    generator = build_generator(**model_config)

    if verbose:
        print('Done!')

    # ── load checkpoint ───────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = osp.join(checkpoint_dir, model_name + '.pth')

    if verbose:
        print(f'  \\__Loading checkpoint: {ckpt_path} ...', end='')

    if not osp.exists(ckpt_path):
        if verbose:
            print(f'\n  \\__Checkpoint not found — downloading from {url} ...')
        subprocess.call(['wget', '--quiet', '-O', ckpt_path, url])

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    key = 'generator_smooth' if 'generator_smooth' in checkpoint else 'generator'
    generator.load_state_dict(checkpoint[key])

    if verbose:
        print('Done!')

    generator.dim_z = generator.z_space_dim
    return generator
