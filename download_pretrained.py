"""Download all pretrained model weights required by FaceAnonyMixer.

Downloads:
    - GenForce StyleGAN2-FFHQ-1024 / 512 generators
    - e4e inversion encoder + auxiliary weights
    - SFD face detector
    - FaRL ViT-B/16 checkpoint (ep64)
    - ArcFace IR-SE50 checkpoint

Usage:
    python download_pretrained.py
"""

import hashlib
import os
import os.path as osp
import sys
import tarfile
import time
import urllib.request

from lib.config import GENFORCE, GENFORCE_MODELS, E4E, SFD, FARL, ARCFACE, FARL_PRETRAIN_MODEL


# ── progress hook ─────────────────────────────────────────────────────────────

_start_time = None


def _reporthook(count, block_size, total_size):
    global _start_time
    if count == 0:
        _start_time = time.time()
        return
    elapsed  = max(time.time() - _start_time, 1e-6)
    progress = count * block_size
    speed    = progress / (1024 * elapsed)
    percent  = min(int(progress * 100 / max(total_size, 1)), 100)
    sys.stdout.write(
        f'\r      \\__ {percent:3d}%  '
        f'{progress / 1024**2:.1f} MB  '
        f'{speed:.0f} KB/s  '
        f'{elapsed:.0f}s'
    )
    sys.stdout.flush()


# ── download helper ───────────────────────────────────────────────────────────

def _download(src: str, sha256sum: str, dest_dir: str):
    """Download *src*, verify SHA-256, and extract the tar archive."""
    os.makedirs(dest_dir, exist_ok=True)
    tmp_tar = osp.join(dest_dir, '.tmp_download.tar')

    try:
        urllib.request.urlretrieve(src, tmp_tar, _reporthook)
    except Exception as exc:
        raise ConnectionError(f'Download failed: {src}') from exc
    print()

    # Verify checksum
    h = hashlib.sha256()
    with open(tmp_tar, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    ok = h.hexdigest() == sha256sum
    print(f'      \\__SHA-256: {"OK" if ok else "MISMATCH — expected " + sha256sum}')
    if not ok:
        os.remove(tmp_tar)
        raise ValueError(f'SHA-256 mismatch for {src}')

    with tarfile.open(tmp_tar, 'r') as tf:
        tf.extractall(dest_dir)
    os.remove(tmp_tar)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pretrained_root = osp.join('models', 'pretrained')

    # ── GenForce generators ───────────────────────────────────────────────────
    print('#. GenForce GAN generators ...')
    genforce_dir = osp.join(pretrained_root, 'genforce')
    missing = any(
        not osp.exists(osp.join(genforce_dir, v[0]))
        for v in GENFORCE_MODELS.values()
    )
    if missing:
        _download(*GENFORCE, dest_dir=pretrained_root)
    else:
        print('   Already present — skipping.')

    # ── e4e encoder ───────────────────────────────────────────────────────────
    print('#. e4e inversion encoder ...')
    e4e_dir = osp.join(pretrained_root, 'e4e')
    e4e_files = [
        osp.join(e4e_dir, 'model_ir_se50.pth'),
        osp.join(e4e_dir, 'e4e_ffhq_encode.pt'),
        osp.join(e4e_dir, 'shape_predictor_68_face_landmarks.dat'),
    ]
    if not all(osp.exists(p) for p in e4e_files):
        _download(*E4E, dest_dir=pretrained_root)
    else:
        print('   Already present — skipping.')

    # ── SFD face detector ─────────────────────────────────────────────────────
    print('#. SFD face detector ...')
    sfd_path = osp.join(pretrained_root, 'sfd', 's3fd-619a316812.pth')
    if not osp.exists(sfd_path):
        _download(*SFD, dest_dir=pretrained_root)
    else:
        print('   Already present — skipping.')

    # ── FaRL ──────────────────────────────────────────────────────────────────
    print('#. FaRL facial representation model ...')
    farl_path = osp.join(pretrained_root, 'farl', FARL_PRETRAIN_MODEL)
    if not osp.exists(farl_path):
        _download(*FARL, dest_dir=pretrained_root)
    else:
        print('   Already present — skipping.')

    # ── ArcFace ───────────────────────────────────────────────────────────────
    print('#. ArcFace IR-SE50 ...')
    arcface_path = osp.join(pretrained_root, 'arcface', 'model_ir_se50.pth')
    if not osp.exists(arcface_path):
        _download(*ARCFACE, dest_dir=pretrained_root)
    else:
        print('   Already present — skipping.')

    print('\n#. All pretrained weights are ready.')


if __name__ == '__main__':
    main()
