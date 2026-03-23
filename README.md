# FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.05636)
[![Conference](https://img.shields.io/badge/IJCB-2025-blue)](https://ijcb2025.ieee-biometrics.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Mohammed Talha Alam¹, Fahad Shamshad¹, Fakhri Karray¹², Karthik Nandakumar¹³**
>
> ¹ Mohamed Bin Zayed University of Artificial Intelligence, UAE  
> ² University of Waterloo, Canada · ³ Michigan State University, USA  
> `{mohammed.alam, fahad.shamshad, fakhri.karray, karthik.nandakumar}@mbzuai.ac.ae`

**Accepted at IEEE International Joint Conference on Biometrics (IJCB) 2025**

---

## Overview

FaceAnonyMixer generates privacy-preserving face images by **mixing a real face's W+ latent code with a key-derived synthetic code** inside StyleGAN2's latent space. It satisfies all four ISO/IEC 24745 requirements simultaneously:

| Requirement | What it means |
|---|---|
| **Revocability** | Change the key to revoke and replace any compromised template |
| **Unlinkability** | Templates from different keys cannot be cross-matched |
| **Irreversibility** | Original identity cannot be recovered even if the key and template are both known |
| **Performance Preservation** | Recognition accuracy on protected faces matches unprotected baselines |

**Inspired by [FALCO](https://github.com/chi0tzp/FALCO).**

---

## How It Works

StyleGAN2's 18 W+ layers decompose as: coarse structure (0–2), identity (3–7), fine details (8–17).

1. **Invert** the real face into W+: `z_r = E(x_r)`
2. **Sample** a synthetic latent from the key: `z_f = S(k)`
3. **Naïve mix** — replace identity layers with those from the fake code:
   ```
   z_p = [ z_r^(0:2),  z_f^(3:7),  z_r^(8:17) ]
   ```
4. **Optimise** for 50 Adam steps under:
   ```
   L_total = λ1·L_anon + λ2·L_idp + λ3·L_attr
   ```

| Loss | Role |
|---|---|
| `L_anon` | Pushes protected face away from original in ArcFace space |
| `L_idp` | Pulls protected faces of the *same person* together (identity preservation) |
| `L_attr` | Aligns FaRL features to retain pose, expression, and other non-identity attributes |

---

## Repository Structure

```
faceanonymixer/
│
├── anonymize.py              # Core: latent mixing + multi-loss optimization
├── invert.py                 # GAN inversion via e4e + Pivot Tuning
├── create_fake_dataset.py    # Generate fake StyleGAN2 image pool
├── extract_features.py       # Extract CLIP/FaRL/DINO/ArcFace features for real images
├── pair_unique.py            # Pair each real identity to a unique fake identity ← use this
├── pair_nn.py                # Random per-image pairing (ablation only)
├── visualize.py              # Grid visualization
├── download_pretrained.py    # Download all pretrained weights
│
├── lib/
│   ├── __init__.py
│   ├── config.py             # Dataset paths + model URLs  ← update DATASETS here
│   ├── celebahq.py           # CelebA-HQ dataset class
│   ├── vggface2.py           # VGGFace2 dataset class
│   ├── latent_code.py        # LatentCode nn.Module
│   ├── id_loss.py            # Anonymity loss (ArcFace cosine)
│   ├── attr_loss.py          # Attribute loss (FaRL/CLIP/DINO)
│   ├── cons_loss.py          # Identity preservation loss
│   ├── augmentations.py      # ImageAugmenter
│   ├── aligner.py            # Face alignment (face_alignment library)
│   ├── arcface.py            # ArcFace feature extractor
│   ├── collate_fn.py         # DataLoader collate helper
│   └── aux.py                # tensor2image, anon_exp_dir, DataParallelPassthrough
│
├── models/
│   ├── __init__.py
│   ├── load_generator.py     # Build + load StyleGAN2 from GenForce
│   ├── psp.py                # e4e / pSp encoder wrapper
│   └── encoders/
│       ├── __init__.py
│       ├── helpers.py        # Shared IR bottleneck blocks
│       └── psp_encoders.py   # GradualStyleEncoder, Encoder4Editing
│
├── models/genforce/          # Populated by scripts/setup_genforce.sh
│
├── utils1/
│   ├── __init__.py
│   ├── ImagesDataset.py      # Simple flat image dataset
│   └── data_utils.py         # make_dataset helper
│
├── scripts/
│   └── setup_genforce.sh     # Clone GenForce into models/genforce/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/talha-alam/faceanonymixer.git
cd faceanonymixer

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set up the GenForce model definitions
bash scripts/setup_genforce.sh

# 4. Download all pretrained weights
#    (StyleGAN2-FFHQ-1024, e4e, ArcFace, FaRL ep64, SFD detector)
python download_pretrained.py
```

---

## Dataset Preparation

Both CelebA-HQ and VGGFace2 must be organised as **one sub-folder per identity**:

```
/path/to/dataset/
└── train/
    ├── n000001/
    │   ├── 0001.jpg
    │   └── 0002.jpg
    └── n000002/
        └── 0001.jpg
```

Edit `lib/config.py` to point to your local paths:

```python
DATASETS = {
    'celebahq': '/path/to/your/dataset',
}
```

---

## Step-by-Step Usage

### Step 1 — GAN Inversion

Project real faces into W+ space using e4e + Pivot Tuning Inversion.

```bash
python invert.py \
    --dataset celebahq \
    --num-steps 150 \
    --learning-rate 0.0005 \
    --cuda --verbose
```

Output: `datasets/inv_pivot/celebahq/{aligned,reconstructed,latents}/`

---

### Step 2 — Generate Fake Dataset

Sample a pool of synthetic faces with feature embeddings.

```bash
python create_fake_dataset.py \
    --gan stylegan2_ffhq1024 \
    --num-samples 60000 \
    --truncation 0.7 \
    --cuda --verbose
```

Output: `datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-60000-CLIP-FaRL-DINO-ArcFace/`

---

### Step 3 — Extract Real Dataset Features

```bash
python extract_features.py \
    --dataset celebahq \
    --batch-size 128 \
    --cuda --verbose
```

Output: `datasets/features/celebahq/`

---

### Step 4 — Pair Identities

Assign each real identity a unique fake identity (same fake latent = the revocable key).

```bash
python pair_unique.py \
    --real-dataset celebahq \
    --fake-dataset-root datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-60000-CLIP-FaRL-DINO-ArcFace \
    --verbose
```

Output: `random_nn_map_celebahq.json` inside the fake dataset directory.

---

### Step 5 — Anonymize

```bash
python anonymize.py \
    --dataset celebahq \
    --fake-nn-map datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-60000-CLIP-FaRL-DINO-ArcFace/random_nn_map_celebahq.json \
    --latent-space W+ \
    --epochs 50 \
    --lr 0.01 \
    --lambda-id 10.0 \
    --lambda-attr 0.15 \
    --lambda-consistency 10.0 \
    --cuda --gpu-id 0 --verbose
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Optimisation steps per identity group |
| `--lambda-id` | 10.0 | Weight for `L_anon` |
| `--lambda-attr` | 0.15 | Weight for `L_attr` |
| `--lambda-consistency` | 10.0 | Weight for `L_idp` |
| `--id-margin` | 0.0 | Cosine margin in `L_anon` (0 = max anonymisation) |

Output: `experiments/<hash>/{data,latent_codes}/<identity>/`

---

### Visualization (optional)

```bash
python visualize.py \
    --dataset celebahq \
    --fake-nn-map datasets/fake/.../random_nn_map_celebahq.json \
    --inv \
    --anon experiments/<hash> \
    --batch-size 4 --save --verbose
```

---

## Results

### Protection Success Rate (%) — Anonymization

| Method | FMR | CelebA-HQ avg | VGGFace2 avg | Overall |
|---|---|---|---|---|
| CanFG | 0.1% | 64.59 | 11.78 | 38.18 |
| **Ours** | **0.1%** | **80.12** | **97.34** | **88.73** |
| CanFG | 0.001% | 41.40 | 5.05 | 23.22 |
| **Ours** | **0.001%** | **60.94** | **86.41** | **73.67** |

### Recognition Performance (CelebA-HQ)

| Method | EER ↓ | AUC ↑ | FID ↓ |
|---|---|---|---|
| Original | 0.027 | 0.990 | — |
| CanFG | 0.045 | 0.988 | 82.52 |
| **Ours** | **0.019** | **0.997** | **37.73** |

### Commercial API — Face++ (lower confidence = better protected)

| Method | CelebA-HQ | VGGFace2 | Average |
|---|---|---|---|
| CanFG | 71.37 | 70.78 | 71.07 |
| **Ours** | **53.73** | **31.51** | **58.37** |

---

## Citation

```bibtex
@inproceedings{alam2025faceanonymixer,
    title     = {FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing},
    author    = {Alam, Mohammed Talha and Shamshad, Fahad and Karray, Fakhri and Nandakumar, Karthik},
    booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
    year      = {2025}
}
```

---

## Acknowledgements

Built on [FALCO](https://github.com/chi0tzp/FALCO) · GAN inversion via [e4e](https://github.com/omertov/encoder4editing) · Generator from [GenForce/StyleGAN2](https://github.com/genforce/genforce) · Identity features from [ArcFace](https://github.com/deepinsight/insightface) · Attribute features from [FaRL](https://github.com/FacePerceiver/FaRL)

---

## License

[MIT License](LICENSE)
