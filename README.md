# FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing

[![Paper](https://img.shields.io/badge/Paper-IJCB%202025-red)](https://github.com/your-repo/faceanonymixer)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-yellow)](https://github.com/your-repo/faceanonymixer)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Accepted at IEEE International Joint Conference on Biometrics (IJCB) 2025**

## Authors

**Mohammed Talha Alam**¹, **Fahad Shamshad**¹, **Fakhri Karray**¹'², **Karthik Nandakumar**¹'³

¹ Mohamed Bin Zayed University of Artificial Intelligence, UAE  
² University of Waterloo, Canada  
³ Michigan State University, USA

## Abstract

Advancements in face recognition technologies have increased privacy concerns. Although many face anonymization techniques have been proposed, they do not satisfy all the biometric template protection requirements such as identity preservation, revocability, unlinkability, and irreversibility. In this work, we leverage the power of synthetic face image generators to design a novel cancelable face generation framework called **FaceAnonyMixer** that works in the image domain. The core idea of FaceAnonyMixer is to irreversibly mix the latent code of the real face image with a synthetic latent code indexed by a revocable key. The mixed latent code is further refined by minimizing a combination of losses designed to achieve all the requirements of cancelable biometrics. FaceAnonyMixer is capable of generating high-quality cancelable faces that can be directly matched using existing face recognition systems without any modification. Experimental results demonstrate that our approach achieves competitive recognition accuracy while significantly improving privacy protection.

## Key Features

- 🔐 **Privacy Protection**: Novel cancelable face generation ensuring irreversibility and unlinkability
- 🔄 **Revocability**: Key-based system allowing template updates when compromised
- 🎯 **Identity Preservation**: Maintains recognition accuracy while protecting biometric templates
- ⚡ **Direct Compatibility**: Works with existing face recognition systems without modification
- 🖼️ **High Quality**: Generates visually appealing cancelable face images
- 🔒 **Security**: Satisfies all biometric template protection requirements

## 🚧 Repository Status

**This repository is currently under development. Code, models, and documentation will be uploaded soon.**

We are in the process of:
- [ ] Cleaning and organizing the codebase
- [ ] Preparing pre-trained models
- [ ] Writing comprehensive documentation
- [ ] Setting up evaluation scripts
- [ ] Preparing dataset information

**Expected Release**: Coming Soon

## 📋 Requirements

The code will be released with detailed requirements. Expected dependencies include:
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- PIL/Pillow
- Additional dependencies will be listed in `requirements.txt`

## 🚀 Quick Start

Once the code is released, you'll be able to:

```bash
# Clone the repository
git clone https://github.com/your-username/faceanonymixer.git
cd faceanonymixer

# Install dependencies
pip install -r requirements.txt

# Run FaceAnonyMixer
python main.py --config configs/default.yaml
```

## 📊 Results

Detailed experimental results and comparisons with state-of-the-art methods will be provided upon code release.

## 📖 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{alam2025faceanonymixer,
    title={FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing},
    author={Alam, Mohammed Talha and Shamshad, Fahad and Karray, Fakhri and Nandakumar, Karthik},
    booktitle={IEEE International Joint Conference on Biometrics (IJCB)},
    year={2025}
}
```

## 📧 Contact

For questions about this work, please contact:

- **Mohammed Talha Alam**: mohammed.alam@mbzuai.ac.ae
- **Fahad Shamshad**: fahad.shamshad@mbzuai.ac.ae

## 📄 License

This project will be released under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Note**: This repository is associated with the paper accepted at IJCB 2025. Stay tuned for the code release!
