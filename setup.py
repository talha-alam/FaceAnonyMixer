from setuptools import setup, find_packages

setup(
    name='faceanonymixer',
    version='1.0.0',
    description='Cancelable Faces via Identity Consistent Latent Space Mixing',
    author='Mohammed Talha Alam, Fahad Shamshad, Fakhri Karray, Karthik Nandakumar',
    url='https://github.com/talha-alam/faceanonymixer',
    packages=find_packages(exclude=['datasets', 'experiments', 'viz']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'numpy',
        'Pillow',
        'opencv-python-headless',
        'scikit-learn',
        'scipy',
        'tqdm',
        'matplotlib',
        'lpips',
        'face-alignment',
        'ninja',
    ],
)
