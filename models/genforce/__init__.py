"""GenForce model zoo for FaceAnonyMixer.

The full GenForce synthesis network is an optional dependency installed by::

    bash scripts/setup_genforce.sh

Without it the mapping network and latent-space operations still work; only
actual image decoding (synthesis) requires the full GenForce tree.
"""

from .models import MODEL_ZOO, build_generator, EqualLinear  # noqa: F401
