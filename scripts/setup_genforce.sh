#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clone the GenForce repository and install the StyleGAN2 model definitions
# into models/genforce/ so that FaceAnonyMixer can import them directly.
#
# This script is idempotent: running it a second time is a no-op.
#
# Usage:
#   bash scripts/setup_genforce.sh
# ---------------------------------------------------------------------------
set -euo pipefail

GENFORCE_URL="https://github.com/genforce/genforce.git"
TARGET_DIR="models/genforce"
MARKER="${TARGET_DIR}/.genforce_installed"

if [ -f "${MARKER}" ]; then
    echo "[setup_genforce] GenForce already installed — skipping."
    exit 0
fi

echo "[setup_genforce] Cloning GenForce (shallow) ..."
TMP_DIR="$(mktemp -d)"
git clone --depth 1 --quiet "${GENFORCE_URL}" "${TMP_DIR}"

echo "[setup_genforce] Copying model definitions into ${TARGET_DIR}/ ..."
# Copy every .py file from genforce/models/
cp -r "${TMP_DIR}/models/"* "${TARGET_DIR}/"

echo "[setup_genforce] Cleaning up ..."
rm -rf "${TMP_DIR}"

# Write marker so re-runs are skipped
touch "${MARKER}"

echo ""
echo "[setup_genforce] Done. GenForce synthesis network is now available."
echo "  You can now run the full pipeline (invert.py, anonymize.py, etc.)."
