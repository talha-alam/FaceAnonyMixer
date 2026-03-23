#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the Roboto Condensed Bold font used by visualize.py.
# The font is released under the Apache 2.0 licence.
#
# Usage:
#   bash scripts/download_fonts.sh
# ---------------------------------------------------------------------------
set -euo pipefail

FONT_DIR="lib/fonts"
FONT_FILE="${FONT_DIR}/RobotoCondensed-Bold.ttf"
FONT_URL="https://github.com/googlefonts/roboto/raw/main/src/hinted/RobotoCondensed-Bold.ttf"

mkdir -p "${FONT_DIR}"

if [ -f "${FONT_FILE}" ]; then
    echo "[download_fonts] ${FONT_FILE} already present — skipping."
    exit 0
fi

echo "[download_fonts] Downloading Roboto Condensed Bold ..."
curl -fsSL -o "${FONT_FILE}" "${FONT_URL}"
echo "[download_fonts] Saved to ${FONT_FILE}"
