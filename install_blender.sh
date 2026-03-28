#!/bin/bash
# Install Blender 3.3 LTS as a standalone tarball (headless-capable, no GPU required).
# The nytimes/blender Docker image requires GLX/GPU and crashes on headless CPU nodes.
# The official Blender Linux binary ships with llvmpipe software rendering and works
# headlessly with LIBGL_ALWAYS_SOFTWARE=1.
#
# Usage: bash install_blender.sh

set -e

BLENDER_VERSION="3.3.21"
BLENDER_DIR="$HOME/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="$BLENDER_DIR/blender-softwaregl"
TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender3.3/${TARBALL}"

if [ -x "$BLENDER_BIN" ]; then
    echo "Already installed: $BLENDER_BIN"
    LIBGL_ALWAYS_SOFTWARE=1 "$BLENDER_BIN" --version
    exit 0
fi

echo "Downloading Blender ${BLENDER_VERSION} from blender.org..."
cd "$HOME"
wget -q --show-progress "$URL" -O "$TARBALL"

echo "Extracting..."
tar -xf "$TARBALL"
rm "$TARBALL"

echo ""
echo "Installed: $BLENDER_BIN"
LIBGL_ALWAYS_SOFTWARE=1 "$BLENDER_BIN" --version
