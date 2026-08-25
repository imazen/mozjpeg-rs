#!/bin/bash
# Setup script for instrumented C mozjpeg with test exports
#
# This script clones and builds the imazen/mozjpeg fork which includes
# special test export functions for granular FFI comparison testing.
#
# After running this script, you can run:
#   cargo test -p sys-local --test ffi_comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MOZJPEG_DIR="$(dirname "$PROJECT_DIR")/mozjpeg"

echo "=== Instrumented mozjpeg Setup ==="
echo "Project: $PROJECT_DIR"
echo "Target:  $MOZJPEG_DIR"
echo ""

# Check if mozjpeg directory already exists
if [ -d "$MOZJPEG_DIR" ]; then
    echo "mozjpeg directory already exists at $MOZJPEG_DIR"
    echo "Updating..."
    cd "$MOZJPEG_DIR"
    git fetch origin
    git checkout main
    git pull origin main
else
    echo "Cloning imazen/mozjpeg fork..."
    cd "$(dirname "$MOZJPEG_DIR")"
    git clone https://github.com/imazen/mozjpeg.git mozjpeg
    cd "$MOZJPEG_DIR"
fi

echo ""
echo "Building mozjpeg C library..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=OFF \
      -DENABLE_STATIC=ON \
      ..

# Build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now run the FFI comparison tests:"
echo "  cargo test -p sys-local --test ffi_comparison"
echo ""
echo "Or run the sys-local crate tests:"
echo "  cargo test -p sys-local"
echo ""
