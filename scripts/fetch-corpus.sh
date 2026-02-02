#!/bin/bash
# Fetch codec-corpus test images for testing
#
# Usage: ./scripts/fetch-corpus.sh [--minimal|--full]
#   --minimal: Just CID22 training images (~15MB)
#   --full: CID22 + sample CLIC images (~100MB)
#
# Images are downloaded to ./corpus/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CORPUS_DIR="$PROJECT_ROOT/corpus"

CODEC_CORPUS_REPO="https://github.com/imazen/codec-corpus"
CODEC_CORPUS_BRANCH="main"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_requirements() {
    local missing=()

    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing required tools: ${missing[*]}"
        exit 1
    fi
}

fetch_minimal() {
    info "Fetching minimal corpus (CID22 training images)..."

    if [ -d "$CORPUS_DIR/CID22/CID22-512/training" ] && [ "$(ls -A "$CORPUS_DIR/CID22/CID22-512/training" 2>/dev/null)" ]; then
        info "CID22 corpus already exists, skipping..."
        return 0
    fi

    mkdir -p "$CORPUS_DIR"

    # Use sparse checkout to get only CID22/CID22-512/training directory
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    info "Cloning codec-corpus (sparse checkout for CID22/CID22-512/training)..."
    cd "$temp_dir"
    git init -q
    git remote add origin "$CODEC_CORPUS_REPO"
    git config core.sparseCheckout true
    echo "CID22/CID22-512/training/*" > .git/info/sparse-checkout
    git fetch --depth=1 origin "$CODEC_CORPUS_BRANCH" -q
    git checkout -q FETCH_HEAD

    # Copy to corpus directory
    if [ -d "CID22" ]; then
        mkdir -p "$CORPUS_DIR/CID22/CID22-512"
        cp -r CID22/CID22-512/training "$CORPUS_DIR/CID22/CID22-512/"
        info "CID22 corpus downloaded to $CORPUS_DIR/CID22/CID22-512/training"
    else
        error "Failed to fetch CID22/CID22-512/training directory"
        exit 1
    fi

    trap - EXIT
    rm -rf "$temp_dir"
}

fetch_full() {
    fetch_minimal

    info "Fetching additional corpus images..."

    if [ -d "$CORPUS_DIR/clic2025" ] && [ "$(ls -A "$CORPUS_DIR/clic2025" 2>/dev/null)" ]; then
        info "CLIC corpus already exists, skipping..."
        return 0
    fi

    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    info "Cloning codec-corpus (sparse checkout for clic2025/validation)..."
    cd "$temp_dir"
    git init -q
    git remote add origin "$CODEC_CORPUS_REPO"
    git config core.sparseCheckout true
    echo "clic2025/validation/*" > .git/info/sparse-checkout
    git fetch --depth=1 origin "$CODEC_CORPUS_BRANCH" -q
    git checkout -q FETCH_HEAD

    if [ -d "clic2025" ]; then
        mkdir -p "$CORPUS_DIR/clic2025"
        cp -r clic2025/validation "$CORPUS_DIR/clic2025/"
        info "CLIC corpus downloaded to $CORPUS_DIR/clic2025/validation"
    else
        warn "CLIC directory not found in corpus, skipping..."
    fi

    trap - EXIT
    rm -rf "$temp_dir"
}

print_usage() {
    echo "Usage: $0 [--minimal|--full]"
    echo ""
    echo "Options:"
    echo "  --minimal  Download only CID22 training images (~15MB)"
    echo "  --full     Download CID22 + CLIC validation images (~100MB)"
    echo ""
    echo "Default: --minimal"
    echo ""
    echo "Images are downloaded to: $CORPUS_DIR"
}

# Main
check_requirements

case "${1:-}" in
    --minimal|"")
        fetch_minimal
        ;;
    --full)
        fetch_full
        ;;
    --help|-h)
        print_usage
        exit 0
        ;;
    *)
        error "Unknown option: $1"
        print_usage
        exit 1
        ;;
esac

info "Corpus fetch complete!"
echo ""
echo "Corpus location: $CORPUS_DIR"
echo ""
if [ -d "$CORPUS_DIR/CID22/CID22-512/training" ]; then
    echo "  CID22/CID22-512/training/: $(ls "$CORPUS_DIR/CID22/CID22-512/training" | wc -l | tr -d ' ') images"
fi
if [ -d "$CORPUS_DIR/clic2025/validation" ]; then
    echo "  clic2025/validation/: $(ls "$CORPUS_DIR/clic2025/validation" | wc -l | tr -d ' ') images"
fi
