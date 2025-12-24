#!/bin/bash
# Run all tests including examples
#
# Usage: ./scripts/test-all.sh [--quick|--full]
#   --quick: Skip examples that require corpus (default if no corpus)
#   --full: Run everything including corpus-dependent examples
#
# Exit codes:
#   0 - All tests passed
#   1 - Some tests failed

# Don't use set -e, we handle errors manually
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

FAILED=0
PASSED=0
SKIPPED=0

run_test() {
    local name="$1"
    shift
    info "Running: $name"
    if "$@" 2>&1; then
        pass "$name"
        PASSED=$((PASSED + 1))
    else
        fail "$name"
        FAILED=$((FAILED + 1))
    fi
}

run_example() {
    local name="$1"
    info "Running example: $name"
    if cargo run --example "$name" --release 2>&1 | tail -20; then
        pass "Example: $name"
        PASSED=$((PASSED + 1))
    else
        fail "Example: $name"
        FAILED=$((FAILED + 1))
    fi
}

skip_test() {
    local name="$1"
    local reason="$2"
    warn "Skipping: $name ($reason)"
    SKIPPED=$((SKIPPED + 1))
}

# Check for corpus
has_corpus() {
    [[ -d "$PROJECT_ROOT/corpus/kodak" ]] || \
    [[ -n "$CODEC_CORPUS_DIR" && -d "$CODEC_CORPUS_DIR/kodak" ]] || \
    [[ -n "$MOZJPEG_CORPUS_DIR" && -d "$MOZJPEG_CORPUS_DIR/kodak" ]]
}

# Parse args
MODE="auto"
case "${1:-}" in
    --quick) MODE="quick" ;;
    --full) MODE="full" ;;
    --help|-h)
        echo "Usage: $0 [--quick|--full]"
        echo "  --quick: Skip corpus-dependent examples"
        echo "  --full: Run all tests (requires corpus)"
        exit 0
        ;;
esac

if [[ "$MODE" == "auto" ]]; then
    if has_corpus; then
        MODE="full"
    else
        MODE="quick"
        warn "No corpus found. Run ./scripts/fetch-corpus.sh for full testing."
    fi
fi

echo ""
echo "=========================================="
echo "  mozjpeg-rs Test Suite"
echo "  Mode: $MODE"
echo "=========================================="
echo ""

# 1. Build check
info "=== Build Check ==="
run_test "cargo check" cargo check --workspace

# 2. Unit tests
info ""
info "=== Unit Tests ==="
run_test "cargo test --lib" cargo test --lib -p mozjpeg

# 3. Integration tests
info ""
info "=== Integration Tests ==="
run_test "ffi_validation" cargo test --test ffi_validation -p mozjpeg
run_test "encoder_validation" cargo test --test encoder_validation -p mozjpeg
run_test "codec_comparison" cargo test --test codec_comparison -p mozjpeg

# 4. Examples that work without corpus (use bundled images)
info ""
info "=== Examples (bundled images) ==="
run_example "benchmark"
run_example "test_edge_cropping"
run_example "encode_permutations"

# Note: test_no_trellis falls back to bundled image
run_example "test_no_trellis"

# 5. Examples requiring corpus
info ""
info "=== Examples (corpus required) ==="
if [[ "$MODE" == "full" ]]; then
    if has_corpus; then
        run_example "benchmark_corpus"
        run_example "test_444"
        # compare_real_images needs CLIC corpus (--full fetch)
        if [[ -d "$PROJECT_ROOT/corpus/clic2025" ]] || \
           [[ -n "$CODEC_CORPUS_DIR" && -d "$CODEC_CORPUS_DIR/clic2025" ]]; then
            run_example "compare_real_images"
        else
            skip_test "compare_real_images" "CLIC corpus not found (run fetch-corpus.sh --full)"
        fi
    else
        skip_test "corpus examples" "no corpus found"
    fi
else
    skip_test "corpus examples" "quick mode"
fi

# 6. FFI comparison tests (require local mozjpeg source)
info ""
info "=== FFI Comparison Tests ==="
if [[ -d "$PROJECT_ROOT/../mozjpeg" ]]; then
    run_test "ffi_comparison" cargo test --test ffi_comparison -p mozjpeg
    run_test "mozjpeg-sys-local" cargo test -p mozjpeg-sys-local
else
    skip_test "ffi_comparison" "local mozjpeg source not found at ../mozjpeg"
    skip_test "mozjpeg-sys-local" "local mozjpeg source not found"
fi

# Summary
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo "=========================================="

if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
