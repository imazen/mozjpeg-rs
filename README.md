# mozjpeg-oxide

Pure Rust JPEG encoder based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg), featuring trellis quantization for optimal compression.

[![Crates.io](https://img.shields.io/crates/v/mozjpeg-oxide.svg)](https://crates.io/crates/mozjpeg-oxide)
[![Documentation](https://docs.rs/mozjpeg-oxide/badge.svg)](https://docs.rs/mozjpeg-oxide)
[![CI](https://github.com/imazen/mozjpeg-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/mozjpeg-oxide/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/mozjpeg-oxide/graph/badge.svg)](https://codecov.io/gh/imazen/mozjpeg-oxide)
[![License](https://img.shields.io/crates/l/mozjpeg-oxide.svg)](LICENSE)

## Why mozjpeg-oxide?

| | mozjpeg-oxide | C mozjpeg | libjpeg-turbo |
|--|---------------|-----------|---------------|
| **Language** | Pure Rust | C | C/asm |
| **Memory safety** | Compile-time guaranteed | Manual | Manual |
| **Trellis quantization** | Yes | Yes | No |
| **File size** | ~Same as C | Baseline | 5-15% larger |
| **Build complexity** | `cargo add` | cmake + nasm + C toolchain | cmake + nasm |

**Choose mozjpeg-oxide when you want:**
- Memory-safe JPEG encoding without C dependencies
- Smaller files than libjpeg-turbo (trellis quantization)
- Simple integration via Cargo

**Choose C mozjpeg when you need:**
- Maximum baseline encoding speed (SIMD-optimized)
- Established C ABI for FFI

## Benchmark Results

Tested on [Kodak](http://r0k.us/graphics/kodak/) + [CLIC](https://www.compression.cc/) corpora (56 images total).
Settings: progressive mode, trellis quantization, optimized Huffman, 4:2:0 subsampling.

| Quality | Rust BPP | C BPP | Size Δ | Rust SSIM2 | C SSIM2 | Quality Δ |
|---------|----------|-------|--------|------------|---------|-----------|
| 20 | 0.251 | 0.252 | **-0.3%** | 21.5 | 21.8 | -0.30 |
| 50 | 0.534 | 0.532 | +0.4% | 57.3 | 57.5 | -0.18 |
| 75 | 0.881 | 0.870 | +1.3% | 71.8 | 72.0 | -0.13 |
| 85 | 1.251 | 1.224 | +2.2% | 77.8 | 77.9 | -0.08 |
| 95 | 2.367 | 2.270 | +4.3% | 86.1 | 86.1 | -0.03 |

**Summary**: Files are 0-4% larger than C mozjpeg with imperceptible quality difference (SSIM2 within 0.3 points).

### Reproduce Benchmarks

```bash
# Fetch test corpus (Kodak images, ~15MB)
./scripts/fetch-corpus.sh

# Run benchmark
cargo bench-corpus

# Or just Kodak images
cargo bench-kodak
```

Results written to `benchmark_results.csv`. See also `cargo bench-micro` for criterion microbenchmarks.

## Usage

```rust
use mozjpeg_oxide::{Encoder, Subsampling};

// Default: trellis quantization + Huffman optimization
let jpeg = Encoder::new()
    .quality(85)
    .encode_rgb(&pixels, width, height)?;

// Maximum compression: progressive + trellis + scan optimization
let jpeg = Encoder::max_compression()
    .quality(85)
    .encode_rgb(&pixels, width, height)?;

// Fastest: no optimizations (libjpeg-turbo compatible output)
let jpeg = Encoder::fastest()
    .quality(85)
    .encode_rgb(&pixels, width, height)?;

// Custom configuration
let jpeg = Encoder::new()
    .quality(75)
    .progressive(true)
    .subsampling(Subsampling::S420)
    .optimize_huffman(true)
    .encode_rgb(&pixels, width, height)?;
```

## Features

- **Trellis quantization** - Rate-distortion optimized coefficient selection
- **Progressive JPEG** - Multi-scan encoding with spectral selection
- **Huffman optimization** - 2-pass encoding for optimal entropy coding
- **Overshoot deringing** - Reduces ringing artifacts at sharp edges
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Safe Rust** - `#![deny(unsafe_code)]` with exceptions only for SIMD intrinsics

## Performance

| Configuration | vs C mozjpeg | Notes |
|---------------|--------------|-------|
| Baseline (no opts) | ~4x slower | C has SIMD entropy coding |
| Trellis enabled | **0.9x (faster)** | Trellis dominates runtime |
| Max compression | **0.6x (faster)** | Progressive + trellis |

With trellis quantization enabled (the default), mozjpeg-oxide matches or exceeds C mozjpeg performance.

## Development

### Running CI Locally

To reproduce the CI checks locally:

```bash
# Format check
cargo fmt --all -- --check

# Clippy lints
cargo clippy --workspace --all-targets -- -D warnings

# Build
cargo build --workspace

# Unit tests
cargo test --lib

# Codec comparison tests
cargo test --test codec_comparison

# FFI validation tests (requires mozjpeg-sys from crates.io)
cargo test --test ffi_validation
```

For full CI reproduction including container isolation, install [act](https://github.com/nektos/act):

```bash
# Install act (macOS)
brew install act

# Run CI locally
act push
```

### Test Coverage

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --lib --html

# Open report
open target/llvm-cov/html/index.html
```

## License

BSD-3-Clause - Same license as the original mozjpeg.

## Acknowledgments

Based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg), which builds on libjpeg-turbo and the Independent JPEG Group's libjpeg.

## AI-Generated Code Notice

This crate was developed with significant assistance from Claude (Anthropic). While the code has been tested against the C mozjpeg reference implementation and passes 191 tests including FFI validation, **not all code has been manually reviewed or human-audited**.

Before using in production:
- Review critical code paths for your use case
- Run your own validation against expected outputs
- Consider the encoder's test suite coverage for your specific requirements

The FFI comparison tests in `tests/ffi_comparison.rs` and `tests/ffi_validation.rs` provide confidence in correctness by comparing outputs against C mozjpeg.
