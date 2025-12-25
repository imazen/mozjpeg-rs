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

Tested on [Kodak](http://r0k.us/graphics/kodak/) corpus (24 images).
Settings: progressive mode, trellis quantization, optimized Huffman, 4:2:0 subsampling.

| Quality | Rust BPP | C BPP | Size Δ | Rust SSIM2 | C SSIM2 | Butteraugli |
|---------|----------|-------|--------|------------|---------|-------------|
| 5 | 0.081 | 0.080 | +1.3% | -58.6 | -58.5 | 18.8 |
| 10 | 0.157 | 0.157 | +0.1% | -23.7 | -23.4 | 11.3 |
| 15 | 0.231 | 0.233 | **-0.5%** | 2.6 | 3.0 | 8.4 |
| 20 | 0.301 | 0.303 | **-0.7%** | 19.6 | 20.0 | 6.9 |
| 25 | 0.368 | 0.370 | **-0.7%** | 31.0 | 31.4 | 6.2 |
| 30 | 0.430 | 0.432 | **-0.6%** | 38.7 | 39.0 | 5.5 |
| 35 | 0.490 | 0.492 | **-0.4%** | 45.2 | 45.5 | 5.0 |
| 40 | 0.546 | 0.548 | **-0.3%** | 49.7 | 50.0 | 4.8 |
| 45 | 0.600 | 0.601 | **-0.2%** | 53.3 | 53.5 | 4.5 |
| 50 | 0.654 | 0.654 | **-0.0%** | 56.9 | 57.2 | 4.3 |
| 55 | 0.715 | 0.713 | +0.2% | 59.7 | 60.0 | 4.1 |
| 60 | 0.774 | 0.771 | +0.3% | 61.9 | 62.1 | 3.8 |
| 65 | 0.856 | 0.851 | +0.6% | 65.4 | 65.5 | 3.8 |
| 70 | 0.948 | 0.943 | +0.5% | 67.9 | 68.1 | 3.5 |
| 75 | 1.077 | 1.068 | +0.9% | 71.8 | 71.9 | 3.3 |
| 80 | 1.277 | 1.259 | +1.4% | 75.2 | 75.3 | 3.0 |
| 85 | 1.519 | 1.492 | +1.8% | 78.5 | 78.5 | 2.6 |
| 90 | 1.968 | 1.915 | +2.8% | 82.7 | 82.8 | 2.2 |
| 92 | 2.204 | 2.140 | +3.0% | 84.1 | 84.1 | 2.0 |
| 95 | 2.826 | 2.721 | +3.9% | 87.0 | 87.0 | 1.7 |
| 97 | 3.772 | 3.580 | +5.4% | 89.3 | 89.3 | 1.5 |

**Summary**: At Q15-Q50, Rust produces **smaller files**. At Q55+, files are 0.2-5% larger. Quality (SSIM2, Butteraugli) is virtually identical across all levels.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmark/pareto_ssimulacra2.svg">
  <source media="(prefers-color-scheme: light)" srcset="benchmark/pareto_ssimulacra2.svg">
  <img alt="SSIMULACRA2 vs BPP Pareto curve" src="benchmark/pareto_ssimulacra2.svg">
</picture>

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
