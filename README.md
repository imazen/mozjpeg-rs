# mozjpeg-rs

Pure Rust JPEG encoder based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg), featuring trellis quantization for optimal compression.

[![Crates.io](https://img.shields.io/crates/v/mozjpeg-rs.svg)](https://crates.io/crates/mozjpeg-rs)
[![Documentation](https://docs.rs/mozjpeg-rs/badge.svg)](https://docs.rs/mozjpeg-rs)
[![CI](https://github.com/imazen/mozjpeg-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/mozjpeg-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/mozjpeg-rs/graph/badge.svg)](https://codecov.io/gh/imazen/mozjpeg-rs)
[![License](https://img.shields.io/crates/l/mozjpeg-rs.svg)](LICENSE)

## Encoder Only

**mozjpeg-rs is a JPEG encoder only.** It does not decode JPEG files.

For decoding, use one of these excellent crates:

| Crate | Type | Notes |
|-------|------|-------|
| **[jpeg-decoder](https://crates.io/crates/jpeg-decoder)** | Pure Rust | Widely used, reliable |
| **[zune-jpeg](https://crates.io/crates/zune-jpeg)** | Pure Rust | Fast, SIMD-optimized |
| **[mozjpeg-sys](https://crates.io/crates/mozjpeg-sys)** | C bindings | Full mozjpeg (encode + decode) |

## Why mozjpeg-rs?

| | mozjpeg-rs | C mozjpeg | libjpeg-turbo |
|--|---------------|-----------|---------------|
| **Language** | Pure Rust | C | C/asm |
| **Memory safety** | Compile-time guaranteed | Manual | Manual |
| **Trellis quantization** | Yes | Yes | No |
| **Build complexity** | `cargo add` | cmake + nasm + C toolchain | cmake + nasm |

**Choose mozjpeg-rs when you want:**
- Memory-safe JPEG encoding without C dependencies
- Smaller files than libjpeg-turbo (trellis quantization)
- Simple integration via Cargo

**Choose C mozjpeg when you need:**
- Maximum baseline encoding speed (SIMD-optimized entropy coding)
- Established C ABI for FFI
- Arithmetic coding (rarely used)

## Compression Results vs C mozjpeg

Tested on full [Kodak](http://r0k.us/graphics/kodak/) corpus (24 images), 4:2:0 subsampling, `fast-yuv` enabled. Six encoder configurations across four quality levels. Positive delta = Rust files are larger; negative = Rust files are smaller.

Reproduce with: `cargo test --release --test parity_benchmark -- --nocapture`

| Config                   |  Q |   Avg Rust |      Avg C |   Delta | Max Dev |
|--------------------------|----|------------|------------|---------|---------|
| Baseline                 | 75 |     60,253 |     60,126 |  +0.21% |   0.35% |
| Baseline                 | 85 |     83,482 |     83,296 |  +0.22% |   0.42% |
| Baseline                 | 90 |    106,716 |    106,479 |  +0.22% |   0.40% |
| Baseline                 | 95 |    150,888 |    150,570 |  +0.21% |   0.45% |
| Baseline + Trellis       | 75 |     53,054 |     53,183 |  -0.24% |   0.97% |
| Baseline + Trellis       | 85 |     74,781 |     74,792 |  -0.01% |   0.54% |
| Baseline + Trellis       | 90 |     96,902 |     96,805 |  +0.10% |   0.56% |
| Baseline + Trellis       | 95 |    139,188 |    138,957 |  +0.17% |   0.57% |
| Full Baseline            | 75 |     53,077 |     53,191 |  -0.21% |   0.94% |
| Full Baseline            | 85 |     74,796 |     74,795 |  +0.00% |   0.53% |
| Full Baseline            | 90 |     96,915 |     96,818 |  +0.10% |   0.55% |
| Full Baseline            | 95 |    139,211 |    139,007 |  +0.15% |   0.37% |
| Progressive              | 75 |     58,998 |     58,873 |  +0.21% |   0.30% |
| Progressive              | 85 |     80,928 |     80,749 |  +0.22% |   0.38% |
| Progressive              | 90 |    102,410 |    102,204 |  +0.20% |   0.37% |
| Progressive              | 95 |    143,747 |    143,446 |  +0.21% |   0.41% |
| Progressive + Trellis    | 75 |     52,774 |     52,866 |  -0.17% |   0.64% |
| Progressive + Trellis    | 85 |     73,652 |     73,642 |  +0.01% |   0.33% |
| Progressive + Trellis    | 90 |     94,364 |     94,302 |  +0.07% |   0.35% |
| Progressive + Trellis    | 95 |    134,226 |    134,051 |  +0.13% |   0.41% |
| Full Progressive         | 75 |     52,789 |     52,869 |  -0.15% |   0.65% |
| Full Progressive         | 85 |     73,654 |     73,652 |  +0.00% |   0.35% |
| Full Progressive         | 90 |     94,380 |     94,308 |  +0.08% |   0.34% |
| Full Progressive         | 95 |    134,253 |    134,074 |  +0.13% |   0.40% |
| Max Compression          | 75 |     52,632 |     52,480 |  +0.29% |   1.08% |
| Max Compression          | 85 |     73,615 |     73,353 |  +0.36% |   0.87% |
| Max Compression          | 90 |     94,487 |     94,120 |  +0.39% |   0.84% |
| Max Compression          | 95 |    134,095 |    133,721 |  +0.28% |   0.64% |

**Configs:** Baseline = huffman opt only. +Trellis = AC trellis. Full = AC trellis + DC trellis + deringing. Max Compression = Full + `optimize_scans: true`. All others use `optimize_scans: false`. All use `force_baseline: true`.

**Key findings:**
- With trellis at Q75, Rust produces **smaller** files than C (-0.15% to -0.24%)
- Without trellis, the consistent +0.21% gap comes from `fast-yuv` color conversion (±1 level rounding)
- Without `optimize_scans`, all configs stay within ±0.25% average, worst-case per-image deviation under 1%
- With `optimize_scans` (Max Compression), within ±0.4% average — different scan search heuristics find different local optima
- Visual quality (SSIMULACRA2, Butteraugli) is equivalent at all settings

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmark/pareto_ssimulacra2.svg">
  <source media="(prefers-color-scheme: light)" srcset="benchmark/pareto_ssimulacra2.svg">
  <img alt="SSIMULACRA2 vs BPP Pareto curve" src="benchmark/pareto_ssimulacra2.svg">
</picture>

## Usage

```rust
use mozjpeg_rs::{Encoder, Subsampling};

// Default: trellis quantization + Huffman optimization
let jpeg = Encoder::new()
    .quality(85)
    .encode_rgb(&pixels, width, height)?;

// Maximum compression: progressive + trellis + deringing
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

- **Trellis quantization** - Rate-distortion optimized coefficient selection (AC + DC)
- **Progressive JPEG** - Multi-scan encoding with spectral selection
- **Huffman optimization** - 2-pass encoding for optimal entropy coding
- **Overshoot deringing** - Reduces ringing artifacts at sharp edges
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Safe Rust** - `#![deny(unsafe_code)]` with exceptions only for SIMD intrinsics

### Encoder Settings Matrix

All combinations of settings are supported and tested:

| Setting | Baseline | Progressive | Notes |
|---------|:--------:|:-----------:|-------|
| **Subsampling** | | | |
| ├─ 4:4:4 | ✅ | ✅ | No chroma subsampling |
| ├─ 4:2:2 | ✅ | ✅ | Horizontal subsampling |
| └─ 4:2:0 | ✅ | ✅ | Full subsampling (default) |
| **Trellis Quantization** | | | |
| ├─ AC trellis | ✅ | ✅ | Rate-distortion optimized AC coefficients |
| └─ DC trellis | ✅ | ✅ | Cross-block DC optimization |
| **Huffman** | | | |
| ├─ Default tables | ✅ | ✅ | Fast, slightly larger files |
| └─ Optimized tables | ✅ | ✅ | 2-pass, smaller files |
| **Progressive-only** | | | |
| └─ optimize_scans | ❌ | ✅ | Per-scan Huffman tables |
| **Other** | | | |
| ├─ Deringing | ✅ | ✅ | Reduce overshoot artifacts |
| ├─ Grayscale | ✅ | ✅ | Single-component encoding |
| ├─ EOB optimization | ✅ | ✅ | Cross-block EOB runs (opt-in) |
| └─ Smoothing | ✅ | ✅ | Noise reduction filter (for dithered images) |

**Presets:**
- `Encoder::new()` - Trellis (AC+DC) + Huffman optimization + Deringing
- `Encoder::max_compression()` - Above + Progressive + optimize_scans
- `Encoder::fastest()` - No optimizations (libjpeg-turbo compatible)

### Quantization Tables

| Table | Description |
|-------|-------------|
| `Robidoux` | **Default.** Nicolas Robidoux's psychovisual tables (used by ImageMagick) |
| `JpegAnnexK` | Standard JPEG tables (libjpeg default) |
| `Flat` | Uniform quantization |
| `MssimTuned` | MSSIM-optimized on Kodak corpus |
| `PsnrHvsM` | PSNR-HVS-M tuned |
| `Klein` | Klein, Silverstein, Carney (1992) |
| `Watson` | DCTune (Watson, Taylor, Borthwick 1997) |
| `Ahumada` | Ahumada, Watson, Peterson (1993) |
| `Peterson` | Peterson, Ahumada, Watson (1993) |

```rust
use mozjpeg_rs::{Encoder, QuantTableIdx};

let jpeg = Encoder::new()
    .qtable(QuantTableIdx::Robidoux)  // or .quant_tables()
    .encode_rgb(&pixels, width, height)?;
```

### Method Aliases

For CLI-style naming (compatible with rimage conventions):

| Alias | Equivalent |
|-------|------------|
| `.baseline(true)` | `.progressive(false)` |
| `.optimize_coding(true)` | `.optimize_huffman(true)` |
| `.chroma_subsampling(mode)` | `.subsampling(mode)` |
| `.qtable(idx)` | `.quant_tables(idx)` |

## Performance

Benchmarked on 512x768 image, 20 iterations, release mode:

| Configuration | Rust | C mozjpeg | Ratio |
|---------------|------|-----------|-------|
| Baseline (huffman opt) | 7.1 ms | 26.8 ms | **3.8x faster** |
| Trellis (AC + DC) | 19.7 ms | 25.3 ms | **1.3x faster** |
| Progressive + trellis | 20.0 ms | - | - |

**Note**: C mozjpeg's baseline encoding is typically faster with its hand-optimized SIMD entropy coding. The benchmark numbers above reflect mozjpeg-sys from crates.io which may not have all optimizations enabled.

### SIMD Support

mozjpeg-rs uses `multiversion` for automatic vectorization by default. Optional hand-written SIMD intrinsics are available:

```toml
[dependencies]
mozjpeg-rs = { version = "0.2", features = ["simd-intrinsics"] }
```

In benchmarks, the difference is minimal (~2%) as `multiversion` autovectorization works well for DCT and color conversion.

## Differences from C mozjpeg

mozjpeg-rs aims for compatibility with C mozjpeg but has some differences:

| Feature | mozjpeg-rs | C mozjpeg |
|---------|---------------|-----------|
| **Progressive scan script** | 9-scan with successive approximation (or optimize_scans) | 9-scan with successive approximation |
| **optimize_scans** | Per-scan Huffman tables | Per-scan Huffman tables |
| **Trellis EOB optimization** | Available (opt-in) | Available (rarely used) |
| **Smoothing filter** | Available | Available |
| **Multipass trellis** | Not implemented (poor tradeoff) | Available |
| **Arithmetic coding** | Not implemented | Available (rarely used) |
| **Grayscale progressive** | Yes | Yes |

### Why multipass (`use_scans_in_trellis`) is not implemented

C mozjpeg's multipass option makes trellis quantization "scan-aware" for progressive encoding by optimizing low and high frequency AC coefficients separately. Benchmarks on the Kodak corpus (Q85, progressive) show this is a poor tradeoff:

| Metric | Without Multipass | With Multipass | Difference |
|--------|-------------------|----------------|------------|
| File size | 1,760 KB | 1,770 KB | **+0.52% larger** |
| Quality (butteraugli) | 2.59 | 2.54 | -0.05 (imperceptible) |
| Encoding time | ~7ms | ~8.5ms | **~20% slower** |

Multipass produces larger files, is slower, and provides no perceptible quality improvement.

### Where does the remaining gap come from?

The consistent +0.21% gap in non-trellis modes comes from the `fast-yuv` feature, which uses the `yuv` crate for SIMD color conversion (AVX-512/AVX2/SSE/NEON). It has ±1 level rounding differences vs C mozjpeg's color conversion, producing slightly different DCT coefficients. This is invisible after JPEG quantization. Without `fast-yuv`, Rust matches or beats C at all quality levels.

With trellis enabled, Rust's trellis optimizer finds slightly better rate-distortion tradeoffs at Q75, producing smaller files than C.

### Matching C mozjpeg output exactly

For near byte-identical output to C mozjpeg, use baseline mode with matching settings:
1. Use baseline (non-progressive) mode with Huffman optimization
2. Match all encoder settings via `TestEncoderConfig`
3. Use the same quantization tables (Robidoux/ImageMagick, the default for both)

The FFI comparison tests in `tests/ffi_comparison.rs` verify component-level parity.

## Development

### Running CI Locally

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

### Reproduce Benchmarks

```bash
# Fetch test corpus (Kodak images, ~15MB)
./scripts/fetch-corpus.sh

# Run full corpus comparison
cargo run --release --example full_corpus_test

# Run pareto benchmark
cargo run --release --example pareto_benchmark
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

This crate was developed with significant assistance from Claude (Anthropic). While the code has been tested against the C mozjpeg reference implementation and passes 248 tests including FFI validation, **not all code has been manually reviewed or human-audited**.

Before using in production:
- Review critical code paths for your use case
- Run your own validation against expected outputs
- Consider the encoder's test suite coverage for your specific requirements

The FFI comparison tests in `tests/ffi_comparison.rs` and `tests/ffi_validation.rs` provide confidence in correctness by comparing outputs against C mozjpeg.
