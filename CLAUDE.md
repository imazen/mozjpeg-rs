# mozjpeg-oxide Development Guide

## Project Overview

Rust port of Mozilla's mozjpeg JPEG encoder, following the jpegli-rs methodology.

**Scope**: Encoder only + mozjpeg extensions (trellis, progressive scans, deringing)
**API**: Idiomatic Rust (not C-compatible)
**Validation**: FFI dual-execution against C mozjpeg

## Current Status

**191 tests passing** (153 unit + 8 codec comparison + 11 FFI comparison + 5 FFI validation + 9 sys-local + 5 encoder validation)

### Compression Results vs C mozjpeg

**With matching settings (progressive + trellis + Huffman opt):**

| Mode | Rust | C mozjpeg | Ratio | Notes |
|------|------|-----------|-------|-------|
| Progressive + Trellis | 71,176 bytes | 72,721 bytes | **0.98x** | Rust is 2.1% smaller! |
| Baseline + Trellis | 73,834 bytes | 72,721 bytes | 1.02x | C defaults to progressive |

**Small test images (16x16):**
| Quality | Rust Size | C Size | Ratio | Rust PSNR | C PSNR |
|---------|-----------|--------|-------|-----------|--------|
| Q50 | 491 bytes | 498 bytes | **0.99x** | 40.33 dB | 40.16 dB |
| Q75 | 518 bytes | 558 bytes | **0.93x** | 44.90 dB | 45.20 dB |
| Q85 | 590 bytes | 631 bytes | **0.94x** | 47.32 dB | 47.49 dB |

**Note:** C mozjpeg uses JCP_MAX_COMPRESSION profile by default which enables progressive
mode. Use `Encoder::max_compression()` for equivalent behavior.

### Performance vs C mozjpeg (release mode, AVX2 enabled)

**2048x2048 image (30 iterations, most accurate):**

| Configuration | Rust (ms) | C (ms) | Ratio | Notes |
|---------------|-----------|--------|-------|-------|
| Baseline (no opts) | 40.04 | 8.46 | 4.74x slower | Entropy encoding bottleneck |
| Trellis AC+DC | 162.88 | 181.07 | **0.90x faster** | **Rust 10% faster!** |

**512x512 image (faster but more system noise):**

| Configuration | Rust (ms) | C (ms) | Ratio | Notes |
|---------------|-----------|--------|-------|-------|
| Baseline (no opts) | 1.73 | 0.45 | 3.9x slower | |
| Trellis AC+DC | 11.13 | 11.72 | **0.95x faster** | |
| Progressive* | 4.58 | 11.25 | **0.41x faster** | See note below |
| Max compression* | 14.44 | 24.04 | **0.60x faster** | See note below |

**Key findings:**
- **Trellis mode: Rust is 10% faster than C mozjpeg** (at 2048x2048)
- Baseline gap is ~4.7x - entropy encoding is the remaining bottleneck
- AVX2 DCT intrinsics provide 26% speedup for baseline encoding
- Color conversion uses i32x8 (AVX2) for 8 pixels at a time

**\* Progressive mode note:** Both Rust and C support `optimize_scans` which tries multiple
scan configurations to find the smallest output. Progressive encoding now works correctly
for all image sizes including non-MCU-aligned dimensions with subsampling.

### Completed Layers
- Layer 0: Constants, types, error handling
- Layer 1: Quantization tables, Huffman table construction
- Layer 2: Forward DCT, color conversion, chroma subsampling
- Layer 3: Bitstream writer with byte stuffing
- Layer 4: Entropy encoder, **trellis quantization (AC + DC)**
- Layer 5: **Progressive scan generation (integrated)**
- Layer 6: Marker emission, **Full encoder pipeline**
- **FFI Validation**: Granular comparison tests against C mozjpeg

### Working Encoder
The encoder produces valid JPEG files with mozjpeg-quality compression:
```rust
use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};

// Default encoding (trellis + Huffman optimization enabled)
let encoder = Encoder::new().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Maximum compression (progressive + trellis + optimized Huffman)
let encoder = Encoder::max_compression();
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Fastest encoding (no optimizations)
let encoder = Encoder::fastest().quality(85);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;

// Custom configuration
let encoder = Encoder::new()
    .quality(75)
    .progressive(true)
    .subsampling(Subsampling::S420)
    .trellis(TrellisConfig::default())
    .optimize_huffman(true);
let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
```

### Implemented Features
- **Baseline JPEG encoding** - Standard sequential DCT
- **Progressive JPEG encoding** - Multi-scan with DC first, then AC bands
- **Trellis quantization** - Rate-distortion optimized AC + DC quantization (mozjpeg core feature)
- **DC trellis optimization** - Dynamic programming across blocks for optimal DC encoding
- **Huffman table optimization** - 2-pass encoding for optimal tables
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Quality presets** - `max_compression()` and `fastest()`
- **Overshoot deringing** - Reduce ringing artifacts at sharp edges (see below)
- **Optimize scans** - Try multiple scan configurations for progressive mode, pick smallest

### Remaining Work
- **Performance optimization (SIMD)** - DCT and color conversion are 7.5x slower than C
- EOB optimization integration (`trellis_eob_opt` - disabled by default in C mozjpeg)
- Arithmetic coding (optional, rarely used)

### Recent Fixes
- **Progressive AC scan block count** (Dec 2024): Fixed bug where non-MCU-aligned images
  with subsampling produced corrupted progressive JPEGs. AC scans now correctly encode
  `ceil(width/8) × ceil(height/8)` blocks instead of MCU-padded block count.

**Both baseline and progressive modes work correctly!** With trellis + Huffman optimization,
Rust produces files with quality matching C mozjpeg across all image sizes and subsampling modes.

### Known Issues / Active Investigations

#### Rust vs C Pixel Difference - RESOLVED ✅

Previously reported "max diff ~11" was due to comparing different encoding modes:
- C mozjpeg defaults to `JCP_MAX_COMPRESSION` profile which enables progressive mode
- Rust was using baseline mode in comparisons

**Investigation findings (Dec 2024):**

| Component | Match Status | Notes |
|-----------|--------------|-------|
| DCT | ✅ Exact | FFI test passes |
| Quantization | ✅ Exact | FFI test passes |
| Color conversion | ✅ ±1 | Rounding variance |
| Downsampling | ✅ Exact | FFI test passes |
| Quant tables | ✅ Identical | Verified in JPEG output |
| Huffman tables | ✅ Identical | With `optimize_huffman=true` |
| **Full encoder** | ✅ **0 diff** | When comparing same mode |

**Key findings:**
- With truly identical settings (baseline + Huffman opt), **0 pixel difference**
- Without Huffman optimization, C uses optimized tables even with `optimize_coding=0`
- File size with Huffman opt: Rust within 5% of C (16 bytes = JFIF density field difference)

**Remaining file size notes:**
- Without Huffman optimization, Rust uses standard Annex K tables while C uses minimal tables
- This causes ~40% size difference in non-optimized mode (not a bug, just different defaults)
- Enable `optimize_huffman(true)` for best compression (default in Rust)

## Workflow Rules

### Commit Strategy
- **Commit when new tests pass** - After fixing/completing a module
- **Commit when new tests are added** - Even if they're failing (documents expected behavior)
- Write descriptive commit messages explaining what was ported

### Validation Approach
- Validate equivalence **layer by layer**, not just end-to-end
- Use `mozjpeg-sys` from crates.io for basic FFI validation
- Use `sys-local` (in `crates/`) for granular internal function testing
  - Builds from local `../mozjpeg` C source with test exports
  - C code has been instrumented with `mozjpeg_test_*` functions
  - Tests in `tests/ffi_comparison.rs` compare Rust vs C implementations

### Golden Rule: Never Delete Instrumentation
**NEVER delete tests, FFI comparisons, or instrumentation code.** These are essential for:
- Validating correctness against C mozjpeg
- Catching regressions during development
- Documenting expected behavior
- Ensuring byte-exact parity with C implementation

If a test seems obsolete, comment it out with explanation rather than deleting.

### Golden Rule: Never Relax Test Tolerances to Avoid Debugging
**NEVER relax test tolerances or skip failing tests to make CI green.** When tests fail:
1. **Debug the root cause** - Don't paper over bugs with looser thresholds
2. **Use DSSIM, not PSNR** - PSNR is unreliable for perceptual quality comparison
3. **Validate byte/coefficient differences** - Compare actual encoded values between Rust and C
4. **Track "off-by-N" errors** - Small systematic differences indicate encoder bugs

If a test is failing, the encoder has a bug. Find it and fix it.

## Key Learnings

### mozjpeg Specifics
1. **Default quant tables**: mozjpeg uses ImageMagick tables (index 3), not JPEG Annex K (index 0)
2. **Quality scaling**: Q50 = 100% scale factor (use tables as-is)
3. **DCT scaling**: Output is scaled by factor of 8 (sqrt(8) per dimension)
4. **Huffman pseudo-symbol**: Symbol 256 ensures no real symbol gets all-ones code

### Trellis Quantization (Critical!)
The trellis algorithm requires raw DCT coefficients (scaled by 8):
1. **Lambda calculation**: `lambda = 2^scale1 / (2^scale2 + block_norm)` with `lambda_base = 1.0`
2. **Per-coefficient weights**: `weight[i] = 1/quantval[i]^2`
3. **Quantization divisor**: `q = 8 * quantval[i]` (includes DCT scaling)
4. **Distortion**: `(candidate * q - original)^2 * lambda * weight`
5. **Cost**: `rate + distortion` where rate is Huffman code size + value bits

### Overshoot Deringing

Reduces visible ringing artifacts near hard edges, especially on white backgrounds.
Source: `jcdctmgr.c:416-550` (~130 lines). Enabled by default with `Encoder::new()`.

**Core Insight:**
- JPEG can encode values outside the displayable range (0-255)
- Decoders clamp results to 0-255
- To encode white (255), any encoded value ≥ 255 works after clamping
- Hard edges create "square wave" patterns that compress poorly
- By allowing values to "overshoot" above 255, we get smoother waveforms

**Algorithm (applied BEFORE DCT, after level shift):**
1. Scan block for pixels at max value (127 after level shift = 255 original)
2. If no max pixels or all max pixels: return unchanged
3. Calculate safe overshoot limit:
   ```
   maxovershoot = maxsample + min(31, 2*DC_quant, (maxsample*64 - sum)/count)
   ```
4. For each run of max-value pixels (in zigzag order):
   - Calculate slopes from neighboring pixels
   - Apply Catmull-Rom spline interpolation
   - Clamp peaks to maxovershoot

**Before (hard edge):**
```
Values: 50, 80, 120, 127, 127, 127, 100, 60
```

**After (smooth curve with overshoot):**
```
Values: 50, 80, 120, 135, 140, 138, 100, 60
                      ↑    ↑    ↑
              These overshoot 127 but clamp to 255 when decoded
```

**When it helps:**
- Images with white backgrounds
- Text and graphics with hard edges
- Any image with saturated regions (pixels at 0 or 255)

**API:**
```rust
// Enabled by default
let encoder = Encoder::new();

// Explicitly enable/disable
let encoder = Encoder::new().overshoot_deringing(true);
let encoder = Encoder::fastest().overshoot_deringing(false); // fastest disables it
```

### Implementation Notes
1. **Huffman tree construction**: Use sentinel values carefully to avoid overflow
   - `FREQ_INITIAL_MAX = 1_000_000_000` for comparison
   - `FREQ_MERGED = 1_000_000_001` for merged nodes
2. **Bitstream stuffing**: 0xFF bytes ALWAYS require 0x00 stuffing in entropy data
3. **Bit buffer**: Use 64-bit buffer, flush when full, pad with 1-bits at end
4. **Progressive encoding**:
   - DC scans can be interleaved (multiple components)
   - AC scans must be single-component
   - Successive approximation uses Ah/Al for bit refinement

### Testing Patterns
1. Use `#[cfg(test)]` modules within each source file
2. FFI validation tests in `tests/ffi_validation.rs`
3. Test both positive cases and error conditions

## Architecture

```
mozjpeg-oxide/                  # Repository root IS the main crate
├── src/
│   ├── lib.rs                  # Module exports, public API
│   ├── consts.rs               # Layer 0: Constants, tables, markers
│   ├── types.rs                # Layer 0: ColorSpace, ScanInfo, etc.
│   ├── error.rs                # Error types
│   ├── quant.rs                # Layer 1: Quantization tables
│   ├── huffman.rs              # Layer 1: Huffman table construction
│   ├── dct.rs                  # Layer 2: Forward DCT (Loeffler)
│   ├── color.rs                # Layer 2: RGB→YCbCr conversion
│   ├── sample.rs               # Layer 2: Chroma subsampling
│   ├── deringing.rs            # Layer 2: Overshoot deringing (pre-DCT)
│   ├── bitstream.rs            # Layer 3: Bit-level I/O
│   ├── entropy.rs              # Layer 4: Huffman encoding
│   ├── trellis.rs              # Layer 4: Trellis quantization
│   ├── progressive.rs          # Layer 5: Progressive scans
│   ├── marker.rs               # Layer 6: JPEG markers
│   ├── encode.rs               # Layer 6: Encoder pipeline
│   └── test_encoder.rs         # Unified test API for Rust vs C comparison
├── tests/
│   ├── ffi_validation.rs       # crates.io mozjpeg-sys tests
│   └── ffi_comparison.rs       # Local FFI granular comparison
├── examples/
│   └── pareto_benchmark.rs     # Benchmark vs C mozjpeg
├── crates/
│   └── sys-local/              # Local FFI bindings (builds from ../mozjpeg)
│       ├── build.rs            # CMake integration
│       └── src/lib.rs          # FFI declarations + test exports
├── benchmark/                  # Reproducible benchmark infrastructure
│   ├── Dockerfile
│   └── plot_pareto.py
└── ../mozjpeg/                 # Instrumented C mozjpeg fork (external)
    ├── mozjpeg_test_exports.c  # Test export implementations
    └── mozjpeg_test_exports.h  # Test export declarations
```

### Unified Test API (`test_encoder.rs`)

For comparing Rust vs C implementations with guaranteed parameter parity:

```rust
use mozjpeg_oxide::test_encoder::{TestEncoderConfig, encode_rust};

// Configuration shared between both implementations
let config = TestEncoderConfig {
    quality: 85,
    subsampling: Subsampling::S420,
    progressive: false,
    optimize_huffman: false,
    trellis_quant: false,
    trellis_dc: false,
    overshoot_deringing: false,
};

// Encode with Rust
let rust_jpeg = encode_rust(&rgb, width, height, &config);

// Encode with C (in examples, implement encode_c using mozjpeg_sys)
let c_jpeg = encode_c(&rgb, width, height, &config);
```

This ensures apples-to-apples comparison by using identical settings.

## Build & Test

```bash
cargo test                           # Run all tests
cargo test huffman                   # Run specific module tests
cargo test --test ffi_validation    # Run crates.io FFI tests
cargo test --test ffi_comparison    # Run local FFI comparison tests
cargo test -p sys-local             # Run sys-local tests
```

### Test Corpus

For extended testing with real images:

```bash
# Fetch Kodak test images (~15MB)
./scripts/fetch-corpus.sh

# Fetch full corpus including CLIC (~100MB)
./scripts/fetch-corpus.sh --full

# Or set environment variable to use existing corpus
export CODEC_CORPUS_DIR=/path/to/codec-corpus
```

The corpus utilities in `mozjpeg_oxide::corpus` handle path resolution:
- Checks `MOZJPEG_CORPUS_DIR` and `CODEC_CORPUS_DIR` environment variables
- Falls back to `./corpus/` in project root
- Bundled test images always available at `tests/images/`

### CI/CD

GitHub Actions workflow runs on push/PR:
- Tests on Linux, macOS, Windows
- Unit tests, codec comparison tests, FFI validation tests
- Excludes `ffi_comparison` tests (require local mozjpeg C source)

## Dependencies

- `mozjpeg-sys = "2.2"` (dev) - FFI validation against C mozjpeg
- `sys-local` (in `crates/`) - Local FFI with granular test exports
- `bytemuck = "1.14"` - Safe transmutes (for future SIMD)
- `dssim`, `ssimulacra2` (dev) - Perceptual quality metrics
- `codec-eval` (dev) - From https://github.com/imazen/codec-comparison
