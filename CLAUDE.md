# mozjpeg-oxide Development Guide

## Development Guidelines

### Never disable features to achieve parity

**BANNED SOLUTIONS:**
- Disabling features in `max_compression()` to hide parity gaps
- Changing defaults to avoid broken code paths
- Commenting out features that don't work correctly

**REQUIRED APPROACH:**
1. If a feature produces wrong output, FIX THE FEATURE
2. Document gaps honestly in README, but keep features enabled

### Never relax test tolerances

If tests fail, find and fix the bug. Never:
- Increase allowed difference thresholds
- Skip failing tests
- Mark tests as `#[ignore]` to make CI green

### Resolved Issues

**optimize_scans producing larger files (FIXED Dec 2024):**
- Root cause: Trial encoder used per-scan Huffman tables, actual encoder used global tables
- Fix: Actual encoder now uses per-scan AC Huffman tables when optimize_scans=true
- Result: optimize_scans now produces smaller files at all quality levels

## Project Overview

Rust port of Mozilla's mozjpeg JPEG encoder, following the jpegli-rs methodology.

**Scope**: Encoder only + mozjpeg extensions (trellis, progressive scans, deringing)
**API**: Idiomatic Rust (not C-compatible)
**Validation**: FFI dual-execution against C mozjpeg

## Current Status

**191 tests passing** (153 unit + 8 codec comparison + 11 FFI comparison + 5 FFI validation + 9 sys-local + 5 encoder validation)

### Compression Results vs C mozjpeg

**Kodak corpus benchmark (24 images, baseline + trellis + Huffman opt):**

| Quality | Rust Total | C Total | Size Δ | Max Individual Δ |
|---------|------------|---------|--------|------------------|
| Q50 | 772,864 | 774,577 | **-0.22%** | -0.72% |
| Q75 | 1,270,536 | 1,276,594 | **-0.47%** | -1.22% |
| Q85 | 1,791,238 | 1,795,089 | **-0.21%** | -0.75% |
| Q90 | 2,320,570 | 2,323,641 | **-0.13%** | -0.79% |
| Q95 | 3,333,392 | 3,336,170 | **-0.08%** | -0.64% |
| Q97 | 4,449,919 | 4,447,419 | **+0.06%** | -0.57% |

**Key findings:**
- **With trellis enabled (default): Rust is within ±0.5% of C mozjpeg**
- **Without trellis: Rust is within ±0.01% of C mozjpeg** (essentially identical)
- Rust often produces **smaller** files than C mozjpeg at Q50-Q95
- Trellis effect: Rust saves 15.7% vs C's 12.9% on typical images
- Visual quality is equivalent (verified via SSIMULACRA2 and Butteraugli)

**Progressive mode with optimize_scans (6 images from Kodak corpus):**

| Quality | Simple Progressive vs C | optimize_scans vs C | Notes |
|---------|-------------------------|---------------------|-------|
| Q50 | -0.83% | **-1.42%** | Rust smaller than C |
| Q75 | +0.72% | **-0.67%** | Rust smaller with opt_scans |
| Q85 | +1.82% | **-0.07%** | Near-identical to C |
| Q90 | +3.01% | +0.58% | Small gap at high quality |
| Q95 | +3.82% | +0.96% | |
| Q97 | +4.95% | +1.15% | |

**Key findings for progressive mode:**
- With `optimize_scans=true`: Rust matches or beats C at Q50-Q85
- Each AC scan gets its own optimal Huffman table (per-scan tables)
- The simple 4-scan script works well, but `optimize_scans` finds even better configurations

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

#### File Size Gap (2-4% with optimize_scans) - ROOT CAUSE IDENTIFIED

**Symptom:** Rust produces ~2-4% larger files with `optimize_scans` enabled.

**Root Cause:** Trial encoding for refinement scans is fundamentally broken.

When `optimize_scans` is enabled, C mozjpeg:
1. Generates 64 candidate scans (matching Rust exactly as of Dec 2024)
2. Encodes ALL scans **in sequence**, storing the bytes in buffers
3. Uses scan sizes to select optimal Al levels and frequency splits
4. **Copies pre-encoded bytes** directly to output (copy_buffer)

Rust's approach differs critically:
1. Generates 64 candidate scans (matching C exactly ✓)
2. Trial-encodes each scan **independently** to get sizes
3. Uses sizes to select configuration (algorithm matches C ✓)
4. **Re-encodes** the selected scans from scratch

**The problem:** Refinement scans (Ah > 0) cannot be encoded independently.
They require the state from previous "first" scans to know which bits to refine.

When we trial-encode a refinement scan alone, it produces garbage sizes:
- Scan 3 (Y refine Ah=1→Al=0): **22,728 bytes** (should be ~200-500)
- This causes Al=1 cost to be ~10x higher than Al=0
- Optimizer always picks Al=0 (no successive approximation)

**Evidence:**
```
Scan sizes: [2466, 2128, 11, 22728, 657, 5, 21922, 24, 2, 21515, ...]
                           ^^^^^ garbage refine scan size
Al=0 cost: 2128 + 11 = 2139
Al=1 cost: 657 + 5 + 22728 = 23390 (10x higher due to garbage refine)
```

**To fully match C mozjpeg's optimize_scans:**
1. Encode all 64 candidate scans in proper sequence (with state)
2. Store encoded bytes for each scan in buffers
3. Select optimal configuration based on sizes
4. Copy selected scan buffers directly to output

This is a significant architectural change. The current implementation still
produces valid, well-optimized progressive JPEGs - just not with successive
approximation, which limits high-quality compression gains.

#### Rust vs C Pixel Difference - RESOLVED ✅

Previously reported "max diff ~11" was due to comparing different encoding modes:
- C mozjpeg defaults to `JCP_MAX_COMPRESSION` profile which enables progressive mode
- Rust was using baseline mode in comparisons

**Investigation findings (Dec 2024):**

| Component | Match Status | Notes |
|-----------|--------------|-------|
| DCT | ✅ Exact | FFI test passes |
| Quantization | ✅ Exact | FFI test passes |
| Trellis | ✅ Exact | FFI test passes (new Dec 2024) |
| Color conversion | ✅ ±1 | Rounding variance |
| Downsampling | ✅ Exact | FFI test passes |
| Quant tables | ✅ Identical | Verified in JPEG output |
| Huffman tables | ✅ Identical | With `optimize_huffman=true` |
| **Full encoder** | ✅ **0 diff** | When comparing same mode |

**Key findings:**
- With truly identical settings (baseline + Huffman opt), **0 pixel difference**
- Trellis quantization produces identical coefficient decisions
- DC clamping to 1023 now matches C behavior
- File size gap at high quality due to progressive scan structure (see above)

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
