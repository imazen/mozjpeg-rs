# mozjpeg-rs Development Guide

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

**AC refinement correction bits tracking (FIXED Jan 2025):**
- Root cause: `count_ac_refine()` didn't track correction bits accumulation, causing DHT
  to miss intermediate EOB symbols when encoder flushed early due to buffer overflow
- Fix: Added `correction_bits_count` field to `ProgressiveSymbolCounter` and matching
  flush threshold (937 bytes) to count_ac_refine()
- Result: ProgressiveBalanced Q90+ now decodes correctly with all decoders (GitHub #2)

**AC refinement encoding (FIXED Dec 2024):**
- Root cause: ZRL loop only ran for newly-nonzero coefficients, not previously-coded ones
- Fix: Restructured encode_ac_refine() to match C mozjpeg's loop structure exactly
- Result: All progressive modes with successive approximation now work correctly

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

**Tests passing**: 164 unit + 8 codec comparison + 5 FFI validation

### Compression Results vs C mozjpeg

**Kodak corpus (24 images), 4:2:0, fast-yuv enabled. 6 configs × 4 quality levels.**
Reproduce: `cargo test --release --test parity_benchmark -- --nocapture`

| Config                   |  Q |   Delta | Max Dev |
|--------------------------|----|---------|---------|
| Baseline                 | 75 |  +0.21% |   0.35% |
| Baseline                 | 85 |  +0.22% |   0.42% |
| Baseline                 | 90 |  +0.22% |   0.40% |
| Baseline                 | 95 |  +0.21% |   0.45% |
| Baseline + Trellis       | 75 |  -0.24% |   0.97% |
| Baseline + Trellis       | 85 |  -0.01% |   0.54% |
| Baseline + Trellis       | 90 |  +0.10% |   0.56% |
| Baseline + Trellis       | 95 |  +0.17% |   0.57% |
| Full Baseline            | 75 |  -0.21% |   0.94% |
| Full Baseline            | 85 |  +0.00% |   0.53% |
| Full Baseline            | 90 |  +0.10% |   0.55% |
| Full Baseline            | 95 |  +0.15% |   0.37% |
| Progressive              | 75 |  +0.21% |   0.30% |
| Progressive              | 85 |  +0.22% |   0.38% |
| Progressive              | 90 |  +0.20% |   0.37% |
| Progressive              | 95 |  +0.21% |   0.41% |
| Progressive + Trellis    | 75 |  -0.17% |   0.64% |
| Progressive + Trellis    | 85 |  +0.01% |   0.33% |
| Progressive + Trellis    | 90 |  +0.07% |   0.35% |
| Progressive + Trellis    | 95 |  +0.13% |   0.41% |
| Full Progressive         | 75 |  -0.15% |   0.65% |
| Full Progressive         | 85 |  +0.00% |   0.35% |
| Full Progressive         | 90 |  +0.08% |   0.34% |
| Full Progressive         | 95 |  +0.13% |   0.40% |
| Max Compression          | 55 |  -0.04% |   1.64% |
| Max Compression          | 65 |  +0.14% |   0.97% |
| Max Compression          | 75 |  +0.29% |   1.08% |
| Max Compression          | 85 |  +0.36% |   0.87% |
| Max Compression          | 90 |  +0.39% |   0.84% |
| Max Compression          | 95 |  +0.28% |   0.64% |

**Configs:** Baseline = huffman opt only. +Trellis = AC trellis. Full = AC trellis + DC trellis + deringing. Max Compression = Full + `optimize_scans: true`. All others use `optimize_scans: false`. All use `force_baseline: true`.

**Key findings:**
- With trellis at Q75, Rust produces **smaller** files than C (-0.15% to -0.24%)
- Without trellis, consistent +0.21% gap from `fast-yuv` color conversion ±1 rounding
- Without `optimize_scans`, all configs within ±0.25% average, worst-case per-image deviation under 1%
- With `optimize_scans` (Max Compression), within ±0.4% average, per-image max ~1.6%
- Rust scan optimizer sometimes finds different local optima than C (different Al/freq split choices)
- Visual quality equivalent (SSIMULACRA2 and Butteraugli verified)

**Mode explanations:**
- **Baseline** (`progressive(false)`): Sequential DCT
- **Progressive** (`progressive(true), optimize_scans(false)`): 9-scan JCP_MAX_COMPRESSION script with successive approximation
- **Max Compression** (`Encoder::max_compression()`): Progressive + `optimize_scans=true` with per-scan Huffman tables

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
- Color conversion uses yuv crate (5.15 Gelem/s, AVX-512/AVX2/SSE/NEON/WASM)

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
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};

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
- **Progressive JPEG encoding** - Multi-scan with DC first, then AC bands (RGB and grayscale)
- **Trellis quantization** - Rate-distortion optimized AC + DC quantization (mozjpeg core feature)
- **Trellis speed optimization** - Adaptive search limiting for high-entropy blocks (Q80-100)
- **DC trellis optimization** - Dynamic programming across blocks for optimal DC encoding
- **Huffman table optimization** - 2-pass encoding for optimal tables
- **Chroma subsampling** - 4:4:4, 4:2:2, 4:2:0 modes
- **Quality presets** - `max_compression()` and `fastest()`
- **Overshoot deringing** - Reduce ringing artifacts at sharp edges (see below)
- **Optimize scans** - Try multiple scan configurations for progressive mode, pick smallest
- **Grayscale progressive** - Full progressive JPEG support for grayscale images
- **Smoothing filter** - Noise reduction for dithered images (`.smoothing(30)`)

### Remaining Work
- **Baseline entropy encoding** - ~4.7x slower than C (trellis mode is 10% faster than C)
- Arithmetic coding (optional, rarely used)

### TrellisConfig Fields Not Yet Wired Up
- **`freq_split`** (default 8) — Field exists, only used in progressive scan generation,
  not in the trellis DP loop. Could split AC trellis into low/high frequency passes.
  `trellis_quantize_block_with_eob_info()` already accepts `ss`/`se` parameters.
- **`q_opt`** (default false) — Field exists but not implemented. Would optimize quant tables
  within trellis loop. Not implemented in C mozjpeg either (experimental/placeholder).

### TrellisConfig Fields — Implemented but No-Op by Default
- **`delta_dc_weight`** (default 0.0) — Wired up in `dc_trellis_optimize_indexed()`.
  When > 0.0, blends vertical DC gradient error into the distortion cost, encouraging
  smoother DC transitions between rows. Matches C mozjpeg `trellis_delta_dc_weight`
  (jcdctmgr.c:1069-1084). Default 0.0 matches C defaults (disabled).
- **`use_lambda_weight_tbl`** — Not used. C mozjpeg hardcodes flat (1/q^2) weights
  regardless of this flag. Rust matches C behavior.
- **`num_loops`** — Not used. Always runs one trellis pass.

### Not Implemented (Poor Tradeoff)
- **Multipass trellis** (`use_scans_in_trellis`) - C mozjpeg benchmarks show +0.52% larger files,
  imperceptible quality improvement (-0.05 butteraugli), 20% slower encoding. Not worth implementing.

### Optional Features (Disabled by Default)
- **EOB cross-block optimization** (`TrellisConfig::eob_optimization(true)`) - Experimental
  cross-block EOBRUN optimization. Disabled by default due to aggressive coefficient zeroing
  in some cases. Enable with `TrellisConfig::default().eob_optimization(true)` if needed.

### Recent Fixes
- **AC refinement correction bits tracking** (Jan 2025): Fixed bug where `count_ac_refine()`
  didn't track correction bits accumulation, causing DHT to miss intermediate EOB symbols.
  The encoder flushes EOBRUN when correction bits exceed 937 bytes, but the counter wasn't
  tracking this threshold. This caused ProgressiveBalanced Q90+ to fail with strict decoders.
- **AC refinement ZRL encoding** (Dec 2024): Fixed bug where ZRL (zero-run-length) symbols
  weren't emitted for previously-coded coefficients, causing decoder errors on 8/24 images.
  Now uses 9-scan JCP_MAX_COMPRESSION script with full successive approximation.
- **Progressive AC scan block count** (Dec 2024): Fixed bug where non-MCU-aligned images
  with subsampling produced corrupted progressive JPEGs. AC scans now correctly encode
  `ceil(width/8) × ceil(height/8)` blocks instead of MCU-padded block count.

**Both baseline and progressive modes work correctly!** With trellis + Huffman optimization,
Rust produces files with quality matching C mozjpeg across all image sizes and subsampling modes.

### Bytewise Parity Analysis (Progressive Q85, Kodak corpus)

**File size comparison (24 images, 9-scan SA script, trellis disabled):**
- Total: C = 1,939,046 bytes, Rust = 1,939,068 bytes (+22 bytes, **+0.00%**)
- Per-image range: -30 to +30 bytes (±0.05%)
- Not a fixed offset - varies per image due to coefficient rounding

**Segment comparison:**
- DQT (quant tables): ✅ Identical
- SOF2 (frame header): ✅ Identical
- DHT (Huffman tables): ✅ Identical
- SOS headers: ✅ Identical
- Entropy data: First 257 bytes match, then diverges

**Decoded pixel comparison:**
- R channel: 0 differences (perfect match)
- G channel: 53 pixels differ by ≤1
- B channel: 122 pixels differ by ≤4

The entropy divergence is caused by minor coefficient rounding differences that
cascade through DC differential encoding. Both produce visually identical images.

### Known Issues / Active Investigations

#### File Size Gap with optimize_scans - FIXED ✅ (Feb 2026)

**Original symptom:** Rust produced ~1-4% larger files with `optimize_scans` at low
quality levels (Q10-Q50), with kodim23 showing +3.37% at Q40.

**Root cause:** `encode_rust()` in `test_encoder.rs` was not passing `optimize_scans`
to the `Encoder` builder chain. The Rust scan optimizer was never called — the encoder
always used the fixed 9-scan script regardless of the `optimize_scans` flag. The C
encoder correctly used its scan search to find simpler scripts (4-5 scans, no SA) at
low quality.

**Fix:** Added `.optimize_scans(config.optimize_scans)` to `encode_rust()` builder chain
in `src/test_encoder.rs`.

**Result:** Max Compression within ±0.4% average at all quality levels. At low quality
(Q10-Q50) Rust is now **smaller** than C. Per-image max deviation ~1.6% (from different
local optima in scan search, not a bug).

**Previous fixes (Dec 2025):** ScanTrialEncoder sequential encoding + per-scan Huffman.
**Previous note (Feb 2025):** C test harness optimize_scans control.

#### AC Refinement Decoder Errors - FIXED ✅ (Dec 2024)

**Symptom:** 8/24 Kodak images failed to decode with "failed to decode huffman code"
when using progressive mode with successive approximation refinement scans.

**Root Cause:** In `encode_ac_refine()` and `count_ac_refine()`, the ZRL (zero-run-length)
loop was only inside the "temp == 1" (newly nonzero) branch, but it needs to run for BOTH
temp > 1 (previously coded) AND temp == 1 cases. When a long run of zeros preceded a
previously coded coefficient, ZRL symbols weren't emitted, corrupting the bitstream.

**Fix:** Restructured the encoding loop to match C mozjpeg's jcphuff.c exactly:
1. Pre-pass to compute `absvalues[]` and find `eob` (last newly-nonzero position)
2. ZRL loop now runs for all non-zero coefficients before the temp > 1 vs temp == 1 check
3. Added `k <= eob` condition to prevent ZRL emission after the last newly-nonzero

**Status:** All 24 Kodak images decode successfully with 9-scan JCP_MAX_COMPRESSION script.
Successive approximation (al_max_luma=3, al_max_chroma=2) is now fully enabled.

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

#### Decoder Chroma Upsampling Variance - DOCUMENTED ✅

When testing decoder round-trips with very small images using chroma subsampling,
we observed large pixel differences (up to 41) between decoders. Investigation
revealed this is expected decoder behavior, not an encoder bug.

**The exact boundary: `chroma_width == 2` (luma width 3-4 with horizontal subsampling)**

| Luma Width | Chroma Width | Rust Decoders vs mozjpeg | Notes |
|------------|--------------|--------------------------|-------|
| 1-2 | 1 | ≤4 diff | Single sample, all agree |
| **3-4** | **2** | **24-41 diff** | Triangle vs simple upsampling |
| 5+ | 3+ | 0 diff | Enough samples, all agree |

**Root cause:** Different chroma upsampling algorithms:
- **jpeg-decoder & zune-jpeg**: Simple replication/linear interpolation
- **mozjpeg**: "Fancy" triangle filter (optimized for perceptual quality)

**Surprisingly, Rust decoders are closer to the original image** at the boundary:

| Case | Rust mean error | mozjpeg mean error | Winner |
|------|-----------------|--------------------| -------|
| 3×4 4:2:2 (chroma 2×4) | 11.6 | 12.1 | Rust by 0.4 |
| 4×4 4:2:0 (chroma 2×2) | 12.3 | 18.0 | Rust by 5.7 |
| 5×4 4:2:2 (chroma 3×4) | 4.7 | 4.7 | Tie (identical) |

The mozjpeg triangle filter is designed for normal images where smooth interpolation
improves perceptual quality. But with only 2 chroma samples, the filter introduces
more deviation from the original than simple replication.

**Test coverage:** `decoder_roundtrip.rs` tests 2496 combinations (26 dimensions ×
4 presets × 3 subsamplings × 8 qualities) with appropriate tolerance for this boundary.

**Analysis tool:** `examples/decoder_chroma_analysis.rs` shows pixel-by-pixel comparison.

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

### Deringing + 16-bit SIMD DCT Overflow (mozilla/mozjpeg#453)

**Bug:** Overshoot deringing pushes level-shifted samples to ±158. The ISLOW forward
DCT's 16-bit SIMD path (`_mm256_add_epi16`) overflows in the column-pass final butterfly:
`8 × 5056 = 40,448 > i16::MAX (32,767)`. Wrapping causes sign flips that invert entire
8×8 blocks.

**mozjpeg-rs status:** Production paths use i32 intermediates — **immune**.
The experimental `SimdOps::avx2_i16()` path (`src/dct.rs:1326`) reproduces the bug.
Use `Encoder::simd_ops()` to select it for testing.

**Trigger conditions:**
- Deringing enabled + i16 SIMD DCT path
- Vertical half-black/half-white edge within an 8×8 block
- Quality ≤ Q57 (DC quant value ≥ 14)

**Horizontal splits do NOT trigger** because each row is uniform (DC-only, zero AC energy).

**Test patterns:** `imazen/codec-corpus` at `imageflow/test_inputs/dct_overflow_patterns/`
- `left_black_right_white.png` — 64×64, triggers overflow
- `left_white_right_black.png` — 64×64, triggers overflow
- `single_8x8_half.png` — 8×8 minimal reproducer
- `top_black_bottom_white.png` — 64×64, does NOT trigger (control)
- `checkerboard_8x8.png` — 64×64, does NOT trigger (control)

**Regression tests:** `tests/encode_tests.rs:1328` (`test_issue444_deringing_overflow_pattern`)
and `tests/encode_tests.rs:1387` (`test_issue444_across_quality_range`).

**Reference:** https://github.com/mozilla/mozjpeg/pull/453

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
5. **AVX2/SIMD intrinsic element ordering** (CRITICAL):
   - `_mm256_set_epi16(e15, e14, ..., e1, e0)` takes arguments in **REVERSE order** (highest element first)
   - NASM `times 4 dw A, B` creates `[A,B,A,B,...]` with A at even indices
   - To match NASM layout, use `_mm256_set_epi16(..., B, A, B, A)` (swap pairs)
   - For `vpmaddwd`: `result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]`
   - If data is `[R, G]` and you want `R*coef_R + G*coef_G`, coefficients must be `[coef_R, coef_G]` in memory
   - See `dct.rs:1384` for documented example

### Testing Patterns
1. Use `#[cfg(test)]` modules within each source file
2. FFI validation tests in `tests/ffi_validation.rs`
3. Test both positive cases and error conditions

## Architecture

```
mozjpeg-rs/                  # Repository root IS the main crate
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
use mozjpeg_rs::test_encoder::{TestEncoderConfig, encode_rust};

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

The corpus utilities in `mozjpeg_rs::corpus` handle path resolution:
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

## Feature Flags

### Published Features (safe to use)

- **`fast-yuv`** (default) - Use the `yuv` crate for SIMD color conversion.
  ~38% faster than our hand-written AVX2 (5.15 vs 3.72 Gelem/s at 1920x1080).
  Supports AVX-512, AVX2, SSE, NEON, and WASM SIMD. Precision difference is ±1 level,
  invisible after JPEG quantization.

- **`mozjpeg-sys-config`** - Encode using C mozjpeg with Rust `Encoder` settings.
  Adds `Encoder::to_c_mozjpeg()` which returns a `CMozjpeg` encoder.
  Uses `mozjpeg-sys` from crates.io.

  ```rust
  let jpeg = encoder.to_c_mozjpeg().encode_rgb(&pixels, w, h)?;
  ```

- **`simd-intrinsics`** - Hand-written AVX2/NEON intrinsics for ~15% better DCT performance.
  Without this, uses `multiversion` autovectorization (safe, ~87% of intrinsics perf).

- **`png`** - Enable PNG loading in corpus module.

### Development-Only Features (not published)

- **`_instrument-c-mozjpeg-internals`** - Enables granular FFI tests against internal C mozjpeg
  functions (DCT, quantization, color conversion, etc.). Requires imazen/mozjpeg fork with
  test exports built locally. Use `scripts/setup-instrumented-mozjpeg.sh` to set up.

## Decoder Alternatives

**mozjpeg-rs is an ENCODER ONLY.** For decoding JPEG files, use one of:

- **[jpeg-decoder](https://crates.io/crates/jpeg-decoder)** - Pure Rust, widely used
- **[zune-jpeg](https://crates.io/crates/zune-jpeg)** - Pure Rust, fast, SIMD-optimized
- **[mozjpeg-sys](https://crates.io/crates/mozjpeg-sys)** - C mozjpeg bindings (encode + decode)
- **[jpegli-rs](https://github.com/psy-repos-rust/jpegli-rs)** - Rust port of Google's jpegli (WIP)

## Round-Trip Decoder Validation

The `decoder_roundtrip` test validates that all major JPEG decoders can decode mozjpeg-rs output
and produce consistent results within expected quality bounds.

**Test matrix:**
- Decoders: jpeg-decoder, zune-jpeg, mozjpeg-sys (C decoder)
- Encoder configs: all 4 Presets × subsampling modes
- Quality levels: Q50, Q60, Q70, Q75, Q80, Q85, Q90, Q95
- Images: 20 test images from corpus

**Validation criteria:**
- All decoders must successfully decode the JPEG
- Decoded pixels must match between decoders (allowing ±1 for rounding)
- Butteraugli distance must be within expected bounds for each Q level

**Expected Butteraugli bounds by quality:**
| Quality | Max Butteraugli |
|---------|-----------------|
| Q50 | 3.0 |
| Q60 | 2.5 |
| Q70 | 2.0 |
| Q75 | 1.5 |
| Q80 | 1.2 |
| Q85 | 1.0 |
| Q90 | 0.7 |
| Q95 | 0.4 |

## CI Test Organization

Tests are organized to handle symbol conflicts between different mozjpeg bindings:

```bash
# Core tests (no FFI)
cargo test -p mozjpeg-rs --lib

# Integration tests using mozjpeg-sys from crates.io
cargo test --test ffi_validation
cargo test --test preset_parity
cargo test --features mozjpeg-sys-config c_mozjpeg

# Decoder round-trip tests
cargo test --test decoder_roundtrip

# Instrumented C mozjpeg tests (local development only)
cargo test --test ffi_comparison --features _instrument-c-mozjpeg-internals
```
