# Context Handoff: C-Compatible Color Conversion Flag

## Task
Add a feature flag or runtime option to use a color conversion algorithm that exactly mirrors C mozjpeg's RGB→YCbCr conversion, eliminating the ±1 rounding differences that cause the 3-5% baseline size gap.

## Background

### Current State
- Default color conversion uses `yuv` crate (fast SIMD: AVX-512/AVX2/SSE/NEON)
- The `yuv` crate has ±1 level rounding differences from C mozjpeg
- This causes 3-5% larger files in **baseline mode only**
- Progressive modes achieve ±0.5% parity (successive approximation masks the differences)

### Verified Size Gaps (10 CID22-512 images)
| Mode | Q75 | Q85 |
|------|-----|-----|
| Baseline (no trellis) | +4.6% | +5.1% |
| Baseline + Trellis | +3.0% | +3.6% |
| Progressive + Trellis | -0.1% | +0.1% |
| MaxCompression | +0.2% | +0.6% |

### Root Cause
C mozjpeg uses specific fixed-point arithmetic for RGB→YCbCr:
```c
// From jccolor.c - uses 16-bit fixed point with specific rounding
#define SCALEBITS  16
#define CBCR_OFFSET  ((INT32) CENTERJSAMPLE << SCALEBITS)
#define ONE_HALF  ((INT32) 1 << (SCALEBITS-1))
#define FIX(x)  ((INT32) ((x) * (1L<<SCALEBITS) + 0.5))

// Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
// Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
// Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE
```

The `yuv` crate uses different fixed-point precision, causing ±1 differences.

## Implementation Plan

### Option 1: Feature Flag (Recommended)
Add `c-compat-color` feature that uses C-compatible conversion:

```toml
# Cargo.toml
[features]
c-compat-color = []  # Use C mozjpeg-compatible color conversion
```

### Option 2: Runtime Flag
Add method to `Encoder`:
```rust
impl Encoder {
    /// Use C mozjpeg-compatible color conversion for exact baseline parity.
    /// Slightly slower than default but produces identical output to C mozjpeg.
    pub fn c_compatible_color(mut self, enable: bool) -> Self {
        self.c_compat_color = enable;
        self
    }
}
```

### Files to Modify

1. **`src/color.rs`** - Add C-compatible conversion function
   - Current: `rgb_to_ycbcr_*` functions use `yuv` crate or hand-written SIMD
   - Add: `rgb_to_ycbcr_c_compat()` matching C mozjpeg exactly

2. **`src/simd/mod.rs`** - Add dispatch for C-compat path
   - `SimdOps` struct has `color_convert_rgb_to_ycbcr` field
   - Add conditional to use C-compat when flag enabled

3. **`src/encode.rs`** - Wire up the flag
   - Add field to `Encoder` struct
   - Pass to `SimdOps::detect()` or select conversion function

4. **`Cargo.toml`** - Add feature flag if using compile-time option

### C mozjpeg Reference Code

From `libjpeg-turbo/jccolor.c`:
```c
// Exact coefficients used by C mozjpeg
rgb_ycc_tab[i + R_Y_OFF] = FIX(0.29900) * i;
rgb_ycc_tab[i + G_Y_OFF] = FIX(0.58700) * i;
rgb_ycc_tab[i + B_Y_OFF] = FIX(0.11400) * i + ONE_HALF;
rgb_ycc_tab[i + R_CB_OFF] = (-FIX(0.16874)) * i;
rgb_ycc_tab[i + G_CB_OFF] = (-FIX(0.33126)) * i;
rgb_ycc_tab[i + B_CB_OFF] = FIX(0.50000) * i + CBCR_OFFSET + ONE_HALF - 1;
rgb_ycc_tab[i + R_CR_OFF] = FIX(0.50000) * i + CBCR_OFFSET + ONE_HALF - 1;
rgb_ycc_tab[i + G_CR_OFF] = (-FIX(0.41869)) * i;
rgb_ycc_tab[i + B_CR_OFF] = (-FIX(0.08131)) * i;
```

Key: `SCALEBITS=16`, `ONE_HALF = 1 << 15`, `CBCR_OFFSET = 128 << 16`

### Scalar Implementation (for reference)
```rust
/// C mozjpeg-compatible RGB to YCbCr conversion.
/// Uses identical fixed-point arithmetic to match output exactly.
pub fn rgb_to_ycbcr_c_compat(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    const SCALE: i32 = 16;
    const ONE_HALF: i32 = 1 << (SCALE - 1);
    const CBCR_OFFSET: i32 = 128 << SCALE;

    // FIX(x) = (x * (1 << 16) + 0.5) as i32
    const FIX_0_29900: i32 = 19595;   // FIX(0.29900)
    const FIX_0_58700: i32 = 38470;   // FIX(0.58700)
    const FIX_0_11400: i32 = 7471;    // FIX(0.11400)
    const FIX_0_16874: i32 = 11059;   // FIX(0.16874)
    const FIX_0_33126: i32 = 21709;   // FIX(0.33126)
    const FIX_0_50000: i32 = 32768;   // FIX(0.50000)
    const FIX_0_41869: i32 = 27439;   // FIX(0.41869)
    const FIX_0_08131: i32 = 5329;    // FIX(0.08131)

    for (i, chunk) in rgb.chunks_exact(3).enumerate() {
        let r = chunk[0] as i32;
        let g = chunk[1] as i32;
        let b = chunk[2] as i32;

        // Y = 0.29900*R + 0.58700*G + 0.11400*B
        y[i] = ((FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALE) as u8;

        // Cb = -0.16874*R - 0.33126*G + 0.50000*B + 128
        cb[i] = (((-FIX_0_16874) * r + (-FIX_0_33126) * g + FIX_0_50000 * b
                  + CBCR_OFFSET + ONE_HALF - 1) >> SCALE) as u8;

        // Cr = 0.50000*R - 0.41869*G - 0.08131*B + 128
        cr[i] = ((FIX_0_50000 * r + (-FIX_0_41869) * g + (-FIX_0_08131) * b
                  + CBCR_OFFSET + ONE_HALF - 1) >> SCALE) as u8;
    }
}
```

Note: The `-1` in Cb/Cr calculation is important for exact match!

### Validation

After implementing, verify with:
```bash
cargo run --release --example quick_baseline_test
```

Expected: Baseline modes should show ±0.5% instead of +3-5%.

Also run FFI comparison tests:
```bash
cargo test --test ffi_validation -- --nocapture
```

### Performance Consideration

The C-compat scalar version will be slower than `yuv` crate SIMD. Options:
1. Accept slower speed for exact parity (likely 2-3x slower for color conversion)
2. Write SIMD version of C-compat algorithm (AVX2/NEON)
3. Make it opt-in only for users who need exact parity

## Files Read This Session

- `/home/lilith/work/mozjpeg-rs/ENTROPY_OPTIMIZATION_NOTES.md` - Documents baseline entropy gap
- `/home/lilith/work/mozjpeg-rs/src/encode.rs` - Main encoder, SIMD entropy paths
- `/home/lilith/work/mozjpeg-rs/src/simd/x86_64/entropy.rs` - SIMD entropy encoder
- `/home/lilith/work/mozjpeg-rs/tests/preset_parity.rs` - Preset comparison tests
- `/home/lilith/work/mozjpeg-rs/examples/cid22_bench.rs` - Benchmark reference

## Current Status

- v0.6.0 just released with imgref support
- CI passing
- All preset parity tests pass with current tolerances
- Baseline gap is documented and understood

## Commands to Start

```bash
# Check current color conversion
grep -n "color_convert\|rgb_to_ycbcr" src/*.rs src/simd/*.rs

# Run baseline comparison after changes
cargo run --release --example quick_baseline_test

# Full test suite
cargo test --release
```
