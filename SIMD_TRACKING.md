# SIMD Optimization Tracking

This document tracks SIMD optimization progress for mozjpeg-oxide.

## Current Performance Gap

| Component | Rust | C mozjpeg | Gap | Priority |
|-----------|------|-----------|-----|----------|
| Baseline encoding (overall) | 3.45ms | 0.46ms | **7.5x slower** | - |
| DCT | No SIMD | SIMD (SSE2/AVX2) | Main bottleneck | **HIGH** |
| Color conversion | **SIMD (wide)** | SIMD | ~Even | Done |
| Downsampling | Scalar | SIMD | Minor | LOW |
| Quantization | Scalar | Scalar | Even | - |

## Completed SIMD Optimizations

### Color Conversion (`src/color.rs`)

**Status: DONE**

Uses `wide::i32x4` to process 4 pixels at a time.

```rust
// SIMD RGB to YCbCr conversion
let fix_y_r = i32x4::splat(FIX_0_29900);
let fix_y_g = i32x4::splat(FIX_0_58700);
let fix_y_b = i32x4::splat(FIX_0_11400);

let y = (fix_y_r * r + fix_y_g * g + fix_y_b * b + half) >> SCALEBITS;
```

Functions optimized:
- `convert_rgb_to_ycbcr()` - 4 pixels/iteration
- `convert_rgb_to_gray()` - 4 pixels/iteration

## Pending SIMD Optimizations

### Forward DCT (`src/dct.rs`) - HIGH PRIORITY

**Status: NOT STARTED**

The forward DCT is the largest performance bottleneck. C mozjpeg uses SSE2/AVX2 for this.

#### Current Implementation
- Loeffler-Ligtenberg-Moschytz algorithm
- 12 multiplies + 32 adds per 1-D DCT
- 2-D DCT = row DCT + column DCT
- Fully scalar

#### SIMD Strategy Options

**Option 1: Row-parallel SIMD (simpler)**
- Process 4 rows simultaneously using `i32x4`
- Each lane is one row
- Same algorithm, just vectorized across rows
- Good for `wide` crate

```rust
// Conceptual: 4 rows at once
let row0_col0 = i32x4::new([data[0], data[8], data[16], data[24]]);
// ... process 4 rows in parallel
```

**Option 2: Column-parallel SIMD (more complex)**
- Process 8 columns as SIMD vector
- Requires transpose operations
- Better cache utilization
- Needs `i32x8` or two `i32x4`

**Option 3: AAN algorithm**
- Alternative DCT algorithm, more SIMD-friendly
- Used by libjpeg-turbo
- More floating-point operations but better parallelism

#### Recommended Approach
Start with Option 1 (row-parallel) using `wide::i32x4`:
1. Process all 8 rows in 2 batches of 4
2. Keep existing Loeffler algorithm
3. Simpler implementation, easier to verify correctness

### Downsampling (`src/sample.rs`) - LOW PRIORITY

**Status: NOT STARTED**

Simple averaging operations that vectorize well.

```rust
// Current scalar
for y in (0..src_height).step_by(v_factor) {
    for x in (0..src_width).step_by(h_factor) {
        // Average h_factor * v_factor samples
    }
}

// SIMD approach: process 4-8 output samples at once
```

Lower priority because:
- Only used for 4:2:0 and 4:2:2 subsampling
- Smaller fraction of total encoding time
- Simple loop already optimizes reasonably well

## Implementation Guidelines

### Using the `wide` crate

The crate already depends on `wide = "1.1"`. Key types:

```rust
use wide::{i32x4, i32x8, i16x8, i16x16};

// Create vectors
let v = i32x4::splat(42);           // All lanes same value
let v = i32x4::new([1, 2, 3, 4]);   // From array

// Arithmetic (element-wise)
let sum = a + b;
let prod = a * b;
let shifted = v >> 16;

// Comparison and selection
let mask = a.cmp_gt(b);
let result = mask.blend(a, b);      // Select a where true, b where false

// Extract
let arr = v.to_array();
```

### Testing SIMD Code

1. **Correctness first**: Compare output to scalar version
2. **Edge cases**: Test with < 4 pixels (remainder handling)
3. **Benchmark**: Use criterion to measure speedup

```rust
#[test]
fn test_simd_dct_matches_scalar() {
    let samples = [/* test data */];
    let mut coeffs_scalar = [0i16; 64];
    let mut coeffs_simd = [0i16; 64];

    forward_dct_scalar(&samples, &mut coeffs_scalar);
    forward_dct_simd(&samples, &mut coeffs_simd);

    assert_eq!(coeffs_scalar, coeffs_simd);
}
```

### Build Configuration

Feature flag for explicit SIMD control:

```toml
[features]
default = []
simd = []  # Enable explicit SIMD paths (wide always enabled)
```

## Performance Measurement

Use the existing benchmark infrastructure:

```bash
# Run benchmarks
cargo bench --bench encoder_benchmark

# Compare with C mozjpeg
cargo run --example pareto_benchmark --release
```

Key metrics to track:
- Overall encoding time (ms)
- DCT time per block (ns)
- Color conversion time (ms for 1MP image)

## References

- [libjpeg-turbo SIMD](https://github.com/libjpeg-turbo/libjpeg-turbo/tree/main/simd)
- [wide crate docs](https://docs.rs/wide/latest/wide/)
- [Loeffler DCT paper](https://www.researchgate.net/publication/224100807_Practical_Fast_1-D_DCT_Algorithms_with_11_Multiplications)
- [AAN DCT algorithm](https://unix4lyfe.org/dct/)

## Next Steps

1. **Profile current encoder** to confirm DCT is the bottleneck
2. **Implement row-parallel DCT** using `i32x4`
3. **Benchmark** against scalar and C mozjpeg
4. **If needed**, implement column-parallel DCT for further gains
