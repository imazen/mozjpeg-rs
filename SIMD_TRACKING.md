# SIMD Optimization Tracking

This document tracks SIMD optimization progress for mozjpeg-oxide.

## Current Performance Gap

| Component | Rust | C mozjpeg | Gap | Priority |
|-----------|------|-----------|-----|----------|
| Baseline encoding (overall) | 2.28ms | 0.46ms | **5.0x slower** | - |
| DCT | **SIMD (wide)** | SIMD (SSE2/AVX2) | Still 5x gap | MEDIUM |
| Color conversion | **SIMD (wide)** | SIMD | ~Even | Done |
| Downsampling | Scalar | SIMD | Minor | LOW |
| Quantization | Scalar | Scalar | Even | - |

## Completed SIMD Optimizations

### Forward DCT (`src/dct.rs`)

**Status: DONE** (Dec 2024)

Three implementations available:
1. `forward_dct_8x8` - Scalar reference (kept for correctness testing)
2. `forward_dct_8x8_simd` - Gather-based SIMD with `i32x4`
3. `forward_dct_8x8_transpose` - **Transpose-based SIMD with `i32x8`** (production)

The transpose-based approach avoids expensive gather/scatter by:
1. Loading rows contiguously (no gather)
2. Transposing to enable row-parallel processing
3. Using 8-wide SIMD (`i32x8`) for full row processing

**Optimizations applied:**
- Pre-computed SIMD constants (`const` - no per-call allocation)
- Pre-negated constants (no runtime negation)
- Inlined 1D DCT helper (`#[inline(always)]`)
- Contiguous memory loads (no gather operations)

**Benchmark Results (single 8x8 block):**
| Implementation | Time | Throughput | vs Scalar |
|----------------|------|------------|-----------|
| Scalar | 81 ns | 786 Melem/s | 1.00x |
| SIMD (gather) | 67 ns | 955 Melem/s | 1.21x |
| **SIMD (transpose)** | **59 ns** | **1078 Melem/s** | **1.37x** |

**Encoder Impact (512x512 image):**
| Mode | Before SIMD | After SIMD | Improvement |
|------|-------------|------------|-------------|
| Baseline | 2.47ms | 2.28ms | **8.3% faster** |
| Trellis | 12.02ms | 11.57ms | **3.8% faster** |

The DCT SIMD reduced the gap vs C mozjpeg from 5.7x to 5.0x slower.

**Why only 1.37x speedup instead of 4x?**
- `wide` crate uses portable SIMD, not native intrinsics
- Transpose overhead (3 transposes per block, done via scalar)
- C mozjpeg uses true SSE2/AVX2 with shuffle-based transpose

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

## Future Optimizations

### Platform-Specific Intrinsics

**Status: NOT STARTED**

To close the remaining 5x gap, platform-specific SIMD is needed:

**Option 1: `std::simd` (nightly)**
- Rust's experimental portable SIMD
- Better codegen than `wide`
- Requires nightly compiler

**Option 2: `core::arch` intrinsics**
- Direct SSE2/AVX2/NEON intrinsics
- Maximum performance, matches C
- Architecture-specific code paths
- More complex maintenance

**Option 3: `pulp` crate**
- Higher-level abstraction over intrinsics
- Runtime CPU feature detection
- Good balance of performance and portability

### Downsampling (`src/sample.rs`) - LOW PRIORITY

**Status: NOT STARTED**

Simple averaging operations. Lower priority because:
- Only used for 4:2:0 and 4:2:2 subsampling
- Smaller fraction of total encoding time
- Simple loop already optimizes reasonably well

## Implementation Guidelines

### Using the `wide` crate

The crate depends on `wide = "1.1"`. Key types:

```rust
use wide::i32x4;

// Create vectors
let v = i32x4::splat(42);           // All lanes same value
let v = i32x4::new([1, 2, 3, 4]);   // From array

// Arithmetic (element-wise)
let sum = a + b;
let prod = a * b;
let shifted = v >> 16;

// Extract
let arr = v.to_array();
```

### Testing SIMD Code

1. **Correctness first**: Compare output to scalar version
2. **Edge cases**: Test with various input patterns
3. **Benchmark**: Use criterion to measure speedup

```rust
#[test]
fn test_simd_dct_matches_scalar() {
    let samples = [/* test data */];
    let mut coeffs_scalar = [0i16; 64];
    let mut coeffs_simd = [0i16; 64];

    forward_dct_8x8(&samples, &mut coeffs_scalar);
    forward_dct_8x8_simd(&samples, &mut coeffs_simd);

    assert_eq!(coeffs_scalar, coeffs_simd);
}
```

## Performance Measurement

```bash
# Run DCT microbenchmark
cargo bench --bench encode -- dct

# Run full encoder benchmark vs C
cargo bench --bench encode -- rust_vs_c
```

Key metrics:
- DCT time per block (ns) - currently 58.5ns SIMD vs 80ns scalar
- Overall encoding time (ms) - currently 2.28ms vs C's 0.46ms
- Throughput (Melem/s) - currently 1094 SIMD vs 800 scalar

## References

- [libjpeg-turbo SIMD](https://github.com/libjpeg-turbo/libjpeg-turbo/tree/main/simd)
- [wide crate docs](https://docs.rs/wide/latest/wide/)
- [Loeffler DCT paper](https://www.researchgate.net/publication/224100807_Practical_Fast_1-D_DCT_Algorithms_with_11_Multiplications)
