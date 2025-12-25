# SIMD Optimization Tracking

This document tracks SIMD optimization progress for mozjpeg-oxide.

## Current Performance Gap (Dec 2024)

**2048x2048 image, 30 iterations, with AVX2** (`RUSTFLAGS="-C target-feature=+avx2"`):

| Mode | Rust | C mozjpeg | Ratio | Notes |
|------|------|-----------|-------|-------|
| Baseline | 40.04 ms | 8.46 ms | **4.74x slower** | Entropy encoding bottleneck |
| Trellis | 162.88 ms | 181.07 ms | **0.90x (10% faster!)** | Rust wins with trellis |

**512x512 image (less accurate due to system noise)**:

| Mode | Rust | C mozjpeg | Ratio | Notes |
|------|------|-----------|-------|-------|
| Baseline | 1.73 ms | 0.45 ms | 3.9x slower | DCT + color optimized |
| Trellis | 11.13 ms | 11.72 ms | 0.95x (faster!) | |

**Key insight**: Larger images give more accurate measurements. The 4.74x gap is the true
baseline performance - entropy encoding dominates at scale.

## Profiling Results (512x512 image)

### Baseline Mode Breakdown (with AVX2)

| Stage | Time (µs) | % of Total | Priority |
|-------|-----------|------------|----------|
| Entropy encoding | 1498 | **58.8%** | **HIGH** |
| Color conversion | 353 | 13.8% | DONE (i32x8) |
| Quantization | 317 | 12.4% | LOW |
| Forward DCT | 222 | 8.7% | DONE (AVX2) |
| Downsampling | 128 | 5.0% | LOW |
| MCU expansion | 31 | 1.2% | - |

**Key Finding:** Entropy encoding is the main remaining bottleneck at 59% of baseline time.
DCT and color conversion have been optimized.

### Trellis Mode Breakdown

| Stage | Time (µs) | % of Total |
|-------|-----------|------------|
| Trellis quantization | 6621 | 77.2% |
| Prep (color+down) | 1046 | 12.2% |
| Entropy encoding | 656 | 7.7% |
| Forward DCT | 251 | 2.9% |

**Key Finding:** Trellis dominates as expected. No optimization needed (already at parity).

## Optimization Priorities

Based on profiling, the optimization priority order for baseline mode:

### 1. Entropy Encoding (49.2% of time) - **HIGH PRIORITY**

Current issues:
- `put_bits` called for every code and value (many function calls)
- `BitWriter` writes through `Write` trait (potential virtual dispatch)
- `emit_byte_stuffed` writes 1 byte at a time when 0xFF detected
- `flush_buffer` checks for 0xFF in every 8-byte chunk

Potential optimizations:
- Buffer multiple codes before writing (reduce function calls)
- Direct Vec<u8> access instead of Write trait
- SIMD-accelerated 0xFF byte detection and stuffing
- Inline `put_bits` more aggressively

Reference: libjpeg-turbo uses optimized assembly for entropy encoding.

### 2. Color Conversion (13.8% of time) - **DONE**

Updated to use `i32x8` (8 pixels at a time) for AVX2 width.
Further optimization would require AVX2 intrinsics for RGB deinterleaving,
but gains would be limited since gather/scatter is the bottleneck.

### 3. Quantization (10.1% of time) - **LOW PRIORITY**

Simple division/rounding. Already fast.

### 4. Forward DCT (8.2% of time) - **DONE**

AVX2 intrinsics implemented. 60ns per block.

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

### AVX2 Intrinsics DCT (`src/dct.rs::avx2`)

**Status: DONE** (Dec 2024)

Uses `core::arch::x86_64` intrinsics directly for maximum performance.

Key optimizations over `wide`-based version:
- `_mm256_cvtepi16_epi32` for proper load+sign-extend (no `vpinsrw` gather)
- Shuffle-based transpose using `_mm256_unpacklo/hi_epi32/64` + `_mm256_permute2x128_si256`
- Proper `_mm256_srai_epi32` with const generic shift amounts
- `_mm_packs_epi32` for efficient i32→i16 packing

**Benchmark Results (with `-C target-cpu=native`):**
| Implementation | Time | Throughput | vs Scalar |
|----------------|------|------------|-----------|
| **AVX2 intrinsics** | **40.1 ns** | **1.59 Gelem/s** | **1.19x faster** |
| Scalar | 47.5 ns | 1.35 Gelem/s | 1.00x |
| SIMD transpose (wide) | 50.3 ns | 1.27 Gelem/s | 0.95x (slower!) |
| SIMD gather (wide) | 56.1 ns | 1.14 Gelem/s | 0.85x (slower!) |

**Key Finding:** The `wide` crate is slower than optimized scalar when the compiler
can auto-vectorize with `-C target-cpu=native`. Explicit `core::arch` intrinsics
are required for actual speedup.

### Color Conversion (`src/color.rs`)

**Status: DONE (Dec 2024)**

Uses `wide::i32x8` to process 8 pixels at a time (AVX2 width).

```rust
// SIMD RGB to YCbCr conversion (8 pixels at a time)
let fix_y_r = i32x8::splat(FIX_0_29900);
let fix_y_g = i32x8::splat(FIX_0_58700);
let fix_y_b = i32x8::splat(FIX_0_11400);

let y = (fix_y_r * r + fix_y_g * g + fix_y_b * b + half) >> SCALEBITS;
```

Functions optimized:
- `convert_rgb_to_ycbcr()` - 8 pixels/iteration (was 4)
- `convert_rgb_to_gray()` - 8 pixels/iteration (was 4)

**Benchmark Results:**
| Image Size | Before (i32x4) | After (i32x8) | Improvement |
|------------|----------------|---------------|-------------|
| 512x512 | 375 µs | 353 µs | **~6% faster** |

**Why only 6% improvement?**
- The gather operation (extracting R, G, B from interleaved format) dominates
- `i32x8::new([...])` still uses element-by-element construction
- True AVX2 gather (`_mm256_i32gather_epi32`) might help but interleaved
  RGB format is inherently unfriendly to SIMD

## Failed Optimizations (What Didn't Work)

### FastEntropyEncoder (Dec 2024) - REVERTED

**Hypothesis:** Streaming bitwriter with precomputed code tables would reduce
per-symbol overhead and improve entropy encoding performance.

**What we tried:**
```rust
// Direct Vec<u8> access instead of BitWriter trait object
struct FastEntropyEncoder {
    output: Vec<u8>,
    bit_buffer: u64,
    bits_in_buffer: u32,
}

// Precomputed code/size tables per block
let mut code_buf: [u16; 64] = [0; 64];
let mut size_buf: [u8; 64] = [0; 64];
```

**Results:**
| Mode | Before | After | Change |
|------|--------|-------|--------|
| Full encoder | 0.95 ms | 2.45 ms | **2.6x SLOWER** |
| Isolated entropy | N/A | N/A | 1.38x faster |

**Why it failed:**
- Micro-optimization hurt macro-performance due to code locality issues
- The "fast" path bloated the encoder binary, hurting instruction cache
- Extra complexity (precomputed tables) added overhead that offset gains
- Isolated benchmarks don't capture instruction cache effects

**Lesson learned:** Always benchmark the full pipeline, not just isolated
functions. Micro-optimizations can destroy macro-performance.

## SIMD Ecosystem in Rust (2025)

Based on research from [zune-image](https://github.com/etemesi254/zune-image),
[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo), and
[Nine Rules for SIMD](https://bardai.ai/2025/02/27/nine-rules-for-simd-acceleration-of-your-rust-code-part-1/):

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| `core::simd` (nightly) | Portable, future standard | Nightly only | Good for experimentation |
| `core::arch` intrinsics | Maximum performance | Arch-specific, unsafe | Best for production perf |
| `wide` crate | Stable, portable | Slower than autovectorized scalar | Avoid for perf-critical code |
| `pulp` crate | Runtime multiversioning | Extra dependency | Good for portable binaries |
| Autovectorization | Zero effort | Unpredictable, often fails for floats | Default fallback |

**Key Lessons:**
1. `wide` crate generates `vpinsrw` (scalar insert) instead of proper SIMD loads
2. With `-C target-cpu=native`, scalar code autovectorizes and beats `wide`
3. For real speedup, use `core::arch` intrinsics with proper load/widen patterns
4. Use `_mm256_cvtepi16_epi32` for i16→i32 widening (not element-by-element)
5. Use shuffle-based transpose (unpack + permute) not array extraction

## Future Optimizations

### Platform-Specific Intrinsics

**Status: PARTIALLY DONE** (AVX2 DCT implemented)

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
