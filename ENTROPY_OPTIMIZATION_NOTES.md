# Entropy Encoding Optimization Notes

## Goal
Speed up baseline JPEG entropy encoding, which is currently ~4.7x slower than C mozjpeg.

## Current State (Before Optimization)
- Entropy encoding is 59% of baseline encoding time
- C mozjpeg uses SSE2/AVX2 intrinsics for entropy encoding
- Rust encoder uses standard iteration with Write trait

## Approaches Tried

### 1. jpegli-rs Style BitWriter (FAILED)
**Hypothesis**: Write trait overhead and bit buffer structure are bottlenecks.

**Changes**:
- Owned `Vec<u8>` instead of `Write` trait
- 64-bit buffer with flush at 32+ bits
- SWAR 0xFF detection
- `#[cold]` annotation on flush path

**Results** (rough, noisy benchmark):
| Config | Standard | Fast | Speedup |
|--------|----------|------|---------|
| Q50 sparse | 0.14ms | 0.14ms | 1.0x |
| Q75 medium | 0.22ms | 0.24ms | 0.9x |

**Conclusion**: Write trait overhead is NOT the bottleneck. The standard BitWriter
is already efficient. This approach added code complexity without benefit.

### 2. tzcnt-based Zero Run Skipping (PARTIAL SUCCESS)
**Hypothesis**: Iterating through all 63 AC coefficients is slow; jumping to
non-zero positions via tzcnt would be faster.

**Changes**:
- Build 64-bit mask of non-zero coefficients in zigzag order
- Use `trailing_zeros()` (compiles to tzcnt) to find next non-zero
- Calculate run length from position difference

**Results**:
| Config | Standard | Fast | Speedup |
|--------|----------|------|---------|
| Q50 sparse | 0.15ms | 0.10ms | **1.5x** |
| Q75 medium | 0.23ms | 0.23ms | 1.0x |
| Q90 dense | 0.29ms | 0.33ms | 0.88x |

**Conclusion**: tzcnt helps for SPARSE blocks but HURTS dense blocks.
The zigzag mask building overhead (~14ns/block) dominates for dense blocks.

### 3. Hybrid Sparse/Dense Approach (MARGINAL)
**Hypothesis**: Use popcount to detect density, choose algorithm accordingly.

**Changes**:
- Build zigzag mask
- If popcount < 20, use tzcnt path
- Otherwise, use linear iteration

**Results**: Similar to tzcnt-only. The mask building overhead still hurts.

### 4. Lookup Table for jpeg_nbits (MARGINAL)
**Hypothesis**: `leading_zeros()` calls are slow.

**Changes**:
- 256-entry lookup table for values 0-255
- Fallback to leading_zeros for larger values

**Results**: Minimal improvement. `leading_zeros()` is already single-cycle on modern CPUs.

## Key Insights

1. **The Write trait is NOT the bottleneck** - jpegli-rs approach didn't help
2. **Bit buffer operations are already efficient** - standard 64-bit buffer is fine
3. **Sparse blocks benefit from tzcnt** - but most real images have mixed density
4. **Mask building has overhead** - ~14ns/block for zigzag-order mask
5. **The standard encoder is already well-optimized** - simple linear iteration
   with SIMD early-exit for all-zero blocks

## What C mozjpeg Does Differently

From `jchuff-sse2.asm`:
1. **Reorders coefficients into temp array** first (zigzag order)
2. **Builds zero-mask during reorder** (amortizes cost)
3. **Uses 64KB lookup table** for jpeg_nbits (trades memory for speed)
4. **Speculative writes** - writes 8 bytes, fixes up if 0xFF found
5. **Tight assembly loop** with register allocation optimized

The key insight: mozjpeg's approach works because the coefficient reordering
and mask building are fused into a single SIMD pass over the data.

## Next Steps to Try

1. **Fused zigzag reorder + mask build** - do both in one SIMD pass
2. **Larger nbits lookup table** - match mozjpeg's 64KB table
3. **Profile the actual hot path** - use `perf` or `flamegraph` to identify real bottleneck
4. **Compare against C mozjpeg's non-SIMD path** - see if we're competitive there

## Criterion Benchmark Results (Accurate)

Configuration: 4096 blocks, varying density

| Density | Standard (µs) | Fast (µs) | Ratio |
|---------|---------------|-----------|-------|
| 10% sparse | 199 | 198 | 1.01x (tie) |
| 20% sparse | 211 | 230 | **0.92x** |
| 40% medium | 301 | 342 | **0.88x** |
| 60% dense | 382 | 434 | **0.88x** |
| 80% dense | 458 | 525 | **0.87x** |
| 2k image (40%) | 4.65 ms | 5.40 ms | **0.86x** |

**The "fast" encoder is 12-14% SLOWER than standard!**

### Why jpegli-rs Approach Didn't Help

1. **Write trait is NOT the bottleneck** - VecBitWriter is already efficient
2. **Bit buffer ops already optimized** - 64-bit buffer with good flush logic
3. **Missing SIMD early-exit** - Fast encoder doesn't check for all-zero blocks
4. **Extra function call overhead** - Separate encode_ac_linear adds indirection

### What the Standard Encoder Does Right

1. **SIMD mask for early all-zero AC detection** - Huge win for sparse blocks
2. **Inline everything** - No function call overhead in hot path
3. **Simple linear iteration** - Branch predictor-friendly
4. **Combined code+extra writes** - Already has `put_bits_combined`

## Critical Finding: Synthetic vs Real Data

**Synthetic benchmark results were MISLEADING!**

| Test Type | ns/block | Notes |
|-----------|----------|-------|
| Synthetic (Criterion) | 49-73 | Uniform coefficient distribution |
| Real image (timing_breakdown) | **220** | After DCT+quantization |

The 3x difference is because:
1. Real DCT output has natural sparsity patterns
2. Coefficient magnitudes follow power-law distribution
3. Non-zero clustering in low frequencies
4. More complex run-length encoding patterns

**The fast encoder approaches didn't work because they were tested on unrealistic data.**

## Revised Benchmark Requirements

1. Use REAL image data through DCT+quantization pipeline
2. Test multiple images with varying content
3. Compare against C mozjpeg's actual entropy encoding time

## Benchmark Configuration

Using Criterion for reliable measurements on busy system:
- Warm-up: 2-3 seconds
- Measurement: 5-8 seconds
- Sample size: 50-100
- Confidence interval: 95%
- **MUST use real DCT/quant pipeline data**

## FINAL RESULTS: Real Image Benchmark (2026-01-19)

Using real PNG image (`tests/images/1.png`, 512x512) through DCT+quantization pipeline.

### Real Image Data (4096 blocks)

| Quality | Standard (µs) | Fast (µs) | Ratio |
|---------|---------------|-----------|-------|
| Q50 | 742 | 869 | **0.85x** (17% slower) |
| Q75 | 877 | 1057 | **0.83x** (20% slower) |
| Q85 | 976 | 1189 | **0.82x** (22% slower) |
| Q95 | 1178 | 1432 | **0.82x** (22% slower) |

### Synthetic Image Data (for comparison)

| Quality | Standard (µs) | Fast (µs) | Ratio |
|---------|---------------|-----------|-------|
| Q50 | 500 | 549 | **0.91x** (10% slower) |
| Q85 | 734 | 732 | **1.00x** (tie) |

### Key Observations

1. **Real image data is harder to encode** - Standard encoder takes 742µs on real
   vs 500µs on synthetic at Q50 (48% more time). This is because real images have
   more complex coefficient distributions.

2. **Fast encoder is consistently slower on real images** - 17-22% slower across
   all quality levels.

3. **Synthetic data was misleading** - At Q85 synthetic, the fast encoder ties.
   But on real Q85 data, fast encoder is 22% slower.

4. **ns/block comparison**:
   - Standard: 181-288 ns/block on real data
   - Fast: 212-350 ns/block on real data

## CONCLUSION

**The jpegli-rs approach and tzcnt-based optimizations have FAILED.**

The "fast" entropy encoder is 17-22% SLOWER than the standard encoder on real
image data. Both approaches:

1. **jpegli-rs BitWriter style** - Write trait is not the bottleneck
2. **tzcnt zero-run skipping** - Mask building overhead dominates

The standard encoder is already well-optimized:
- SIMD check for all-zero AC blocks (huge win for sparse blocks)
- Everything inlined (no function call overhead)
- Simple linear iteration (branch predictor friendly)
- Combined code+extra bit writes

## Potential Next Steps (if pursuing further)

1. **Fused zigzag reorder + mask build in single SIMD pass** - This is what C mozjpeg does
2. **Speculative 8-byte writes with 0xFF fixup** - Avoids per-byte checking
3. **Direct port of jchuff-sse2.asm** - Match C mozjpeg's exact approach
4. **Profile with `perf`** - Identify actual hot spots in the encoding loop

However, given that:
- Trellis mode (where most encoding time is spent) already beats C mozjpeg by 10%
- Baseline mode gap is 4.7x, entropy is only part of that
- Further optimization may have diminishing returns

It may be more productive to focus on other bottlenecks (color conversion, DCT)
or accept that the current entropy encoder is "good enough" for production use.

## PERF Analysis: Rust vs C mozjpeg (2026-01-19)

### Baseline Mode (no trellis) - 512x512 image

| Component | Rust Cycles | C Cycles | Slowdown |
|-----------|-------------|----------|----------|
| **Total** | 6.1B | 1.25B | **4.9x** |
| Color conversion | 1.7B (28%) | 0.17B (14%) | **10x** |
| Entropy encoding | 2.2B (36%) | 0.46B (37%) | **4.8x** |
| DCT | 0.24B (4%) | 0.11B (9%) | **2.2x** |

### Key Finding: Color Conversion Bug

The encoder uses `simd/scalar.rs` for color conversion, NOT the `yuv` crate!
The `fast-yuv` feature only affects `color.rs` which isn't used by the encoder.
**This alone accounts for 10x of the slowdown.**

### C mozjpeg SIMD Functions (from perf)

| Function | % Time | Description |
|----------|--------|-------------|
| `jsimd_huff_encode_one_block_sse2` | 37% | SIMD entropy encoding |
| `jsimd_rgb_ycc_convert_avx2` | 14% | AVX2 color conversion |
| `jsimd_fdct_islow_avx2` | 9% | AVX2 DCT |
| `jsimd_quantize_avx2` | 6% | AVX2 quantization |
| `jsimd_convsamp_avx2` | 7% | AVX2 sample conversion |

### jchuff-sse2.asm Key Techniques

From `/home/lilith/work/jpegli-rs/internal/jpegli-cpp/third_party/libjpeg-turbo/simd/x86_64/jchuff-sse2.asm`:

1. **Fused zigzag reorder + sign handling**
   - SSE2 shuffles (punpckldq, pshuflw, pinsrw) for zigzag
   - pcmpgtw + paddw for sign handling in same pass

2. **64KB lookup table for nbits**
   - Direct mapping: value → bit count
   - Avoids leading_zeros() at runtime

3. **Speculative 8-byte writes with SIMD 0xFF detection**
   - Write 8 bytes first (optimistic)
   - pcmpeqb + pmovmskb to find 0xFF bytes
   - Only fixup when stuffing needed

4. **tzcnt for zero-run detection**
   - Uses trailing zero count to skip zeros efficiently

### Priority Fixes

1. ~~**Fix color conversion (10x potential)** - Make encoder use yuv crate~~ ✅ DONE
2. **Port jchuff-sse2 (5.6x potential)** - SIMD entropy encoding
3. ~~**Improve DCT (2.2x potential)** - Better AVX2 intrinsics~~ ✅ DONE (1.6x remaining)

## PERF Analysis: After Color + DCT Fixes (2026-01-19)

### Baseline Mode (no trellis) - 512x512 image, with fast-yuv + simd-intrinsics

| Component | Rust Cycles | C Cycles | Slowdown |
|-----------|-------------|----------|----------|
| **Total** | 4.46B | 1.72B | **2.59x** (was 4.9x) |
| Color conversion | 0.14B (3%) | 0.22B (13%) | **Rust 1.6x faster!** |
| Entropy encoding | 3.57B (80%) | 0.64B (37%) | **5.6x** |
| DCT | 0.24B (5%) | 0.15B (9%) | **1.6x** |

### What Was Fixed

1. **Color conversion now uses yuv crate** - SimdOps::detect() routes to yuv crate's
   AVX2/SSE/NEON implementation when `fast-yuv` feature is enabled (default).
   Result: Rust is now 1.6x FASTER than C mozjpeg for color conversion!

2. **DCT uses hand-written AVX2 intrinsics** - With `simd-intrinsics` feature,
   uses `simd::x86_64::avx2::forward_dct_8x8` instead of multiversion scalar.
   Result: DCT improved from 2.2x to 1.6x slower.

### Remaining Bottleneck: Entropy Encoding

The 2.59x slowdown is now **80% entropy encoding**. The jchuff-sse2 port is the
only remaining optimization with significant impact potential (5.6x improvement).

### Key Insight: fast-yuv vs simd-intrinsics

- `fast-yuv` (default): Uses yuv crate for color conversion - essential, 10x speedup
- `simd-intrinsics` (optional): Uses hand-written DCT intrinsics - 3% overall improvement

The `simd-intrinsics` feature is NOT in defaults because the 3% improvement is marginal
and the multiversion autovectorization is almost as good (93% of intrinsics perf).

## SIMD Entropy Encoder Port (2026-01-19)

### Implementation

Ported key techniques from jchuff-sse2.asm to Rust:

1. **Fused zigzag reorder + sign handling** - SSE2 shuffles for zigzag order,
   pcmpgtw + paddw for sign handling in same pass
2. **64KB nbits lookup table** - Direct mapping: value → bit count
3. **64-bit non-zero mask** - Built during zigzag reorder
4. **tzcnt-based iteration** - `trailing_zeros()` to skip zeros efficiently

### Benchmark Results (Real Image Data, 512x512)

| Quality | Standard (µs) | SIMD (µs) | Speedup |
|---------|---------------|-----------|---------|
| Q50 | 775.81 | 330.10 | **2.35x** |
| Q75 | ~880 | ~370 | **2.38x** |
| Q85 | ~980 | ~420 | **2.33x** |
| Q95 | ~1180 | ~550 | **2.15x** |

### Integration

- Safe wrapper `encode_block()` calls unsafe SSE2 intrinsics internally
- `cfg(target_arch = "x86_64")` with fallback to standard encoder
- All tests pass, produces valid JPEG files
