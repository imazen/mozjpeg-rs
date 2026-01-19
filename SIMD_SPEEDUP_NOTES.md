# SIMD Speedup Investigation Notes

Date: 2026-01-19
Branch: simd-speedup-exploration

## Summary

Investigated speeding up sequential mode JPEG encoding using:
1. mozjpeg's AVX2 DCT assembly as reference
2. `vpmaddwd` instruction for multiply-accumulate optimization
3. archmage/simd-compare for safe SIMD patterns

## Key Findings

### Current Performance (DCT, batch of 1000 blocks)

| Implementation | Time (µs) | Throughput (Melem/s) | vs AVX2 |
|----------------|-----------|----------------------|---------|
| scalar (multiversion) | 20.9 | 47.6 | 1.28x slower |
| transpose_i32x8 | 27.0 | 37.0 | 1.66x slower |
| **avx2_intrinsics** | **16.3** | **61.4** | **baseline** |
| avx2_i16_vpmaddwd | 12.2 | 82.1 | **1.34x faster** |

### vpmaddwd Optimization (25-35% Potential Speedup)

The `vpmaddwd` instruction is key to mozjpeg's AVX2 DCT performance:

```nasm
; vpmaddwd computes: result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
; This maps perfectly to DCT patterns like:
;   data2 = tmp13 * FIX_A + tmp12 * FIX_B
;   data6 = tmp13 * FIX_C + tmp12 * FIX_D
; With interleaved data: [tmp13, tmp12, tmp13, tmp12, ...]
; And constants: [FIX_A, FIX_B, FIX_C, FIX_D, ...]
```

**Implementation status:** Shows 25-35% speedup in benchmarks but produces incorrect output.
The bug is in data layout/flow - needs careful debugging.

### mozjpeg Assembly Analysis

mozjpeg's `jfdctint-avx2.asm` uses:
- 16-bit data packed as (row0|row4), (row1|row5), (row2|row6), (row3|row7)
- 16-bit transpose with `vpunpcklwd/hi` (faster than 32-bit)
- `vpmaddwd` for all multiply-accumulate operations
- After transpose: (col1|col0), (col3|col2), (col4|col5), (col6|col7)

This layout enables efficient butterfly operations:
- `tmp1_0 = data1_0 + data6_7` processes two butterflies per instruction

### simd-compare Findings

From `~/work/simd-compare/SIMD-PATTERNS-GUIDE.md`:
- `pulp` is best for safe SIMD (full AVX2, FMA support)
- `wide` limited to 128-bit even with AVX2
- For shuffles/transposes, raw intrinsics are necessary
- FMA can provide 6x speedup for multiply-add chains

### archmage Integration

archmage provides type-safe SIMD tokens but is not published to crates.io yet.
For now, `multiversion` provides adequate dispatch. archmage would add:
- Type-safe capability proofs
- `#[simd_fn]` macro for safe intrinsics use
- Better composability with wide/pulp crates

## Files Modified

- `Cargo.toml`: Added DCT benchmark
- `benches/dct.rs`: Criterion benchmarks for all DCT variants
- `src/dct.rs`: Added `forward_dct_8x8_avx2_i16` (experimental, buggy)

## Next Steps

1. **Fix i16 DCT algorithm:**
   - Trace transpose output to verify column pairing
   - Print intermediate values and compare to scalar
   - The 25-35% speedup is worth the debugging effort

2. **Alternative optimization:**
   - Keep 32-bit DCT but use vpmaddwd for key operations
   - Less complex, may capture 50-75% of the speedup

3. **Entropy encoding:**
   - DCT is now fast (~16µs/1000 blocks with AVX2)
   - Entropy encoding is likely the new bottleneck
   - Consider Huffman table lookups optimization

## Benchmark Command

```bash
cd ~/work/mozjpeg-rs-simd-speedup
RUSTFLAGS="-C target-cpu=native" cargo bench --bench dct -- --noplot
```
