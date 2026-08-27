//! What the aarch64 SIMD DCT path is actually worth.
//!
//! The crate advertises "safe SIMD (archmage) on x86_64 (AVX2) and aarch64
//! (NEON)". This measures the dispatched forward DCT against the scalar
//! reference. Colour conversion is deliberately not benched here — see the
//! note at the bottom of `bench`.
//!
//! Run: `cargo bench --bench neon_tiers`

use criterion::{Criterion, criterion_group, criterion_main};
use mozjpeg_rs::simd::SimdOps;

fn bench(c: &mut Criterion) {
    let ops = SimdOps::detect();
    eprintln!("[neon_tiers] dct variant: {}", ops.dct_variant_name());

    // DCT: per 8x8 block.
    let mut s = 0x9e37_79b9u32;
    let mut samples = [0i16; 64];
    for v in samples.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((s >> 20) as i16 & 0x1FF) - 256;
    }
    let mut coeffs = [0i16; 64];
    c.bench_function("forward_dct_8x8/dispatched", |b| {
        b.iter(|| ops.do_forward_dct(std::hint::black_box(&samples), &mut coeffs))
    });
    c.bench_function("forward_dct_8x8/scalar_ref", |b| {
        b.iter(|| {
            mozjpeg_rs::dct::forward_dct_8x8_i32_multiversion(
                std::hint::black_box(&samples),
                &mut coeffs,
            )
        })
    });

    // Colour conversion is NOT benched here: `SimdOps::color_fn` is private,
    // and on aarch64 the default `fast-yuv` feature routes it through the
    // `zenyuv` crate, which carries its own NEON path (`mod neon_encode` plus
    // a magetypes `neon` tier). So it is already vectorized — an earlier read
    // of `detect()` mistook the scalar fallthrough on line 154 for the active
    // arm, missing that the fast-yuv cfg branch takes precedence.
}

criterion_group!(benches, bench);
criterion_main!(benches);
