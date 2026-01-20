//! DCT benchmarks comparing different implementations.

#![allow(deprecated)] // Benchmarks compare deprecated implementations

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mozjpeg_rs::dct::{
    forward_dct_8x8_i32_multiversion, forward_dct_8x8_i32_wide_gather,
    forward_dct_8x8_i32_wide_transpose, forward_dct_u8_i32_multiversion,
};
use mozjpeg_rs::simd::SimdOps;

/// Generate pseudo-random test data
fn generate_test_data() -> [u8; 64] {
    let mut data = [0u8; 64];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 73 + 17) % 256) as u8;
    }
    data
}

fn generate_shifted_data() -> [i16; 64] {
    let mut data = [0i16; 64];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 73 + 17) % 256) as i16 - 128;
    }
    data
}

fn bench_dct_multiversion(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1)); // 1 block

    group.bench_function("multiversion", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i32_multiversion(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

fn bench_dct_wide_gather(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("wide_gather", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i32_wide_gather(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

fn bench_dct_wide_transpose(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("wide_transpose", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i32_wide_transpose(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

fn bench_dct_combined(c: &mut Criterion) {
    let samples = generate_test_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("forward_dct_full", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_u8_i32_multiversion(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_dct_archmage(c: &mut Criterion) {
    use archmage::tokens::x86::Avx2Token;
    use archmage::SimdToken;
    use mozjpeg_rs::dct::avx2_archmage::forward_dct_8x8_i32;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping benchmark");
        return;
    };

    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("archmage_i32", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i32(token, black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_dct_archmage_i16(c: &mut Criterion) {
    use archmage::tokens::x86::Avx2Token;
    use archmage::SimdToken;
    use mozjpeg_rs::dct::avx2_archmage::forward_dct_8x8_i16;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping benchmark");
        return;
    };

    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("archmage_i16", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i16(token, black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

#[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
fn bench_dct_avx2_intrinsics(c: &mut Criterion) {
    use mozjpeg_rs::simd::x86_64::avx2::forward_dct_8x8_i32_avx2_intrinsics;

    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("avx2_intrinsics", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_i32_avx2_intrinsics(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

/// Benchmark SimdOps dispatch methods
fn bench_simdops_dispatch(c: &mut Criterion) {
    let samples = generate_shifted_data();
    let simd = SimdOps::detect();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    // Show which variant is being used
    eprintln!("SimdOps variant: {}", simd.dct_variant_name());

    group.bench_function("simdops_dispatch", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            simd.do_forward_dct(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

/// Benchmark multiple blocks in a batch (more realistic)
fn bench_dct_batch_1000(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT_batch_1000");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("multiversion", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            for _ in 0..1000 {
                forward_dct_8x8_i32_multiversion(black_box(&samples), black_box(&mut coeffs));
            }
            coeffs
        })
    });

    group.bench_function("wide_transpose", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            for _ in 0..1000 {
                forward_dct_8x8_i32_wide_transpose(black_box(&samples), black_box(&mut coeffs));
            }
            coeffs
        })
    });

    // SimdOps dispatch (uses archmage on x86_64 by default)
    let simd = SimdOps::detect();
    group.bench_function("simdops_dispatch", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            for _ in 0..1000 {
                simd.do_forward_dct(black_box(&samples), black_box(&mut coeffs));
            }
            coeffs
        })
    });

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::tokens::x86::Avx2Token;
        use archmage::SimdToken;
        use mozjpeg_rs::dct::avx2_archmage::{forward_dct_8x8_i16, forward_dct_8x8_i32};

        if let Some(token) = Avx2Token::try_new() {
            group.bench_function("archmage_i32", |b| {
                b.iter(|| {
                    let mut coeffs = [0i16; 64];
                    for _ in 0..1000 {
                        forward_dct_8x8_i32(token, black_box(&samples), black_box(&mut coeffs));
                    }
                    coeffs
                })
            });

            group.bench_function("archmage_i16", |b| {
                b.iter(|| {
                    let mut coeffs = [0i16; 64];
                    for _ in 0..1000 {
                        forward_dct_8x8_i16(token, black_box(&samples), black_box(&mut coeffs));
                    }
                    coeffs
                })
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
    {
        use mozjpeg_rs::simd::x86_64::avx2::forward_dct_8x8_i32_avx2_intrinsics;

        group.bench_function("avx2_intrinsics", |b| {
            b.iter(|| {
                let mut coeffs = [0i16; 64];
                for _ in 0..1000 {
                    forward_dct_8x8_i32_avx2_intrinsics(
                        black_box(&samples),
                        black_box(&mut coeffs),
                    );
                }
                coeffs
            })
        });
    }
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_dct_multiversion,
    bench_dct_wide_gather,
    bench_dct_wide_transpose,
    bench_dct_combined,
    bench_dct_archmage,
    bench_dct_archmage_i16,
    bench_simdops_dispatch,
    bench_dct_batch_1000,
);

#[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
criterion_group!(benches_intrinsics, bench_dct_avx2_intrinsics,);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(
    benches,
    bench_dct_multiversion,
    bench_dct_wide_gather,
    bench_dct_wide_transpose,
    bench_dct_combined,
    bench_simdops_dispatch,
    bench_dct_batch_1000,
);

#[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
criterion_main!(benches, benches_intrinsics);

#[cfg(all(target_arch = "x86_64", not(feature = "simd-intrinsics")))]
criterion_main!(benches);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(benches);
