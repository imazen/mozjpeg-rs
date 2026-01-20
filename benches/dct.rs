//! DCT benchmarks comparing different implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mozjpeg_rs::dct::{
    forward_dct, forward_dct_8x8, forward_dct_8x8_simd, forward_dct_8x8_transpose,
};

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

fn bench_dct_scalar(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1)); // 1 block

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

fn bench_dct_simd_i32x4(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("simd_i32x4", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_simd(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

fn bench_dct_transpose(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("transpose_i32x8", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_transpose(black_box(&samples), black_box(&mut coeffs));
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
            forward_dct(black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_dct_avx2_intrinsics(c: &mut Criterion) {
    use archmage::SimdToken;
    use archmage::tokens::x86::Avx2Token;
    use mozjpeg_rs::dct::avx2::forward_dct_8x8_avx2;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping benchmark");
        return;
    };

    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("avx2_intrinsics", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_avx2(token, black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_dct_avx2_i16(c: &mut Criterion) {
    use archmage::SimdToken;
    use archmage::tokens::x86::Avx2Token;
    use mozjpeg_rs::dct::avx2::forward_dct_8x8_avx2_i16;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping benchmark");
        return;
    };

    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT");
    group.throughput(Throughput::Elements(1));

    group.bench_function("avx2_i16_vpmaddwd", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            forward_dct_8x8_avx2_i16(token, black_box(&samples), black_box(&mut coeffs));
            coeffs
        })
    });
}

/// Benchmark multiple blocks in a batch (more realistic)
fn bench_dct_batch_1000(c: &mut Criterion) {
    let samples = generate_shifted_data();

    let mut group = c.benchmark_group("DCT_batch_1000");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            for _ in 0..1000 {
                forward_dct_8x8(black_box(&samples), black_box(&mut coeffs));
            }
            coeffs
        })
    });

    group.bench_function("transpose_i32x8", |b| {
        b.iter(|| {
            let mut coeffs = [0i16; 64];
            for _ in 0..1000 {
                forward_dct_8x8_transpose(black_box(&samples), black_box(&mut coeffs));
            }
            coeffs
        })
    });

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        use archmage::tokens::x86::Avx2Token;
        use mozjpeg_rs::dct::avx2::{forward_dct_8x8_avx2, forward_dct_8x8_avx2_i16};

        if let Some(token) = Avx2Token::try_new() {
            group.bench_function("avx2_intrinsics", |b| {
                b.iter(|| {
                    let mut coeffs = [0i16; 64];
                    for _ in 0..1000 {
                        forward_dct_8x8_avx2(token, black_box(&samples), black_box(&mut coeffs));
                    }
                    coeffs
                })
            });

            group.bench_function("avx2_i16_vpmaddwd", |b| {
                b.iter(|| {
                    let mut coeffs = [0i16; 64];
                    for _ in 0..1000 {
                        forward_dct_8x8_avx2_i16(token, black_box(&samples), black_box(&mut coeffs));
                    }
                    coeffs
                })
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_dct_scalar,
    bench_dct_simd_i32x4,
    bench_dct_transpose,
    bench_dct_combined,
    bench_dct_avx2_intrinsics,
    bench_dct_avx2_i16,
    bench_dct_batch_1000,
);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(
    benches,
    bench_dct_scalar,
    bench_dct_simd_i32x4,
    bench_dct_transpose,
    bench_dct_combined,
    bench_dct_batch_1000,
);

criterion_main!(benches);
