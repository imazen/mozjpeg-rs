//! DCT profiling example - run with perf to analyze hotspots.
//!
//! Usage:
//!   cargo build --release --example profile_dct
//!   perf record -g ./target/release/examples/profile_dct
//!   perf report
//!
//! Or for detailed assembly:
//!   perf annotate -s forward_dct_8x8

#![allow(deprecated)] // Profiling deprecated implementations for comparison

use mozjpeg_rs::dct::{
    forward_dct_8x8_i32_multiversion, forward_dct_8x8_i32_wide_gather,
    forward_dct_8x8_i32_wide_transpose,
};
use std::hint::black_box;
use std::time::Instant;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use mozjpeg_rs::dct::avx2_archmage::{forward_dct_8x8_i16, forward_dct_8x8_i32};

const ITERATIONS: usize = 10_000_000;

fn generate_test_data() -> [i16; 64] {
    let mut data = [0i16; 64];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 73 + 17) % 256) as i16 - 128;
    }
    data
}

fn main() {
    let samples = generate_test_data();
    let mut coeffs = [0i16; 64];

    println!("DCT Profiling - {} iterations each\n", ITERATIONS);

    // Warm up
    for _ in 0..1000 {
        forward_dct_8x8_i32_multiversion(black_box(&samples), black_box(&mut coeffs));
    }

    // Profile multiversion (auto-vectorized)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        forward_dct_8x8_i32_multiversion(black_box(&samples), black_box(&mut coeffs));
    }
    let scalar_time = start.elapsed();
    println!(
        "multiversion:{:>8.2} ms ({:.1} ns/block)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_time.as_nanos() as f64 / ITERATIONS as f64
    );

    // Profile wide gather i32x4
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        forward_dct_8x8_i32_wide_gather(black_box(&samples), black_box(&mut coeffs));
    }
    let simd_time = start.elapsed();
    println!(
        "wide_gather: {:>8.2} ms ({:.1} ns/block)",
        simd_time.as_secs_f64() * 1000.0,
        simd_time.as_nanos() as f64 / ITERATIONS as f64
    );

    // Profile wide transpose i32x8
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        forward_dct_8x8_i32_wide_transpose(black_box(&samples), black_box(&mut coeffs));
    }
    let transpose_time = start.elapsed();
    println!(
        "wide_transp: {:>8.2} ms ({:.1} ns/block)",
        transpose_time.as_secs_f64() * 1000.0,
        transpose_time.as_nanos() as f64 / ITERATIONS as f64
    );

    // Profile AVX2 archmage intrinsics
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use archmage::{SimdToken, X64V3Token};

        if let Some(token) = X64V3Token::try_new() {
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                forward_dct_8x8_i32(token, black_box(&samples), black_box(&mut coeffs));
            }
            let avx2_time = start.elapsed();
            println!(
                "archmage_i32:{:>8.2} ms ({:.1} ns/block)",
                avx2_time.as_secs_f64() * 1000.0,
                avx2_time.as_nanos() as f64 / ITERATIONS as f64
            );

            // Profile AVX2 i16 (experimental - may produce wrong results)
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                forward_dct_8x8_i16(token, black_box(&samples), black_box(&mut coeffs));
            }
            let avx2_i16_time = start.elapsed();
            println!(
                "archmage_i16:{:>8.2} ms ({:.1} ns/block) [experimental]",
                avx2_i16_time.as_secs_f64() * 1000.0,
                avx2_i16_time.as_nanos() as f64 / ITERATIONS as f64
            );
        } else {
            println!("AVX2 not available for archmage tests");
        }
    }

    println!("\nDone. Use 'perf report' to analyze.");
}
