//! Benchmarks for color conversion

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use mozjpeg_rs::color::{convert_rgb_to_gray, convert_rgb_to_ycbcr, rgb_to_ycbcr};
#[cfg(target_arch = "x86_64")]
use mozjpeg_rs::color_avx2::convert_rgb_to_ycbcr_dispatch;
use yuv::{
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
    rgb_to_yuv444,
};

fn bench_rgb_to_ycbcr_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_scalar");

    // Single pixel
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_pixel", |b| {
        b.iter(|| rgb_to_ycbcr(black_box(128), black_box(64), black_box(192)))
    });

    // 64 pixels (8x8 block)
    group.throughput(Throughput::Elements(64));
    let rgb_block: Vec<u8> = (0..192).map(|i| (i % 256) as u8).collect();
    group.bench_function("8x8_block", |b| {
        b.iter(|| {
            for i in 0..64 {
                let _ = rgb_to_ycbcr(
                    black_box(rgb_block[i * 3]),
                    black_box(rgb_block[i * 3 + 1]),
                    black_box(rgb_block[i * 3 + 2]),
                );
            }
        })
    });

    group.finish();
}

fn bench_convert_rgb_to_ycbcr(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_convert");

    // Various image sizes
    for (name, width, height) in [
        ("64x64", 64, 64),
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1920x1080", 1920, 1080),
    ] {
        let num_pixels = width * height;
        let rgb: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();
        let mut y_out = vec![0u8; num_pixels];
        let mut cb_out = vec![0u8; num_pixels];
        let mut cr_out = vec![0u8; num_pixels];

        group.throughput(Throughput::Elements(num_pixels as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                convert_rgb_to_ycbcr(
                    black_box(&rgb),
                    &mut y_out,
                    &mut cb_out,
                    &mut cr_out,
                    width,
                    height,
                );
            })
        });
    }

    group.finish();
}

fn bench_convert_rgb_to_gray(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_gray");

    for (name, width, height) in [
        ("64x64", 64, 64),
        ("512x512", 512, 512),
        ("1920x1080", 1920, 1080),
    ] {
        let num_pixels = width * height;
        let rgb: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();
        let mut gray_out = vec![0u8; num_pixels];

        group.throughput(Throughput::Elements(num_pixels as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                convert_rgb_to_gray(black_box(&rgb), &mut gray_out, width, height);
            })
        });
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_convert_rgb_to_ycbcr_avx2(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_avx2");

    for (name, width, height) in [
        ("64x64", 64, 64),
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1920x1080", 1920, 1080),
    ] {
        let num_pixels = width * height;
        let rgb: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();
        let mut y_out = vec![0u8; num_pixels];
        let mut cb_out = vec![0u8; num_pixels];
        let mut cr_out = vec![0u8; num_pixels];

        group.throughput(Throughput::Elements(num_pixels as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                convert_rgb_to_ycbcr_dispatch(
                    black_box(&rgb),
                    &mut y_out,
                    &mut cb_out,
                    &mut cr_out,
                    num_pixels,
                );
            })
        });
    }

    group.finish();
}

fn bench_convert_rgb_to_ycbcr_yuv_crate(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_yuv_crate");

    for (name, width, height) in [
        ("64x64", 64u32, 64u32),
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1920x1080", 1920, 1080),
    ] {
        let num_pixels = (width * height) as usize;
        let rgb: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Pre-allocate the YUV image to avoid allocation in the benchmark loop
        let mut yuv_image = YuvPlanarImageMut::alloc(width, height, YuvChromaSubsampling::Yuv444);

        group.throughput(Throughput::Elements(num_pixels as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                rgb_to_yuv444(
                    &mut yuv_image,
                    black_box(&rgb),
                    width * 3,
                    YuvRange::Full,
                    YuvStandardMatrix::Bt601,
                    YuvConversionMode::default(), // Balanced mode
                )
                .expect("yuv conversion failed");
            })
        });
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_rgb_to_ycbcr_scalar,
    bench_convert_rgb_to_ycbcr,
    bench_convert_rgb_to_gray,
    bench_convert_rgb_to_ycbcr_avx2,
    bench_convert_rgb_to_ycbcr_yuv_crate,
);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(
    benches,
    bench_rgb_to_ycbcr_scalar,
    bench_convert_rgb_to_ycbcr,
    bench_convert_rgb_to_gray,
    bench_convert_rgb_to_ycbcr_yuv_crate,
);

criterion_main!(benches);
