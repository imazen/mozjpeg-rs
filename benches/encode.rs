//! Encoding benchmarks for mozjpeg-oxide using criterion.
//!
//! Compares Rust encoder performance across configurations and against C mozjpeg.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mozjpeg_rs::test_encoder::{encode_rust, TestEncoderConfig};
use mozjpeg_rs::{Encoder, Subsampling};
use mozjpeg_sys::*;
use std::ptr;

/// Create a synthetic test image with gradient and noise.
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let noise = ((x * 7 + y * 13) % 50) as u8;
            rgb[idx] = ((x * 255 / width) as u8).saturating_add(noise);
            rgb[idx + 1] = ((y * 255 / height) as u8).saturating_add(noise);
            rgb[idx + 2] = (((x + y) * 255 / (width + height)) as u8).saturating_add(noise);
        }
    }
    rgb
}

/// Encode with C mozjpeg FFI.
fn encode_c(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_compress_struct>(),
        );

        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);

        if config.progressive {
            jpeg_simple_progression(&mut cinfo);
        } else {
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        }

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => (1, 1),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT,
            if config.trellis_quant { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT_DC,
            if config.trellis_dc { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OVERSHOOT_DERINGING,
            if config.overshoot_deringing { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OPTIMIZE_SCANS,
            if config.optimize_scans { 1 } else { 0 },
        );

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = rgb.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}

/// Benchmark different encoder configurations.
fn bench_encoder_configs(c: &mut Criterion) {
    let width = 512u32;
    let height = 512u32;
    let rgb = create_test_image(width as usize, height as usize);

    let configs = [
        ("baseline", TestEncoderConfig::baseline()),
        (
            "huffman_opt",
            TestEncoderConfig {
                optimize_huffman: true,
                ..TestEncoderConfig::default()
            },
        ),
        (
            "trellis_ac",
            TestEncoderConfig {
                optimize_huffman: true,
                trellis_quant: true,
                ..TestEncoderConfig::default()
            },
        ),
        (
            "trellis_ac_dc",
            TestEncoderConfig {
                optimize_huffman: true,
                trellis_quant: true,
                trellis_dc: true,
                ..TestEncoderConfig::default()
            },
        ),
        (
            "progressive",
            TestEncoderConfig {
                progressive: true,
                optimize_huffman: true,
                ..TestEncoderConfig::default()
            },
        ),
        ("max_compression", TestEncoderConfig::max_compression()),
    ];

    let mut group = c.benchmark_group("rust_configs");
    group.throughput(Throughput::Elements((width * height) as u64));

    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::new("rust", name), config, |b, cfg| {
            b.iter(|| encode_rust(black_box(&rgb), width, height, cfg))
        });
    }

    group.finish();
}

/// Benchmark Rust vs C at different configurations.
fn bench_rust_vs_c(c: &mut Criterion) {
    let width = 512u32;
    let height = 512u32;
    let rgb = create_test_image(width as usize, height as usize);

    let configs = [
        ("baseline", TestEncoderConfig::baseline()),
        (
            "trellis",
            TestEncoderConfig {
                optimize_huffman: true,
                trellis_quant: true,
                trellis_dc: true,
                ..TestEncoderConfig::default()
            },
        ),
    ];

    let mut group = c.benchmark_group("rust_vs_c");
    group.throughput(Throughput::Elements((width * height) as u64));

    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::new("rust", name), config, |b, cfg| {
            b.iter(|| encode_rust(black_box(&rgb), width, height, cfg))
        });
        group.bench_with_input(BenchmarkId::new("c", name), config, |b, cfg| {
            b.iter(|| encode_c(black_box(&rgb), width, height, cfg))
        });
    }

    group.finish();
}

/// Benchmark across different image sizes.
fn bench_image_sizes(c: &mut Criterion) {
    let sizes: [(u32, u32); 4] = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    let mut group = c.benchmark_group("image_sizes");

    for (width, height) in sizes {
        let rgb = create_test_image(width as usize, height as usize);
        let size_label = format!("{}x{}", width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        // Fastest mode (no optimizations)
        group.bench_with_input(
            BenchmarkId::new("fastest", &size_label),
            &rgb,
            |b, rgb_data| {
                let encoder = Encoder::fastest().quality(85);
                b.iter(|| {
                    encoder
                        .encode_rgb(black_box(rgb_data), width, height)
                        .unwrap()
                })
            },
        );

        // Baseline optimized mode (trellis + huffman opt)
        group.bench_with_input(
            BenchmarkId::new("baseline_opt", &size_label),
            &rgb,
            |b, rgb_data| {
                let encoder = Encoder::baseline_optimized().quality(85);
                b.iter(|| {
                    encoder
                        .encode_rgb(black_box(rgb_data), width, height)
                        .unwrap()
                })
            },
        );

        // Max compression
        group.bench_with_input(
            BenchmarkId::new("max_compression", &size_label),
            &rgb,
            |b, rgb_data| {
                let encoder = Encoder::max_compression().quality(85);
                b.iter(|| {
                    encoder
                        .encode_rgb(black_box(rgb_data), width, height)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark subsampling modes.
fn bench_subsampling(c: &mut Criterion) {
    let width = 512u32;
    let height = 512u32;
    let rgb = create_test_image(width as usize, height as usize);

    let subsamplings = [
        ("4:4:4", Subsampling::S444),
        ("4:2:2", Subsampling::S422),
        ("4:2:0", Subsampling::S420),
    ];

    let mut group = c.benchmark_group("subsampling");
    group.throughput(Throughput::Elements((width * height) as u64));

    for (name, subsampling) in subsamplings {
        group.bench_with_input(BenchmarkId::from_parameter(name), &subsampling, |b, &ss| {
            let encoder = Encoder::baseline_optimized().quality(85).subsampling(ss);
            b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
        });
    }

    group.finish();
}

/// Benchmark DCT implementations (scalar vs SIMD).
fn bench_dct(c: &mut Criterion) {
    use mozjpeg_rs::dct::{forward_dct_8x8, forward_dct_8x8_simd, forward_dct_8x8_transpose};
    use mozjpeg_rs::simd::SimdOps;

    // Create test data
    let mut samples = [0i16; 64];
    for i in 0..64 {
        samples[i] = ((i as i32 * 73 + 17) % 256 - 128) as i16;
    }

    let mut group = c.benchmark_group("dct");
    group.throughput(Throughput::Elements(64)); // 64 coefficients per block

    group.bench_function("scalar", |b| {
        let mut coeffs = [0i16; 64];
        b.iter(|| {
            forward_dct_8x8(black_box(&samples), &mut coeffs);
            black_box(coeffs)
        })
    });

    group.bench_function("simd_gather", |b| {
        let mut coeffs = [0i16; 64];
        b.iter(|| {
            forward_dct_8x8_simd(black_box(&samples), &mut coeffs);
            black_box(coeffs)
        })
    });

    group.bench_function("simd_transpose", |b| {
        let mut coeffs = [0i16; 64];
        b.iter(|| {
            forward_dct_8x8_transpose(black_box(&samples), &mut coeffs);
            black_box(coeffs)
        })
    });

    // New SIMD dispatch module (uses AVX2 intrinsics on x86_64)
    let simd_ops = SimdOps::detect();
    group.bench_function("simd_dispatch", |b| {
        let mut coeffs = [0i16; 64];
        b.iter(|| {
            (simd_ops.forward_dct)(black_box(&samples), &mut coeffs);
            black_box(coeffs)
        })
    });

    // Scalar-only from simd module (for comparison)
    let scalar_ops = SimdOps::scalar();
    group.bench_function("simd_scalar_ref", |b| {
        let mut coeffs = [0i16; 64];
        b.iter(|| {
            (scalar_ops.forward_dct)(black_box(&samples), &mut coeffs);
            black_box(coeffs)
        })
    });

    group.finish();
}

/// Benchmark the cost of individual optimization flags.
///
/// This measures the INCREMENTAL cost of each flag by building up
/// from fastest() step by step.
fn bench_optimization_flags(c: &mut Criterion) {
    use mozjpeg_rs::TrellisConfig;

    let width = 512u32;
    let height = 512u32;
    let rgb = create_test_image(width as usize, height as usize);

    let mut group = c.benchmark_group("optimization_flags");
    group.throughput(Throughput::Elements((width * height) as u64));

    // Step 0: fastest() - no optimizations
    group.bench_function("0_fastest", |b| {
        let encoder = Encoder::fastest()
            .quality(75)
            .subsampling(Subsampling::S420);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 1: +huffman (incremental)
    group.bench_function("1_+huffman", |b| {
        let encoder = Encoder::fastest()
            .quality(75)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 2: +huffman +deringing
    group.bench_function("2_+deringing", |b| {
        let encoder = Encoder::fastest()
            .quality(75)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true)
            .overshoot_deringing(true);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 3: +huffman +deringing +trellis_ac
    group.bench_function("3_+trellis_ac", |b| {
        let encoder = Encoder::fastest()
            .quality(75)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true)
            .overshoot_deringing(true)
            .trellis(TrellisConfig::default().dc_trellis(false));
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 4: +huffman +deringing +trellis_ac_dc = new(false)
    group.bench_function("4_new(false)", |b| {
        let encoder = Encoder::baseline_optimized()
            .quality(75)
            .subsampling(Subsampling::S420);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 5: new(false) + progressive
    group.bench_function("5_+progressive", |b| {
        let encoder = Encoder::baseline_optimized()
            .quality(75)
            .subsampling(Subsampling::S420)
            .progressive(true);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    // Step 6: new(false) + progressive + optimize_scans = new(true)
    group.bench_function("6_new(true)", |b| {
        let encoder = Encoder::max_compression()
            .quality(75)
            .subsampling(Subsampling::S420);
        b.iter(|| encoder.encode_rgb(black_box(&rgb), width, height).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encoder_configs,
    bench_rust_vs_c,
    bench_image_sizes,
    bench_subsampling,
    bench_dct,
    bench_optimization_flags
);
criterion_main!(benches);
