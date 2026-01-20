//! Criterion benchmarks for entropy encoding.
//!
//! Compares standard vs fast entropy encoder implementations with REAL image data.
//! Uses actual PNG images through DCT+quantization to match real-world coefficient distributions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mozjpeg_rs::bitstream::VecBitWriter;
use mozjpeg_rs::consts::{
    AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES, DCTSIZE,
    DCTSIZE2, QuantTableIdx,
};
use mozjpeg_rs::dct;
use mozjpeg_rs::entropy::EntropyEncoder;
use mozjpeg_rs::fast_entropy::FastEntropyEncoder;
use mozjpeg_rs::huffman::{DerivedTable, HuffTable};
use mozjpeg_rs::quant;

#[cfg(target_arch = "x86_64")]
use mozjpeg_rs::simd::x86_64::entropy::SimdEntropyEncoder;
use png::ColorType;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

fn create_dc_luma_table() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, true).unwrap()
}

fn create_ac_luma_table() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, false).unwrap()
}

/// Load a real PNG image and convert to Y plane (grayscale).
fn load_real_image(path: &Path) -> (Vec<u8>, usize, usize) {
    let file = File::open(path).expect("Failed to open test image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to Y (grayscale) using standard coefficients
    let y_plane: Vec<u8> = match info.color_type {
        ColorType::Grayscale | ColorType::GrayscaleAlpha => {
            let step = if info.color_type == ColorType::GrayscaleAlpha { 2 } else { 1 };
            buf.iter().step_by(step).copied().collect()
        }
        ColorType::Rgb => {
            buf.chunks_exact(3)
                .map(|rgb| {
                    // Y = 0.299*R + 0.587*G + 0.114*B (scaled integer math)
                    let y = (19595 * rgb[0] as u32 + 38470 * rgb[1] as u32 + 7471 * rgb[2] as u32 + 32768) >> 16;
                    y.min(255) as u8
                })
                .collect()
        }
        ColorType::Rgba => {
            buf.chunks_exact(4)
                .map(|rgba| {
                    let y = (19595 * rgba[0] as u32 + 38470 * rgba[1] as u32 + 7471 * rgba[2] as u32 + 32768) >> 16;
                    y.min(255) as u8
                })
                .collect()
        }
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    (y_plane, width, height)
}

/// Create a synthetic test image with gradient and noise (fallback).
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut y = vec![0u8; width * height];
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let noise = ((col * 7 + row * 13) % 50) as u8;
            y[idx] = ((col * 255 / width) as u8).saturating_add(noise);
        }
    }
    y
}

/// Generate DCT+quantized blocks from a Y plane.
fn generate_blocks_from_y(y_plane: &[u8], width: usize, height: usize, quality: u8) -> Vec<[i16; DCTSIZE2]> {
    let (luma_qtable, _) = quant::create_quant_tables(quality, QuantTableIdx::ImageMagick, true);

    let mcu_width = (width + 7) / 8 * 8;
    let mcu_height = (height + 7) / 8 * 8;
    let blocks_h = mcu_width / DCTSIZE;
    let blocks_v = mcu_height / DCTSIZE;
    let num_blocks = blocks_h * blocks_v;

    let mut blocks = Vec::with_capacity(num_blocks);
    let mut dct_block = [0i16; DCTSIZE2];

    for block_row in 0..blocks_v {
        for block_col in 0..blocks_h {
            // Extract 8x8 block with level shift
            let mut samples = [0i16; DCTSIZE2];
            for row in 0..DCTSIZE {
                for col in 0..DCTSIZE {
                    let y = block_row * DCTSIZE + row;
                    let x = block_col * DCTSIZE + col;
                    let pixel = if y < height && x < width {
                        y_plane[y * width + x] as i16
                    } else {
                        128 // Edge padding
                    };
                    samples[row * DCTSIZE + col] = pixel - 128;
                }
            }

            // Forward DCT
            dct::forward_dct_8x8_transpose(&samples, &mut dct_block);

            // Quantize
            let mut quant_block = [0i16; DCTSIZE2];
            let raw: [i32; DCTSIZE2] = std::array::from_fn(|i| dct_block[i] as i32);
            quant::quantize_block(&raw, &luma_qtable.values, &mut quant_block);

            blocks.push(quant_block);
        }
    }

    blocks
}

/// Generate REALISTIC DCT+quantized blocks from synthetic image data.
/// This matches the coefficient distribution seen in real encoding.
fn generate_realistic_blocks(width: usize, height: usize, quality: u8) -> Vec<[i16; DCTSIZE2]> {
    let y_plane = create_test_image(width, height);
    let (luma_qtable, _) = quant::create_quant_tables(quality, QuantTableIdx::ImageMagick, true);

    let mcu_width = (width + 7) / 8 * 8;
    let mcu_height = (height + 7) / 8 * 8;
    let blocks_h = mcu_width / DCTSIZE;
    let blocks_v = mcu_height / DCTSIZE;
    let num_blocks = blocks_h * blocks_v;

    let mut blocks = Vec::with_capacity(num_blocks);
    let mut dct_block = [0i16; DCTSIZE2];

    for block_row in 0..blocks_v {
        for block_col in 0..blocks_h {
            // Extract 8x8 block with level shift
            let mut samples = [0i16; DCTSIZE2];
            for row in 0..DCTSIZE {
                for col in 0..DCTSIZE {
                    let y = block_row * DCTSIZE + row;
                    let x = block_col * DCTSIZE + col;
                    let pixel = if y < height && x < width {
                        y_plane[y * width + x] as i16
                    } else {
                        128 // Edge padding
                    };
                    samples[row * DCTSIZE + col] = pixel - 128;
                }
            }

            // Forward DCT
            dct::forward_dct_8x8_transpose(&samples, &mut dct_block);

            // Quantize
            let mut quant_block = [0i16; DCTSIZE2];
            let raw: [i32; DCTSIZE2] = std::array::from_fn(|i| dct_block[i] as i32);
            quant::quantize_block(&raw, &luma_qtable.values, &mut quant_block);

            blocks.push(quant_block);
        }
    }

    blocks
}

fn bench_standard_encoder(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
) -> Vec<u8> {
    let mut writer = VecBitWriter::new_vec();
    {
        let mut encoder = EntropyEncoder::new(&mut writer);
        for block in blocks {
            encoder.encode_block(block, 0, dc_table, ac_table).unwrap();
        }
        encoder.flush().unwrap();
    }
    writer.into_bytes()
}

fn bench_fast_encoder(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
) -> Vec<u8> {
    let mut encoder = FastEntropyEncoder::new();
    for block in blocks {
        encoder.encode_block(block, 0, dc_table, ac_table);
    }
    encoder.into_bytes()
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_encoder(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
) -> Vec<u8> {
    let mut encoder = SimdEntropyEncoder::new();
    for block in blocks {
        unsafe {
            encoder.encode_block_sse2(block, 0, dc_table, ac_table);
        }
    }
    encoder.finish()
}

fn entropy_benchmark(c: &mut Criterion) {
    let dc_table = create_dc_luma_table();
    let ac_table = create_ac_luma_table();

    // ===== REAL IMAGE BENCHMARK (PRIMARY) =====
    // Load actual test image from repo
    let test_image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/images/1.png");
    let (y_plane, img_width, img_height) = load_real_image(&test_image_path);
    println!("Loaded real image: {}x{}", img_width, img_height);

    let mut real_group = c.benchmark_group("entropy_real_image");
    real_group.warm_up_time(Duration::from_secs(3));
    real_group.measurement_time(Duration::from_secs(8));
    real_group.sample_size(50);

    // Test real image at different quality levels
    for quality in [50, 75, 85, 95] {
        let blocks = generate_blocks_from_y(&y_plane, img_width, img_height, quality);
        let num_blocks = blocks.len();
        let name = format!("Q{}_real", quality);

        real_group.throughput(Throughput::Elements(num_blocks as u64));

        real_group.bench_with_input(
            BenchmarkId::new("standard", &name),
            &blocks,
            |b, blocks| {
                b.iter(|| bench_standard_encoder(black_box(blocks), &dc_table, &ac_table))
            },
        );

        real_group.bench_with_input(
            BenchmarkId::new("fast", &name),
            &blocks,
            |b, blocks| {
                b.iter(|| bench_fast_encoder(black_box(blocks), &dc_table, &ac_table))
            },
        );

        #[cfg(target_arch = "x86_64")]
        real_group.bench_with_input(
            BenchmarkId::new("simd_sse2", &name),
            &blocks,
            |b, blocks| {
                b.iter(|| bench_simd_encoder(black_box(blocks), &dc_table, &ac_table))
            },
        );
    }

    real_group.finish();

    // ===== SYNTHETIC IMAGE BENCHMARK (FOR COMPARISON) =====
    let configs = [
        ("Q50_512x512", 512, 512, 50),
        ("Q85_512x512", 512, 512, 85),
    ];

    let mut synth_group = c.benchmark_group("entropy_synthetic");
    synth_group.warm_up_time(Duration::from_secs(2));
    synth_group.measurement_time(Duration::from_secs(5));
    synth_group.sample_size(50);

    for (name, width, height, quality) in configs {
        let blocks = generate_realistic_blocks(width, height, quality);
        let num_blocks = blocks.len();

        synth_group.throughput(Throughput::Elements(num_blocks as u64));

        synth_group.bench_with_input(
            BenchmarkId::new("standard", name),
            &blocks,
            |b, blocks| {
                b.iter(|| bench_standard_encoder(black_box(blocks), &dc_table, &ac_table))
            },
        );

        synth_group.bench_with_input(
            BenchmarkId::new("fast", name),
            &blocks,
            |b, blocks| {
                b.iter(|| bench_fast_encoder(black_box(blocks), &dc_table, &ac_table))
            },
        );
    }

    synth_group.finish();
}

criterion_group!(benches, entropy_benchmark);
criterion_main!(benches);
