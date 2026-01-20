//! Timing breakdown test to identify encoder hotspots.
//!
//! This test measures each stage of the encoding pipeline separately
//! to identify which operations are slowest compared to C mozjpeg.
//!
//! Run with: cargo test --release timing_breakdown -- --nocapture

use mozjpeg_rs::{
    bitstream::BitWriter,
    color,
    consts::{
        AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DCTSIZE, DCTSIZE2, DC_LUMINANCE_BITS,
        DC_LUMINANCE_VALUES,
    },
    dct,
    entropy::EntropyEncoder,
    huffman::{DerivedTable, HuffTable},
    quant, sample, trellis, QuantTableIdx, TrellisConfig,
};
use std::time::Instant;

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

fn create_std_dc_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    htbl.huffval[..DC_LUMINANCE_VALUES.len()].copy_from_slice(&DC_LUMINANCE_VALUES);
    htbl
}

fn create_std_ac_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    htbl.huffval[..AC_LUMINANCE_VALUES.len()].copy_from_slice(&AC_LUMINANCE_VALUES);
    htbl
}

/// Measure time for an operation, returning (result, duration_micros)
fn timed<T, F: FnOnce() -> T>(f: F) -> (T, u64) {
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_micros() as u64;
    (result, elapsed)
}

#[test]
fn timing_breakdown_baseline() {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;
    const ITERATIONS: usize = 10;

    println!(
        "\n=== Encoder Timing Breakdown ({}x{}, {} iterations) ===\n",
        WIDTH, HEIGHT, ITERATIONS
    );

    let rgb = create_test_image(WIDTH, HEIGHT);
    let num_pixels = WIDTH * HEIGHT;

    // Pre-allocate buffers
    let mut y_plane = vec![0u8; num_pixels];
    let mut cb_plane = vec![0u8; num_pixels];
    let mut cr_plane = vec![0u8; num_pixels];

    // Subsampling 4:2:0
    let (luma_h, luma_v) = (2usize, 2usize);
    let (chroma_width, chroma_height) =
        sample::subsampled_dimensions(WIDTH, HEIGHT, luma_h, luma_v);
    let chroma_size = chroma_width * chroma_height;
    let mut cb_subsampled = vec![0u8; chroma_size];
    let mut cr_subsampled = vec![0u8; chroma_size];

    // MCU-aligned dimensions
    let (mcu_width, mcu_height) = sample::mcu_aligned_dimensions(WIDTH, HEIGHT, luma_h, luma_v);
    let (mcu_chroma_w, mcu_chroma_h) = (mcu_width / luma_h, mcu_height / luma_v);
    let mut y_mcu = vec![0u8; mcu_width * mcu_height];
    let mut cb_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];
    let mut cr_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];

    // Quant tables
    let (luma_qtable, _chroma_qtable) =
        quant::create_quant_tables(85, QuantTableIdx::ImageMagick, true);

    // Huffman tables
    let dc_luma_huff = create_std_dc_luma_table();
    let ac_luma_huff = create_std_ac_luma_table();
    let dc_luma_derived = DerivedTable::from_huff_table(&dc_luma_huff, true).unwrap();
    let ac_luma_derived = DerivedTable::from_huff_table(&ac_luma_huff, false).unwrap();

    // Block counts
    let mcu_rows = mcu_height / (DCTSIZE * luma_v);
    let mcu_cols = mcu_width / (DCTSIZE * luma_h);
    let y_blocks_per_mcu = luma_h * luma_v;
    let total_y_blocks = mcu_rows * mcu_cols * y_blocks_per_mcu;

    // Storage for DCT blocks (i32 for raw DCT, i16 for quantized)
    let mut raw_dct: Vec<[i32; DCTSIZE2]> = vec![[0i32; DCTSIZE2]; total_y_blocks];
    let mut y_blocks: Vec<[i16; DCTSIZE2]> = vec![[0i16; DCTSIZE2]; total_y_blocks];
    let mut dct_block = [0i16; DCTSIZE2];

    // Timing accumulators
    let mut color_time = 0u64;
    let mut downsample_time = 0u64;
    let mut expand_time = 0u64;
    let mut dct_time = 0u64;
    let mut quant_time = 0u64;
    let mut entropy_time = 0u64;

    for _ in 0..ITERATIONS {
        // Stage 1: Color conversion (RGB -> YCbCr)
        let (_, t) = timed(|| {
            color::convert_rgb_to_ycbcr(
                &rgb,
                &mut y_plane,
                &mut cb_plane,
                &mut cr_plane,
                WIDTH,
                HEIGHT,
            );
        });
        color_time += t;

        // Stage 2: Downsampling
        let (_, t) = timed(|| {
            sample::downsample_plane(&cb_plane, WIDTH, HEIGHT, luma_h, luma_v, &mut cb_subsampled);
            sample::downsample_plane(&cr_plane, WIDTH, HEIGHT, luma_h, luma_v, &mut cr_subsampled);
        });
        downsample_time += t;

        // Stage 3: MCU expansion
        let (_, t) = timed(|| {
            sample::expand_to_mcu(&y_plane, WIDTH, HEIGHT, &mut y_mcu, mcu_width, mcu_height);
            sample::expand_to_mcu(
                &cb_subsampled,
                chroma_width,
                chroma_height,
                &mut cb_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
            );
            sample::expand_to_mcu(
                &cr_subsampled,
                chroma_width,
                chroma_height,
                &mut cr_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
            );
        });
        expand_time += t;

        // Stage 4: Forward DCT (Y channel only for timing)
        let (_, t) = timed(|| {
            let mut block_idx = 0;
            for mcu_row in 0..mcu_rows {
                for mcu_col in 0..mcu_cols {
                    for v_block in 0..luma_v {
                        for h_block in 0..luma_h {
                            let block_row = mcu_row * luma_v + v_block;
                            let block_col = mcu_col * luma_h + h_block;
                            let y_offset = block_row * DCTSIZE * mcu_width + block_col * DCTSIZE;

                            // Extract block and level-shift
                            let mut samples = [0i16; DCTSIZE2];
                            for row in 0..DCTSIZE {
                                for col in 0..DCTSIZE {
                                    let pixel = y_mcu[y_offset + row * mcu_width + col] as i16;
                                    samples[row * DCTSIZE + col] = pixel - 128;
                                }
                            }

                            // Forward DCT
                            dct::forward_dct_8x8_i32_wide_transpose(&samples, &mut dct_block);

                            // Store as i32 for quantization
                            for i in 0..DCTSIZE2 {
                                raw_dct[block_idx][i] = dct_block[i] as i32;
                            }
                            block_idx += 1;
                        }
                    }
                }
            }
        });
        dct_time += t;

        // Stage 5: Quantization (simple, no trellis)
        let (_, t) = timed(|| {
            for (block_idx, raw) in raw_dct.iter().enumerate() {
                quant::quantize_block(raw, &luma_qtable.values, &mut y_blocks[block_idx]);
            }
        });
        quant_time += t;

        // Stage 6: Entropy encoding
        let (_, t) = timed(|| {
            let mut output = Vec::with_capacity(total_y_blocks * 64);
            let mut bit_writer = BitWriter::new(&mut output);
            let mut encoder = EntropyEncoder::new(&mut bit_writer);

            for block in &y_blocks {
                encoder
                    .encode_block(block, 0, &dc_luma_derived, &ac_luma_derived)
                    .unwrap();
            }
            bit_writer.flush().unwrap();
        });
        entropy_time += t;
    }

    // Calculate averages
    let color_avg = color_time / ITERATIONS as u64;
    let downsample_avg = downsample_time / ITERATIONS as u64;
    let expand_avg = expand_time / ITERATIONS as u64;
    let dct_avg = dct_time / ITERATIONS as u64;
    let quant_avg = quant_time / ITERATIONS as u64;
    let entropy_avg = entropy_time / ITERATIONS as u64;
    let total_avg = color_avg + downsample_avg + expand_avg + dct_avg + quant_avg + entropy_avg;

    println!("| Stage            | Time (µs) | % of Total |");
    println!("|------------------|-----------|------------|");
    println!(
        "| Color conversion | {:>9} | {:>9.1}% |",
        color_avg,
        100.0 * color_avg as f64 / total_avg as f64
    );
    println!(
        "| Downsampling     | {:>9} | {:>9.1}% |",
        downsample_avg,
        100.0 * downsample_avg as f64 / total_avg as f64
    );
    println!(
        "| MCU expansion    | {:>9} | {:>9.1}% |",
        expand_avg,
        100.0 * expand_avg as f64 / total_avg as f64
    );
    println!(
        "| Forward DCT      | {:>9} | {:>9.1}% |",
        dct_avg,
        100.0 * dct_avg as f64 / total_avg as f64
    );
    println!(
        "| Quantization     | {:>9} | {:>9.1}% |",
        quant_avg,
        100.0 * quant_avg as f64 / total_avg as f64
    );
    println!(
        "| Entropy encoding | {:>9} | {:>9.1}% |",
        entropy_avg,
        100.0 * entropy_avg as f64 / total_avg as f64
    );
    println!("|------------------|-----------|------------|");
    println!("| **Total**        | {:>9} | {:>9.1}% |", total_avg, 100.0);
    println!();
    println!("Block count: {} Y blocks", total_y_blocks);
    println!(
        "DCT per block: {:.1} ns",
        (dct_avg as f64 * 1000.0) / total_y_blocks as f64
    );
    println!();
}

#[test]
fn timing_breakdown_trellis() {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;
    const ITERATIONS: usize = 5;

    println!(
        "\n=== Trellis Timing Breakdown ({}x{}, {} iterations) ===\n",
        WIDTH, HEIGHT, ITERATIONS
    );

    let rgb = create_test_image(WIDTH, HEIGHT);
    let num_pixels = WIDTH * HEIGHT;

    // Pre-allocate buffers
    let mut y_plane = vec![0u8; num_pixels];
    let mut cb_plane = vec![0u8; num_pixels];
    let mut cr_plane = vec![0u8; num_pixels];

    // Subsampling 4:2:0
    let (luma_h, luma_v) = (2usize, 2usize);
    let (chroma_width, chroma_height) =
        sample::subsampled_dimensions(WIDTH, HEIGHT, luma_h, luma_v);
    let chroma_size = chroma_width * chroma_height;
    let mut cb_subsampled = vec![0u8; chroma_size];
    let mut cr_subsampled = vec![0u8; chroma_size];

    // MCU-aligned dimensions
    let (mcu_width, mcu_height) = sample::mcu_aligned_dimensions(WIDTH, HEIGHT, luma_h, luma_v);
    let (mcu_chroma_w, mcu_chroma_h) = (mcu_width / luma_h, mcu_height / luma_v);
    let mut y_mcu = vec![0u8; mcu_width * mcu_height];
    let mut cb_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];
    let mut cr_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];

    // Quant tables
    let (luma_qtable, _chroma_qtable) =
        quant::create_quant_tables(85, QuantTableIdx::ImageMagick, true);

    // Huffman tables
    let dc_luma_huff = create_std_dc_luma_table();
    let ac_luma_huff = create_std_ac_luma_table();
    let dc_luma_derived = DerivedTable::from_huff_table(&dc_luma_huff, true).unwrap();
    let ac_luma_derived = DerivedTable::from_huff_table(&ac_luma_huff, false).unwrap();

    // Block counts
    let mcu_rows = mcu_height / (DCTSIZE * luma_v);
    let mcu_cols = mcu_width / (DCTSIZE * luma_h);
    let y_blocks_per_mcu = luma_h * luma_v;
    let total_y_blocks = mcu_rows * mcu_cols * y_blocks_per_mcu;

    // Storage for DCT blocks
    let mut raw_dct: Vec<[i32; DCTSIZE2]> = vec![[0i32; DCTSIZE2]; total_y_blocks];
    let mut y_blocks: Vec<[i16; DCTSIZE2]> = vec![[0i16; DCTSIZE2]; total_y_blocks];
    let mut dct_block = [0i16; DCTSIZE2];

    // Trellis config
    let trellis_config = TrellisConfig::default();

    // Timing accumulators
    let mut prep_time = 0u64;
    let mut dct_time = 0u64;
    let mut trellis_time = 0u64;
    let mut entropy_time = 0u64;

    for _ in 0..ITERATIONS {
        // Stage 1: Preparation (color + downsample + expand)
        let (_, t) = timed(|| {
            color::convert_rgb_to_ycbcr(
                &rgb,
                &mut y_plane,
                &mut cb_plane,
                &mut cr_plane,
                WIDTH,
                HEIGHT,
            );
            sample::downsample_plane(&cb_plane, WIDTH, HEIGHT, luma_h, luma_v, &mut cb_subsampled);
            sample::downsample_plane(&cr_plane, WIDTH, HEIGHT, luma_h, luma_v, &mut cr_subsampled);
            sample::expand_to_mcu(&y_plane, WIDTH, HEIGHT, &mut y_mcu, mcu_width, mcu_height);
            sample::expand_to_mcu(
                &cb_subsampled,
                chroma_width,
                chroma_height,
                &mut cb_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
            );
            sample::expand_to_mcu(
                &cr_subsampled,
                chroma_width,
                chroma_height,
                &mut cr_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
            );
        });
        prep_time += t;

        // Stage 2: Forward DCT (Y channel only for timing)
        let (_, t) = timed(|| {
            let mut block_idx = 0;
            for mcu_row in 0..mcu_rows {
                for mcu_col in 0..mcu_cols {
                    for v_block in 0..luma_v {
                        for h_block in 0..luma_h {
                            let block_row = mcu_row * luma_v + v_block;
                            let block_col = mcu_col * luma_h + h_block;
                            let y_offset = block_row * DCTSIZE * mcu_width + block_col * DCTSIZE;

                            // Extract block and level-shift
                            let mut samples = [0i16; DCTSIZE2];
                            for row in 0..DCTSIZE {
                                for col in 0..DCTSIZE {
                                    let pixel = y_mcu[y_offset + row * mcu_width + col] as i16;
                                    samples[row * DCTSIZE + col] = pixel - 128;
                                }
                            }

                            // Forward DCT
                            dct::forward_dct_8x8_i32_wide_transpose(&samples, &mut dct_block);

                            // Store raw DCT as i32 for trellis
                            for i in 0..DCTSIZE2 {
                                raw_dct[block_idx][i] = dct_block[i] as i32;
                            }
                            block_idx += 1;
                        }
                    }
                }
            }
        });
        dct_time += t;

        // Stage 3: Trellis quantization
        let (_, t) = timed(|| {
            for (block_idx, raw) in raw_dct.iter().enumerate() {
                trellis::trellis_quantize_block(
                    raw,
                    &mut y_blocks[block_idx],
                    &luma_qtable.values,
                    &ac_luma_derived,
                    &trellis_config,
                );
            }
        });
        trellis_time += t;

        // Stage 4: Entropy encoding
        let (_, t) = timed(|| {
            let mut output = Vec::with_capacity(total_y_blocks * 64);
            let mut bit_writer = BitWriter::new(&mut output);
            let mut encoder = EntropyEncoder::new(&mut bit_writer);

            for block in &y_blocks {
                encoder
                    .encode_block(block, 0, &dc_luma_derived, &ac_luma_derived)
                    .unwrap();
            }
            bit_writer.flush().unwrap();
        });
        entropy_time += t;
    }

    // Calculate averages
    let prep_avg = prep_time / ITERATIONS as u64;
    let dct_avg = dct_time / ITERATIONS as u64;
    let trellis_avg = trellis_time / ITERATIONS as u64;
    let entropy_avg = entropy_time / ITERATIONS as u64;
    let total_avg = prep_avg + dct_avg + trellis_avg + entropy_avg;

    println!("| Stage              | Time (µs) | % of Total |");
    println!("|--------------------|-----------|------------|");
    println!(
        "| Prep (color+down)  | {:>9} | {:>9.1}% |",
        prep_avg,
        100.0 * prep_avg as f64 / total_avg as f64
    );
    println!(
        "| Forward DCT        | {:>9} | {:>9.1}% |",
        dct_avg,
        100.0 * dct_avg as f64 / total_avg as f64
    );
    println!(
        "| Trellis quant      | {:>9} | {:>9.1}% |",
        trellis_avg,
        100.0 * trellis_avg as f64 / total_avg as f64
    );
    println!(
        "| Entropy encoding   | {:>9} | {:>9.1}% |",
        entropy_avg,
        100.0 * entropy_avg as f64 / total_avg as f64
    );
    println!("|--------------------|-----------|------------|");
    println!(
        "| **Total**          | {:>9} | {:>9.1}% |",
        total_avg, 100.0
    );
    println!();
    println!("Block count: {} Y blocks", total_y_blocks);
    println!(
        "Trellis per block: {:.1} µs",
        trellis_avg as f64 / total_y_blocks as f64
    );
    println!();
}
