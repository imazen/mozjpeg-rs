//! Trace through the encoding pipeline step by step, comparing Rust and C at each stage.

#![allow(dead_code)]

use mozjpeg_oxide::consts::{
    AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    STD_LUMINANCE_QUANT_TBL,
};
use mozjpeg_oxide::huffman::{DerivedTable, HuffTable};
use mozjpeg_oxide::TrellisConfig;
use mozjpeg_oxide::{color, dct, deringing, quant, trellis};
use std::fs;

fn main() {
    // Load test image
    let source_path = "/home/lilith/work/mozjpeg-rs/corpus/kodak/10.png";
    let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width as usize;
    let height = info.height as usize;

    println!("=== Pipeline Trace Comparison ===");
    println!("Image: {}x{}", width, height);
    println!();

    // Test at Q97 where we see the biggest difference
    let quality = 97u8;
    let scale_factor = quant::quality_to_scale_factor(quality);

    // Scale quantization table
    let mut qtable_natural = [0u16; 64];
    for i in 0..64 {
        let base = STD_LUMINANCE_QUANT_TBL[3][i] as i32; // ImageMagick table (index 3)
        let scaled = (base * scale_factor as i32 + 50) / 100;
        qtable_natural[i] = scaled.clamp(1, 255) as u16;
    }

    println!("Quality: Q{}", quality);
    println!("DC quant value: {}", qtable_natural[0]);
    println!();

    // Build Huffman tables
    let ac_table = build_ac_table();
    let dc_table = build_dc_table();

    let row_stride = width * 3;

    println!("=== Stage 1: RGB to YCbCr Conversion ===");

    // Convert first 8 pixels
    let mut rust_y = [0u8; 8];
    let mut c_y = [0i32; 8];

    for i in 0..8 {
        let r = rgb_data[i * 3];
        let g = rgb_data[i * 3 + 1];
        let b = rgb_data[i * 3 + 2];

        // Rust
        let (y, _, _) = color::rgb_to_ycbcr(r, g, b);
        rust_y[i] = y;

        // C
        let mut cy = 0i32;
        let mut ccb = 0i32;
        let mut ccr = 0i32;
        unsafe {
            sys_local::mozjpeg_test_rgb_to_ycbcr(
                r as i32, g as i32, b as i32, &mut cy, &mut ccb, &mut ccr,
            );
        }
        c_y[i] = cy;
    }

    println!("First 8 Y values (Rust): {:?}", rust_y);
    println!("First 8 Y values (C):    {:?}", c_y);

    let y_match = (0..8).all(|i| rust_y[i] as i32 == c_y[i]);
    println!("Y conversion match: {}", if y_match { "✓" } else { "✗" });

    if !y_match {
        for i in 0..8 {
            if rust_y[i] as i32 != c_y[i] {
                println!("  Diff at {}: Rust={}, C={}", i, rust_y[i], c_y[i]);
            }
        }
    }
    println!();

    // Extract 8x8 block with level shift
    println!("=== Stage 2: Extract and Level-Shift 8x8 Block ===");

    let mut block_samples = [0i16; 64];
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * row_stride + col * 3;
            let r = rgb_data[idx];
            let g = rgb_data[idx + 1];
            let b = rgb_data[idx + 2];
            let (y, _, _) = color::rgb_to_ycbcr(r, g, b);
            block_samples[row * 8 + col] = y as i16 - 128;
        }
    }

    println!(
        "Level-shifted block (first row): {:?}",
        &block_samples[0..8]
    );
    println!();

    // Apply deringing
    println!("=== Stage 3: Overshoot Deringing ===");

    let mut rust_dering = block_samples.clone();
    deringing::preprocess_deringing(&mut rust_dering, qtable_natural[0]);

    let mut c_dering = block_samples.clone();
    unsafe {
        sys_local::mozjpeg_test_preprocess_deringing(c_dering.as_mut_ptr(), qtable_natural[0]);
    }

    let dering_changed = (0..64).any(|i| rust_dering[i] != block_samples[i]);
    println!("Deringing made changes: {}", dering_changed);

    let dering_match = rust_dering == c_dering;
    println!("Deringing match: {}", if dering_match { "✓" } else { "✗" });

    if !dering_match {
        for i in 0..64 {
            if rust_dering[i] != c_dering[i] {
                println!(
                    "  Diff at {}: Rust={}, C={}",
                    i, rust_dering[i], c_dering[i]
                );
            }
        }
    }
    println!();

    // Forward DCT
    println!("=== Stage 4: Forward DCT ===");

    let mut rust_dct = [0i16; 64];
    dct::forward_dct_8x8(&block_samples, &mut rust_dct);

    let mut c_dct = block_samples.clone();
    unsafe {
        sys_local::mozjpeg_test_fdct_islow(c_dct.as_mut_ptr());
    }

    println!("DCT DC (Rust): {}, (C): {}", rust_dct[0], c_dct[0]);
    println!("DCT first 8 AC (Rust): {:?}", &rust_dct[1..9]);
    println!("DCT first 8 AC (C):    {:?}", &c_dct[1..9]);

    let dct_match = (0..64).all(|i| rust_dct[i] == c_dct[i]);
    println!("DCT match: {}", if dct_match { "✓" } else { "✗" });

    if !dct_match {
        let diffs: Vec<_> = (0..64)
            .filter(|&i| rust_dct[i] != c_dct[i])
            .map(|i| (i, rust_dct[i], c_dct[i]))
            .collect();
        println!(
            "DCT differences (idx, rust, c): {:?}",
            &diffs[..diffs.len().min(10)]
        );
    }
    println!();

    // AC Trellis quantization
    println!("=== Stage 5: AC Trellis Quantization ===");

    let config = TrellisConfig::default();
    let dct_i32: [i32; 64] = std::array::from_fn(|i| rust_dct[i] as i32);

    let mut rust_trellis_q = [0i16; 64];
    trellis::trellis_quantize_block(
        &dct_i32,
        &mut rust_trellis_q,
        &qtable_natural,
        &ac_table,
        &config,
    );

    // Get AC huffman sizes for C
    let mut ac_huffsi = [0i8; 256];
    for i in 0..256 {
        let (_, size) = ac_table.get_code(i as u8);
        ac_huffsi[i] = size as i8;
    }

    let mut c_trellis_q = [0i16; 64];
    unsafe {
        sys_local::mozjpeg_test_trellis_quantize_block(
            c_dct.as_ptr(),
            c_trellis_q.as_mut_ptr(),
            qtable_natural.as_ptr(),
            ac_huffsi.as_ptr(),
            config.lambda_log_scale1,
            config.lambda_log_scale2,
        );
    }

    println!(
        "Trellis DC (Rust): {}, (C): {}",
        rust_trellis_q[0], c_trellis_q[0]
    );
    println!("Trellis first 8 AC (Rust): {:?}", &rust_trellis_q[1..9]);
    println!("Trellis first 8 AC (C):    {:?}", &c_trellis_q[1..9]);

    let trellis_match = (0..64).all(|i| rust_trellis_q[i] == c_trellis_q[i]);
    println!(
        "AC Trellis match: {}",
        if trellis_match { "✓" } else { "✗" }
    );

    if !trellis_match {
        let diffs: Vec<_> = (0..64)
            .filter(|&i| rust_trellis_q[i] != c_trellis_q[i])
            .map(|i| (i, rust_trellis_q[i], c_trellis_q[i]))
            .collect();
        println!("Trellis differences: {:?}", &diffs[..diffs.len().min(10)]);
    }

    let rust_nonzero = rust_trellis_q.iter().filter(|&&x| x != 0).count();
    let c_nonzero = c_trellis_q.iter().filter(|&&x| x != 0).count();
    println!(
        "Non-zero coefficients: Rust={}, C={}",
        rust_nonzero, c_nonzero
    );
    println!();

    // Process full row of blocks
    println!("=== Stage 6: Full Row of Blocks (AC + DC Trellis) ===");

    let blocks_in_row = width / 8;
    println!("Processing {} blocks in first row", blocks_in_row);

    let mut raw_dct_blocks: Vec<[i32; 64]> = Vec::with_capacity(blocks_in_row);
    let mut rust_quantized_blocks: Vec<[i16; 64]> = Vec::with_capacity(blocks_in_row);
    let mut c_quantized_blocks: Vec<[i16; 64]> = Vec::with_capacity(blocks_in_row);

    for block_col in 0..blocks_in_row {
        let x_start = block_col * 8;

        // Extract 8x8 block
        let mut block = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                let idx = row * row_stride + (x_start + col) * 3;
                let r = rgb_data[idx];
                let g = rgb_data[idx + 1];
                let b = rgb_data[idx + 2];
                let (y, _, _) = color::rgb_to_ycbcr(r, g, b);
                block[row * 8 + col] = y as i16 - 128;
            }
        }

        // DCT
        let mut dct_out = [0i16; 64];
        dct::forward_dct_8x8(&block, &mut dct_out);

        // Store raw DCT
        let dct_i32: [i32; 64] = std::array::from_fn(|i| dct_out[i] as i32);
        raw_dct_blocks.push(dct_i32);

        // AC trellis quantize (Rust)
        let mut rust_q = [0i16; 64];
        trellis::trellis_quantize_block(&dct_i32, &mut rust_q, &qtable_natural, &ac_table, &config);
        rust_quantized_blocks.push(rust_q);

        // AC trellis quantize (C)
        let mut c_q = [0i16; 64];
        unsafe {
            sys_local::mozjpeg_test_trellis_quantize_block(
                dct_out.as_ptr(),
                c_q.as_mut_ptr(),
                qtable_natural.as_ptr(),
                ac_huffsi.as_ptr(),
                config.lambda_log_scale1,
                config.lambda_log_scale2,
            );
        }
        c_quantized_blocks.push(c_q);
    }

    // Check AC trellis results match for all blocks
    let ac_blocks_match = (0..blocks_in_row)
        .all(|b| (0..64).all(|i| rust_quantized_blocks[b][i] == c_quantized_blocks[b][i]));
    println!(
        "AC trellis for all {} blocks match: {}",
        blocks_in_row,
        if ac_blocks_match { "✓" } else { "✗" }
    );

    if !ac_blocks_match {
        let mismatched_blocks: Vec<_> = (0..blocks_in_row)
            .filter(|&b| (0..64).any(|i| rust_quantized_blocks[b][i] != c_quantized_blocks[b][i]))
            .collect();
        println!("Number of mismatched blocks: {}", mismatched_blocks.len());

        // Show details of first mismatched block
        if let Some(&b) = mismatched_blocks.first() {
            let diffs: Vec<_> = (0..64)
                .filter(|&i| rust_quantized_blocks[b][i] != c_quantized_blocks[b][i])
                .map(|i| (i, rust_quantized_blocks[b][i], c_quantized_blocks[b][i]))
                .collect();
            println!(
                "Block {} differences: {:?}",
                b,
                &diffs[..diffs.len().min(10)]
            );
        }
    }

    // DC trellis
    let dc_quantval = qtable_natural[0];

    let mut dc_huffsi = [0i8; 17];
    for bits in 0..=16 {
        let (_, size) = dc_table.get_code(bits as u8);
        dc_huffsi[bits] = size as i8;
    }

    // Rust DC trellis
    trellis::dc_trellis_optimize(
        &raw_dct_blocks,
        &mut rust_quantized_blocks,
        dc_quantval,
        &dc_table,
        0,
        config.lambda_log_scale1,
        config.lambda_log_scale2,
    );

    // C DC trellis
    let raw_dc: Vec<i32> = raw_dct_blocks.iter().map(|b| b[0]).collect();
    let ac_norms: Vec<f32> = raw_dct_blocks
        .iter()
        .map(|b| {
            let sum: f32 = (1..64).map(|i| (b[i] as f32).powi(2)).sum();
            sum / 63.0
        })
        .collect();
    let mut c_dc_out = vec![0i16; blocks_in_row];

    unsafe {
        sys_local::mozjpeg_test_dc_trellis_optimize(
            raw_dc.as_ptr(),
            ac_norms.as_ptr(),
            c_dc_out.as_mut_ptr(),
            blocks_in_row as i32,
            dc_quantval,
            dc_huffsi.as_ptr(),
            0,
            config.lambda_log_scale1,
            config.lambda_log_scale2,
        );
    }

    // Update C blocks with DC trellis results
    for (i, dc) in c_dc_out.iter().enumerate() {
        c_quantized_blocks[i][0] = *dc;
    }

    // Compare DC values
    let rust_dc_values: Vec<i16> = rust_quantized_blocks.iter().map(|b| b[0]).collect();
    let c_dc_values: Vec<i16> = c_quantized_blocks.iter().map(|b| b[0]).collect();

    let dc_match = rust_dc_values == c_dc_values;
    println!("DC trellis match: {}", if dc_match { "✓" } else { "✗" });

    if !dc_match {
        let dc_diffs: Vec<_> = (0..blocks_in_row)
            .filter(|&i| rust_dc_values[i] != c_dc_values[i])
            .take(20)
            .map(|i| (i, rust_dc_values[i], c_dc_values[i]))
            .collect();
        println!("DC differences (block, rust, c): {:?}", dc_diffs);
    }

    println!(
        "First 16 DC values (Rust): {:?}",
        &rust_dc_values[..16.min(blocks_in_row)]
    );
    println!(
        "First 16 DC values (C):    {:?}",
        &c_dc_values[..16.min(blocks_in_row)]
    );

    // Count total non-zero coefficients
    let rust_total_nonzero: usize = rust_quantized_blocks
        .iter()
        .map(|b| b.iter().filter(|&&x| x != 0).count())
        .sum();
    let c_total_nonzero: usize = c_quantized_blocks
        .iter()
        .map(|b| b.iter().filter(|&&x| x != 0).count())
        .sum();
    println!();
    println!(
        "Total non-zero coefficients in row: Rust={}, C={}",
        rust_total_nonzero, c_total_nonzero
    );

    // Final summary
    println!();
    println!("=== SUMMARY ===");
    println!(
        "YCbCr conversion: {}",
        if y_match { "✓ MATCH" } else { "✗ DIFFERS" }
    );
    println!(
        "Deringing: {}",
        if dering_match {
            "✓ MATCH"
        } else {
            "✗ DIFFERS"
        }
    );
    println!(
        "DCT: {}",
        if dct_match {
            "✓ MATCH"
        } else {
            "✗ DIFFERS"
        }
    );
    println!(
        "AC trellis (single block): {}",
        if trellis_match {
            "✓ MATCH"
        } else {
            "✗ DIFFERS"
        }
    );
    println!(
        "AC trellis (row of {} blocks): {}",
        blocks_in_row,
        if ac_blocks_match {
            "✓ MATCH"
        } else {
            "✗ DIFFERS"
        }
    );
    println!(
        "DC trellis (row): {}",
        if dc_match { "✓ MATCH" } else { "✗ DIFFERS" }
    );

    if y_match && dering_match && dct_match && trellis_match && ac_blocks_match && dc_match {
        println!();
        println!("All stages match! The difference must be in the encoder integration.");
    }
}

fn build_ac_table() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, false).unwrap()
}

fn build_dc_table() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, true).unwrap()
}
