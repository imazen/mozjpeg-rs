//! Direct FFI comparison of quantization (non-trellis path)
//! Uses sys-local for C DCT function

use mozjpeg_rs::consts::DCTSIZE2;
use mozjpeg_rs::dct::forward_dct_8x8;
use mozjpeg_rs::quant::{create_quant_table, get_luminance_quant_table, quantize_block};
use mozjpeg_rs::QuantTableIdx;
use std::fs;

fn main() {
    let source_path_buf = mozjpeg_rs::corpus::cid22_dir()
        .expect("CID22 corpus not found. Set MOZJPEG_CORPUS_DIR or CODEC_CORPUS_DIR.")
        .join("10.png");
    let source_path = source_path_buf.to_str().expect("corpus path");
    let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    println!("=== Quantization FFI Comparison ===");
    println!("Image: {}x{}", width, height);
    println!();

    let quality = 85u8;

    // Create quantization table using the actual encoder API
    let base_table = get_luminance_quant_table(QuantTableIdx::ImageMagick);
    let qtable = create_quant_table(base_table, quality, true);

    println!("Quantization table (Q{}):", quality);
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:3} ", qtable.values[row * 8 + col]);
        }
        println!();
    }
    println!();

    // Test multiple blocks
    let mut total_diffs = 0;
    let mut total_coeffs = 0;
    let blocks_to_test = 100;

    for block_idx in 0..blocks_to_test {
        let block_x = (block_idx % (width as usize / 8)) as usize;
        let block_y = (block_idx / (width as usize / 8)) as usize;

        if block_y >= height as usize / 8 {
            break;
        }

        // Extract block from Y plane
        let mut block = [0i16; DCTSIZE2];
        for row in 0..8 {
            for col in 0..8 {
                let px = (block_y * 8 + row) * width as usize + (block_x * 8 + col);
                let r = rgb_data[px * 3] as f32;
                let g = rgb_data[px * 3 + 1] as f32;
                let b = rgb_data[px * 3 + 2] as f32;
                let y = (0.299 * r + 0.587 * g + 0.114 * b).round() as i16;
                block[row * 8 + col] = y - 128;
            }
        }

        // Rust DCT + quantization
        let rust_quant = rust_dct_and_quantize(&block, &qtable.values);

        // C DCT + quantization via FFI
        let c_quant = c_dct_and_quantize(&block, &qtable.values);

        // Compare
        for i in 0..DCTSIZE2 {
            if rust_quant[i] != c_quant[i] {
                total_diffs += 1;
            }
            total_coeffs += 1;
        }
    }

    println!(
        "Tested {} blocks ({} coefficients)",
        blocks_to_test, total_coeffs
    );
    println!(
        "Different coefficients: {} ({:.2}%)",
        total_diffs,
        100.0 * total_diffs as f64 / total_coeffs as f64
    );

    // Show a sample block with differences
    println!();
    println!("=== Sample Block with Differences ===");

    let block_x = 10;
    let block_y = 10;
    let mut block = [0i16; DCTSIZE2];
    for row in 0..8 {
        for col in 0..8 {
            let px = (block_y * 8 + row) * width as usize + (block_x * 8 + col);
            let r = rgb_data[px * 3] as f32;
            let g = rgb_data[px * 3 + 1] as f32;
            let b = rgb_data[px * 3 + 2] as f32;
            let y = (0.299 * r + 0.587 * g + 0.114 * b).round() as i16;
            block[row * 8 + col] = y - 128;
        }
    }

    let rust_quant = rust_dct_and_quantize(&block, &qtable.values);
    let c_quant = c_dct_and_quantize(&block, &qtable.values);

    println!("Block ({}, {})", block_x, block_y);
    println!();
    println!("Rust quantized:");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:4} ", rust_quant[row * 8 + col]);
        }
        println!();
    }
    println!();
    println!("C quantized:");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:4} ", c_quant[row * 8 + col]);
        }
        println!();
    }
    println!();
    println!("Differences (Rust - C):");
    let mut diff_count = 0;
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            let diff = rust_quant[row * 8 + col] as i32 - c_quant[row * 8 + col] as i32;
            if diff != 0 {
                print!("{:+4} ", diff);
                diff_count += 1;
            } else {
                print!("   . ");
            }
        }
        println!();
    }
    println!();
    println!("Total differences in block: {}/64", diff_count);

    // Also compare DCT output directly
    println!();
    println!("=== DCT Output Comparison ===");

    let mut rust_dct = [0i16; DCTSIZE2];
    forward_dct_8x8(&block, &mut rust_dct);

    let c_dct = c_dct_only(&block);

    println!("Rust DCT output (before descale):");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:6} ", rust_dct[row * 8 + col]);
        }
        println!();
    }
    println!();
    println!("C DCT output:");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:6} ", c_dct[row * 8 + col]);
        }
        println!();
    }

    let mut dct_diff_count = 0;
    for i in 0..DCTSIZE2 {
        if rust_dct[i] != c_dct[i] {
            dct_diff_count += 1;
        }
    }
    println!();
    println!("DCT differences: {}/64", dct_diff_count);
}

fn rust_dct_and_quantize(block: &[i16; DCTSIZE2], qtable: &[u16; DCTSIZE2]) -> [i16; DCTSIZE2] {
    // Forward DCT (output scaled by 8)
    let mut dct_block = [0i16; DCTSIZE2];
    forward_dct_8x8(block, &mut dct_block);

    // Convert to i32 and descale
    let mut dct_i32 = [0i32; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        // Descale the DCT output (remove factor of 8)
        dct_i32[i] = (dct_block[i] as i32 + 4) >> 3;
    }

    // Quantize
    let mut output = [0i16; DCTSIZE2];
    quantize_block(&dct_i32, qtable, &mut output);
    output
}

fn c_dct_and_quantize(block: &[i16; DCTSIZE2], qtable: &[u16; DCTSIZE2]) -> [i16; DCTSIZE2] {
    use sys_local::*;

    unsafe {
        // Copy block to workspace
        let mut workspace = [0i16; DCTSIZE2];
        workspace.copy_from_slice(block);

        // Call C's forward DCT
        mozjpeg_test_fdct_islow(workspace.as_mut_ptr());

        // Descale and quantize (same as Rust)
        let mut output = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            let descaled = (workspace[i] as i32 + 4) >> 3;
            let q = qtable[i] as i32;
            output[i] = if descaled >= 0 {
                ((descaled + q / 2) / q) as i16
            } else {
                ((descaled - q / 2) / q) as i16
            };
        }
        output
    }
}

fn c_dct_only(block: &[i16; DCTSIZE2]) -> [i16; DCTSIZE2] {
    use sys_local::*;

    unsafe {
        let mut workspace = [0i16; DCTSIZE2];
        workspace.copy_from_slice(block);
        mozjpeg_test_fdct_islow(workspace.as_mut_ptr());
        workspace
    }
}
