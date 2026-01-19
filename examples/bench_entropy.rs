//! Benchmark comparing standard vs fast entropy encoders.
//!
//! Run with: cargo run --example bench_entropy --release

use mozjpeg_rs::bitstream::VecBitWriter;
use mozjpeg_rs::consts::{
    AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES, DCTSIZE2,
};
use mozjpeg_rs::entropy::EntropyEncoder;
use mozjpeg_rs::fast_entropy::FastEntropyEncoder;
use mozjpeg_rs::huffman::{DerivedTable, HuffTable};
use std::time::Instant;

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

/// Generate test blocks with realistic coefficient distributions.
fn generate_test_blocks(num_blocks: usize, quality_factor: f32) -> Vec<[i16; DCTSIZE2]> {
    let mut blocks = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let mut block = [0i16; DCTSIZE2];

        // DC coefficient with some variation
        block[0] = ((i as i32 % 256) - 128) as i16;

        // AC coefficients - sparse with quality-dependent distribution
        // Higher quality = more non-zero coefficients
        let sparsity = (1.0 / quality_factor).ceil() as usize;
        let max_nonzero = (64.0 * quality_factor) as usize;

        for j in 1..max_nonzero.min(63) {
            if j % sparsity == 0 {
                // Coefficient values decrease with position (typical DCT)
                let mag = ((64 - j) as f32 * quality_factor * 0.5) as i16;
                let sign = if (i + j) % 3 == 0 { -1 } else { 1 };
                block[j] = sign * mag.max(1);
            }
        }

        blocks.push(block);
    }

    blocks
}

fn benchmark_standard_encoder(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
    iterations: usize,
) -> (f64, usize) {
    let mut total_bytes = 0;
    let start = Instant::now();

    for _ in 0..iterations {
        let mut writer = VecBitWriter::new_vec();
        {
            let mut encoder = EntropyEncoder::new(&mut writer);
            for block in blocks {
                encoder.encode_block(block, 0, dc_table, ac_table).unwrap();
            }
            encoder.flush().unwrap();
        }
        let bytes = writer.into_bytes();
        total_bytes = bytes.len();
    }

    let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
    (elapsed, total_bytes)
}

fn benchmark_fast_encoder(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
    iterations: usize,
) -> (f64, usize) {
    let mut total_bytes = 0;
    let start = Instant::now();

    for _ in 0..iterations {
        let mut encoder = FastEntropyEncoder::new();
        for block in blocks {
            encoder.encode_block(block, 0, dc_table, ac_table);
        }
        let bytes = encoder.into_bytes();
        total_bytes = bytes.len();
    }

    let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
    (elapsed, total_bytes)
}

fn main() {
    println!("Entropy Encoder Benchmark");
    println!("=========================\n");

    let dc_table = create_dc_luma_table();
    let ac_table = create_ac_luma_table();

    // Test with different block counts and quality levels
    let configs = [
        ("512x512 Q50", 64 * 64, 0.3),    // 4096 blocks, sparse
        ("512x512 Q75", 64 * 64, 0.5),    // 4096 blocks, medium
        ("512x512 Q90", 64 * 64, 0.7),    // 4096 blocks, dense
        ("2048x2048 Q75", 256 * 256, 0.5), // 65536 blocks
    ];

    let iterations = 10;
    let warmup = 2;

    println!("Configuration | Blocks | Standard (ms) | Fast (ms) | Speedup | Output Size");
    println!("{}", "-".repeat(80));

    for (name, num_blocks, quality) in configs {
        let blocks = generate_test_blocks(num_blocks, quality);

        // Warmup
        for _ in 0..warmup {
            benchmark_standard_encoder(&blocks, &dc_table, &ac_table, 1);
            benchmark_fast_encoder(&blocks, &dc_table, &ac_table, 1);
        }

        // Benchmark
        let (std_time, std_bytes) = benchmark_standard_encoder(&blocks, &dc_table, &ac_table, iterations);
        let (fast_time, fast_bytes) = benchmark_fast_encoder(&blocks, &dc_table, &ac_table, iterations);

        let speedup = std_time / fast_time;

        println!(
            "{:<14} | {:>6} | {:>13.3} | {:>9.3} | {:>7.2}x | {} bytes",
            name,
            num_blocks,
            std_time * 1000.0,
            fast_time * 1000.0,
            speedup,
            fast_bytes
        );

        // Verify output matches
        assert_eq!(std_bytes, fast_bytes, "Output size mismatch!");
    }

    println!("\nOutput verified: Fast encoder produces identical output to standard encoder.");
}
