//! Profile entropy encoders with perf.
//!
//! Run with:
//!   cargo build --release --example profile_entropy
//!   perf record -g ./target/release/examples/profile_entropy standard
//!   perf report

use mozjpeg_rs::bitstream::VecBitWriter;
use mozjpeg_rs::consts::{
    QuantTableIdx, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DCTSIZE, DCTSIZE2, DC_LUMINANCE_BITS,
    DC_LUMINANCE_VALUES,
};
use mozjpeg_rs::dct;
use mozjpeg_rs::entropy::EntropyEncoder;
use mozjpeg_rs::fast_entropy::FastEntropyEncoder;
use mozjpeg_rs::huffman::{DerivedTable, HuffTable};
use mozjpeg_rs::quant;
use png::ColorType;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::path::Path;

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

fn load_real_image(path: &Path) -> (Vec<u8>, usize, usize) {
    let file = File::open(path).expect("Failed to open test image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let width = info.width as usize;
    let height = info.height as usize;

    let y_plane: Vec<u8> = match info.color_type {
        ColorType::Grayscale | ColorType::GrayscaleAlpha => {
            let step = if info.color_type == ColorType::GrayscaleAlpha {
                2
            } else {
                1
            };
            buf.iter().step_by(step).copied().collect()
        }
        ColorType::Rgb => buf
            .chunks_exact(3)
            .map(|rgb| {
                let y =
                    (19595 * rgb[0] as u32 + 38470 * rgb[1] as u32 + 7471 * rgb[2] as u32 + 32768)
                        >> 16;
                y.min(255) as u8
            })
            .collect(),
        ColorType::Rgba => buf
            .chunks_exact(4)
            .map(|rgba| {
                let y = (19595 * rgba[0] as u32
                    + 38470 * rgba[1] as u32
                    + 7471 * rgba[2] as u32
                    + 32768)
                    >> 16;
                y.min(255) as u8
            })
            .collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    (y_plane, width, height)
}

fn generate_blocks(
    y_plane: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Vec<[i16; DCTSIZE2]> {
    let (luma_qtable, _) = quant::create_quant_tables(quality, QuantTableIdx::ImageMagick, true);

    let mcu_width = (width + 7) / 8 * 8;
    let mcu_height = (height + 7) / 8 * 8;
    let blocks_h = mcu_width / DCTSIZE;
    let blocks_v = mcu_height / DCTSIZE;

    let mut blocks = Vec::with_capacity(blocks_h * blocks_v);
    let mut dct_block = [0i16; DCTSIZE2];

    for block_row in 0..blocks_v {
        for block_col in 0..blocks_h {
            let mut samples = [0i16; DCTSIZE2];
            for row in 0..DCTSIZE {
                for col in 0..DCTSIZE {
                    let y = block_row * DCTSIZE + row;
                    let x = block_col * DCTSIZE + col;
                    let pixel = if y < height && x < width {
                        y_plane[y * width + x] as i16
                    } else {
                        128
                    };
                    samples[row * DCTSIZE + col] = pixel - 128;
                }
            }

            dct::forward_dct_8x8_i32_wide_transpose(&samples, &mut dct_block);

            let mut quant_block = [0i16; DCTSIZE2];
            let raw: [i32; DCTSIZE2] = std::array::from_fn(|i| dct_block[i] as i32);
            quant::quantize_block(&raw, &luma_qtable.values, &mut quant_block);

            blocks.push(quant_block);
        }
    }

    blocks
}

fn run_standard(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
    iterations: usize,
) {
    for _ in 0..iterations {
        let mut writer = VecBitWriter::new_vec();
        {
            let mut encoder = EntropyEncoder::new(&mut writer);
            for block in blocks {
                encoder
                    .encode_block(black_box(block), 0, dc_table, ac_table)
                    .unwrap();
            }
            encoder.flush().unwrap();
        }
        black_box(writer.into_bytes());
    }
}

fn run_fast(
    blocks: &[[i16; DCTSIZE2]],
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
    iterations: usize,
) {
    for _ in 0..iterations {
        let mut encoder = FastEntropyEncoder::new();
        for block in blocks {
            encoder.encode_block(black_box(block), 0, dc_table, ac_table);
        }
        black_box(encoder.into_bytes());
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("standard");

    let dc_table = create_dc_luma_table();
    let ac_table = create_ac_luma_table();

    // Load real image
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_image = Path::new(manifest_dir).join("tests/images/1.png");
    let (y_plane, width, height) = load_real_image(&test_image);

    // Generate blocks at Q85
    let blocks = generate_blocks(&y_plane, width, height, 85);

    println!("Image: {}x{}, {} blocks", width, height, blocks.len());
    println!("Mode: {}", mode);
    println!("Running 1000 iterations for profiling...");

    let iterations = 1000;

    match mode {
        "standard" => run_standard(&blocks, &dc_table, &ac_table, iterations),
        "fast" => run_fast(&blocks, &dc_table, &ac_table, iterations),
        _ => {
            eprintln!("Usage: profile_entropy [standard|fast]");
            std::process::exit(1);
        }
    }

    println!("Done.");
}
