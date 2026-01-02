//! Encode test image with all common permutations of encoder settings.
//!
//! This generates JPEG files with DSSIM values in the filename for easy
//! human comparison of quality vs compression tradeoffs.
//!
//! Usage: cargo run --example encode_permutations
//!
//! Output files are saved to mozjpeg/tests/images/encoded/

use dssim::Dssim;
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use png::ColorType;
use rgb::RGB;
use std::fs;
use std::path::Path;

fn main() {
    let input_path = Path::new("mozjpeg/tests/images/1.png");
    let output_dir = Path::new("mozjpeg/tests/images/encoded");

    // Create output directory
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Load the PNG image
    let file = fs::File::open(input_path).expect("Failed to open input image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    let width = info.width;
    let height = info.height;

    // Convert to RGB if necessary
    let rgb_data: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    println!("Loaded {}x{} image from {:?}", width, height, input_path);
    println!("Output directory: {:?}\n", output_dir);

    // Define permutations
    let qualities = [50, 75, 85, 95];
    let subsamplings = [
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];
    let progressive_opts = [(false, "baseline"), (true, "progressive")];
    let trellis_opts = [(false, "notrellis"), (true, "trellis")];
    let huffman_opts = [(false, "stdhuff"), (true, "opthuff")];

    // Prepare DSSIM calculator
    let dssim = Dssim::new();

    // Convert to RGB<u8> slice for DSSIM
    let rgb_pixels: Vec<RGB<u8>> = rgb_data
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();

    let original = dssim
        .create_image_rgb(&rgb_pixels, width as usize, height as usize)
        .expect("Failed to create DSSIM reference");

    let mut results: Vec<(String, usize, f64)> = Vec::new();

    // Generate all permutations
    for quality in &qualities {
        for (subsampling, sub_name) in &subsamplings {
            for (progressive, prog_name) in &progressive_opts {
                for (trellis, trellis_name) in &trellis_opts {
                    for (optimize_huffman, huff_name) in &huffman_opts {
                        // Encode
                        let trellis_config = if *trellis {
                            TrellisConfig::default()
                        } else {
                            TrellisConfig::disabled()
                        };

                        let jpeg_data = Encoder::new(false)
                            .quality(*quality)
                            .subsampling(*subsampling)
                            .progressive(*progressive)
                            .trellis(trellis_config)
                            .optimize_huffman(*optimize_huffman)
                            .encode_rgb(&rgb_data, width, height)
                            .expect("Encoding failed");

                        // Decode and calculate DSSIM
                        let mut decoder =
                            jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
                        let decoded = decoder.decode().expect("Decode failed");
                        let dec_info = decoder.info().unwrap();

                        // Convert decoded bytes to RGB pixels
                        let decoded_pixels: Vec<RGB<u8>> = decoded
                            .chunks(3)
                            .map(|c| RGB::new(c[0], c[1], c[2]))
                            .collect();

                        let decoded_img = dssim
                            .create_image_rgb(
                                &decoded_pixels,
                                dec_info.width as usize,
                                dec_info.height as usize,
                            )
                            .expect("Failed to create decoded image");

                        let (dssim_val, _) = dssim.compare(&original, decoded_img);
                        let dssim_f64: f64 = dssim_val.into();

                        // Create filename with all parameters and DSSIM
                        let filename = format!(
                            "q{:02}_{}_{}_{}_{}_{:.6}.jpg",
                            quality, sub_name, prog_name, trellis_name, huff_name, dssim_f64
                        );

                        let output_path = output_dir.join(&filename);
                        fs::write(&output_path, &jpeg_data).expect("Failed to write JPEG");

                        results.push((filename.clone(), jpeg_data.len(), dssim_f64));

                        println!(
                            "{:50} {:>8} bytes  DSSIM: {:.6}",
                            filename,
                            jpeg_data.len(),
                            dssim_f64
                        );
                    }
                }
            }
        }
    }

    // Print summary sorted by DSSIM
    println!("\n=== Summary (sorted by DSSIM, lower is better) ===\n");
    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    println!("{:50} {:>10} {:>12}", "Filename", "Size", "DSSIM");
    println!("{}", "-".repeat(74));
    for (filename, size, dssim_val) in &results {
        println!("{:50} {:>10} {:>12.6}", filename, size, dssim_val);
    }

    // Print summary sorted by size
    println!("\n=== Summary (sorted by size, smaller is better) ===\n");
    results.sort_by_key(|a| a.1);

    println!("{:50} {:>10} {:>12}", "Filename", "Size", "DSSIM");
    println!("{}", "-".repeat(74));
    for (filename, size, dssim_val) in &results {
        println!("{:50} {:>10} {:>12.6}", filename, size, dssim_val);
    }

    println!(
        "\n{} files written to {:?}",
        results.len(),
        output_dir
            .canonicalize()
            .unwrap_or(output_dir.to_path_buf())
    );
}
