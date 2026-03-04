#![allow(deprecated)] // Uses deprecated avx2_archmage module for overflow testing
//! Detect i16 DCT overflow with synthetic worst-case patterns.
//!
//! Half-black/half-white blocks within a single 8x8 DCT block produce the largest
//! possible AC coefficients. When using i16 DCT + overshoot deringing at low quality,
//! intermediate values can overflow i16 range, causing sign flips visible as
//! catastrophic pixel errors.
//!
//! Compares i16 (AVX2) vs i32 (reference) DCT output to detect overflow artifacts.

use mozjpeg_rs::simd::SimdOps;
use mozjpeg_rs::{Encoder, Preset, Subsampling, TrellisConfig};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

fn mozjpeg_output_dir() -> String {
    std::env::var("MOZJPEG_RS_OUTPUT_DIR").unwrap_or_else(|_| "/mnt/v/output/mozjpeg-rs".into())
}

fn output_dir() -> String {
    format!("{}/synth-overflow", mozjpeg_output_dir())
}

fn main() {
    let out_dir = output_dir();
    fs::create_dir_all(&out_dir).unwrap();

    let i16_ops = match SimdOps::avx2_i16() {
        Some(ops) => ops,
        None => {
            eprintln!("AVX2 not available on this CPU, cannot run i16 overflow test.");
            std::process::exit(1);
        }
    };
    // Use avx2_archmage (i32 DCT, same scalar color conversion as i16)
    // NOT detect() which uses the yuv crate for color conversion —
    // different color conversion would mask DCT overflow with ±1 rounding diffs.
    let i32_ops = SimdOps::avx2_archmage().expect("AVX2 not available");

    let patterns: Vec<(&str, Vec<u8>, u32, u32)> = vec![
        (
            "left_black_right_white",
            make_vertical_split(64, 64, false),
            64,
            64,
        ),
        (
            "left_white_right_black",
            make_vertical_split(64, 64, true),
            64,
            64,
        ),
        (
            "top_black_bottom_white",
            make_horizontal_split(64, 64, false),
            64,
            64,
        ),
        (
            "top_white_bottom_black",
            make_horizontal_split(64, 64, true),
            64,
            64,
        ),
        ("checkerboard_8x8", make_checkerboard(64, 64), 64, 64),
        ("single_8x8_half", make_single_block_half(8, 8), 8, 8),
    ];

    // Save patterns as PNG
    let output_dir = Path::new(&out_dir);
    for (name, rgb, w, h) in &patterns {
        save_png(&output_dir.join(format!("{}.png", name)), rgb, *w, *h);
    }

    println!("=== Synthetic DCT Overflow Test ===");
    println!("Pattern: half-black/half-white blocks that trigger PR #453 overflow");
    println!("i16 variant: {}", i16_ops.dct_variant_name());
    println!("i32 variant: {}", i32_ops.dct_variant_name());
    println!();

    // Debug: compare raw DCT output for the failing pattern
    {
        use mozjpeg_rs::consts::DCTSIZE2;
        use mozjpeg_rs::dct::{avx2_archmage, level_shift};
        use mozjpeg_rs::deringing::preprocess_deringing;

        // left_white_right_black: cols 0-3 = 255, cols 4-7 = 0
        let mut pixels = [0u8; DCTSIZE2];
        for row in 0..8usize {
            for col in 0..4usize {
                pixels[row * 8 + col] = 255;
            }
        }

        let mut shifted = [0i16; DCTSIZE2];
        level_shift(&pixels, &mut shifted);
        // Apply deringing with q25 DC quant (typical value ~16)
        preprocess_deringing(&mut shifted, 16);

        println!("After deringing (left_white_right_black):");
        for row in 0..8 {
            println!("  {:?}", &shifted[row * 8..(row + 1) * 8]);
        }

        use archmage::SimdToken;
        let token = archmage::X64V3Token::try_new().unwrap();
        let mut coeffs_i16 = [0i16; DCTSIZE2];
        let mut coeffs_i32 = [0i16; DCTSIZE2];
        #[allow(deprecated)]
        {
            avx2_archmage::forward_dct_8x8_i16(token, &shifted, &mut coeffs_i16);
            avx2_archmage::forward_dct_8x8_i32(token, &shifted, &mut coeffs_i32);
        }

        println!("\ni16 coefficients:");
        for row in 0..8 {
            print!("  ");
            for col in 0..8 {
                print!("{:7}", coeffs_i16[row * 8 + col]);
            }
            println!();
        }
        println!("i32 coefficients:");
        for row in 0..8 {
            print!("  ");
            for col in 0..8 {
                print!("{:7}", coeffs_i32[row * 8 + col]);
            }
            println!();
        }
        println!("\nDCT differences (i16 vs i32):");
        let mut max_diff = 0i32;
        for i in 0..DCTSIZE2 {
            let d = coeffs_i16[i] as i32 - coeffs_i32[i] as i32;
            if d.abs() > 1 {
                println!(
                    "  [{},{}]: i16={:6} i32={:6} diff={:+}",
                    i / 8,
                    i % 8,
                    coeffs_i16[i],
                    coeffs_i32[i],
                    d
                );
            }
            max_diff = max_diff.max(d.abs());
        }
        println!("Max DCT coeff difference: {}", max_diff);
        println!();
    }

    for quality in [25u8, 50] {
        println!("========== Quality {} ==========", quality);
        println!();

        for (name, rgb, w, h) in &patterns {
            // i16 + deringing (vulnerable path)
            let i16_jpg = Encoder::new(Preset::BaselineBalanced)
                .quality(quality)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::disabled())
                .overshoot_deringing(true)
                .subsampling(Subsampling::S444)
                .simd_ops(i16_ops)
                .encode_rgb(rgb, *w, *h)
                .expect("i16 encode failed");

            // i32 + deringing (reference path)
            let i32_jpg = Encoder::new(Preset::BaselineBalanced)
                .quality(quality)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::disabled())
                .overshoot_deringing(true)
                .subsampling(Subsampling::S444)
                .simd_ops(i32_ops)
                .encode_rgb(rgb, *w, *h)
                .expect("i32 encode failed");

            let p_i16 = output_dir.join(format!("{}_q{}_{}.jpg", name, quality, "i16"));
            let p_i32 = output_dir.join(format!("{}_q{}_{}.jpg", name, quality, "i32"));
            File::create(&p_i16).unwrap().write_all(&i16_jpg).unwrap();
            File::create(&p_i32).unwrap().write_all(&i32_jpg).unwrap();

            // Decode both and compare
            let dec_i16 = decode_jpeg(&i16_jpg);
            let dec_i32 = decode_jpeg(&i32_jpg);

            let mut max_diff = 0u8;
            let mut diff_count = 0usize;
            for (a, b) in dec_i16.iter().zip(dec_i32.iter()) {
                let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
                if d > max_diff {
                    max_diff = d;
                }
                if d > 64 {
                    diff_count += 1;
                }
            }

            let status = if max_diff > 128 {
                "*** SIGN FLIP ***"
            } else if max_diff > 10 {
                "differs"
            } else {
                "OK"
            };
            println!(
                "  {:<30} max_diff={:>3}  pixels>64={:>5}  {}",
                name, max_diff, diff_count, status
            );
        }
        println!();
    }

    println!("Files saved to {:?}", output_dir);
    println!();
    println!("To view comparisons:");
    println!("  feh {}/*_q25_*.jpg", out_dir);
}

fn make_vertical_split(w: u32, h: u32, invert: bool) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let block_x = x % 8;
            let bright = if invert { block_x < 4 } else { block_x >= 4 };
            let val = if bright { 255u8 } else { 0u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_horizontal_split(w: u32, h: u32, invert: bool) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let block_y = y % 8;
            let bright = if invert { block_y < 4 } else { block_y >= 4 };
            let val = if bright { 255u8 } else { 0u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_checkerboard(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let bx = x / 8;
            let by = y / 8;
            let bright = (bx + by) % 2 == 0;
            let val = if bright { 255u8 } else { 0u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_single_block_half(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 4..w {
            let i = (y * w + x) as usize * 3;
            rgb[i] = 255;
            rgb[i + 1] = 255;
            rgb[i + 2] = 255;
        }
    }
    rgb
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("Failed to decode JPEG")
}

fn save_png(path: &Path, rgb: &[u8], w: u32, h: u32) {
    let file = File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgb).unwrap();
}
