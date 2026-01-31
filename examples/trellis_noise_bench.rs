//! Benchmark TrellisSpeedMode on noise-only images where trellis works hardest.
//!
//! Noise images have maximum entropy (many nonzero coefficients), so the speed
//! limiter fires on nearly every block. This isolates the trellis speed tradeoff.
//!
//! No corpus required - uses 12 deterministic synthetic noise images.

use dssim::Dssim;
use mozjpeg_rs::{Encoder, Preset, Subsampling, TrellisConfig, TrellisSpeedMode};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const OUTPUT_DIR: &str = "/mnt/v/output/mozjpeg-rs/trellis-noise-bench";
const WARMUP_ITERS: usize = 2;
const BENCH_ITERS: usize = 7;

fn main() {
    fs::create_dir_all(OUTPUT_DIR).unwrap();

    let images: Vec<(&str, Vec<u8>, u32, u32)> = vec![
        (
            "uniform_noise",
            make_uniform_noise(512, 512, 0xDEADBEEF),
            512,
            512,
        ),
        (
            "uniform_noise_2",
            make_uniform_noise(512, 512, 0x12345678),
            512,
            512,
        ),
        (
            "uniform_noise_3",
            make_uniform_noise(512, 512, 0xFEEDFACE),
            512,
            512,
        ),
        (
            "gaussian_mid",
            make_gaussian_noise(512, 512, 128, 40, 0xCAFEBABE),
            512,
            512,
        ),
        (
            "gaussian_bright",
            make_gaussian_noise(512, 512, 200, 30, 0xABCD1234),
            512,
            512,
        ),
        (
            "gaussian_dark",
            make_gaussian_noise(512, 512, 50, 25, 0x98765432),
            512,
            512,
        ),
        (
            "high_freq_checker",
            make_high_freq_checker(512, 512),
            512,
            512,
        ),
        (
            "high_freq_diag",
            make_high_freq_diagonal(512, 512),
            512,
            512,
        ),
        (
            "noisy_gradient",
            make_noisy_gradient(512, 512, 0xBEEF),
            512,
            512,
        ),
        (
            "film_grain_light",
            make_film_grain(512, 512, 180, 12, 0x1111),
            512,
            512,
        ),
        (
            "film_grain_heavy",
            make_film_grain(512, 512, 128, 40, 0x2222),
            512,
            512,
        ),
        (
            "salt_pepper",
            make_salt_pepper(512, 512, 0.05, 0x3333),
            512,
            512,
        ),
    ];

    println!("=== Trellis Speed Mode: Noise Images Only ===");
    println!("{} images, {}x{}", images.len(), 512, 512);
    println!(
        "{} warmup + {} timed iterations per encode",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!();

    let modes: Vec<(&str, TrellisSpeedMode)> = vec![
        ("Thorough", TrellisSpeedMode::Thorough),
        ("Adaptive", TrellisSpeedMode::Adaptive),
        ("Level(0)", TrellisSpeedMode::Level(0)),
        ("Level(3)", TrellisSpeedMode::Level(3)),
        ("Level(5)", TrellisSpeedMode::Level(5)),
        ("Level(7)", TrellisSpeedMode::Level(7)),
        ("Level(10)", TrellisSpeedMode::Level(10)),
    ];

    let qualities = [75u8, 85, 90, 95, 100];

    let csv_path = Path::new(OUTPUT_DIR).join("noise_results.csv");
    let mut csv = File::create(&csv_path).unwrap();
    writeln!(
        csv,
        "image,quality,mode,size_bytes,time_us,dssim,size_vs_thorough_pct,pixel_diff_count,pixel_max_diff"
    )
    .unwrap();

    let dssim = Dssim::new();

    for quality in qualities {
        println!("================ Quality {} ================", quality);
        println!();

        println!(
            "{:<22} {:<12} {:>8} {:>9} {:>10} {:>10} {:>8} {:>6}",
            "Image", "Mode", "Size", "Time us", "DSSIM", "Size %", "PixD", "MaxD"
        );
        println!("{}", "-".repeat(100));

        let mut mode_totals: Vec<(usize, u128, f64, usize, i32)> =
            vec![(0, 0, 0.0, 0, 0); modes.len()];

        for (img_name, rgb, w, h) in &images {
            let rgb_pixels: Vec<rgb::RGB<u8>> = rgb
                .chunks(3)
                .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
                .collect();
            let original = dssim
                .create_image_rgb(&rgb_pixels, *w as usize, *h as usize)
                .unwrap();

            let thorough_jpeg = encode_with_mode(rgb, *w, *h, quality, TrellisSpeedMode::Thorough);

            for (mi, (mode_name, mode)) in modes.iter().enumerate() {
                for _ in 0..WARMUP_ITERS {
                    let _ = encode_with_mode(rgb, *w, *h, quality, *mode);
                }

                let start = Instant::now();
                let mut jpeg = Vec::new();
                for _ in 0..BENCH_ITERS {
                    jpeg = encode_with_mode(rgb, *w, *h, quality, *mode);
                }
                let elapsed_us = start.elapsed().as_micros() / BENCH_ITERS as u128;

                let decoded = decode_jpeg(&jpeg);
                let dec_pixels: Vec<rgb::RGB<u8>> = decoded
                    .chunks(3)
                    .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
                    .collect();
                let decoded_img = dssim
                    .create_image_rgb(&dec_pixels, *w as usize, *h as usize)
                    .unwrap();
                let (dssim_val, _) = dssim.compare(&original, decoded_img);
                let dssim_f: f64 = dssim_val.into();

                let (diff_count, max_diff) = compare_pixels(&thorough_jpeg, &jpeg);

                let size_pct = if mi == 0 {
                    0.0
                } else {
                    (jpeg.len() as f64 / thorough_jpeg.len() as f64 - 1.0) * 100.0
                };

                mode_totals[mi].0 += jpeg.len();
                mode_totals[mi].1 += elapsed_us;
                mode_totals[mi].2 += dssim_f;
                mode_totals[mi].3 += diff_count;
                mode_totals[mi].4 = mode_totals[mi].4.max(max_diff);

                println!(
                    "{:<22} {:<12} {:>8} {:>9} {:>10.6} {:>+9.3}% {:>8} {:>6}",
                    img_name,
                    mode_name,
                    jpeg.len(),
                    elapsed_us,
                    dssim_f,
                    size_pct,
                    diff_count,
                    max_diff
                );

                writeln!(
                    csv,
                    "{},{},{},{},{},{:.8},{:.4},{},{}",
                    img_name,
                    quality,
                    mode_name,
                    jpeg.len(),
                    elapsed_us,
                    dssim_f,
                    size_pct,
                    diff_count,
                    max_diff
                )
                .unwrap();
            }
        }

        // Summary
        let n = images.len();
        let thorough_size = mode_totals[0].0;
        let thorough_time = mode_totals[0].1;

        println!();
        println!("--- SUMMARY Q{} ({} noise images) ---", quality, n);
        println!(
            "{:<12} {:>10} {:>9} {:>10} {:>10} {:>10} {:>6}",
            "Mode", "Avg Size", "Avg us", "Avg DSSIM", "Size %", "Tot PixD", "MaxD"
        );
        println!("{}", "-".repeat(72));

        for (mi, (mode_name, _)) in modes.iter().enumerate() {
            let avg_size = mode_totals[mi].0 / n;
            let avg_time = mode_totals[mi].1 / n as u128;
            let avg_dssim = mode_totals[mi].2 / n as f64;
            let size_pct = if mi == 0 {
                0.0
            } else {
                (mode_totals[mi].0 as f64 / thorough_size as f64 - 1.0) * 100.0
            };
            let speedup = if mi == 0 {
                1.0
            } else {
                thorough_time as f64 / mode_totals[mi].1 as f64
            };

            println!(
                "{:<12} {:>10} {:>8} {:>10.6} {:>+9.3}% {:>10} {:>6}  ({:.2}x speed)",
                mode_name,
                avg_size,
                avg_time,
                avg_dssim,
                size_pct,
                mode_totals[mi].3,
                mode_totals[mi].4,
                speedup
            );
        }
        println!();
    }

    println!("CSV written to {:?}", csv_path);
}

fn encode_with_mode(rgb: &[u8], w: u32, h: u32, quality: u8, mode: TrellisSpeedMode) -> Vec<u8> {
    Encoder::new(Preset::BaselineBalanced)
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .subsampling(Subsampling::S420)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default().speed_mode(mode))
        .encode_rgb(rgb, w, h)
        .expect("encode failed")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode failed")
}

fn compare_pixels(ref_jpeg: &[u8], test_jpeg: &[u8]) -> (usize, i32) {
    let ref_dec = decode_jpeg(ref_jpeg);
    let test_dec = decode_jpeg(test_jpeg);
    let mut diff_count = 0usize;
    let mut max_diff = 0i32;
    for (a, b) in ref_dec.iter().zip(test_dec.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > 0 {
            diff_count += 1;
        }
        max_diff = max_diff.max(d);
    }
    (diff_count, max_diff)
}

// ============ Noise generators ============

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed >> 33
}

fn make_uniform_noise(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for byte in rgb.iter_mut() {
        *byte = lcg(&mut s) as u8;
    }
    rgb
}

fn make_gaussian_noise(w: u32, h: u32, mean: u8, stddev: u8, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for byte in rgb.iter_mut() {
        let sum: i32 = (0..4).map(|_| lcg(&mut s) as u8 as i32).sum();
        let centered = sum / 4;
        let val = mean as i32 + ((centered - 128) * stddev as i32) / 64;
        *byte = val.clamp(0, 255) as u8;
    }
    rgb
}

fn make_high_freq_checker(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let val = if (x + y) % 2 == 0 { 200u8 } else { 55u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_high_freq_diagonal(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let val = if (x + y * 3) % 4 < 2 { 220u8 } else { 35u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_noisy_gradient(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let base = ((x as f32 / w as f32) * 255.0) as i32;
            let noise = (lcg(&mut s) % 30) as i32 - 15;
            let val = (base + noise).clamp(0, 255) as u8;
            let i = (y * w + x) as usize * 3;
            rgb[i] = val;
            rgb[i + 1] = val;
            rgb[i + 2] = val;
        }
    }
    rgb
}

fn make_film_grain(w: u32, h: u32, base: u8, grain: u8, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for byte in rgb.iter_mut() {
        let noise = (lcg(&mut s) % (grain as u64 * 2 + 1)) as i32 - grain as i32;
        *byte = (base as i32 + noise).clamp(0, 255) as u8;
    }
    rgb
}

fn make_salt_pepper(w: u32, h: u32, density: f64, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut rgb = vec![128u8; (w * h * 3) as usize];
    let threshold = (density * u32::MAX as f64) as u64;
    for pixel in rgb.chunks_mut(3) {
        let r = lcg(&mut s);
        if r < threshold {
            let val = if r % 2 == 0 { 0u8 } else { 255u8 };
            pixel[0] = val;
            pixel[1] = val;
            pixel[2] = val;
        }
    }
    rgb
}
