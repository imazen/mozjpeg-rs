//! Benchmark TrellisSpeedMode variants across CID22-512 corpus + synthetic noise.
//!
//! Measures: file size, DSSIM, encoding time, pixel differences vs Thorough.
//!
//! Requires CID22-512 corpus. Set CODEC_CORPUS_DIR or MOZJPEG_CORPUS_DIR to the
//! codec-corpus root (expects CID22/CID22-512/validation/ underneath).

use dssim::Dssim;
use mozjpeg_rs::corpus::{corpus_dir, png_files_in_dir};
use mozjpeg_rs::{Encoder, Preset, Subsampling, TrellisConfig, TrellisSpeedMode};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const OUTPUT_DIR: &str = "/mnt/v/output/mozjpeg-rs/trellis-speed-bench";
const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 5;

fn main() {
    fs::create_dir_all(OUTPUT_DIR).unwrap();

    let mut images: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();

    // Load CID22-512 validation set via corpus module
    let corpus = corpus_dir().unwrap_or_else(|| {
        eprintln!("ERROR: No corpus directory found.");
        eprintln!("Set CODEC_CORPUS_DIR to your codec-corpus root");
        eprintln!("  (expects CID22/CID22-512/validation/ underneath).");
        eprintln!("Example: CODEC_CORPUS_DIR=~/work/codec-eval/codec-corpus cargo run ...");
        std::process::exit(1);
    });
    let cid22_dir = corpus.join("CID22").join("CID22-512").join("validation");
    if !cid22_dir.is_dir() {
        eprintln!(
            "ERROR: CID22-512 validation dir not found at {:?}",
            cid22_dir
        );
        eprintln!("Corpus root is {:?}", corpus);
        std::process::exit(1);
    }

    let png_files = png_files_in_dir(&cid22_dir);
    for path in &png_files {
        if let Some((rgb, w, h)) = load_png(path) {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            images.push((name, rgb, w, h));
        }
    }
    println!("Loaded {} CID22-512 images", images.len());

    // Generate synthetic noise images (512x512)
    images.push((
        "synth_uniform_noise.png".into(),
        make_uniform_noise(512, 512),
        512,
        512,
    ));
    images.push((
        "synth_gaussian_noise.png".into(),
        make_gaussian_noise(512, 512),
        512,
        512,
    ));
    images.push((
        "synth_high_freq.png".into(),
        make_high_freq(512, 512),
        512,
        512,
    ));
    images.push((
        "synth_flat_gray.png".into(),
        make_flat(512, 512, 128),
        512,
        512,
    ));

    println!("Total images: {}", images.len());
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

    let qualities = [75u8, 85, 90, 95];

    let csv_path = Path::new(OUTPUT_DIR).join("results.csv");
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
            "{:<30} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Mode", "Size", "Time us", "DSSIM", "Size %", "PixDiff", "MaxD"
        );
        println!("{}", "-".repeat(92));

        // Aggregate stats per mode
        let mut mode_totals: Vec<(usize, u128, f64, usize, i32)> =
            vec![(0, 0, 0.0, 0, 0); modes.len()];

        for (img_name, rgb, w, h) in &images {
            let rgb_pixels: Vec<rgb::RGB<u8>> = rgb
                .chunks(3)
                .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
                .collect();
            let original = dssim
                .create_image_rgb(&rgb_pixels, *w as usize, *h as usize)
                .expect("dssim create_image failed");

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
                    .expect("dssim decoded image failed");
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

        // Print aggregated summary
        let n = images.len();
        let thorough_size = mode_totals[0].0;

        for (mi, (mode_name, _)) in modes.iter().enumerate() {
            let avg_size = mode_totals[mi].0 / n;
            let avg_time = mode_totals[mi].1 / n as u128;
            let avg_dssim = mode_totals[mi].2 / n as f64;
            let total_diff = mode_totals[mi].3;
            let max_diff = mode_totals[mi].4;
            let size_pct = if mi == 0 {
                0.0
            } else {
                (mode_totals[mi].0 as f64 / thorough_size as f64 - 1.0) * 100.0
            };

            println!(
                "{:<30} {:>8} {:>8} {:>10.6} {:>+9.3}% {:>8} {:>8}",
                mode_name, avg_size, avg_time, avg_dssim, size_pct, total_diff, max_diff
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

fn load_png(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

// ============ Synthetic image generators ============

fn make_uniform_noise(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed: u64 = 0xDEADBEEF;
    for byte in rgb.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *byte = (seed >> 33) as u8;
    }
    rgb
}

fn make_gaussian_noise(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed: u64 = 0xCAFEBABE;
    for chunk in rgb.chunks_mut(6) {
        for byte in chunk.iter_mut() {
            let mut sum = 0u32;
            for _ in 0..3 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                sum += (seed >> 33) as u32 & 0xFF;
            }
            *byte = (sum / 3).min(255) as u8;
        }
    }
    rgb
}

fn make_high_freq(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let base = if (x + y) % 2 == 0 { 200u8 } else { 55u8 };
            let i = (y * w + x) as usize * 3;
            rgb[i] = base;
            rgb[i + 1] = base;
            rgb[i + 2] = base;
        }
    }
    rgb
}

fn make_flat(w: u32, h: u32, val: u8) -> Vec<u8> {
    vec![val; (w * h * 3) as usize]
}
