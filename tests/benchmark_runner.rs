//! Benchmark runner for tracking compression performance across commits.
//!
//! Tracks 3 configurations:
//! 1. Baseline + Trellis (sequential DCT)
//! 2. Progressive + Trellis (progressive DCT)
//! 3. Max Compression (progressive + trellis + optimize_scans)
//!
//! Run with: cargo test --test benchmark_runner --release -- --nocapture

use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use rgb::RGB;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;

/// Quality levels to test (20 values)
const QUALITY_LEVELS: [u8; 20] = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99,
];

/// Encoder configuration mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderMode {
    /// Baseline (sequential) + Trellis + Huffman optimization
    Baseline,
    /// Progressive + Trellis + Huffman optimization
    Progressive,
    /// Max compression: Progressive + Trellis + optimize_scans
    MaxCompression,
}

impl EncoderMode {
    fn name(&self) -> &'static str {
        match self {
            EncoderMode::Baseline => "baseline",
            EncoderMode::Progressive => "progressive",
            EncoderMode::MaxCompression => "max_compression",
        }
    }
}

/// Single quality level result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    pub quality: u8,
    pub size: usize,
    pub dssim: f64,
}

/// Complete benchmark results for one encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub commit: String,
    pub timestamp: String,
    pub image: String,
    pub encoder: String,
    pub mode: String,
    pub results: Vec<QualityResult>,
}

/// Load a PNG image and return RGB data
fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(buf.len() * 3 / 4);
            for chunk in buf.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return Err("Unsupported color type".into()),
    };

    Ok((rgb, info.width, info.height))
}

/// Decode a JPEG and return RGB data
fn decode_jpeg(jpeg_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(jpeg_data));
    let pixels = decoder.decode()?;
    Ok(pixels)
}

/// Calculate DSSIM between original and decoded image
fn calculate_dssim(
    original: &[u8],
    decoded: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    use dssim::Dssim;

    let attr = Dssim::new();

    let orig_rgb: Vec<RGB<u8>> = original
        .chunks(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();

    let dec_rgb: Vec<RGB<u8>> = decoded
        .chunks(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();

    let orig_img = attr
        .create_image_rgb(&orig_rgb, width as usize, height as usize)
        .ok_or("Failed to create original image")?;

    let dec_img = attr
        .create_image_rgb(&dec_rgb, width as usize, height as usize)
        .ok_or("Failed to create decoded image")?;

    let (dssim_val, _) = attr.compare(&orig_img, dec_img);
    Ok(dssim_val.into())
}

/// Encode with Rust mozjpeg-oxide using specified mode
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, mode: EncoderMode) -> Vec<u8> {
    match mode {
        EncoderMode::Baseline => Encoder::baseline_optimized()
            .quality(quality)
            .subsampling(Subsampling::S420)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .overshoot_deringing(true)
            .encode_rgb(rgb, width, height)
            .expect("Rust encoding failed"),

        EncoderMode::Progressive => Encoder::baseline_optimized()
            .quality(quality)
            .subsampling(Subsampling::S420)
            .progressive(true)
            .optimize_scans(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .overshoot_deringing(true)
            .encode_rgb(rgb, width, height)
            .expect("Rust encoding failed"),

        EncoderMode::MaxCompression => Encoder::max_compression()
            .quality(quality)
            .encode_rgb(rgb, width, height)
            .expect("Rust encoding failed"),
    }
}

/// Encode with C mozjpeg using specified mode
#[allow(unsafe_code)]
fn encode_c_mozjpeg(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    mode: EncoderMode,
) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::ptr;

    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_compress_struct>(),
        );

        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);

        // ImageMagick quant tables (index 3)
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Common optimizations
        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);

        // Mode-specific settings
        match mode {
            EncoderMode::Baseline => {
                // No progressive
            }
            EncoderMode::Progressive => {
                jpeg_simple_progression(&mut cinfo);
            }
            EncoderMode::MaxCompression => {
                jpeg_simple_progression(&mut cinfo);
                jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1);
            }
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_ptr = rgb.as_ptr().add(cinfo.next_scanline as usize * row_stride);
            let row_array: [*const u8; 1] = [row_ptr];
            jpeg_write_scanlines(&mut cinfo, row_array.as_ptr(), 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);

        result
    }
}

/// Get current git commit hash
fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get current timestamp
fn get_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Run benchmark for specified encoder and mode
pub fn run_benchmark(
    image_path: &Path,
    encoder_name: &str,
    mode: EncoderMode,
    encode_fn: impl Fn(&[u8], u32, u32, u8, EncoderMode) -> Vec<u8>,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let (rgb, width, height) = load_png(image_path)?;
    let mut results = Vec::new();

    for &quality in &QUALITY_LEVELS {
        let jpeg = encode_fn(&rgb, width, height, quality, mode);
        let decoded = decode_jpeg(&jpeg)?;
        let dssim = calculate_dssim(&rgb, &decoded, width, height)?;

        results.push(QualityResult {
            quality,
            size: jpeg.len(),
            dssim,
        });
    }

    let commit = if encoder_name == "c-mozjpeg" {
        format!("c-mozjpeg-{}", mode.name())
    } else {
        get_git_commit()
    };

    Ok(BenchmarkResults {
        commit,
        timestamp: get_timestamp(),
        image: image_path.to_string_lossy().to_string(),
        encoder: encoder_name.to_string(),
        mode: mode.name().to_string(),
        results,
    })
}

/// Save results to JSON file
pub fn save_results(
    results: &BenchmarkResults,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;

    let filename = format!(
        "{}_{}_{}.json",
        results.encoder, results.mode, results.commit
    );
    let path = output_dir.join(filename);

    let json = serde_json::to_string_pretty(results)?;
    fs::write(&path, json)?;

    println!("Saved: {}", path.display());
    Ok(())
}

/// Print comparison table
pub fn print_comparison(rust: &BenchmarkResults, c: &BenchmarkResults) {
    println!("\n{:=<85}", "");
    println!("{} [{}] vs C mozjpeg [{}]", rust.commit, rust.mode, c.mode);
    println!("{:=<85}", "");
    println!(
        "{:>5} {:>10} {:>10} {:>8} {:>12} {:>12} {:>10}",
        "Q", "Rust", "C", "Δ Size", "Rust DSSIM", "C DSSIM", "Winner"
    );
    println!("{:-<85}", "");

    for (r, c) in rust.results.iter().zip(c.results.iter()) {
        let size_ratio = (r.size as f64 / c.size as f64 - 1.0) * 100.0;
        let winner = if r.size < c.size && r.dssim <= c.dssim {
            "Rust"
        } else if c.size < r.size && c.dssim <= r.dssim {
            "C"
        } else if r.dssim < c.dssim {
            "Rust (q)"
        } else if c.dssim < r.dssim {
            "C (q)"
        } else {
            "="
        };

        println!(
            "{:>5} {:>10} {:>10} {:>+7.2}% {:>12.6} {:>12.6} {:>10}",
            r.quality, r.size, c.size, size_ratio, r.dssim, c.dssim, winner
        );
    }
    println!("{:-<85}", "");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_all_benchmarks() {
        let image_path = Path::new("tests/images/1.png");
        if !image_path.exists() {
            eprintln!("Test image not found, skipping benchmark");
            return;
        }

        let results_dir = Path::new("tests/benchmark_tracking/results");
        let modes = [
            EncoderMode::Baseline,
            EncoderMode::Progressive,
            EncoderMode::MaxCompression,
        ];

        for mode in modes {
            println!("\n### {} mode ###", mode.name().to_uppercase());

            // C mozjpeg
            let c_results = run_benchmark(image_path, "c-mozjpeg", mode, encode_c_mozjpeg)
                .expect("C benchmark failed");
            save_results(&c_results, results_dir).expect("Failed to save C results");

            // Rust
            let rust_results = run_benchmark(image_path, "mozjpeg-oxide", mode, encode_rust)
                .expect("Rust benchmark failed");
            save_results(&rust_results, results_dir).expect("Failed to save Rust results");

            print_comparison(&rust_results, &c_results);
        }
    }
}
