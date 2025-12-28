//! Benchmark runner for tracking compression performance across commits.
//!
//! Run with: cargo test --test benchmark_runner --release -- --nocapture

use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
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

/// Single quality level result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    pub quality: u8,
    pub size: usize,
    pub dssim: f64,
}

/// Complete benchmark results for one encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub commit: String,
    pub timestamp: String,
    pub image: String,
    pub encoder: String,
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

    // Convert to RGB if necessary
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

    // Convert to RGB<u8> slices
    let orig_rgb: Vec<RGB<u8>> = original
        .chunks(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();

    let dec_rgb: Vec<RGB<u8>> = decoded
        .chunks(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
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

/// Encode with Rust mozjpeg-oxide
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::new()
        .quality(quality)
        .subsampling(Subsampling::S420)
        .progressive(false)
        .optimize_huffman(true)
        .trellis(TrellisConfig::default())
        .overshoot_deringing(true)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

/// Encode with C mozjpeg
#[allow(unsafe_code)]
fn encode_c_mozjpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // Optimizations (matching Rust config)
        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);

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

/// Run benchmark for Rust encoder
pub fn run_rust_benchmark(
    image_path: &Path,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let (rgb, width, height) = load_png(image_path)?;
    let mut results = Vec::new();

    for &quality in &QUALITY_LEVELS {
        let jpeg = encode_rust(&rgb, width, height, quality);
        let decoded = decode_jpeg(&jpeg)?;
        let dssim = calculate_dssim(&rgb, &decoded, width, height)?;

        results.push(QualityResult {
            quality,
            size: jpeg.len(),
            dssim,
        });
    }

    Ok(BenchmarkResults {
        commit: get_git_commit(),
        timestamp: get_timestamp(),
        image: image_path.to_string_lossy().to_string(),
        encoder: "mozjpeg-oxide".to_string(),
        results,
    })
}

/// Run benchmark for C mozjpeg
pub fn run_c_benchmark(
    image_path: &Path,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let (rgb, width, height) = load_png(image_path)?;
    let mut results = Vec::new();

    for &quality in &QUALITY_LEVELS {
        let jpeg = encode_c_mozjpeg(&rgb, width, height, quality);
        let decoded = decode_jpeg(&jpeg)?;
        let dssim = calculate_dssim(&rgb, &decoded, width, height)?;

        results.push(QualityResult {
            quality,
            size: jpeg.len(),
            dssim,
        });
    }

    Ok(BenchmarkResults {
        commit: "c-mozjpeg-baseline".to_string(),
        timestamp: get_timestamp(),
        image: image_path.to_string_lossy().to_string(),
        encoder: "c-mozjpeg".to_string(),
        results,
    })
}

/// Save results to JSON file
pub fn save_results(
    results: &BenchmarkResults,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;

    let filename = format!("{}_{}.json", results.encoder, results.commit);
    let path = output_dir.join(filename);

    let json = serde_json::to_string_pretty(results)?;
    fs::write(&path, json)?;

    println!("Saved results to: {}", path.display());
    Ok(())
}

/// Print results as a table
pub fn print_results(rust: &BenchmarkResults, c: &BenchmarkResults) {
    println!("\n{:=<80}", "");
    println!("Benchmark Results: {} vs C mozjpeg", rust.commit);
    println!("{:=<80}", "");
    println!(
        "{:>5} {:>10} {:>10} {:>8} {:>12} {:>12} {:>8}",
        "Q", "Rust Size", "C Size", "Ratio", "Rust DSSIM", "C DSSIM", "Quality"
    );
    println!("{:-<80}", "");

    for (r, c) in rust.results.iter().zip(c.results.iter()) {
        let ratio = r.size as f64 / c.size as f64;
        let quality_cmp = if r.dssim < c.dssim {
            "Rust+"
        } else if r.dssim > c.dssim {
            "C+"
        } else {
            "="
        };

        println!(
            "{:>5} {:>10} {:>10} {:>7.2}% {:>12.6} {:>12.6} {:>8}",
            r.quality,
            r.size,
            c.size,
            (ratio - 1.0) * 100.0,
            r.dssim,
            c.dssim,
            quality_cmp
        );
    }
    println!("{:-<80}", "");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_benchmark_and_save() {
        let image_path = Path::new("tests/images/1.png");
        if !image_path.exists() {
            eprintln!("Test image not found, skipping benchmark");
            return;
        }

        let results_dir = Path::new("tests/benchmark_tracking/results");

        // Run C baseline
        println!("Running C mozjpeg baseline...");
        let c_results = run_c_benchmark(image_path).expect("C benchmark failed");
        save_results(&c_results, results_dir).expect("Failed to save C results");

        // Run Rust benchmark
        println!("Running Rust benchmark...");
        let rust_results = run_rust_benchmark(image_path).expect("Rust benchmark failed");
        save_results(&rust_results, results_dir).expect("Failed to save Rust results");

        // Print comparison
        print_results(&rust_results, &c_results);
    }
}
