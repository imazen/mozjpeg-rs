//! Pareto Front Benchmark: Rust mozjpeg-oxide vs C mozjpeg
//!
//! Generates quality vs file size data for Pareto front visualization.
//! Measures SSIMULACRA2 and DSSIM perceptual quality metrics.
//!
//! Usage:
//!   cargo run --release --example pareto_benchmark -- [OPTIONS]
//!
//! Options:
//!   --corpus PATH    Path to corpus directory (default: ./corpus)
//!   --output PATH    Output CSV file (default: benchmark_results.csv)
//!   --qualities      Comma-separated quality levels (default: 20,30,40,50,60,70,75,80,85,90,95)
//!   --kodak-only     Only use Kodak corpus
//!   --clic-only      Only use CLIC corpus

use mozjpeg_oxide::Encoder;
use mozjpeg_sys::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::Instant;

// Re-export ssimulacra2 types
use ssimulacra2::{ColorPrimaries, Rgb as Ssim2Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

/// Result of encoding a single image at a specific quality
#[derive(Debug, Clone)]
struct EncodingResult {
    corpus: String,
    image: String,
    encoder: String,
    quality: u8,
    file_size: usize,
    bpp: f64, // bits per pixel
    ssimulacra2: f64,
    dssim: f64,
    encode_time_ms: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut corpus_path = PathBuf::from("corpus");
    let mut output_path = PathBuf::from("benchmark_results.csv");
    let mut qualities: Vec<u8> = vec![20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95];
    let mut kodak_only = false;
    let mut clic_only = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                i += 1;
                corpus_path = PathBuf::from(&args[i]);
            }
            "--output" => {
                i += 1;
                output_path = PathBuf::from(&args[i]);
            }
            "--qualities" => {
                i += 1;
                qualities = args[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--kodak-only" => kodak_only = true,
            "--clic-only" => clic_only = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("Pareto Front Benchmark: mozjpeg-oxide vs C mozjpeg");
    println!("===================================================");
    println!();

    // Collect images from corpora
    let mut images: Vec<(String, PathBuf)> = Vec::new();

    if !clic_only {
        let kodak_dir = corpus_path.join("kodak");
        if kodak_dir.is_dir() {
            for entry in fs::read_dir(&kodak_dir).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().map(|e| e == "png").unwrap_or(false) {
                    images.push(("kodak".to_string(), path));
                }
            }
            println!("Kodak corpus: {} images", images.iter().filter(|(c, _)| c == "kodak").count());
        } else {
            eprintln!("Warning: Kodak corpus not found at {:?}", kodak_dir);
        }
    }

    if !kodak_only {
        let clic_dir = corpus_path.join("clic2025").join("validation");
        if clic_dir.is_dir() {
            let clic_count_before = images.len();
            for entry in fs::read_dir(&clic_dir).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().map(|e| e == "png").unwrap_or(false) {
                    images.push(("clic".to_string(), path));
                }
            }
            println!("CLIC corpus: {} images", images.len() - clic_count_before);
        } else {
            eprintln!("Warning: CLIC corpus not found at {:?}", clic_dir);
        }
    }

    if images.is_empty() {
        eprintln!("Error: No images found in corpus");
        std::process::exit(1);
    }

    // Sort for reproducibility
    images.sort_by(|a, b| a.1.file_name().cmp(&b.1.file_name()));

    println!("Quality levels: {:?}", qualities);
    println!("Total configurations: {}", images.len() * qualities.len() * 2);
    println!();

    let mut results: Vec<EncodingResult> = Vec::new();
    let total = images.len() * qualities.len() * 2;
    let mut completed = 0;

    for (corpus, image_path) in &images {
        // Load image
        let (rgb, width, height) = match load_png(image_path) {
            Some(data) => data,
            None => {
                eprintln!("Failed to load: {:?}", image_path);
                continue;
            }
        };

        let image_name = image_path.file_name().unwrap().to_string_lossy().to_string();
        let pixels = (width * height) as f64;

        for &quality in &qualities {
            // Encode with Rust
            let start = Instant::now();
            let rust_jpeg = encode_rust(&rgb, width, height, quality);
            let rust_time = start.elapsed().as_secs_f64() * 1000.0;

            // Decode Rust JPEG and compute metrics
            let rust_decoded = decode_jpeg(&rust_jpeg);
            let (rust_ssim2, rust_dssim) = compute_metrics(&rgb, &rust_decoded, width, height);

            results.push(EncodingResult {
                corpus: corpus.clone(),
                image: image_name.clone(),
                encoder: "rust".to_string(),
                quality,
                file_size: rust_jpeg.len(),
                bpp: (rust_jpeg.len() * 8) as f64 / pixels,
                ssimulacra2: rust_ssim2,
                dssim: rust_dssim,
                encode_time_ms: rust_time,
            });

            completed += 1;
            print!("\rProgress: {}/{} ({:.1}%)", completed, total, 100.0 * completed as f64 / total as f64);
            std::io::stdout().flush().ok();

            // Encode with C mozjpeg
            let start = Instant::now();
            let c_jpeg = encode_c(&rgb, width, height, quality);
            let c_time = start.elapsed().as_secs_f64() * 1000.0;

            // Decode C JPEG and compute metrics
            let c_decoded = decode_jpeg(&c_jpeg);
            let (c_ssim2, c_dssim) = compute_metrics(&rgb, &c_decoded, width, height);

            results.push(EncodingResult {
                corpus: corpus.clone(),
                image: image_name.clone(),
                encoder: "c".to_string(),
                quality,
                file_size: c_jpeg.len(),
                bpp: (c_jpeg.len() * 8) as f64 / pixels,
                ssimulacra2: c_ssim2,
                dssim: c_dssim,
                encode_time_ms: c_time,
            });

            completed += 1;
            print!("\rProgress: {}/{} ({:.1}%)", completed, total, 100.0 * completed as f64 / total as f64);
            std::io::stdout().flush().ok();
        }
    }

    println!("\n");

    // Write CSV
    write_csv(&output_path, &results).expect("Failed to write CSV");
    println!("Results written to: {:?}", output_path);

    // Print summary
    print_summary(&results, &qualities);
}

fn print_usage() {
    println!("Pareto Front Benchmark: mozjpeg-oxide vs C mozjpeg");
    println!();
    println!("Usage: cargo run --release --example pareto_benchmark -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --corpus PATH    Path to corpus directory (default: ./corpus)");
    println!("  --output PATH    Output CSV file (default: benchmark_results.csv)");
    println!("  --qualities Q    Comma-separated quality levels");
    println!("  --kodak-only     Only use Kodak corpus");
    println!("  --clic-only      Only use CLIC corpus");
    println!("  --help, -h       Show this help");
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
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

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use mozjpeg_oxide::TrellisConfig;

    // Use settings that match C mozjpeg configuration:
    // - Progressive mode with standard scans
    // - Trellis quantization (AC + DC)
    // - Optimized Huffman tables
    // - Overshoot deringing
    // Note: optimize_scans is disabled for both to ensure fair comparison
    Encoder::new()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default()) // Enable trellis AC + DC
        .optimize_scans(false) // Disabled for fair comparison
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // Use JCP_MAX_COMPRESSION equivalent settings
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling (default)
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Enable optimizations
        cinfo.optimize_coding = 1;

        // Enable trellis quantization
        // Note: JBOOLEAN_OPTIMIZE_SCANS requires more complex setup and causes decode issues
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

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    decoder.decode().expect("JPEG decode failed")
}

fn compute_metrics(original: &[u8], decoded: &[u8], width: u32, height: u32) -> (f64, f64) {
    use dssim::Dssim;
    use rgb::RGB8;

    // Convert to RGB8 format for dssim
    let orig_rgb: Vec<RGB8> = original.chunks(3)
        .map(|c| RGB8 { r: c[0], g: c[1], b: c[2] })
        .collect();
    let dec_rgb: Vec<RGB8> = decoded.chunks(3)
        .map(|c| RGB8 { r: c[0], g: c[1], b: c[2] })
        .collect();

    // Compute DSSIM
    let dssim = Dssim::new();
    let orig_img = dssim.create_image_rgb(&orig_rgb, width as usize, height as usize)
        .expect("Failed to create DSSIM image");
    let dec_img = dssim.create_image_rgb(&dec_rgb, width as usize, height as usize)
        .expect("Failed to create DSSIM image");
    let (dssim_val, _) = dssim.compare(&orig_img, dec_img);

    // Compute SSIMULACRA2 - convert u8 RGB to f32 RGB (0.0-1.0 range)
    let orig_f32: Vec<[f32; 3]> = original.chunks(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();
    let dec_f32: Vec<[f32; 3]> = decoded.chunks(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();

    let orig_ssim = Ssim2Rgb::new(
        orig_f32,
        width as usize,
        height as usize,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    ).expect("Failed to create SSIM2 source image");

    let dec_ssim = Ssim2Rgb::new(
        dec_f32,
        width as usize,
        height as usize,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    ).expect("Failed to create SSIM2 distorted image");

    let ssim2 = compute_frame_ssimulacra2(orig_ssim, dec_ssim).unwrap_or(0.0);

    (ssim2, dssim_val.into())
}

fn write_csv(path: &Path, results: &[EncodingResult]) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header
    writeln!(file, "corpus,image,encoder,quality,file_size,bpp,ssimulacra2,dssim,encode_time_ms")?;

    for r in results {
        writeln!(file, "{},{},{},{},{},{:.6},{:.6},{:.8},{:.3}",
            r.corpus, r.image, r.encoder, r.quality, r.file_size,
            r.bpp, r.ssimulacra2, r.dssim, r.encode_time_ms)?;
    }

    Ok(())
}

fn print_summary(results: &[EncodingResult], qualities: &[u8]) {
    println!("Summary by Quality Level");
    println!("========================");
    println!();
    println!("{:>5} {:>12} {:>12} {:>10} {:>10} {:>12} {:>12}",
        "Q", "Rust Size", "C Size", "Size Δ%", "SSIM2 Δ", "Rust DSSIM", "C DSSIM");
    println!("{}", "-".repeat(85));

    for &q in qualities {
        let rust: Vec<_> = results.iter().filter(|r| r.encoder == "rust" && r.quality == q).collect();
        let c: Vec<_> = results.iter().filter(|r| r.encoder == "c" && r.quality == q).collect();

        if rust.is_empty() || c.is_empty() { continue; }

        let rust_size: f64 = rust.iter().map(|r| r.file_size as f64).sum::<f64>() / rust.len() as f64;
        let c_size: f64 = c.iter().map(|r| r.file_size as f64).sum::<f64>() / c.len() as f64;
        let rust_ssim2: f64 = rust.iter().map(|r| r.ssimulacra2).sum::<f64>() / rust.len() as f64;
        let c_ssim2: f64 = c.iter().map(|r| r.ssimulacra2).sum::<f64>() / c.len() as f64;
        let rust_dssim: f64 = rust.iter().map(|r| r.dssim).sum::<f64>() / rust.len() as f64;
        let c_dssim: f64 = c.iter().map(|r| r.dssim).sum::<f64>() / c.len() as f64;

        let size_diff = (rust_size - c_size) / c_size * 100.0;
        let ssim2_diff = rust_ssim2 - c_ssim2;

        println!("{:>5} {:>12.0} {:>12.0} {:>+9.2}% {:>+10.4} {:>12.6} {:>12.6}",
            q, rust_size, c_size, size_diff, ssim2_diff, rust_dssim, c_dssim);
    }

    println!();
    println!("Note: Positive Size Δ% means Rust is larger, negative means smaller.");
    println!("      Positive SSIM2 Δ means Rust has better quality.");
    println!("      Lower DSSIM is better (0 = identical).");
}
