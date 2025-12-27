//! Pareto Front Benchmark: Rust mozjpeg-oxide vs C mozjpeg
//!
//! Uses codec-eval for unified codec comparison with perceptual quality metrics.
//!
//! Usage:
//!   cargo run --release --example pareto_benchmark -- [OPTIONS]
//!
//! Options:
//!   --corpus PATH    Path to corpus directory (default: ./corpus)
//!   --output PATH    Output CSV file (default: benchmark_results.csv)
//!   --qualities      Comma-separated quality levels (default: fine-grained 5-95)
//!   --kodak-only     Only use Kodak corpus
//!   --clic-only      Only use CLIC corpus
//!   --xyb-roundtrip  Enable XYB roundtrip for fair comparison (isolates compression error)

use codec_eval::{
    EvalConfig, EvalSession, ImageData, MetricConfig, ViewingCondition, decode::jpeg_decode_callback,
};
use mozjpeg_oxide::Encoder;
use mozjpeg_sys::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr;

/// Result of encoding a single image at a specific quality
#[derive(Debug)]
struct EncodingResult {
    corpus: String,
    image: String,
    encoder: String,
    quality: u8,
    file_size: usize,
    bpp: f64,
    ssimulacra2: f64,
    dssim: f64,
    butteraugli: f64,
    encode_time_ms: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut corpus_path = PathBuf::from("corpus");
    let mut output_path = PathBuf::from("benchmark_results.csv");
    // Fine-grained quality levels for smooth Pareto curves
    let mut qualities: Vec<u8> = vec![
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 95, 97,
    ];
    let mut kodak_only = false;
    let mut clic_only = false;
    let mut xyb_roundtrip = false;

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
            "--xyb-roundtrip" => xyb_roundtrip = true,
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
            println!(
                "Kodak corpus: {} images",
                images.iter().filter(|(c, _)| c == "kodak").count()
            );
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
    println!(
        "Total configurations: {}",
        images.len() * qualities.len() * 2
    );
    if xyb_roundtrip {
        println!("XYB roundtrip: ENABLED (isolating compression error from color space error)");
    }
    println!();

    // Create codec-eval session with perceptual metrics
    let metrics = if xyb_roundtrip {
        MetricConfig::perceptual_xyb() // Perceptual metrics with XYB roundtrip
    } else {
        MetricConfig::perceptual() // DSSIM, SSIMULACRA2, Butteraugli
    };

    let config = EvalConfig::builder()
        .report_dir("./benchmark")
        .viewing(ViewingCondition::desktop())
        .metrics(metrics)
        .quality_levels(qualities.iter().map(|&q| q as f64).collect())
        .build();

    let mut session = EvalSession::new(config);

    // Register Rust mozjpeg-oxide encoder
    // Uses ICC-aware decode callback for proper color management
    session.add_codec_with_decode(
        "rust",
        env!("CARGO_PKG_VERSION"),
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;
            encode_rust(&rgb, width, height, quality).map_err(|e: mozjpeg_oxide::Error| {
                codec_eval::Error::Codec {
                    codec: "rust".to_string(),
                    message: e.to_string(),
                }
            })
        }),
        jpeg_decode_callback(),
    );

    // Register C mozjpeg encoder
    session.add_codec_with_decode(
        "c",
        "4.1.1", // mozjpeg-sys version
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;
            Ok(encode_c(&rgb, width, height, quality))
        }),
        jpeg_decode_callback(),
    );

    // Process each image and collect results
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

        let image_name = image_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let image_data = ImageData::RgbSlice {
            data: rgb,
            width: width as usize,
            height: height as usize,
        };

        // Evaluate with codec-eval
        match session.evaluate_image(&image_name, image_data) {
            Ok(report) => {
                for result in &report.results {
                    let quality = result.quality as u8;

                    results.push(EncodingResult {
                        corpus: corpus.clone(),
                        image: image_name.clone(),
                        encoder: result.codec_id.clone(),
                        quality,
                        file_size: result.file_size,
                        bpp: result.bits_per_pixel,
                        ssimulacra2: result.metrics.ssimulacra2.unwrap_or(0.0),
                        dssim: result.metrics.dssim.unwrap_or(0.0),
                        butteraugli: result.metrics.butteraugli.unwrap_or(0.0),
                        encode_time_ms: result.encode_time.as_secs_f64() * 1000.0,
                    });

                    completed += 1;
                    print!(
                        "\rProgress: {}/{} ({:.1}%)",
                        completed,
                        total,
                        100.0 * completed as f64 / total as f64
                    );
                    std::io::stdout().flush().ok();
                }
            }
            Err(e) => {
                eprintln!("\nError evaluating {}: {}", image_name, e);
            }
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
    println!("  --xyb-roundtrip  Enable XYB roundtrip (isolates compression error)");
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

fn encode_rust(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, mozjpeg_oxide::Error> {
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
}

#[allow(unsafe_code)]
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

fn write_csv(path: &Path, results: &[EncodingResult]) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header
    writeln!(
        file,
        "corpus,image,encoder,quality,file_size,bpp,ssimulacra2,dssim,butteraugli,encode_time_ms"
    )?;

    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{:.6},{:.6},{:.8},{:.6},{:.3}",
            r.corpus,
            r.image,
            r.encoder,
            r.quality,
            r.file_size,
            r.bpp,
            r.ssimulacra2,
            r.dssim,
            r.butteraugli,
            r.encode_time_ms
        )?;
    }

    Ok(())
}

fn print_summary(results: &[EncodingResult], qualities: &[u8]) {
    println!("Summary by Quality Level");
    println!("========================");
    println!();
    println!(
        "{:>5} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10}",
        "Q", "Rust BPP", "C BPP", "Size Δ%", "SSIM2 Δ", "Rust BA", "C BA"
    );
    println!("{}", "-".repeat(75));

    for &q in qualities {
        let rust: Vec<_> = results
            .iter()
            .filter(|r| r.encoder == "rust" && r.quality == q)
            .collect();
        let c: Vec<_> = results
            .iter()
            .filter(|r| r.encoder == "c" && r.quality == q)
            .collect();

        if rust.is_empty() || c.is_empty() {
            continue;
        }

        let rust_bpp: f64 = rust.iter().map(|r| r.bpp).sum::<f64>() / rust.len() as f64;
        let c_bpp: f64 = c.iter().map(|r| r.bpp).sum::<f64>() / c.len() as f64;
        let rust_ssim2: f64 = rust.iter().map(|r| r.ssimulacra2).sum::<f64>() / rust.len() as f64;
        let c_ssim2: f64 = c.iter().map(|r| r.ssimulacra2).sum::<f64>() / c.len() as f64;
        let rust_ba: f64 = rust.iter().map(|r| r.butteraugli).sum::<f64>() / rust.len() as f64;
        let c_ba: f64 = c.iter().map(|r| r.butteraugli).sum::<f64>() / c.len() as f64;

        let size_diff = (rust_bpp - c_bpp) / c_bpp * 100.0;
        let ssim2_diff = rust_ssim2 - c_ssim2;

        println!(
            "{:>5} {:>10.4} {:>10.4} {:>+8.2}% {:>+9.3} {:>10.4} {:>10.4}",
            q, rust_bpp, c_bpp, size_diff, ssim2_diff, rust_ba, c_ba
        );
    }

    println!();
    println!("Legend:");
    println!("  BPP: Bits per pixel (lower = smaller file)");
    println!("  Size Δ%: Positive = Rust larger, negative = Rust smaller");
    println!("  SSIM2 Δ: Positive = Rust better quality (SSIMULACRA2 score)");
    println!("  BA: Butteraugli score (lower = better, <1.0 = imperceptible)");
}
