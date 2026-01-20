//! Comprehensive quality comparison: Rust vs C mozjpeg
//!
//! Tests 3 modes × 20 quality levels × 15 images with DSSIM, Butteraugli, SSIMULACRA2
//!
//! Uses codec-eval for perceptual quality metrics.

use codec_eval::{
    decode::jpeg_decode_callback, EvalConfig, EvalSession, ImageData, MetricConfig,
    ViewingCondition,
};
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::fs::File;
use std::path::Path;

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
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

// ============= Rust encoders =============

fn encode_rust_baseline(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::baseline_optimized()
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

fn encode_rust_progressive(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::baseline_optimized()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(false)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

fn encode_rust_max_compression(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::max_compression()
        .quality(quality)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

// ============= C mozjpeg encoders =============

fn encode_c_baseline(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // Disable progressive - baseline mode
        cinfo.num_scans = 0;
        cinfo.scan_info = ptr::null();

        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

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

fn encode_c_progressive(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Disable optimize_scans BEFORE jpeg_simple_progression
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);
        jpeg_simple_progression(&mut cinfo);

        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

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

fn encode_c_max_compression(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Enable optimize_scans for max compression
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1);
        jpeg_simple_progression(&mut cinfo);

        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

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

#[derive(Debug, Clone)]
struct Result {
    quality: u8,
    rust_size: usize,
    c_size: usize,
    size_diff_pct: f64,
    rust_ssim2: f64,
    c_ssim2: f64,
    rust_dssim: f64,
    c_dssim: f64,
    rust_butteraugli: f64,
    c_butteraugli: f64,
}

fn run_mode<F1, F2>(
    images: &[(Vec<u8>, u32, u32, String)],
    qualities: &[u8],
    rust_encoder: F1,
    c_encoder: F2,
    mode_name: &str,
) -> Vec<Result>
where
    F1: Fn(&[u8], u32, u32, u8) -> Vec<u8>,
    F2: Fn(&[u8], u32, u32, u8) -> Vec<u8>,
{
    println!("\n=== {} ===", mode_name);
    println!(
        "Processing {} images × {} qualities...\n",
        images.len(),
        qualities.len()
    );

    let metrics = MetricConfig::perceptual();
    let config = EvalConfig::builder()
        .report_dir("./comparison_outputs")
        .viewing(ViewingCondition::desktop())
        .metrics(metrics)
        .quality_levels(qualities.iter().map(|&q| q as f64).collect())
        .build();

    let mut results = Vec::new();

    for &quality in qualities {
        let mut rust_size_total = 0usize;
        let mut c_size_total = 0usize;
        let mut rust_ssim2_sum = 0.0f64;
        let mut c_ssim2_sum = 0.0f64;
        let mut rust_dssim_sum = 0.0f64;
        let mut c_dssim_sum = 0.0f64;
        let mut rust_ba_sum = 0.0f64;
        let mut c_ba_sum = 0.0f64;
        let mut count = 0;

        for (rgb, width, height, name) in images {
            let rust_jpeg = rust_encoder(rgb, *width, *height, quality);
            let c_jpeg = c_encoder(rgb, *width, *height, quality);

            rust_size_total += rust_jpeg.len();
            c_size_total += c_jpeg.len();

            // Create sessions for each encoding
            let mut rust_session = EvalSession::new(config.clone());
            let mut c_session = EvalSession::new(config.clone());

            // Add encoders that return pre-encoded data
            let rust_data = rust_jpeg.clone();
            let c_data = c_jpeg.clone();

            rust_session.add_codec_with_decode(
                "rust",
                "0.1",
                Box::new(move |_image, _request| Ok(rust_data.clone())),
                jpeg_decode_callback(),
            );

            c_session.add_codec_with_decode(
                "c",
                "0.1",
                Box::new(move |_image, _request| Ok(c_data.clone())),
                jpeg_decode_callback(),
            );

            let image_data = ImageData::RgbSlice {
                data: rgb.clone(),
                width: *width as usize,
                height: *height as usize,
            };

            if let Ok(rust_report) = rust_session.evaluate_image(name, image_data.clone()) {
                for r in &rust_report.results {
                    rust_ssim2_sum += r.metrics.ssimulacra2.unwrap_or(0.0);
                    rust_dssim_sum += r.metrics.dssim.unwrap_or(0.0);
                    rust_ba_sum += r.metrics.butteraugli.unwrap_or(0.0);
                }
            }

            if let Ok(c_report) = c_session.evaluate_image(name, image_data) {
                for r in &c_report.results {
                    c_ssim2_sum += r.metrics.ssimulacra2.unwrap_or(0.0);
                    c_dssim_sum += r.metrics.dssim.unwrap_or(0.0);
                    c_ba_sum += r.metrics.butteraugli.unwrap_or(0.0);
                }
            }

            count += 1;
        }

        let n = count as f64;
        let size_diff = ((rust_size_total as f64 / c_size_total as f64) - 1.0) * 100.0;

        results.push(Result {
            quality,
            rust_size: rust_size_total,
            c_size: c_size_total,
            size_diff_pct: size_diff,
            rust_ssim2: rust_ssim2_sum / n,
            c_ssim2: c_ssim2_sum / n,
            rust_dssim: rust_dssim_sum / n,
            c_dssim: c_dssim_sum / n,
            rust_butteraugli: rust_ba_sum / n,
            c_butteraugli: c_ba_sum / n,
        });

        print!("Q{:02} ", quality);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    println!("\n");

    results
}

fn print_results(results: &[Result], mode_name: &str) {
    println!("### {} Results\n", mode_name);

    println!("| Q | Rust Size | C Size | Diff | R SSIM2 | C SSIM2 | R DSSIM | C DSSIM | R Butter | C Butter |");
    println!("|---|-----------|--------|------|---------|---------|---------|---------|----------|----------|");
    for r in results {
        println!(
            "| {:2} | {:>9} | {:>6} | {:>+.2}% | {:>7.2} | {:>7.2} | {:.6} | {:.6} | {:>8.4} | {:>8.4} |",
            r.quality,
            r.rust_size, r.c_size, r.size_diff_pct,
            r.rust_ssim2, r.c_ssim2,
            r.rust_dssim, r.c_dssim,
            r.rust_butteraugli, r.c_butteraugli
        );
    }
    println!();
}

fn main() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Corpus not found at corpus/kodak");
        eprintln!("Run: ./scripts/fetch-corpus.sh");
        return;
    }

    // Load first 15 images
    let mut images: Vec<(Vec<u8>, u32, u32, String)> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries.iter().take(15) {
        if let Some((rgb, w, h)) = load_png(&entry.path()) {
            let name = entry.file_name().to_string_lossy().to_string();
            println!("Loaded: {}", name);
            images.push((rgb, w, h, name));
        }
    }

    println!("\nLoaded {} images\n", images.len());

    // 20 quality levels spread across the range
    let qualities: Vec<u8> = vec![
        30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 78, 80, 83, 85, 88, 90, 92, 94, 96, 98,
    ];

    println!("Quality levels: {:?}\n", qualities);

    // Run all three modes
    let baseline = run_mode(
        &images,
        &qualities,
        encode_rust_baseline,
        encode_c_baseline,
        "BASELINE",
    );
    print_results(&baseline, "Baseline");

    let progressive = run_mode(
        &images,
        &qualities,
        encode_rust_progressive,
        encode_c_progressive,
        "PROGRESSIVE (optimize_scans=false)",
    );
    print_results(&progressive, "Progressive");

    let max_comp = run_mode(
        &images,
        &qualities,
        encode_rust_max_compression,
        encode_c_max_compression,
        "MAX COMPRESSION (optimize_scans=true)",
    );
    print_results(&max_comp, "Max Compression");

    // Summary
    println!("\n## Size Comparison Summary\n");
    println!("| Q | Baseline | Progressive | Max Comp |");
    println!("|---|----------|-------------|----------|");
    for i in 0..qualities.len() {
        println!(
            "| {:2} | {:>+.2}% | {:>+.2}% | {:>+.2}% |",
            qualities[i],
            baseline[i].size_diff_pct,
            progressive[i].size_diff_pct,
            max_comp[i].size_diff_pct
        );
    }
}
