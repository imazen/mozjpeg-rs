//! Corpus validation tests.
//!
//! Tests against the Kodak corpus to ensure consistent quality vs C mozjpeg.
//! Run `./scripts/fetch-corpus.sh` first to download test images.

use mozjpeg_oxide::corpus::all_corpus_dirs;
use mozjpeg_oxide::test_encoder::{encode_rust, TestEncoderConfig};
use mozjpeg_oxide::Subsampling;
use mozjpeg_sys::*;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::ptr;

/// Skip if no corpus available (don't fail CI if corpus not downloaded).
fn has_corpus() -> bool {
    !all_corpus_dirs().is_empty()
}

/// Load a PNG and return RGB data.
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => {
            bytes.chunks(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect()
        }
        png::ColorType::Grayscale => {
            bytes.iter()
                .flat_map(|&g| [g, g, g])
                .collect()
        }
        png::ColorType::GrayscaleAlpha => {
            bytes.chunks(2)
                .flat_map(|c| [c[0], c[0], c[0]])
                .collect()
        }
        _ => return None,
    };

    Some((rgb_data, width, height))
}

/// Encode with C mozjpeg using TestEncoderConfig settings.
fn encode_c_with_config(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
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
        cinfo.in_color_space = JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        cinfo.num_scans = 0;
        cinfo.scan_info = ptr::null();

        jpeg_set_quality(&mut cinfo, config.quality as i32, if config.force_baseline { 1 } else { 0 });

        // Set subsampling
        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => panic!("Gray not supported"),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT,
            if config.trellis_quant { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC,
            if config.trellis_dc { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING,
            if config.overshoot_deringing { 1 } else { 0 });

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = rgb.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}

/// Calculate PSNR between original and decoded JPEG.
fn psnr(original: &[u8], jpeg: &[u8]) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(jpeg));
    let decoded = decoder.decode().expect("Decode failed");

    let mse: f64 = original.iter().zip(decoded.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>() / original.len() as f64;

    if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// Test Rust vs C mozjpeg on Kodak corpus.
///
/// Requirements:
/// - File size ratio within 5% of C
/// - PSNR within 0.5 dB of C
#[test]
fn test_kodak_corpus_quality() {
    if !has_corpus() {
        println!("Skipping corpus test - run ./scripts/fetch-corpus.sh first");
        return;
    }

    let corpus_dirs = all_corpus_dirs();
    let mut passed = 0;
    let mut failed = 0;
    let mut size_ratios = Vec::new();
    let mut psnr_diffs = Vec::new();

    for corpus_dir in &corpus_dirs {
        let dir = match fs::read_dir(&corpus_dir) {
            Ok(d) => d,
            Err(_) => continue,
        };

        for entry in dir.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map(|e| e != "png").unwrap_or(true) {
                continue;
            }

            let (rgb, width, height) = match load_png(&path) {
                Some(data) => data,
                None => continue,
            };

            // Use unified config to ensure identical settings for both encoders
            let config = TestEncoderConfig::baseline_huffman_opt().with_quality(75);

            let rust_jpeg = encode_rust(&rgb, width, height, &config);
            let c_jpeg = encode_c_with_config(&rgb, width, height, &config);

            let rust_psnr = psnr(&rgb, &rust_jpeg);
            let c_psnr = psnr(&rgb, &c_jpeg);
            let psnr_diff = (rust_psnr - c_psnr).abs();

            let size_ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

            size_ratios.push(size_ratio);
            psnr_diffs.push(psnr_diff);

            let filename = path.file_name().unwrap().to_string_lossy();

            // Allow up to 5% larger and 0.5 dB worse PSNR
            if size_ratio > 1.05 {
                println!("FAIL: {} size ratio {:.3} > 1.05", filename, size_ratio);
                failed += 1;
            } else if psnr_diff > 0.5 {
                println!("FAIL: {} PSNR diff {:.2} dB > 0.5 dB", filename, psnr_diff);
                failed += 1;
            } else {
                passed += 1;
            }
        }
    }

    if !size_ratios.is_empty() {
        let avg_size_ratio: f64 = size_ratios.iter().sum::<f64>() / size_ratios.len() as f64;
        let avg_psnr_diff: f64 = psnr_diffs.iter().sum::<f64>() / psnr_diffs.len() as f64;

        println!("\nKodak corpus results:");
        println!("  Passed: {}, Failed: {}", passed, failed);
        println!("  Average size ratio: {:.3}x", avg_size_ratio);
        println!("  Average PSNR diff: {:.2} dB", avg_psnr_diff);

        assert!(
            failed == 0,
            "Corpus validation failed: {} images out of tolerance",
            failed
        );
    }
}

/// Test file size consistency across quality levels.
#[test]
fn test_corpus_quality_sweep() {
    if !has_corpus() {
        println!("Skipping corpus test - run ./scripts/fetch-corpus.sh first");
        return;
    }

    let corpus_dirs = all_corpus_dirs();
    let qualities = [50, 75, 85, 95];

    // Take first image from corpus
    for corpus_dir in &corpus_dirs {
        let dir = match fs::read_dir(&corpus_dir) {
            Ok(d) => d,
            Err(_) => continue,
        };

        for entry in dir.filter_map(|e| e.ok()).take(1) {
            let path = entry.path();
            if path.extension().map(|e| e != "png").unwrap_or(true) {
                continue;
            }

            let (rgb, width, height) = match load_png(&path) {
                Some(data) => data,
                None => continue,
            };

            println!("Testing quality sweep on {:?}", path.file_name().unwrap());

            for &quality in &qualities {
                // Use unified config for identical settings
                let config = TestEncoderConfig::baseline_huffman_opt().with_quality(quality);

                let rust_jpeg = encode_rust(&rgb, width, height, &config);
                let c_jpeg = encode_c_with_config(&rgb, width, height, &config);

                let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

                println!(
                    "  Q{}: Rust={} C={} ratio={:.3}",
                    quality, rust_jpeg.len(), c_jpeg.len(), ratio
                );

                // Higher quality = more tolerance for ratio differences
                let max_ratio = if quality >= 90 { 1.10 } else { 1.05 };
                assert!(
                    ratio < max_ratio,
                    "Q{} size ratio {:.3} exceeds {:.2}",
                    quality, ratio, max_ratio
                );
            }
            return; // Only test one image
        }
    }
}
