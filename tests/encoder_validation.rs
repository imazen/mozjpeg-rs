//! Encoder validation tests converted from examples.
//!
//! These tests verify encoder correctness across various configurations.
//! Some tests require corpus images and are skipped if unavailable.

use dssim::Dssim;
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::io::Cursor;

/// Helper to decode JPEG and return pixels
fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(Cursor::new(data))
        .decode()
        .expect("Failed to decode JPEG")
}

/// Helper to calculate PSNR between original and decoded
fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len(), "Size mismatch");
    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Helper to calculate DSSIM between original and decoded RGB data.
/// Returns DSSIM value (0 = identical, higher = worse).
/// DSSIM thresholds: <0.0003 imperceptible, <0.001 marginal, <0.003 noticeable.
fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    use rgb::RGB8;

    let attr = Dssim::new();

    let orig_rgb: Vec<RGB8> = original
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = attr
        .create_image_rgb(&orig_rgb, width as usize, height as usize)
        .expect("Failed to create original image");

    let dec_rgb: Vec<RGB8> = decoded
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = attr
        .create_image_rgb(&dec_rgb, width as usize, height as usize)
        .expect("Failed to create decoded image");

    let (dssim_val, _) = attr.compare(&orig_img, dec_img);
    dssim_val.into()
}

/// Load bundled test image (1.png from tests/images)
fn load_bundled_image() -> (Vec<u8>, u32, u32) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/images/1.png");
    let file = std::fs::File::open(&path).expect("Failed to open bundled test image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let rgb_data: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    (rgb_data, info.width, info.height)
}

// =============================================================================
// Edge Cropping Tests (from test_edge_cropping.rs)
// =============================================================================

/// Test that we can encode images with non-8-aligned dimensions.
/// The encoder should handle edge padding correctly.
#[test]
fn test_edge_cropping_non_aligned_dimensions() {
    let (rgb_data, orig_width, orig_height) = load_bundled_image();

    // Test various non-8-aligned sizes
    let test_sizes = [
        (100, 100), // Neither dimension aligned
        (128, 100), // Only width aligned
        (100, 128), // Only height aligned
        (127, 127), // Just under 8-aligned
        (129, 129), // Just over 8-aligned
        (17, 33),   // Small odd sizes
    ];

    for (target_w, target_h) in test_sizes {
        if target_w > orig_width || target_h > orig_height {
            continue; // Skip if larger than source
        }

        // Crop to target size
        let mut cropped = Vec::with_capacity((target_w * target_h * 3) as usize);
        for y in 0..target_h {
            let src_offset = (y * orig_width * 3) as usize;
            let row_bytes = (target_w * 3) as usize;
            cropped.extend_from_slice(&rgb_data[src_offset..src_offset + row_bytes]);
        }

        // Encode with various settings
        let configs = [
            ("baseline", Encoder::baseline_optimized().quality(85)),
            (
                "progressive",
                Encoder::baseline_optimized().quality(85).progressive(true),
            ),
            (
                "trellis",
                Encoder::baseline_optimized()
                    .quality(85)
                    .trellis(TrellisConfig::default()),
            ),
        ];

        for (config_name, encoder) in configs {
            let result = encoder.encode_rgb(&cropped, target_w, target_h);
            assert!(
                result.is_ok(),
                "Failed to encode {}x{} with {}: {:?}",
                target_w,
                target_h,
                config_name,
                result.err()
            );

            let jpeg_data = result.unwrap();
            assert!(
                jpeg_data.len() > 100,
                "JPEG too small for {}x{} with {}",
                target_w,
                target_h,
                config_name
            );

            // Verify it decodes correctly
            let decoded = decode_jpeg(&jpeg_data);
            let expected_pixels = (target_w * target_h * 3) as usize;
            assert_eq!(
                decoded.len(),
                expected_pixels,
                "Decoded size mismatch for {}x{} with {}",
                target_w,
                target_h,
                config_name
            );

            // Verify reasonable quality - PSNR varies widely by content and size
            // Small cropped regions with edges may have lower PSNR
            let psnr = calculate_psnr(&cropped, &decoded);
            assert!(
                psnr > 15.0,
                "PSNR extremely low ({:.2} dB) for {}x{} with {} - possible encoding bug",
                psnr,
                target_w,
                target_h,
                config_name
            );

            // DSSIM check - perceptual quality metric
            // <0.003 noticeable, <0.01 acceptable for small crops with edge artifacts
            let dssim = calculate_dssim(&cropped, &decoded, target_w, target_h);
            assert!(
                dssim < 0.01,
                "DSSIM too high ({:.6}) for {}x{} with {} - perceptual quality issue",
                dssim,
                target_w,
                target_h,
                config_name
            );
        }
    }
}

/// Test that edge pixels are preserved correctly (not corrupted by padding).
#[test]
fn test_edge_pixel_preservation() {
    // Create a small image with known edge values
    let width = 17u32;
    let height = 17u32;
    let mut rgb = vec![128u8; (width * height * 3) as usize];

    // Set distinctive edge patterns
    // Top-left corner: red
    rgb[0] = 255;
    rgb[1] = 0;
    rgb[2] = 0;
    // Top-right corner: green
    let tr = ((width - 1) * 3) as usize;
    rgb[tr] = 0;
    rgb[tr + 1] = 255;
    rgb[tr + 2] = 0;
    // Bottom-left corner: blue
    let bl = ((height - 1) * width * 3) as usize;
    rgb[bl] = 0;
    rgb[bl + 1] = 0;
    rgb[bl + 2] = 255;
    // Bottom-right corner: yellow
    let br = (((height - 1) * width + (width - 1)) * 3) as usize;
    rgb[br] = 255;
    rgb[br + 1] = 255;
    rgb[br + 2] = 0;

    // Encode at high quality to minimize compression loss
    let encoder = Encoder::baseline_optimized().quality(95);
    let jpeg = encoder
        .encode_rgb(&rgb, width, height)
        .expect("Encoding failed");

    let decoded = decode_jpeg(&jpeg);

    // Check corners are approximately preserved (within JPEG tolerance)
    // JPEG compression can cause significant color shift at edges due to:
    // - YCbCr conversion rounding
    // - DCT block boundary effects on small images
    // - Chroma subsampling (if enabled)
    // Use a very generous tolerance since we mainly want to verify
    // the encoder doesn't completely corrupt edge pixels
    let tolerance = 100; // Wide tolerance for edge effects

    // Top-left (red) - should be predominantly red
    assert!(
        decoded[0] > decoded[1] && decoded[0] > decoded[2],
        "Top-left should be reddish: R={} G={} B={}",
        decoded[0],
        decoded[1],
        decoded[2]
    );

    // Top-right (green) - should be predominantly green
    assert!(
        decoded[tr + 1] > decoded[tr] && decoded[tr + 1] > decoded[tr + 2],
        "Top-right should be greenish: R={} G={} B={}",
        decoded[tr],
        decoded[tr + 1],
        decoded[tr + 2]
    );

    // Bottom-left (blue) - should be predominantly blue
    assert!(
        decoded[bl + 2] > decoded[bl] && decoded[bl + 2] > decoded[bl + 1],
        "Bottom-left should be bluish: R={} G={} B={}",
        decoded[bl],
        decoded[bl + 1],
        decoded[bl + 2]
    );

    // Bottom-right (yellow = R+G) - R and G should both be significant
    assert!(
        decoded[br] > tolerance && decoded[br + 1] > tolerance,
        "Bottom-right should be yellowish: R={} G={} B={}",
        decoded[br],
        decoded[br + 1],
        decoded[br + 2]
    );
}

// =============================================================================
// Encode Permutations Tests (from encode_permutations.rs)
// =============================================================================

/// Test that all common encoder setting permutations work without error.
#[test]
fn test_all_encoder_permutations_work() {
    let (rgb_data, width, height) = load_bundled_image();

    let qualities = [50, 75, 85, 95];
    let subsamplings = [Subsampling::S444, Subsampling::S422, Subsampling::S420];
    let progressive_opts = [false, true];
    let trellis_opts = [false, true];
    let huffman_opts = [false, true];

    let mut success_count = 0;
    let mut total_count = 0;

    for quality in &qualities {
        for subsampling in &subsamplings {
            for progressive in &progressive_opts {
                for trellis in &trellis_opts {
                    for optimize_huffman in &huffman_opts {
                        total_count += 1;

                        let trellis_config = if *trellis {
                            TrellisConfig::default()
                        } else {
                            TrellisConfig::disabled()
                        };

                        let result = Encoder::baseline_optimized()
                            .quality(*quality)
                            .subsampling(*subsampling)
                            .progressive(*progressive)
                            .trellis(trellis_config)
                            .optimize_huffman(*optimize_huffman)
                            .encode_rgb(&rgb_data, width, height);

                        assert!(
                            result.is_ok(),
                            "Failed: Q{} {:?} prog={} trellis={} huff={}: {:?}",
                            quality,
                            subsampling,
                            progressive,
                            trellis,
                            optimize_huffman,
                            result.err()
                        );

                        let jpeg = result.unwrap();
                        assert!(
                            jpeg.len() > 100,
                            "JPEG too small: Q{} {:?} prog={} trellis={} huff={}",
                            quality,
                            subsampling,
                            progressive,
                            trellis,
                            optimize_huffman
                        );

                        // Verify it decodes
                        let decode_result = jpeg_decoder::Decoder::new(Cursor::new(&jpeg)).decode();
                        assert!(
                            decode_result.is_ok(),
                            "Decode failed: Q{} {:?} prog={} trellis={} huff={}: {:?}",
                            quality,
                            subsampling,
                            progressive,
                            trellis,
                            optimize_huffman,
                            decode_result.err()
                        );

                        success_count += 1;
                    }
                }
            }
        }
    }

    assert_eq!(success_count, total_count, "Not all permutations succeeded");
    println!("All {} encoder permutations worked correctly", total_count);
}

// =============================================================================
// 4:4:4 Subsampling Tests (from test_444.rs)
// Requires corpus - these tests check against C mozjpeg
// =============================================================================

#[cfg(feature = "ffi-test")]
mod corpus_tests {
    use super::*;
    use mozjpeg_rs::corpus::kodak_dir;

    /// Test 4:4:4 subsampling produces reasonable output compared to C mozjpeg.
    #[test]
    fn test_444_subsampling_vs_c() {
        let corpus_dir = match kodak_dir() {
            Some(dir) => dir,
            None => {
                eprintln!("Skipping test_444_subsampling_vs_c: no corpus");
                return;
            }
        };

        let entries: Vec<_> = std::fs::read_dir(&corpus_dir)
            .expect("Failed to read corpus")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "png")
                    .unwrap_or(false)
            })
            .take(3) // Test first 3 images
            .collect();

        assert!(!entries.is_empty(), "No PNG files in corpus");

        for entry in entries {
            let path = entry.path();
            let file = std::fs::File::open(&path).expect("Failed to open image");
            let decoder = png::Decoder::new(file);
            let mut reader = decoder.read_info().expect("PNG read failed");
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).expect("PNG decode failed");

            let rgb_data: Vec<u8> = match info.color_type {
                png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
                png::ColorType::Rgba => buf[..info.buffer_size()]
                    .chunks(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect(),
                _ => continue,
            };

            // Rust encoder with 4:4:4
            let rust_jpeg = Encoder::baseline_optimized()
                .quality(75)
                .subsampling(Subsampling::S444)
                .encode_rgb(&rgb_data, info.width, info.height)
                .expect("Rust encoding failed");

            // C encoder with 4:4:4
            let c_jpeg =
                encode_c_444(&rgb_data, info.width, info.height, 75).expect("C encoding failed");

            let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

            // Rust should be within 20% of C (reasonable tolerance for different optimizations)
            assert!(
                ratio > 0.8 && ratio < 1.2,
                "4:4:4 ratio out of bounds for {:?}: {:.3} (Rust: {} C: {})",
                path.file_name(),
                ratio,
                rust_jpeg.len(),
                c_jpeg.len()
            );
        }
    }

    fn encode_c_444(
        rgb: &[u8],
        width: u32,
        height: u32,
        quality: i32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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
            let mut outsize: u64 = 0;
            jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = 3;
            cinfo.in_color_space = JCS_RGB;

            jpeg_set_defaults(&mut cinfo);
            jpeg_set_quality(&mut cinfo, quality, 1);

            // Set 4:4:4 subsampling
            (*cinfo.comp_info.offset(0)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(0)).v_samp_factor = 1;
            (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

            cinfo.optimize_coding = 1;

            jpeg_start_compress(&mut cinfo, 1);

            let row_stride = (width * 3) as usize;
            let mut row_pointer: [*const u8; 1] = [ptr::null()];

            while cinfo.next_scanline < cinfo.image_height {
                let offset = cinfo.next_scanline as usize * row_stride;
                row_pointer[0] = rgb.as_ptr().add(offset);
                jpeg_write_scanlines(&mut cinfo, row_pointer.as_ptr(), 1);
            }

            jpeg_finish_compress(&mut cinfo);
            jpeg_destroy_compress(&mut cinfo);

            let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
            libc::free(outbuffer as *mut libc::c_void);

            Ok(result)
        }
    }
}

// =============================================================================
// Progressive vs Baseline Quality Tests
// =============================================================================

/// Test that progressive and baseline modes produce similar quality.
#[test]
fn test_progressive_baseline_quality_parity() {
    let (rgb_data, width, height) = load_bundled_image();

    let baseline = Encoder::baseline_optimized()
        .quality(85)
        .progressive(false)
        .encode_rgb(&rgb_data, width, height)
        .expect("Baseline encoding failed");

    let progressive = Encoder::baseline_optimized()
        .quality(85)
        .progressive(true)
        .encode_rgb(&rgb_data, width, height)
        .expect("Progressive encoding failed");

    let baseline_decoded = decode_jpeg(&baseline);
    let progressive_decoded = decode_jpeg(&progressive);

    let baseline_psnr = calculate_psnr(&rgb_data, &baseline_decoded);
    let progressive_psnr = calculate_psnr(&rgb_data, &progressive_decoded);

    // Both should have reasonable quality at Q85
    // PSNR varies by image content; natural photos typically get 30-40+ dB
    assert!(
        baseline_psnr > 28.0,
        "Baseline PSNR too low: {:.2}",
        baseline_psnr
    );
    assert!(
        progressive_psnr > 28.0,
        "Progressive PSNR too low: {:.2}",
        progressive_psnr
    );

    // Quality difference should be minimal (< 1 dB)
    let psnr_diff = (baseline_psnr - progressive_psnr).abs();
    assert!(
        psnr_diff < 1.0,
        "PSNR difference too large: {:.2} dB (baseline: {:.2}, progressive: {:.2})",
        psnr_diff,
        baseline_psnr,
        progressive_psnr
    );

    // DSSIM perceptual quality checks
    // <0.001 is marginal (hard to see difference)
    let baseline_dssim = calculate_dssim(&rgb_data, &baseline_decoded, width, height);
    let progressive_dssim = calculate_dssim(&rgb_data, &progressive_decoded, width, height);

    assert!(
        baseline_dssim < 0.003,
        "Baseline DSSIM too high: {:.6} (should be < 0.003 for Q85)",
        baseline_dssim
    );
    assert!(
        progressive_dssim < 0.003,
        "Progressive DSSIM too high: {:.6} (should be < 0.003 for Q85)",
        progressive_dssim
    );
}

/// Test that progressive mode produces multiple scans.
#[test]
fn test_progressive_has_multiple_scans() {
    let (rgb_data, width, height) = load_bundled_image();

    let progressive = Encoder::baseline_optimized()
        .quality(85)
        .progressive(true)
        .encode_rgb(&rgb_data, width, height)
        .expect("Progressive encoding failed");

    // Count SOS markers (0xFF 0xDA)
    let scan_count = progressive
        .windows(2)
        .filter(|w| *w == [0xFF, 0xDA])
        .count();

    // Progressive should have multiple scans (typically 10+ for 3-component)
    assert!(
        scan_count > 1,
        "Progressive should have multiple scans, found {}",
        scan_count
    );
}
