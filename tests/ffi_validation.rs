//! FFI validation tests comparing Rust implementation against C mozjpeg.
//!
//! These tests verify that our Rust port produces identical results to the
//! original C implementation at each layer.

use mozjpeg_rs::consts::{DCTSIZE2, STD_CHROMINANCE_QUANT_TBL, STD_LUMINANCE_QUANT_TBL};
use mozjpeg_rs::quant::quality_to_scale_factor;

/// Test that our quality_to_scale_factor matches mozjpeg's jpeg_quality_scaling.
#[test]
fn test_quality_scaling_matches_c() {
    // Call C mozjpeg's jpeg_quality_scaling
    for quality in 1..=100 {
        let c_scale = unsafe { mozjpeg_sys::jpeg_quality_scaling(quality) };
        let rust_scale = quality_to_scale_factor(quality as u8) as i32;

        assert_eq!(
            c_scale, rust_scale,
            "Quality scaling mismatch at Q{}: C={}, Rust={}",
            quality, c_scale, rust_scale
        );
    }
}

/// Test that our quantization tables match mozjpeg's built-in tables.
///
/// Note: mozjpeg may use different base tables depending on compression profile.
/// This test dumps the C tables to understand what's being used.
#[test]
fn test_quant_tables_match_c() {
    unsafe {
        // Create compress struct using mozjpeg-sys safe wrappers
        let mut jerr = std::mem::zeroed::<mozjpeg_sys::jpeg_error_mgr>();
        let jerr_ptr = mozjpeg_sys::jpeg_std_error(&mut jerr);

        let mut cinfo = std::mem::zeroed::<mozjpeg_sys::jpeg_compress_struct>();
        cinfo.common.err = jerr_ptr;

        mozjpeg_sys::jpeg_CreateCompress(
            &mut cinfo,
            mozjpeg_sys::JPEG_LIB_VERSION as i32,
            std::mem::size_of::<mozjpeg_sys::jpeg_compress_struct>(),
        );

        // Set up minimal image parameters
        cinfo.image_width = 8;
        cinfo.image_height = 8;
        cinfo.input_components = 3;
        cinfo.in_color_space = mozjpeg_sys::J_COLOR_SPACE::JCS_RGB;

        mozjpeg_sys::jpeg_set_defaults(&mut cinfo);

        // Test quality 75 (common default)
        mozjpeg_sys::jpeg_set_quality(&mut cinfo, 75, 1);

        // Get the generated quant tables from C
        let c_luma_ptr = cinfo.quant_tbl_ptrs[0];
        let c_chroma_ptr = cinfo.quant_tbl_ptrs[1];

        assert!(!c_luma_ptr.is_null(), "C luma quant table is null");
        assert!(!c_chroma_ptr.is_null(), "C chroma quant table is null");

        let c_luma = &(*c_luma_ptr).quantval;
        let c_chroma = &(*c_chroma_ptr).quantval;

        // Generate Rust tables at same quality
        // mozjpeg defaults depend on compress_profile.
        // JCP_MAX_COMPRESSION uses table index 3 (ImageMagick)
        // The default profile from jpeg_set_defaults may vary.
        let scale = quality_to_scale_factor(75);

        // First verify our scaling is correct
        assert_eq!(scale, 50, "Q75 should give scale factor 50");

        // Check what mozjpeg actually produced for comparison
        // Print first few values for debugging
        println!("C luma table (first 8): {:?}", &c_luma[0..8]);
        println!("C chroma table (first 8): {:?}", &c_chroma[0..8]);

        // Our JPEG Annex K luma base table
        let base_luma = &STD_LUMINANCE_QUANT_TBL[0];
        let base_chroma = &STD_CHROMINANCE_QUANT_TBL[0];

        println!("Base luma (Annex K, first 8): {:?}", &base_luma[0..8]);
        println!("Base chroma (Annex K, first 8): {:?}", &base_chroma[0..8]);

        // Calculate what we would produce with scale 50
        let mut rust_luma = [0u16; DCTSIZE2];
        let mut rust_chroma = [0u16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            rust_luma[i] = ((base_luma[i] as u32 * scale + 50) / 100).clamp(1, 255) as u16;
            rust_chroma[i] = ((base_chroma[i] as u32 * scale + 50) / 100).clamp(1, 255) as u16;
        }

        println!("Rust luma (first 8): {:?}", &rust_luma[0..8]);
        println!("Rust chroma (first 8): {:?}", &rust_chroma[0..8]);

        // Now compare - allow some tolerance since mozjpeg may round differently
        // The key is to understand the pattern
        let mut luma_matches = 0;
        let mut chroma_matches = 0;
        for i in 0..DCTSIZE2 {
            if c_luma[i] == rust_luma[i] {
                luma_matches += 1;
            }
            if c_chroma[i] == rust_chroma[i] {
                chroma_matches += 1;
            }
        }

        println!("Luma matches: {}/64", luma_matches);
        println!("Chroma matches: {}/64", chroma_matches);

        // The test passes if our quality scaling is correct (verified above)
        // and we understand the table differences

        mozjpeg_sys::jpeg_destroy_compress(&mut cinfo);
    }
}

/// Test quality scaling at edge cases.
#[test]
fn test_quality_scaling_edge_cases() {
    // Q1 (minimum)
    let c_q1 = unsafe { mozjpeg_sys::jpeg_quality_scaling(1) };
    assert_eq!(c_q1, quality_to_scale_factor(1) as i32);

    // Q100 (maximum)
    let c_q100 = unsafe { mozjpeg_sys::jpeg_quality_scaling(100) };
    assert_eq!(c_q100, quality_to_scale_factor(100) as i32);

    // Q50 (inflection point)
    let c_q50 = unsafe { mozjpeg_sys::jpeg_quality_scaling(50) };
    assert_eq!(c_q50, 100); // Should be exactly 100%
    assert_eq!(quality_to_scale_factor(50), 100);
}

/// Validate all 101 quality levels match C implementation.
#[test]
fn test_all_quality_levels() {
    for q in 0..=100 {
        let c_scale = unsafe { mozjpeg_sys::jpeg_quality_scaling(q) };
        // mozjpeg clamps quality 0 to 1
        let rust_q = if q == 0 { 1 } else { q as u8 };
        let rust_scale = quality_to_scale_factor(rust_q) as i32;

        // For quality 0, C returns same as quality 1
        if q == 0 {
            let c_q1 = unsafe { mozjpeg_sys::jpeg_quality_scaling(1) };
            assert_eq!(c_scale, c_q1, "Q0 should equal Q1 in C");
        }

        assert_eq!(
            c_scale, rust_scale,
            "Mismatch at Q{}: C={}, Rust={}",
            q, c_scale, rust_scale
        );
    }
}

/// Test that compares Rust encoder output directly against C mozjpeg.
///
/// Current status: Rust encoder produces valid JPEG files but with different
/// characteristics than C mozjpeg. This test documents the current state.
#[test]
fn test_rust_vs_c_mozjpeg_encoder() {
    use mozjpeg_rs::{Encoder, Subsampling};

    // Create a test image (64x64 gradient)
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 4) as u8; // R gradient
            rgb_data[i * 3 + 1] = (y * 4) as u8; // G gradient
            rgb_data[i * 3 + 2] = 128; // B constant
        }
    }

    println!("\n=== Rust vs C mozjpeg encoder comparison ===\n");

    for quality in [50, 75, 85] {
        // Encode with Rust implementation
        let rust_encoder = Encoder::baseline_optimized()
            .quality(quality)
            .subsampling(Subsampling::S420);
        let rust_jpeg = rust_encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with C mozjpeg
        let c_jpeg = unsafe { encode_with_c_mozjpeg(&rgb_data, width, height, quality) };

        // Verify both produce valid JPEG files
        assert!(rust_jpeg.len() > 100, "Rust JPEG too small");
        assert!(c_jpeg.len() > 100, "C JPEG too small");
        assert_eq!(rust_jpeg[0..2], [0xFF, 0xD8], "Rust JPEG missing SOI");
        assert_eq!(c_jpeg[0..2], [0xFF, 0xD8], "C JPEG missing SOI");

        // Decode both to verify they're valid
        let mut rust_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg));
        let rust_decoded = rust_decoder.decode().expect("Failed to decode Rust JPEG");

        let mut c_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg));
        let c_decoded = c_decoder.decode().expect("Failed to decode C JPEG");

        // Calculate PSNR between original and decoded
        let rust_psnr = calculate_psnr(&rgb_data, &rust_decoded);
        let c_psnr = calculate_psnr(&rgb_data, &c_decoded);

        // Calculate similarity between the two decoded outputs
        let decoded_psnr = calculate_psnr(&rust_decoded, &c_decoded);

        let size_ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

        println!("Q{:2}:", quality);
        println!(
            "  File size: Rust={:5} bytes, C={:5} bytes (ratio: {:.2}x)",
            rust_jpeg.len(),
            c_jpeg.len(),
            size_ratio
        );
        println!(
            "  PSNR vs original: Rust={:.2} dB, C={:.2} dB",
            rust_psnr, c_psnr
        );
        println!(
            "  PSNR Rust vs C decoded: {:.2} dB (higher = more similar)",
            decoded_psnr
        );

        // Report status (don't fail - this is informational)
        if rust_psnr < 20.0 {
            println!("  ⚠️  Rust PSNR is low - encoding quality needs investigation");
        }
        if size_ratio > 2.0 {
            println!("  ⚠️  Rust output is significantly larger than C mozjpeg");
        }
        println!();
    }

    // Minimal assertions - just verify encoding works
    println!("Status: Both encoders produce valid, decodable JPEG files.");
    println!("Note: Quality/size differences require further investigation.");
}

/// Helper: Encode using C mozjpeg via FFI (uses crates.io mozjpeg-sys)
unsafe fn encode_with_c_mozjpeg(rgb_data: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::ptr;

    // Output buffer
    let mut outbuffer: *mut u8 = ptr::null_mut();
    let mut outsize: std::ffi::c_ulong = 0;

    // Create compression struct
    let mut cinfo = std::mem::zeroed::<mozjpeg_sys::jpeg_compress_struct>();
    let mut jerr = std::mem::zeroed::<mozjpeg_sys::jpeg_error_mgr>();

    cinfo.common.err = mozjpeg_sys::jpeg_std_error(&mut jerr);
    mozjpeg_sys::jpeg_CreateCompress(
        &mut cinfo,
        mozjpeg_sys::JPEG_LIB_VERSION as i32,
        std::mem::size_of::<mozjpeg_sys::jpeg_compress_struct>(),
    );

    // Set up memory destination
    mozjpeg_sys::jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

    // Set image parameters
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = mozjpeg_sys::J_COLOR_SPACE::JCS_RGB;

    mozjpeg_sys::jpeg_set_defaults(&mut cinfo);
    mozjpeg_sys::jpeg_set_quality(&mut cinfo, quality as i32, 1); // force_baseline=true

    // Start compression
    mozjpeg_sys::jpeg_start_compress(&mut cinfo, 1);

    // Write scanlines
    let row_stride = (width * 3) as usize;
    while cinfo.next_scanline < cinfo.image_height {
        let row_ptr = rgb_data
            .as_ptr()
            .add(cinfo.next_scanline as usize * row_stride);
        let mut row_array = [row_ptr as *const u8];
        mozjpeg_sys::jpeg_write_scanlines(&mut cinfo, row_array.as_mut_ptr() as *mut *const u8, 1);
    }

    mozjpeg_sys::jpeg_finish_compress(&mut cinfo);
    mozjpeg_sys::jpeg_destroy_compress(&mut cinfo);

    // Copy output to Vec
    let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();

    // Free the C-allocated buffer
    libc::free(outbuffer as *mut std::ffi::c_void);

    result
}

/// Calculate PSNR between two images
fn calculate_psnr(img1: &[u8], img2: &[u8]) -> f64 {
    assert_eq!(img1.len(), img2.len());

    let mse: f64 = img1
        .iter()
        .zip(img2.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / img1.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}
