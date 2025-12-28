//! FFI comparison tests for validating Rust implementations against C mozjpeg.
//!
//! These tests compare our Rust implementations against the C mozjpeg library
//! at a granular level to ensure correctness.
//!
//! Uses sys-local which builds from local C source at ~/work/mozjpeg.
//! To run these tests, uncomment sys-local in Cargo.toml dev-dependencies.
//!
//! This entire file is disabled in CI (sys-local is not included in dev-deps).

// This file will fail to compile if sys-local is not in dev-dependencies.
// That's intentional - these tests only run locally with the C source available.
#![cfg(feature = "__sys_local_available")]

use mozjpeg_oxide::{color, dct, quant, sample};
use sys_local as ffi;

/// Test forward DCT matches C implementation
#[test]
fn test_dct_matches_c() {
    // Test with various input patterns
    let test_patterns: &[[i16; 64]] = &[
        // All zeros
        [0i16; 64],
        // Uniform value (DC only)
        [100i16; 64],
        // Pattern that exercises AC coefficients
        {
            let mut arr = [0i16; 64];
            for i in 0..64 {
                arr[i] = (i as i16 - 32) * 2;
            }
            arr
        },
    ];

    for (idx, pattern) in test_patterns.iter().enumerate() {
        // Rust implementation
        let mut rust_coeffs = [0i16; 64];
        dct::forward_dct_8x8(pattern, &mut rust_coeffs);

        // C implementation (mutates in place)
        let mut c_data = *pattern;
        unsafe {
            ffi::mozjpeg_test_fdct_islow(c_data.as_mut_ptr());
        }

        // Compare
        for i in 0..64 {
            assert_eq!(
                rust_coeffs[i], c_data[i],
                "DCT mismatch at coefficient {} for pattern {}: Rust={}, C={}",
                i, idx, rust_coeffs[i], c_data[i]
            );
        }
    }
}

/// Test forward DCT with random-ish data
#[test]
fn test_dct_matches_c_varied() {
    // Generate varied input data
    for seed in 0..10 {
        let mut pattern = [0i16; 64];
        for i in 0..64 {
            // Simple deterministic pseudo-random values in typical sample range
            pattern[i] = (((seed * 7 + i * 13) % 255) as i16) - 128;
        }

        // Rust implementation
        let mut rust_coeffs = [0i16; 64];
        dct::forward_dct_8x8(&pattern, &mut rust_coeffs);

        // C implementation
        let mut c_data = pattern;
        unsafe {
            ffi::mozjpeg_test_fdct_islow(c_data.as_mut_ptr());
        }

        // Compare
        for i in 0..64 {
            assert_eq!(
                rust_coeffs[i], c_data[i],
                "DCT mismatch at coeff {} for seed {}: Rust={}, C={}",
                i, seed, rust_coeffs[i], c_data[i]
            );
        }
    }
}

/// Test quality to scale factor conversion matches C
#[test]
fn test_quality_scaling_matches_c() {
    // Test all quality values 1-100
    for q in 1..=100 {
        let rust_scale = quant::quality_to_scale_factor(q as u8);
        let c_scale = unsafe { ffi::mozjpeg_test_quality_scaling(q) };

        assert_eq!(
            rust_scale as i32, c_scale,
            "Quality scaling mismatch for q={}: Rust={}, C={}",
            q, rust_scale, c_scale
        );
    }
}

/// Test RGB to YCbCr conversion matches C
#[test]
fn test_rgb_to_ycbcr_matches_c() {
    // Test corner cases and various colors
    let test_colors: &[(u8, u8, u8)] = &[
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (255, 255, 0),   // Yellow
        (0, 255, 255),   // Cyan
        (255, 0, 255),   // Magenta
        (128, 128, 128), // Gray
        (100, 150, 200), // Random
        (1, 1, 1),       // Near black
        (254, 254, 254), // Near white
    ];

    for &(r, g, b) in test_colors {
        // Rust implementation
        let (rust_y, rust_cb, rust_cr) = color::rgb_to_ycbcr(r, g, b);

        // C implementation
        let mut c_y: i32 = 0;
        let mut c_cb: i32 = 0;
        let mut c_cr: i32 = 0;
        unsafe {
            ffi::mozjpeg_test_rgb_to_ycbcr(
                r as i32, g as i32, b as i32, &mut c_y, &mut c_cb, &mut c_cr,
            );
        }

        // Exact match required - Rust now uses identical formula to C mozjpeg
        assert!(
            rust_y as i32 == c_y && rust_cb as i32 == c_cb && rust_cr as i32 == c_cr,
            "RGB({},{},{}) -> YCbCr mismatch: Rust=({},{},{}), C=({},{},{})",
            r,
            g,
            b,
            rust_y,
            rust_cb,
            rust_cr,
            c_y,
            c_cb,
            c_cr
        );
    }
}

/// Test RGB to YCbCr for all possible values
#[test]
fn test_rgb_to_ycbcr_exhaustive() {
    let mut max_y_diff = 0;
    let mut max_cb_diff = 0;
    let mut max_cr_diff = 0;

    // Sample a grid of RGB values
    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                let (rust_y, rust_cb, rust_cr) = color::rgb_to_ycbcr(r, g, b);

                let mut c_y: i32 = 0;
                let mut c_cb: i32 = 0;
                let mut c_cr: i32 = 0;
                unsafe {
                    ffi::mozjpeg_test_rgb_to_ycbcr(
                        r as i32, g as i32, b as i32, &mut c_y, &mut c_cb, &mut c_cr,
                    );
                }

                max_y_diff = max_y_diff.max((rust_y as i32 - c_y).abs());
                max_cb_diff = max_cb_diff.max((rust_cb as i32 - c_cb).abs());
                max_cr_diff = max_cr_diff.max((rust_cr as i32 - c_cr).abs());
            }
        }
    }

    // Exact match required - Rust now uses identical formula to C mozjpeg
    assert!(
        max_y_diff == 0 && max_cb_diff == 0 && max_cr_diff == 0,
        "Max differences: Y={}, Cb={}, Cr={} (should all be 0)",
        max_y_diff,
        max_cb_diff,
        max_cr_diff
    );
}

/// Test h2v2 downsampling matches C
#[test]
fn test_downsample_h2v2_matches_c() {
    // Test with various input patterns
    let test_cases: &[([u8; 8], [u8; 8])] = &[
        // Uniform
        ([128; 8], [128; 8]),
        // Gradient
        (
            [100, 110, 120, 130, 140, 150, 160, 170],
            [105, 115, 125, 135, 145, 155, 165, 175],
        ),
        // Alternating
        (
            [0, 255, 0, 255, 0, 255, 0, 255],
            [255, 0, 255, 0, 255, 0, 255, 0],
        ),
        // Edge case
        (
            [0, 0, 255, 255, 0, 0, 255, 255],
            [0, 0, 255, 255, 0, 0, 255, 255],
        ),
    ];

    for (row0, row1) in test_cases {
        // Rust implementation
        let mut rust_output = [0u8; 4];
        sample::downsample_h2v2_rows(row0, row1, &mut rust_output);

        // C implementation
        let mut c_output = [0u8; 4];
        unsafe {
            ffi::mozjpeg_test_downsample_h2v2(
                row0.as_ptr(),
                row1.as_ptr(),
                c_output.as_mut_ptr(),
                8,
            );
        }

        // Compare
        for i in 0..4 {
            assert_eq!(
                rust_output[i], c_output[i],
                "h2v2 downsample mismatch at {}: Rust={}, C={} (input rows: {:?}, {:?})",
                i, rust_output[i], c_output[i], row0, row1
            );
        }
    }
}

/// Test coefficient quantization matches C
#[test]
fn test_quantize_coef_matches_c() {
    // Test various coefficient/quantval combinations
    let test_cases: &[(i16, u16)] = &[
        (0, 16),
        (100, 16),
        (-100, 16),
        (1000, 16),
        (-1000, 16),
        (8, 16),
        (-8, 16),
        (7, 16),
        (-7, 16),
        (1, 1),
        (255, 10),
        (-255, 10),
        (32767, 100),
        (-32767, 100),
    ];

    for &(coef, quantval) in test_cases {
        let c_result = unsafe { ffi::mozjpeg_test_quantize_coef(coef, quantval) };

        // Rust implementation of same algorithm (round to nearest)
        let rust_result = if coef < 0 {
            let temp = (-coef as u32 + quantval as u32 / 2) / quantval as u32;
            -(temp as i16)
        } else {
            let temp = (coef as u32 + quantval as u32 / 2) / quantval as u32;
            temp as i16
        };

        assert_eq!(
            rust_result, c_result,
            "Quantize mismatch for coef={}, quantval={}: Rust={}, C={}",
            coef, quantval, rust_result, c_result
        );
    }
}

/// Test nbits calculation matches C
#[test]
fn test_nbits_matches_c() {
    // Test various values
    for val in 0..=1024 {
        let c_result = unsafe { ffi::mozjpeg_test_nbits(val) };

        // Rust implementation
        let rust_result = if val == 0 {
            0
        } else {
            32 - (val as u32).leading_zeros() as i32
        };

        assert_eq!(
            rust_result, c_result,
            "nbits mismatch for {}: Rust={}, C={}",
            val, rust_result, c_result
        );
    }

    // Also test some larger values
    for val in [
        2000, 4095, 4096, 8191, 8192, 16383, 16384, 32767, 32768, 65535,
    ] {
        let c_result = unsafe { ffi::mozjpeg_test_nbits(val) };
        let rust_result = 32 - (val as u32).leading_zeros() as i32;

        assert_eq!(
            rust_result, c_result,
            "nbits mismatch for {}: Rust={}, C={}",
            val, rust_result, c_result
        );
    }
}

/// Test that Rust encoder produces valid JPEG that decodes correctly
#[test]
fn test_rust_encoder_quality() {
    use mozjpeg_oxide::{Encoder, Subsampling};

    // Create a simple test image (smooth gradient is easier to compress)
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    // Create a smooth gradient
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let r = (x * 4) as u8;
            let g = (y * 4) as u8;
            let b = ((x + y) * 2) as u8;
            rgb_data[i * 3] = r;
            rgb_data[i * 3 + 1] = g;
            rgb_data[i * 3 + 2] = b;
        }
    }

    // Encode with Rust implementation
    let rust_encoder = Encoder::new().quality(75).subsampling(Subsampling::S420);
    let rust_jpeg = rust_encoder.encode_rgb(&rgb_data, width, height).unwrap();

    // Verify Rust output is valid JPEG
    assert!(rust_jpeg.len() > 100, "Rust JPEG too small");
    assert_eq!(rust_jpeg[0], 0xFF);
    assert_eq!(rust_jpeg[1], 0xD8); // SOI
    assert_eq!(rust_jpeg[rust_jpeg.len() - 2], 0xFF);
    assert_eq!(rust_jpeg[rust_jpeg.len() - 1], 0xD9); // EOI

    // Decode Rust output
    let mut rust_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg));
    let rust_decoded = rust_decoder.decode().expect("Failed to decode Rust JPEG");
    let rust_info = rust_decoder.info().unwrap();

    assert_eq!(rust_info.width, width as u16);
    assert_eq!(rust_info.height, height as u16);
    assert_eq!(rust_decoded.len(), rgb_data.len());

    println!("Rust encoder Q75 results:");
    println!("  JPEG size: {} bytes", rust_jpeg.len());
    println!("  Image size: {}x{}", width, height);
}

/// Test Rust encoder at different quality levels (standard qualities)
#[test]
fn test_rust_encoder_quality_levels() {
    use mozjpeg_oxide::{Encoder, Subsampling};

    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (x * 8) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    // Test common quality levels (skip Q90+ as they may have edge cases)
    let quality_levels = [25, 50, 75, 85];

    println!("\nQuality level comparison:");
    for quality in quality_levels {
        let encoder = Encoder::new()
            .quality(quality)
            .subsampling(Subsampling::S420);
        let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Verify it's decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        decoder
            .decode()
            .expect(&format!("Failed to decode Q{} JPEG", quality));

        println!("  Q{:3}: {} bytes", quality, jpeg.len());
    }
}

/// Test that Rust deringing matches C deringing.
///
/// This test compares the output of preprocess_deringing between Rust and C
/// implementations to ensure they produce identical results.
#[test]
fn test_deringing_matches_c() {
    use mozjpeg_oxide::consts::JPEG_NATURAL_ORDER;
    use mozjpeg_oxide::deringing::preprocess_deringing;

    const MAX_SAMPLE: i16 = 127; // 255 - 128

    // Test case 1: No max pixels (should be unchanged in both)
    {
        let mut rust_data = [64i16; 64];
        let mut c_data = [64i16; 64];

        preprocess_deringing(&mut rust_data, 16);
        unsafe {
            ffi::mozjpeg_test_preprocess_deringing(c_data.as_mut_ptr(), 16);
        }

        assert_eq!(rust_data, c_data, "No max pixels case should match");
    }

    // Test case 2: All max pixels (should be unchanged in both)
    {
        let mut rust_data = [MAX_SAMPLE; 64];
        let mut c_data = [MAX_SAMPLE; 64];

        preprocess_deringing(&mut rust_data, 16);
        unsafe {
            ffi::mozjpeg_test_preprocess_deringing(c_data.as_mut_ptr(), 16);
        }

        assert_eq!(rust_data, c_data, "All max pixels case should match");
    }

    // Test case 3: Run of max pixels with surrounding slope
    {
        let mut rust_data = [0i16; 64];
        let mut c_data = [0i16; 64];

        // Set some pixels to max value (indices 10-15 in natural order)
        for i in 10..16 {
            rust_data[JPEG_NATURAL_ORDER[i]] = MAX_SAMPLE;
            c_data[JPEG_NATURAL_ORDER[i]] = MAX_SAMPLE;
        }
        // Set surrounding pixels to create a slope
        rust_data[JPEG_NATURAL_ORDER[8]] = 80;
        rust_data[JPEG_NATURAL_ORDER[9]] = 100;
        rust_data[JPEG_NATURAL_ORDER[16]] = 100;
        rust_data[JPEG_NATURAL_ORDER[17]] = 80;

        c_data[JPEG_NATURAL_ORDER[8]] = 80;
        c_data[JPEG_NATURAL_ORDER[9]] = 100;
        c_data[JPEG_NATURAL_ORDER[16]] = 100;
        c_data[JPEG_NATURAL_ORDER[17]] = 80;

        preprocess_deringing(&mut rust_data, 16);
        unsafe {
            ffi::mozjpeg_test_preprocess_deringing(c_data.as_mut_ptr(), 16);
        }

        // Compare each coefficient - exact match required
        let mut max_diff = 0i16;
        for i in 0..64 {
            let diff = (rust_data[i] - c_data[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff == 0,
                "Deringing mismatch at index {} (natural order): Rust={}, C={}, diff={}",
                i,
                rust_data[i],
                c_data[i],
                diff
            );
        }

        println!("Deringing run test max_diff: {}", max_diff);
    }

    // Test case 4: Different DC quant values
    for dc_quant in [2u16, 8, 16, 32] {
        let mut rust_data = [0i16; 64];
        let mut c_data = [0i16; 64];

        // Create a pattern with max pixels
        for i in 5..15 {
            rust_data[JPEG_NATURAL_ORDER[i]] = MAX_SAMPLE;
            c_data[JPEG_NATURAL_ORDER[i]] = MAX_SAMPLE;
        }
        // Surrounding slopes
        rust_data[JPEG_NATURAL_ORDER[3]] = 50;
        rust_data[JPEG_NATURAL_ORDER[4]] = 90;
        rust_data[JPEG_NATURAL_ORDER[15]] = 90;
        rust_data[JPEG_NATURAL_ORDER[16]] = 50;

        c_data[JPEG_NATURAL_ORDER[3]] = 50;
        c_data[JPEG_NATURAL_ORDER[4]] = 90;
        c_data[JPEG_NATURAL_ORDER[15]] = 90;
        c_data[JPEG_NATURAL_ORDER[16]] = 50;

        preprocess_deringing(&mut rust_data, dc_quant);
        unsafe {
            ffi::mozjpeg_test_preprocess_deringing(c_data.as_mut_ptr(), dc_quant);
        }

        // Compare - exact match required
        for i in 0..64 {
            let diff = (rust_data[i] - c_data[i]).abs();
            assert!(
                diff == 0,
                "DC quant {} mismatch at {}: Rust={}, C={}, diff={}",
                dc_quant,
                i,
                rust_data[i],
                c_data[i],
                diff
            );
        }
    }

    println!("All deringing comparison tests passed!");
}

/// Test that Rust trellis quantization matches C trellis quantization.
///
/// This is a critical test for file size parity - the trellis DP algorithm
/// must produce identical decisions to C mozjpeg.
#[test]
fn test_trellis_matches_c() {
    use mozjpeg_oxide::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, STD_LUMINANCE_QUANT_TBL};
    use mozjpeg_oxide::huffman::{DerivedTable, HuffTable};
    use mozjpeg_oxide::trellis::trellis_quantize_block;
    use mozjpeg_oxide::TrellisConfig;

    // Build the AC Huffman table to get code sizes
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    let ac_table = DerivedTable::from_huff_table(&htbl, false).unwrap();

    // Extract code sizes for C function
    let mut ac_huffsi = [0i8; 256];
    for i in 0..256 {
        let (_, size) = ac_table.get_code(i as u8);
        ac_huffsi[i] = size as i8;
    }

    // Use standard luminance quantization table at Q75
    let qtbl = STD_LUMINANCE_QUANT_TBL[0];

    // Default trellis config
    let config = TrellisConfig::default();

    // Test with various input patterns
    let test_patterns: Vec<[i16; 64]> = vec![
        // Pattern 1: Moderate coefficients
        {
            let mut arr = [0i16; 64];
            arr[0] = 1000; // DC
            arr[1] = 200;
            arr[8] = 150;
            arr[2] = 100;
            arr[9] = 80;
            arr[16] = 50;
            arr
        },
        // Pattern 2: High DC, sparse AC
        {
            let mut arr = [0i16; 64];
            arr[0] = 2000;
            arr[1] = 50;
            arr[8] = 30;
            arr
        },
        // Pattern 3: Gradient-like pattern
        {
            let mut arr = [0i16; 64];
            for i in 0..64 {
                arr[i] = (1000 - i as i16 * 15).max(0);
            }
            arr
        },
        // Pattern 4: Negative coefficients
        {
            let mut arr = [0i16; 64];
            arr[0] = 800;
            arr[1] = -150;
            arr[8] = 100;
            arr[2] = -75;
            arr
        },
        // Pattern 5: Large coefficients (high detail)
        {
            let mut arr = [0i16; 64];
            arr[0] = 3000;
            arr[1] = 500;
            arr[8] = 400;
            arr[2] = 350;
            arr[9] = 300;
            arr[16] = 250;
            arr[3] = 200;
            arr[10] = 180;
            arr[17] = 160;
            arr[24] = 140;
            arr
        },
    ];

    let mut total_diffs = 0;
    let mut total_coeffs = 0;

    for (pattern_idx, pattern) in test_patterns.iter().enumerate() {
        // Scale input by 8 to match raw DCT format (DCT output is scaled by 8)
        let mut src = [0i32; 64];
        for i in 0..64 {
            src[i] = pattern[i] as i32 * 8;
        }

        // Rust trellis
        let mut rust_quantized = [0i16; 64];
        trellis_quantize_block(&src, &mut rust_quantized, &qtbl, &ac_table, &config);

        // C trellis (needs i16 input matching JCOEF type)
        let mut c_src = [0i16; 64];
        for i in 0..64 {
            c_src[i] = (src[i] / 8) as i16; // C expects pre-divided values? Let's check
        }

        // Actually the C test export expects the same scaled format
        let mut c_src_scaled = [0i16; 64];
        for i in 0..64 {
            c_src_scaled[i] = src[i] as i16; // Scaled by 8
        }

        let mut c_quantized = [0i16; 64];
        unsafe {
            ffi::mozjpeg_test_trellis_quantize_block(
                c_src_scaled.as_ptr(),
                c_quantized.as_mut_ptr(),
                qtbl.as_ptr(),
                ac_huffsi.as_ptr(),
                config.lambda_log_scale1,
                config.lambda_log_scale2,
            );
        }

        // Compare results
        let mut pattern_diffs = 0;
        for i in 0..64 {
            if rust_quantized[i] != c_quantized[i] {
                pattern_diffs += 1;
                println!(
                    "Pattern {} coef {}: Rust={}, C={}",
                    pattern_idx, i, rust_quantized[i], c_quantized[i]
                );
            }
        }

        total_diffs += pattern_diffs;
        total_coeffs += 64;

        if pattern_diffs > 0 {
            println!(
                "Pattern {} had {} differences out of 64 coefficients",
                pattern_idx, pattern_diffs
            );
            println!("  Rust: {:?}", &rust_quantized[..16]);
            println!("  C:    {:?}", &c_quantized[..16]);
        }
    }

    println!(
        "\nTrellis comparison: {} differences out of {} total coefficients",
        total_diffs, total_coeffs
    );

    assert_eq!(
        total_diffs, 0,
        "Trellis quantization should produce identical results to C mozjpeg"
    );
}

/// Test trellis with random-ish realistic DCT coefficients.
///
/// Uses pseudo-random values that mimic actual DCT output patterns.
#[test]
fn test_trellis_matches_c_random() {
    use mozjpeg_oxide::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, STD_LUMINANCE_QUANT_TBL};
    use mozjpeg_oxide::huffman::{DerivedTable, HuffTable};
    use mozjpeg_oxide::trellis::trellis_quantize_block;
    use mozjpeg_oxide::TrellisConfig;

    // Build AC Huffman table
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    let ac_table = DerivedTable::from_huff_table(&htbl, false).unwrap();

    // Extract code sizes
    let mut ac_huffsi = [0i8; 256];
    for i in 0..256 {
        let (_, size) = ac_table.get_code(i as u8);
        ac_huffsi[i] = size as i8;
    }

    let qtbl = STD_LUMINANCE_QUANT_TBL[0];
    let config = TrellisConfig::default();

    let mut total_diffs = 0;
    let mut total_coeffs = 0;

    // Test with many pseudo-random patterns
    for seed in 0..50 {
        // Generate pseudo-random DCT-like coefficients
        // DC is large, AC coefficients decay with frequency (typical of natural images)
        // Note: Values must fit in i16 after scaling by 8, so max raw value is ~4000
        let mut pattern = [0i16; 64];
        pattern[0] = 500 + (seed * 20) as i16; // DC coefficient (max ~1500)

        for i in 1..64 {
            // Coefficients decay with frequency position
            // Add some randomness
            let freq_factor = 1.0 / (1.0 + (i as f32).sqrt());
            let random_part = ((seed * 7 + i * 13 + i * i) % 256) as i16 - 128;
            pattern[i] = ((random_part as f32 * freq_factor * 2.0) as i16).clamp(-1000, 1000);
        }

        // Scale by 8 for raw DCT format (matches how DCT output is scaled)
        // Values must fit in i16 for FFI call
        let mut src = [0i32; 64];
        for i in 0..64 {
            src[i] = pattern[i] as i32 * 8;
        }

        // Verify no overflow when converting to i16
        for i in 0..64 {
            assert!(
                src[i] >= -32768 && src[i] <= 32767,
                "src[{}] = {} overflows i16",
                i,
                src[i]
            );
        }

        // Rust trellis
        let mut rust_quantized = [0i16; 64];
        trellis_quantize_block(&src, &mut rust_quantized, &qtbl, &ac_table, &config);

        // C trellis
        let mut c_src_scaled = [0i16; 64];
        for i in 0..64 {
            c_src_scaled[i] = src[i] as i16;
        }

        let mut c_quantized = [0i16; 64];
        unsafe {
            ffi::mozjpeg_test_trellis_quantize_block(
                c_src_scaled.as_ptr(),
                c_quantized.as_mut_ptr(),
                qtbl.as_ptr(),
                ac_huffsi.as_ptr(),
                config.lambda_log_scale1,
                config.lambda_log_scale2,
            );
        }

        // Compare
        let mut pattern_diffs = 0;
        for i in 0..64 {
            if rust_quantized[i] != c_quantized[i] {
                pattern_diffs += 1;
                if pattern_diffs <= 5 {
                    println!(
                        "Seed {} coef {}: Rust={}, C={}",
                        seed, i, rust_quantized[i], c_quantized[i]
                    );
                }
            }
        }

        total_diffs += pattern_diffs;
        total_coeffs += 64;
    }

    println!(
        "\nRandom trellis comparison: {} differences out of {} coefficients",
        total_diffs, total_coeffs
    );

    assert_eq!(
        total_diffs, 0,
        "Random trellis test: expected 0 differences"
    );
}

/// Test trellis at different quality levels (which affects quantization tables).
#[test]
fn test_trellis_matches_c_quality_levels() {
    use mozjpeg_oxide::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, STD_LUMINANCE_QUANT_TBL};
    use mozjpeg_oxide::huffman::{DerivedTable, HuffTable};
    use mozjpeg_oxide::quant::quality_to_scale_factor;
    use mozjpeg_oxide::trellis::trellis_quantize_block;
    use mozjpeg_oxide::TrellisConfig;

    // Build AC Huffman table
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    let ac_table = DerivedTable::from_huff_table(&htbl, false).unwrap();

    let mut ac_huffsi = [0i8; 256];
    for i in 0..256 {
        let (_, size) = ac_table.get_code(i as u8);
        ac_huffsi[i] = size as i8;
    }

    let config = TrellisConfig::default();

    // Test quality levels that are of interest (especially high qualities where the gap was observed)
    let quality_levels = [50, 75, 85, 90, 95, 97];

    for quality in quality_levels {
        let scale_factor = quality_to_scale_factor(quality);

        // Scale the standard quantization table by quality
        let mut qtbl = [0u16; 64];
        for i in 0..64 {
            let val = (STD_LUMINANCE_QUANT_TBL[0][i] as u32 * scale_factor as u32 + 50) / 100;
            qtbl[i] = val.clamp(1, 255) as u16;
        }

        let mut total_diffs = 0;

        // Test with 20 patterns per quality level
        // Values must fit in i16 after scaling by 8
        for seed in 0..20 {
            let mut pattern = [0i16; 64];
            pattern[0] = 800 + (seed * 30) as i16; // max 800+570=1370, *8=10960, fits in i16

            for i in 1..64 {
                let freq_factor = 1.0 / (1.0 + (i as f32).sqrt());
                let random_part = ((seed * 11 + i * 17) % 200) as i16 - 100;
                pattern[i] = ((random_part as f32 * freq_factor * 2.0) as i16).clamp(-500, 500);
            }

            let mut src = [0i32; 64];
            for i in 0..64 {
                src[i] = pattern[i] as i32 * 8;
            }

            // Verify no overflow
            for i in 0..64 {
                assert!(
                    src[i] >= -32768 && src[i] <= 32767,
                    "Q{} seed {} src[{}] = {} overflows i16",
                    quality,
                    seed,
                    i,
                    src[i]
                );
            }

            let mut rust_quantized = [0i16; 64];
            trellis_quantize_block(&src, &mut rust_quantized, &qtbl, &ac_table, &config);

            let mut c_src_scaled = [0i16; 64];
            for i in 0..64 {
                c_src_scaled[i] = src[i] as i16;
            }

            let mut c_quantized = [0i16; 64];
            unsafe {
                ffi::mozjpeg_test_trellis_quantize_block(
                    c_src_scaled.as_ptr(),
                    c_quantized.as_mut_ptr(),
                    qtbl.as_ptr(),
                    ac_huffsi.as_ptr(),
                    config.lambda_log_scale1,
                    config.lambda_log_scale2,
                );
            }

            for i in 0..64 {
                if rust_quantized[i] != c_quantized[i] {
                    total_diffs += 1;
                    if total_diffs <= 5 {
                        println!(
                            "  Q{} seed {} coef {}: Rust={}, C={}, qtbl[{}]={}",
                            quality, seed, i, rust_quantized[i], c_quantized[i], i, qtbl[i]
                        );
                    }
                }
            }
        }

        println!("Q{}: {} differences", quality, total_diffs);
        if total_diffs > 0 {
            println!("  Q{} DC quant value: {}", quality, qtbl[0]);
            println!(
                "  Lambda: scale1={}, scale2={}",
                config.lambda_log_scale1, config.lambda_log_scale2
            );
        }
        assert_eq!(
            total_diffs, 0,
            "Q{} trellis should match C exactly",
            quality
        );
    }
}

/// Test that Rust DC trellis optimization matches C DC trellis optimization.
///
/// This is critical for file size parity - DC coefficients use DPCM coding
/// and the trellis optimizes the entire chain of DC values.
#[test]
fn test_dc_trellis_matches_c() {
    use mozjpeg_oxide::consts::{DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES, DCTSIZE2};
    use mozjpeg_oxide::huffman::{DerivedTable, HuffTable};
    use mozjpeg_oxide::trellis::dc_trellis_optimize;
    use mozjpeg_oxide::TrellisConfig;

    // Build the DC Huffman table
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    let dc_table = DerivedTable::from_huff_table(&htbl, true).unwrap();

    // Extract DC code sizes (for bits 0-16)
    let mut dc_huffsi = [0i8; 17];
    for bits in 0..=16 {
        let (_, size) = dc_table.get_code(bits as u8);
        dc_huffsi[bits] = size as i8;
    }

    let config = TrellisConfig::default();

    // Test with various block sequences
    let num_blocks = 16; // Test with 16 blocks (one row)
    let dc_quantval: u16 = 8; // Typical DC quant value at Q75

    // Test pattern: pseudo-random DC values like a real image row
    for seed in 0..10 {
        // Generate raw DC values (scaled by 8)
        let mut raw_dc = vec![0i32; num_blocks];
        let mut ac_norms = vec![0.0f32; num_blocks];

        for i in 0..num_blocks {
            // DC value varies smoothly like a real image
            let base_dc = 1000 + (seed * 50) as i32;
            let variation = ((i as i32 * 7 + seed as i32 * 13) % 200) - 100;
            raw_dc[i] = (base_dc + variation) * 8; // Scale by 8

            // AC norm (block energy) - typical values
            ac_norms[i] = 5000.0 + ((i * 1000) % 3000) as f32;
        }

        // Create mock raw DCT blocks for Rust (only DC is used)
        let mut raw_dct_blocks: Vec<[i32; DCTSIZE2]> = vec![[0i32; DCTSIZE2]; num_blocks];
        for i in 0..num_blocks {
            raw_dct_blocks[i][0] = raw_dc[i];
            // Fill AC with values that produce the same norm
            let ac_val = (ac_norms[i] * 63.0).sqrt() as i32;
            raw_dct_blocks[i][1] = ac_val;
        }

        // Pre-quantize blocks (DC trellis modifies existing quantized DC values)
        let mut rust_quantized: Vec<[i16; DCTSIZE2]> = vec![[0i16; DCTSIZE2]; num_blocks];
        let q = 8 * dc_quantval as i32;
        for i in 0..num_blocks {
            let x = raw_dc[i].abs();
            let sign = if raw_dc[i] < 0 { -1i16 } else { 1i16 };
            let qval = ((x + q / 2) / q).min(1023) as i16;
            rust_quantized[i][0] = qval * sign;
        }

        // Run Rust DC trellis
        dc_trellis_optimize(
            &raw_dct_blocks,
            &mut rust_quantized,
            dc_quantval,
            &dc_table,
            0, // last_dc = 0
            config.lambda_log_scale1,
            config.lambda_log_scale2,
        );

        // Extract Rust DC results
        let rust_dc: Vec<i16> = rust_quantized.iter().map(|b| b[0]).collect();

        // Run C DC trellis
        let mut c_quantized_dc = vec![0i16; num_blocks];
        unsafe {
            ffi::mozjpeg_test_dc_trellis_optimize(
                raw_dc.as_ptr(),
                ac_norms.as_ptr(),
                c_quantized_dc.as_mut_ptr(),
                num_blocks as i32,
                dc_quantval,
                dc_huffsi.as_ptr(),
                0, // last_dc = 0
                config.lambda_log_scale1,
                config.lambda_log_scale2,
            );
        }

        // Compare results
        let mut diffs = 0;
        for i in 0..num_blocks {
            if rust_dc[i] != c_quantized_dc[i] {
                diffs += 1;
                if diffs <= 5 {
                    println!(
                        "Seed {} block {}: Rust={}, C={}, raw_dc={}, ac_norm={}",
                        seed, i, rust_dc[i], c_quantized_dc[i], raw_dc[i], ac_norms[i]
                    );
                }
            }
        }

        if diffs > 0 {
            println!("Seed {}: {} differences out of {} blocks", seed, diffs, num_blocks);
            println!("  DC quant value: {}", dc_quantval);
            println!("  Lambda: scale1={}, scale2={}", config.lambda_log_scale1, config.lambda_log_scale2);
        }

        assert_eq!(
            diffs, 0,
            "Seed {}: DC trellis should produce identical results to C",
            seed
        );
    }

    println!("DC trellis comparison: all tests passed!");
}
