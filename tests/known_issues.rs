//! Tests for known issues in mozjpeg-oxide.
//!
//! These tests document bugs that need to be fixed.
//! They are marked #[ignore] until the issues are resolved.

/// Create a gradient test image
fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 4) % 256) as u8);
            rgb.push(((y * 4) % 256) as u8);
            rgb.push((((x + y) * 2) % 256) as u8);
        }
    }
    rgb
}

/// Test progressive grayscale encoding.
///
/// When `.progressive(true)` is set for grayscale encoding, the output
/// should contain SOF2 marker (0xFFC2) for progressive DCT.
///
/// This was previously Issue #1 - now fixed and this test validates the feature.
#[test]
fn test_progressive_grayscale_encoding() {
    let width = 64usize;
    let height = 64usize;

    // Create grayscale gradient
    let gray: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let jpeg = mozjpeg_rs::Encoder::new()
        .quality(80)
        .progressive(true)
        .encode_gray(&gray, width as u32, height as u32)
        .expect("Encoding failed");

    // Check for SOF2 marker (progressive DCT)
    let mut found_sof2 = false;
    let mut found_sof0 = false;
    for i in 0..jpeg.len().saturating_sub(1) {
        if jpeg[i] == 0xFF {
            if jpeg[i + 1] == 0xC2 {
                found_sof2 = true;
            }
            if jpeg[i + 1] == 0xC0 {
                found_sof0 = true;
            }
        }
    }

    assert!(
        found_sof2,
        "Progressive grayscale should have SOF2 marker. Found SOF0={}, SOF2={}",
        found_sof0, found_sof2
    );

    // Verify it decodes correctly
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    let decoded = decoder.decode().expect("Decode failed");
    assert_eq!(decoded.len(), width * height, "Wrong decoded size");
}

/// Issue: Non-monotonic file sizes around Q70.
///
/// Higher quality settings should produce equal or larger files.
/// Currently, Q70 produces smaller files than Q50 for some images,
/// which violates the expected monotonic relationship.
///
/// Expected: size(Q70) >= size(Q50)
/// Actual: size(Q70) < size(Q50) for gradient images
///
/// See: https://github.com/imazen/mozjpeg-oxide/issues/1
#[test]
#[ignore = "mozjpeg-oxide has non-monotonic file sizes around Q70 - issue #1"]
fn test_quality_monotonicity() {
    let width = 64usize;
    let height = 64usize;
    let rgb = create_gradient_image(width, height);

    let qualities = [30u8, 50, 70, 85, 95];
    let mut prev_size = 0usize;
    let mut prev_q = 0u8;

    for q in qualities {
        let jpeg = mozjpeg_rs::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_rs::Subsampling::S444)
            .optimize_huffman(true)
            .encode_rgb(&rgb, width as u32, height as u32)
            .expect("Encoding failed");

        if prev_q > 0 {
            assert!(
                jpeg.len() >= prev_size,
                "Quality monotonicity violated: Q{} ({} bytes) < Q{} ({} bytes)",
                q,
                jpeg.len(),
                prev_q,
                prev_size
            );
        }

        prev_size = jpeg.len();
        prev_q = q;
    }
}

/// Demonstrate the Q70 issue with specific output.
#[test]
fn test_quality_sizes_for_debugging() {
    let width = 64usize;
    let height = 64usize;
    let rgb = create_gradient_image(width, height);

    println!("\n=== Quality vs File Size (64x64 gradient) ===");
    println!("Settings: 4:4:4, optimize_huffman=true\n");

    for q in [30u8, 50, 70, 85, 95] {
        let jpeg = mozjpeg_rs::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_rs::Subsampling::S444)
            .optimize_huffman(true)
            .encode_rgb(&rgb, width as u32, height as u32)
            .expect("Encoding failed");

        println!("Q{:2}: {} bytes", q, jpeg.len());
    }

    // This test always passes - it's for debugging output
}
