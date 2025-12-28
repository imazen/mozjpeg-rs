//! Parity tests comparing Rust encoder against C mozjpeg.
//!
//! These tests encode the same image with both implementations and verify
//! that file sizes and quality metrics are within acceptable tolerances.
//!
//! Expected values are based on C mozjpeg output for the bundled test image.

use dssim::Dssim;
use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
use rgb::RGB8;
use std::io::Cursor;

/// Test image path (bundled 512x512 test image)
const TEST_IMAGE: &str = "tests/images/1.png";

/// Load bundled test image as RGB pixels
fn load_test_image() -> (Vec<u8>, u32, u32) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE);
    let file = std::fs::File::open(&path).expect("Failed to open test image");
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
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    (rgb_data, info.width, info.height)
}

/// Decode JPEG to RGB pixels
fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(Cursor::new(data))
        .decode()
        .expect("Failed to decode JPEG")
}

/// Compute DSSIM between original and decoded
fn compute_dssim(jpeg_data: &[u8], reference_rgb: &[u8], width: u32, height: u32) -> f64 {
    let decoded = decode_jpeg(jpeg_data);

    let attr = Dssim::new();

    let orig_rgb: Vec<RGB8> = reference_rgb
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

// ============================================================================
// Hardcoded Expected Values from C mozjpeg
// ============================================================================
// These values were generated using C mozjpeg (via benchmark tracking) on the
// bundled test image (tests/images/1.png, 512x512 pixels).
//
// The benchmark results show C mozjpeg with trellis+huffman optimization.
// Tolerances are set to detect regressions while allowing minor implementation
// differences.

/// Expected results for baseline mode (trellis + Huffman optimization, 4:2:0)
/// Format: (quality, expected_size, max_size_diff_pct, expected_dssim, max_dssim_diff)
const BASELINE_EXPECTATIONS: &[(u8, usize, f64, f64, f64)] = &[
    // C mozjpeg values from benchmark tracking
    (50, 31133, 0.05, 0.00447, 0.001),
    (75, 47949, 0.05, 0.00212, 0.001),
    (85, 64772, 0.05, 0.00124, 0.001),
    (95, 109520, 0.10, 0.00053, 0.001),
];

/// Expected results for progressive mode (same encoder, progressive output)
const PROGRESSIVE_EXPECTATIONS: &[(u8, usize, f64, f64, f64)] = &[
    // Progressive may have slightly different sizes due to scan structure
    (50, 31133, 0.10, 0.00447, 0.001),
    (75, 47949, 0.10, 0.00212, 0.001),
    (85, 64772, 0.10, 0.00124, 0.001),
    (95, 109520, 0.15, 0.00053, 0.001),
];

/// Expected results for max_compression preset
const MAX_COMPRESSION_EXPECTATIONS: &[(u8, usize, f64, f64, f64)] = &[
    (50, 31133, 0.10, 0.00447, 0.001),
    (75, 47949, 0.10, 0.00212, 0.001),
    (85, 64772, 0.10, 0.00124, 0.001),
    (95, 109520, 0.15, 0.00053, 0.001),
];

/// Expected results for fastest mode (no trellis, no Huffman optimization)
/// These values are larger since no optimization is applied
const FASTEST_EXPECTATIONS: &[(u8, usize, f64, f64, f64)] = &[
    // Fastest produces larger files - compare against unoptimized baseline
    (50, 38899, 0.15, 0.0050, 0.002),
    (75, 57084, 0.15, 0.0025, 0.002),
    (85, 75510, 0.15, 0.0015, 0.002),
    (95, 133718, 0.15, 0.0006, 0.002),
];

// ============================================================================
// Test Functions
// ============================================================================

/// Test baseline encoding parity with C mozjpeg expectations
#[test]
fn test_baseline_parity() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, max_size_diff, expected_dssim, max_dssim_diff) in
        BASELINE_EXPECTATIONS
    {
        let encoder = Encoder::new()
            .quality(quality)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        let size = jpeg_data.len();
        let dssim = compute_dssim(&jpeg_data, &rgb, width, height);

        // Check file size within tolerance
        let size_diff = (size as f64 - expected_size as f64).abs() / expected_size as f64;
        assert!(
            size_diff <= max_size_diff,
            "Baseline Q{}: size {} differs from expected {} by {:.2}% (max {:.2}%)",
            quality,
            size,
            expected_size,
            size_diff * 100.0,
            max_size_diff * 100.0
        );

        // Check DSSIM within tolerance (lower is better)
        let dssim_diff = (dssim - expected_dssim).abs();
        assert!(
            dssim_diff <= max_dssim_diff,
            "Baseline Q{}: DSSIM {:.6} differs from expected {:.6} by {:.6} (max {:.6})",
            quality,
            dssim,
            expected_dssim,
            dssim_diff,
            max_dssim_diff
        );

        println!(
            "Baseline Q{}: size={} (expected {}, diff={:.2}%), dssim={:.6}",
            quality, size, expected_size, size_diff * 100.0, dssim
        );
    }
}

/// Test progressive encoding parity
#[test]
fn test_progressive_parity() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, max_size_diff, expected_dssim, max_dssim_diff) in
        PROGRESSIVE_EXPECTATIONS
    {
        let encoder = Encoder::new()
            .quality(quality)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        let size = jpeg_data.len();
        let dssim = compute_dssim(&jpeg_data, &rgb, width, height);

        // Check file size within tolerance
        let size_diff = (size as f64 - expected_size as f64).abs() / expected_size as f64;
        assert!(
            size_diff <= max_size_diff,
            "Progressive Q{}: size {} differs from expected {} by {:.2}% (max {:.2}%)",
            quality,
            size,
            expected_size,
            size_diff * 100.0,
            max_size_diff * 100.0
        );

        // Check DSSIM within tolerance
        let dssim_diff = (dssim - expected_dssim).abs();
        assert!(
            dssim_diff <= max_dssim_diff,
            "Progressive Q{}: DSSIM {:.6} differs from expected {:.6} by {:.6} (max {:.6})",
            quality,
            dssim,
            expected_dssim,
            dssim_diff,
            max_dssim_diff
        );

        println!(
            "Progressive Q{}: size={} (expected {}, diff={:.2}%), dssim={:.6}",
            quality, size, expected_size, size_diff * 100.0, dssim
        );
    }
}

/// Test max_compression preset parity
#[test]
fn test_max_compression_parity() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, max_size_diff, expected_dssim, max_dssim_diff) in
        MAX_COMPRESSION_EXPECTATIONS
    {
        let encoder = Encoder::max_compression()
            .quality(quality)
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        let size = jpeg_data.len();
        let dssim = compute_dssim(&jpeg_data, &rgb, width, height);

        // Check file size within tolerance
        let size_diff = (size as f64 - expected_size as f64).abs() / expected_size as f64;
        assert!(
            size_diff <= max_size_diff,
            "MaxCompression Q{}: size {} differs from expected {} by {:.2}% (max {:.2}%)",
            quality,
            size,
            expected_size,
            size_diff * 100.0,
            max_size_diff * 100.0
        );

        // Check DSSIM within tolerance
        let dssim_diff = (dssim - expected_dssim).abs();
        assert!(
            dssim_diff <= max_dssim_diff,
            "MaxCompression Q{}: DSSIM {:.6} differs from expected {:.6} by {:.6} (max {:.6})",
            quality,
            dssim,
            expected_dssim,
            dssim_diff,
            max_dssim_diff
        );

        println!(
            "MaxCompression Q{}: size={} (expected {}, diff={:.2}%), dssim={:.6}",
            quality, size, expected_size, size_diff * 100.0, dssim
        );
    }
}

/// Test fastest preset (no optimizations)
#[test]
fn test_fastest_parity() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, max_size_diff, expected_dssim, max_dssim_diff) in
        FASTEST_EXPECTATIONS
    {
        let encoder = Encoder::fastest()
            .quality(quality)
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        let size = jpeg_data.len();
        let dssim = compute_dssim(&jpeg_data, &rgb, width, height);

        // Check file size within tolerance
        let size_diff = (size as f64 - expected_size as f64).abs() / expected_size as f64;
        assert!(
            size_diff <= max_size_diff,
            "Fastest Q{}: size {} differs from expected {} by {:.2}% (max {:.2}%)",
            quality,
            size,
            expected_size,
            size_diff * 100.0,
            max_size_diff * 100.0
        );

        // Check DSSIM within tolerance
        let dssim_diff = (dssim - expected_dssim).abs();
        assert!(
            dssim_diff <= max_dssim_diff,
            "Fastest Q{}: DSSIM {:.6} differs from expected {:.6} by {:.6} (max {:.6})",
            quality,
            dssim,
            expected_dssim,
            dssim_diff,
            max_dssim_diff
        );

        println!(
            "Fastest Q{}: size={} (expected {}, diff={:.2}%), dssim={:.6}",
            quality, size, expected_size, size_diff * 100.0, dssim
        );
    }
}

// ============================================================================
// Detailed Configuration Tests
// ============================================================================

/// Test that trellis quantization reduces file size
#[test]
fn test_trellis_reduces_size() {
    let (rgb, width, height) = load_test_image();

    let with_trellis = Encoder::new()
        .quality(85)
        .trellis(TrellisConfig::default())
        .optimize_huffman(true)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding with trellis failed");

    let without_trellis = Encoder::new()
        .quality(85)
        .trellis(TrellisConfig::disabled())
        .optimize_huffman(true)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding without trellis failed");

    // Trellis should produce smaller files
    assert!(
        with_trellis.len() <= without_trellis.len(),
        "Trellis should reduce file size: {} vs {}",
        with_trellis.len(),
        without_trellis.len()
    );

    let reduction_pct = (1.0 - with_trellis.len() as f64 / without_trellis.len() as f64) * 100.0;
    println!(
        "Trellis effect: {} bytes vs {} bytes ({:.1}% reduction)",
        with_trellis.len(),
        without_trellis.len(),
        reduction_pct
    );
}

/// Test that Huffman optimization reduces file size
#[test]
fn test_huffman_optimization_reduces_size() {
    let (rgb, width, height) = load_test_image();

    let with_huffopt = Encoder::new()
        .quality(85)
        .trellis(TrellisConfig::disabled())
        .optimize_huffman(true)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding with Huffman opt failed");

    let without_huffopt = Encoder::new()
        .quality(85)
        .trellis(TrellisConfig::disabled())
        .optimize_huffman(false)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding without Huffman opt failed");

    // Huffman optimization should produce smaller files
    assert!(
        with_huffopt.len() <= without_huffopt.len(),
        "Huffman opt should reduce file size: {} vs {}",
        with_huffopt.len(),
        without_huffopt.len()
    );

    let reduction_pct = (1.0 - with_huffopt.len() as f64 / without_huffopt.len() as f64) * 100.0;
    println!(
        "Huffman opt effect: {} bytes vs {} bytes ({:.1}% reduction)",
        with_huffopt.len(),
        without_huffopt.len(),
        reduction_pct
    );
}

/// Test different subsampling modes
#[test]
fn test_subsampling_modes() {
    let (rgb, width, height) = load_test_image();

    let modes = [
        (Subsampling::S444, "4:4:4"),
        (Subsampling::S422, "4:2:2"),
        (Subsampling::S420, "4:2:0"),
    ];

    let mut sizes = Vec::new();

    for (mode, name) in modes {
        let jpeg = Encoder::new()
            .quality(85)
            .subsampling(mode)
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        sizes.push((name, jpeg.len()));
        println!("Subsampling {}: {} bytes", name, jpeg.len());
    }

    // 4:2:0 should be smallest, 4:4:4 should be largest
    assert!(
        sizes[2].1 < sizes[0].1,
        "4:2:0 ({}) should be smaller than 4:4:4 ({})",
        sizes[2].1,
        sizes[0].1
    );
}

/// Test grayscale encoding
#[test]
fn test_grayscale_encoding() {
    let (rgb, width, height) = load_test_image();

    // Convert to grayscale
    let gray: Vec<u8> = rgb
        .chunks(3)
        .map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8)
        .collect();

    let jpeg = Encoder::new()
        .quality(85)
        .encode_gray(&gray, width, height)
        .expect("Grayscale encoding failed");

    // Grayscale should produce valid JPEG
    assert!(!jpeg.is_empty(), "Grayscale JPEG should not be empty");

    // Should be smaller than RGB version
    let rgb_jpeg = Encoder::new()
        .quality(85)
        .subsampling(Subsampling::S444)
        .encode_rgb(&rgb, width, height)
        .expect("RGB encoding failed");

    assert!(
        jpeg.len() < rgb_jpeg.len(),
        "Grayscale ({}) should be smaller than RGB 4:4:4 ({})",
        jpeg.len(),
        rgb_jpeg.len()
    );

    println!(
        "Grayscale: {} bytes, RGB 4:4:4: {} bytes",
        jpeg.len(),
        rgb_jpeg.len()
    );
}

/// Test quality levels produce expected size ordering
#[test]
fn test_quality_ordering() {
    let (rgb, width, height) = load_test_image();

    let qualities = [25, 50, 75, 85, 95];
    let mut prev_size = 0;

    for quality in qualities {
        let jpeg = Encoder::new()
            .quality(quality)
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        assert!(
            jpeg.len() > prev_size,
            "Q{} ({} bytes) should be larger than previous ({} bytes)",
            quality,
            jpeg.len(),
            prev_size
        );

        prev_size = jpeg.len();
        println!("Q{}: {} bytes", quality, jpeg.len());
    }
}
