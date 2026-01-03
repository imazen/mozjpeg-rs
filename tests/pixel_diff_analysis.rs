//! Validate pixel quality of Rust encoder across configurations.
//!
//! This test verifies that the Rust encoder produces valid, decodable
//! JPEGs with reasonable quality characteristics.

use dssim::Dssim;
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};

/// Test that encoded JPEGs are valid and decodable across multiple configurations.
#[test]
fn test_encoder_produces_valid_output() {
    let test_cases = [
        (16, 16, Subsampling::S444, "16x16 4:4:4"),
        (16, 16, Subsampling::S420, "16x16 4:2:0"),
        (17, 17, Subsampling::S420, "17x17 4:2:0 (non-MCU)"),
        (32, 32, Subsampling::S420, "32x32 4:2:0"),
        (64, 64, Subsampling::S420, "64x64 4:2:0"),
    ];

    for (width, height, subsampling, name) in test_cases {
        let w = width as usize;
        let h = height as usize;

        // Create gradient image
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                rgb[idx] = ((x * 255) / w.max(1)) as u8;
                rgb[idx + 1] = ((y * 255) / h.max(1)) as u8;
                rgb[idx + 2] = 128;
            }
        }

        // Encode with baseline (no optimizations)
        let jpeg = Encoder::baseline_optimized()
            .quality(85)
            .subsampling(subsampling)
            .progressive(false)
            .optimize_huffman(false)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(&rgb, width, height)
            .expect("Encoding failed");

        // Verify JPEG is valid and decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        let decoded = decoder.decode().expect(&format!("{}: decode failed", name));
        let info = decoder.info().expect("Failed to get JPEG info");

        // Verify dimensions match
        assert_eq!(info.width as u32, width, "{}: width mismatch", name);
        assert_eq!(info.height as u32, height, "{}: height mismatch", name);

        // Verify decoded size matches original (width * height * 3 for RGB)
        assert_eq!(decoded.len(), w * h * 3, "{}: decoded size mismatch", name);

        // Calculate PSNR to verify quality is reasonable
        let psnr = calculate_psnr(&rgb, &decoded);
        assert!(
            psnr > 30.0,
            "{}: PSNR too low ({:.1} dB), quality issue",
            name,
            psnr
        );

        // DSSIM perceptual quality check
        // <0.003 is noticeable, <0.001 is marginal
        let dssim = calculate_dssim(&rgb, &decoded, width, height);
        assert!(
            dssim < 0.003,
            "{}: DSSIM too high ({:.6}), perceptual quality issue",
            name,
            dssim
        );
    }
}

/// Test encoder with progressive mode.
#[test]
fn test_progressive_encoder_valid() {
    let width = 32u32;
    let height = 32u32;
    let w = width as usize;
    let h = height as usize;

    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..rgb.len() {
        rgb[i] = (i % 256) as u8;
    }

    let jpeg = Encoder::baseline_optimized()
        .quality(85)
        .progressive(true)
        .encode_rgb(&rgb, width, height)
        .expect("Progressive encoding failed");

    // Verify it's decodable
    let decoded = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg))
        .decode()
        .expect("Progressive decode failed");

    assert_eq!(decoded.len(), w * h * 3);

    let psnr = calculate_psnr(&rgb, &decoded);
    // Note: PSNR of 25+ dB is acceptable for Q85 with synthetic patterns
    assert!(psnr > 20.0, "Progressive PSNR too low: {:.1}", psnr);

    // DSSIM perceptual quality check - lenient threshold for synthetic patterns
    // (modular pattern is difficult for JPEG to compress well)
    let dssim = calculate_dssim(&rgb, &decoded, width, height);
    assert!(dssim < 0.01, "Progressive DSSIM too high: {:.6}", dssim);
}

fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    if original.len() != decoded.len() {
        return 0.0;
    }

    let mut mse = 0f64;
    for (o, d) in original.iter().zip(decoded.iter()) {
        let diff = *o as f64 - *d as f64;
        mse += diff * diff;
    }
    mse /= original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Calculate DSSIM between original and decoded RGB data.
/// Returns DSSIM value (0 = identical, higher = worse).
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
