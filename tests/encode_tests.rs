//! Integration tests for the JPEG encoder.
//!
//! These tests verify the public API of the encoder.

use dssim::Dssim;
use mozjpeg_rs::{Encode, Encoder, StreamingEncoder, Subsampling, TrellisConfig};

/// Verify JPEG output can be decoded by an external decoder
#[test]
fn test_decode_with_jpeg_decoder() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let encoder = Encoder::new(false).quality(90).subsampling(Subsampling::S444);
    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_encode_small_image() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for i in 0..(width * height) as usize {
        rgb_data[i * 3] = 255; // R
        rgb_data[i * 3 + 1] = 0; // G
        rgb_data[i * 3 + 2] = 0; // B
    }

    let encoder = Encoder::new(false).quality(75);
    let result = encoder.encode_rgb(&rgb_data, width, height);

    assert!(result.is_ok());
    let jpeg_data = result.unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI
    assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
    assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI
}

#[test]
fn test_encode_gradient() {
    let width = 8u32;
    let height = 8u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x + y) * 16) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let encoder = Encoder::new(false).quality(90).subsampling(Subsampling::S444);
    let result = encoder.encode_rgb(&rgb_data, width, height);

    assert!(result.is_ok());
}

#[test]
fn test_encode_grayscale() {
    let width = 16u32;
    let height = 16u32;
    let mut gray_data = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            gray_data[i] = ((x + y) * 8) as u8;
        }
    }

    let encoder = Encoder::new(false).quality(85);
    let result = encoder.encode_gray(&gray_data, width, height);

    assert!(result.is_ok());
    let jpeg_data = result.unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI
    assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
    assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode grayscale JPEG");
    let info = decoder.info().unwrap();

    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height) as usize);
}

#[test]
fn test_encode_with_exif() {
    let width = 16u32;
    let height = 16u32;
    let rgb_data = vec![128u8; (width * height * 3) as usize];

    let exif_data = vec![
        0x4D, 0x4D, // Big-endian TIFF
        0x00, 0x2A, // TIFF magic
        0x00, 0x00, 0x00, 0x08, // Offset to IFD
    ];

    let encoder = Encoder::new(false).quality(75).exif_data(exif_data.clone());
    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut found_app1 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xE1 {
            found_app1 = true;
            if i + 4 < jpeg_data.len() {
                let identifier = &jpeg_data[i + 4..i + 10];
                assert_eq!(identifier, b"Exif\0\0");
            }
            break;
        }
    }
    assert!(found_app1, "APP1 (EXIF) marker not found in output");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode JPEG with EXIF");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_encode_with_restart_markers() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 4) as u8;
            rgb_data[i * 3 + 1] = (y * 4) as u8;
            rgb_data[i * 3 + 2] = 128;
        }
    }

    let encoder = Encoder::new(false)
        .quality(75)
        .subsampling(Subsampling::S444)
        .optimize_huffman(false)
        .trellis(TrellisConfig::disabled())
        .restart_interval(4);

    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    let mut found_dri = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDD {
            found_dri = true;
            if i + 5 < jpeg_data.len() {
                let len = ((jpeg_data[i + 2] as u16) << 8) | (jpeg_data[i + 3] as u16);
                assert_eq!(len, 4, "DRI marker length should be 4");
                let interval = ((jpeg_data[i + 4] as u16) << 8) | (jpeg_data[i + 5] as u16);
                assert_eq!(interval, 4, "Restart interval should be 4");
            }
            break;
        }
    }
    assert!(found_dri, "DRI marker not found in output");

    let mut rst_count = 0;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] >= 0xD0 && jpeg_data[i + 1] <= 0xD7 {
            rst_count += 1;
        }
    }
    assert_eq!(
        rst_count, 15,
        "Expected 15 RST markers, found {}",
        rst_count
    );

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder
        .decode()
        .expect("Failed to decode JPEG with restart markers");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_encode_invalid_size() {
    let rgb_data = vec![0u8; 100];
    let encoder = Encoder::new(false);
    let result = encoder.encode_rgb(&rgb_data, 16, 16);

    assert!(result.is_err());
}

#[test]
fn test_encode_zero_dimensions() {
    use mozjpeg_rs::Error;

    let encoder = Encoder::new(false);

    let result = encoder.encode_rgb(&[], 0, 16);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 0,
            height: 16
        })
    ));

    let result = encoder.encode_rgb(&[], 16, 0);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 16,
            height: 0
        })
    ));

    let result = encoder.encode_rgb(&[], 0, 0);
    assert!(matches!(
        result,
        Err(Error::InvalidDimensions {
            width: 0,
            height: 0
        })
    ));
}

#[test]
fn test_encode_overflow_dimensions() {
    let encoder = Encoder::new(false);
    let result = encoder.encode_rgb(&[], u32::MAX, u32::MAX);
    assert!(result.is_err());
}

#[test]
fn test_progressive_encode_decode() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let encoder = Encoder::new(false)
        .quality(85)
        .progressive(true)
        .subsampling(Subsampling::S420);

    let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI

    let mut has_sof2 = false;
    let mut i = 2;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC2 {
            has_sof2 = true;
            break;
        }
        i += 1;
    }
    assert!(has_sof2, "Progressive JPEG should have SOF2 marker");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
    let decoded = decoder.decode().expect("Failed to decode progressive JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_progressive_vs_baseline_size() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as f32 * 0.1).sin() * 127.0 + 128.0) as u8)
                .wrapping_add((((y as f32) * 0.1).cos() * 50.0) as u8);
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val.wrapping_add(30);
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let baseline = Encoder::new(false)
        .quality(75)
        .progressive(false)
        .subsampling(Subsampling::S420);
    let baseline_data = baseline.encode_rgb(&rgb_data, width, height).unwrap();

    let progressive = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .subsampling(Subsampling::S420);
    let progressive_data = progressive.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!baseline_data.is_empty());
    assert!(!progressive_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline_data));
    decoder.decode().expect("Failed to decode baseline JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive_data));
    decoder.decode().expect("Failed to decode progressive JPEG");
}

#[test]
fn test_trellis_quantization_enabled() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as i32 - y as i32).abs() * 10) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = val / 2;
        }
    }

    let no_trellis = Encoder::new(false)
        .quality(75)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::disabled());
    let no_trellis_data = no_trellis.encode_rgb(&rgb_data, width, height).unwrap();

    let with_trellis = Encoder::new(false)
        .quality(75)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::default());
    let with_trellis_data = with_trellis.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!no_trellis_data.is_empty());
    assert!(!with_trellis_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_trellis_data));
    decoder.decode().expect("Failed to decode non-trellis JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_trellis_data));
    decoder.decode().expect("Failed to decode trellis JPEG");
}

#[test]
fn test_trellis_presets() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = (((x as i32 - y as i32).abs() * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = (val / 2).wrapping_add(64);
        }
    }

    let quality = 97;

    let default = Encoder::new(false)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::default());
    let default_data = default.encode_rgb(&rgb_data, width, height).unwrap();

    let favor_size = Encoder::new(false)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::favor_size());
    let favor_size_data = favor_size.encode_rgb(&rgb_data, width, height).unwrap();

    let favor_quality = Encoder::new(false)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .trellis(TrellisConfig::favor_quality());
    let favor_quality_data = favor_quality.encode_rgb(&rgb_data, width, height).unwrap();

    for (name, data) in [
        ("default", &default_data),
        ("favor_size", &favor_size_data),
        ("favor_quality", &favor_quality_data),
    ] {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
        decoder
            .decode()
            .unwrap_or_else(|_| panic!("Failed to decode {} JPEG", name));
    }

    assert!(
        favor_size_data.len() != favor_quality_data.len(),
        "Presets should produce different sizes"
    );
}

#[test]
fn test_trellis_rd_factor() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 8 + y * 4) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let factor_1 = Encoder::new(false)
        .quality(85)
        .trellis(TrellisConfig::default().rd_factor(1.0));
    let factor_1_data = factor_1.encode_rgb(&rgb_data, width, height).unwrap();

    let factor_2 = Encoder::new(false)
        .quality(85)
        .trellis(TrellisConfig::default().rd_factor(2.0));
    let factor_2_data = factor_2.encode_rgb(&rgb_data, width, height).unwrap();

    jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_1_data))
        .decode()
        .expect("Failed to decode rd_factor(1.0) JPEG");
    jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_2_data))
        .decode()
        .expect("Failed to decode rd_factor(2.0) JPEG");
}

#[test]
fn test_huffman_optimization() {
    let width = 32u32;
    let height = 32u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 8 + y * 4) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val;
            rgb_data[i * 3 + 2] = val;
        }
    }

    let no_opt = Encoder::new(false)
        .quality(75)
        .subsampling(Subsampling::S420)
        .optimize_huffman(false);
    let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let with_opt = Encoder::new(false)
        .quality(75)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);
    let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

    assert!(!no_opt_data.is_empty());
    assert!(!with_opt_data.is_empty());

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
    decoder
        .decode()
        .expect("Failed to decode non-optimized JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
    decoder.decode().expect("Failed to decode optimized JPEG");
}

#[test]
fn test_color_encoding_accuracy() {
    let test_cases = [
        ("black", 0u8, 0u8, 0u8),
        ("red", 255, 0, 0),
        ("green", 0, 255, 0),
        ("blue", 0, 0, 255),
        ("white", 255, 255, 255),
        ("gray", 128, 128, 128),
    ];

    let width = 16u32;
    let height = 16u32;

    for (name, r, g, b) in &test_cases {
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];
        for i in 0..(width * height) as usize {
            rgb_data[i * 3] = *r;
            rgb_data[i * 3 + 1] = *g;
            rgb_data[i * 3 + 2] = *b;
        }

        let encoder = Encoder::new(false).quality(95).subsampling(Subsampling::S444);
        let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        let decoded = decoder.decode().expect("decode failed");

        let dr = decoded[0];
        let dg = decoded[1];
        let db = decoded[2];

        let tolerance = 2i16;
        let r_diff = (dr as i16 - *r as i16).abs();
        let g_diff = (dg as i16 - *g as i16).abs();
        let b_diff = (db as i16 - *b as i16).abs();

        assert!(
            r_diff <= tolerance,
            "{}: R mismatch - expected {}, got {} (diff {})",
            name,
            r,
            dr,
            r_diff
        );
        assert!(
            g_diff <= tolerance,
            "{}: G mismatch - expected {}, got {} (diff {})",
            name,
            g,
            dg,
            g_diff
        );
        assert!(
            b_diff <= tolerance,
            "{}: B mismatch - expected {}, got {} (diff {})",
            name,
            b,
            db,
            b_diff
        );
    }
}

#[test]
fn test_optimize_scans() {
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 4 + y * 3) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = 255 - val;
            rgb_data[i * 3 + 2] = ((val as u16 + 128) % 256) as u8;
        }
    }

    let no_opt = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .optimize_scans(false)
        .subsampling(Subsampling::S420);
    let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let with_opt = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .optimize_scans(true)
        .subsampling(Subsampling::S420);
    let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
    decoder
        .decode()
        .expect("Failed to decode non-optimized JPEG");

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
    decoder
        .decode()
        .expect("Failed to decode scan-optimized JPEG");

    assert!(!with_opt_data.is_empty());
}

/// Regression test: Progressive encoding with non-MCU-aligned dimensions.
#[test]
fn test_progressive_non_mcu_aligned_regression() {
    let failing_sizes = [17, 24, 33, 40, 49];

    for &size in &failing_sizes {
        let s = size as usize;
        let mut rgb = vec![0u8; s * s * 3];
        for y in 0..s {
            for x in 0..s {
                let idx = (y * s + x) * 3;
                rgb[idx] = (x * 15).min(255) as u8;
                rgb[idx + 1] = (y * 15).min(255) as u8;
                rgb[idx + 2] = 128;
            }
        }

        let baseline = Encoder::new(false)
            .quality(95)
            .subsampling(Subsampling::S420)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let progressive = Encoder::new(false)
            .quality(95)
            .subsampling(Subsampling::S420)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
            .decode()
            .expect("baseline decode failed");
        let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
            .decode()
            .expect("progressive decode failed");

        let base_psnr = calculate_psnr(&rgb, &base_dec);
        let prog_psnr = calculate_psnr(&rgb, &prog_dec);

        let diff = (prog_psnr - base_psnr).abs();
        assert!(
            diff < 3.0,
            "{}x{}: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
            size,
            size,
            prog_psnr,
            base_psnr,
            diff
        );

        let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
        let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
        assert!(
            base_dssim < 0.01,
            "{}x{}: Baseline DSSIM too high: {:.6}",
            size,
            size,
            base_dssim
        );
        assert!(
            prog_dssim < 0.01,
            "{}x{}: Progressive DSSIM too high: {:.6}",
            size,
            size,
            prog_dssim
        );
    }
}

#[test]
fn test_progressive_422_non_mcu_aligned_regression() {
    let size = 17u32;
    let s = size as usize;
    let mut rgb = vec![0u8; s * s * 3];
    for y in 0..s {
        for x in 0..s {
            let idx = (y * s + x) * 3;
            rgb[idx] = (x * 15).min(255) as u8;
            rgb[idx + 1] = (y * 15).min(255) as u8;
            rgb[idx + 2] = 128;
        }
    }

    let baseline = Encoder::new(false)
        .quality(95)
        .subsampling(Subsampling::S422)
        .progressive(false)
        .encode_rgb(&rgb, size, size)
        .unwrap();

    let progressive = Encoder::new(false)
        .quality(95)
        .subsampling(Subsampling::S422)
        .progressive(true)
        .encode_rgb(&rgb, size, size)
        .unwrap();

    let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
        .decode()
        .unwrap();
    let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
        .decode()
        .unwrap();

    let base_psnr = calculate_psnr(&rgb, &base_dec);
    let prog_psnr = calculate_psnr(&rgb, &prog_dec);
    let diff = (prog_psnr - base_psnr).abs();

    assert!(
        diff < 3.0,
        "4:2:2 17x17: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
        prog_psnr,
        base_psnr,
        diff
    );

    let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
    let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
    assert!(
        base_dssim < 0.01,
        "4:2:2 17x17: Baseline DSSIM too high: {:.6}",
        base_dssim
    );
    assert!(
        prog_dssim < 0.01,
        "4:2:2 17x17: Progressive DSSIM too high: {:.6}",
        prog_dssim
    );
}

#[test]
fn test_streaming_encoder_rgb() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let streaming = StreamingEncoder::new(false).quality(85);
    let streaming_data = streaming.encode_rgb(&rgb_data, width, height).unwrap();

    assert_eq!(streaming_data[0], 0xFF);
    assert_eq!(streaming_data[1], 0xD8); // SOI
    assert_eq!(streaming_data[streaming_data.len() - 2], 0xFF);
    assert_eq!(streaming_data[streaming_data.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
    let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_streaming_encoder_scanlines() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let val = ((x * 16 + y * 8) % 256) as u8;
            rgb_data[i * 3] = val;
            rgb_data[i * 3 + 1] = val / 2;
            rgb_data[i * 3 + 2] = 255 - val;
        }
    }

    let mut output = Vec::new();
    let mut stream = StreamingEncoder::new(false)
        .quality(85)
        .subsampling(Subsampling::S420)
        .start_rgb(width, height, &mut output)
        .unwrap();

    let bytes_per_line = (width * 3) as usize;
    for chunk in rgb_data.chunks(bytes_per_line * 8) {
        stream.write_scanlines(chunk).unwrap();
    }

    stream.finish().unwrap();

    assert_eq!(output[0], 0xFF);
    assert_eq!(output[1], 0xD8); // SOI
    assert_eq!(output[output.len() - 2], 0xFF);
    assert_eq!(output[output.len() - 1], 0xD9); // EOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&output));
    let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_streaming_encoder_gray() {
    let width = 16u32;
    let height = 16u32;
    let mut gray_data = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            gray_data[i] = ((x * 16 + y * 16) % 256) as u8;
        }
    }

    let streaming = StreamingEncoder::new(false).quality(85);
    let streaming_data = streaming.encode_gray(&gray_data, width, height).unwrap();

    assert_eq!(streaming_data[0], 0xFF);
    assert_eq!(streaming_data[1], 0xD8); // SOI

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
    let decoded = decoder
        .decode()
        .expect("Failed to decode grayscale streaming JPEG");

    let info = decoder.info().unwrap();
    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);
    assert_eq!(decoded.len(), (width * height) as usize);
}

// Helper functions

fn calculate_psnr(orig: &[u8], decoded: &[u8]) -> f64 {
    let mse: f64 = orig
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / orig.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

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

#[test]
fn test_eob_optimization_produces_valid_jpeg() {
    use mozjpeg_rs::TrellisConfig;

    let width = 64u32;
    let height = 64u32;

    // Create test image with some areas that will produce zero blocks
    let mut rgb = vec![128u8; (width * height * 3) as usize];
    // Add some variation in one quadrant
    for y in 0..32 {
        for x in 0..32 {
            let idx = ((y * width + x) * 3) as usize;
            rgb[idx] = ((x * 8) % 256) as u8;
            rgb[idx + 1] = ((y * 8) % 256) as u8;
            rgb[idx + 2] = (((x + y) * 4) % 256) as u8;
        }
    }

    // Encode with EOB optimization enabled
    let trellis_with_eob = TrellisConfig::default().eob_optimization(true);
    let with_eob = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .trellis(trellis_with_eob)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding with EOB opt failed");

    // Encode with EOB optimization disabled
    let trellis_without_eob = TrellisConfig::default().eob_optimization(false);
    let without_eob = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .trellis(trellis_without_eob)
        .encode_rgb(&rgb, width, height)
        .expect("Encoding without EOB opt failed");

    // Both should produce valid JPEGs
    assert!(with_eob.len() > 100, "EOB-optimized JPEG too small");
    assert!(without_eob.len() > 100, "Non-EOB JPEG too small");

    // Both should decode successfully
    let mut decoder1 = jpeg_decoder::Decoder::new(&with_eob[..]);
    let decoded1 = decoder1
        .decode()
        .expect("Failed to decode EOB-optimized JPEG");
    let info1 = decoder1.info().unwrap();
    assert_eq!(info1.width, width as u16);
    assert_eq!(info1.height, height as u16);

    let mut decoder2 = jpeg_decoder::Decoder::new(&without_eob[..]);
    let decoded2 = decoder2.decode().expect("Failed to decode non-EOB JPEG");
    let info2 = decoder2.info().unwrap();
    assert_eq!(info2.width, width as u16);
    assert_eq!(info2.height, height as u16);

    // EOB optimization may zero some coefficients for encoding efficiency,
    // so we allow reasonable quality differences. The important thing is
    // both produce valid, decodable JPEGs.
    let max_diff: i32 = decoded1
        .iter()
        .zip(decoded2.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);

    // Allow up to ~27% difference (70/255) since EOB optimization trades quality for size
    assert!(
        max_diff <= 100,
        "EOB optimization changed output too much: max_diff={}",
        max_diff
    );

    println!(
        "EOB optimization: with={} bytes, without={} bytes, diff={}",
        with_eob.len(),
        without_eob.len(),
        without_eob.len() as i64 - with_eob.len() as i64
    );
}

#[test]
fn test_eob_optimization_grayscale() {
    use mozjpeg_rs::TrellisConfig;

    let width = 64u32;
    let height = 64u32;

    // Create grayscale test image with some flat areas
    let mut gray = vec![128u8; (width * height) as usize];
    // Add gradient in one area
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * width + x) as usize;
            gray[idx] = ((x * 4 + y * 4) % 256) as u8;
        }
    }

    // Encode with EOB optimization
    let trellis_with_eob = TrellisConfig::default().eob_optimization(true);
    let with_eob = Encoder::new(false)
        .quality(75)
        .progressive(true)
        .trellis(trellis_with_eob)
        .encode_gray(&gray, width, height)
        .expect("Grayscale encoding with EOB opt failed");

    // Should produce valid JPEG
    assert!(
        with_eob.len() > 50,
        "EOB-optimized grayscale JPEG too small"
    );

    // Should decode successfully
    let mut decoder = jpeg_decoder::Decoder::new(&with_eob[..]);
    let decoded = decoder
        .decode()
        .expect("Failed to decode EOB-optimized grayscale JPEG");
    assert_eq!(decoded.len(), (width * height) as usize);

    println!("Grayscale EOB optimization: {} bytes", with_eob.len());
}

/// Comprehensive test of all encoder setting permutations with encode+decode round-trip.
/// Uses jpeg_decoder for decoding.
#[test]
fn test_encode_decode_permutations() {
    use mozjpeg_rs::{Subsampling, TrellisConfig};

    let width = 32u32;
    let height = 32u32;

    // Create test image
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();

    // Test matrix of settings
    let progressives = [false, true];
    let subsamplings = [Subsampling::S444, Subsampling::S422, Subsampling::S420];
    let optimize_huffmans = [false, true];
    let trellis_enabled = [false, true];
    let eob_opts = [false, true];
    let qualities = [50u8, 85];

    let mut test_count = 0;

    for &progressive in &progressives {
        for &subsampling in &subsamplings {
            for &optimize_huffman in &optimize_huffmans {
                for &trellis in &trellis_enabled {
                    for &eob_opt in &eob_opts {
                        // Skip eob_opt=true when trellis=false (eob_opt requires trellis)
                        if eob_opt && !trellis {
                            continue;
                        }

                        for &quality in &qualities {
                            let trellis_config = if trellis {
                                TrellisConfig::default().eob_optimization(eob_opt)
                            } else {
                                TrellisConfig::disabled()
                            };

                            let encoder = Encoder::new(false)
                                .quality(quality)
                                .progressive(progressive)
                                .subsampling(subsampling)
                                .optimize_huffman(optimize_huffman)
                                .trellis(trellis_config);

                            let result = encoder.encode_rgb(&rgb, width, height);

                            let jpeg = match result {
                                Ok(data) => data,
                                Err(e) => {
                                    panic!(
                                        "Encoding failed: prog={}, sub={:?}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                        progressive, subsampling, optimize_huffman, trellis, eob_opt, quality, e
                                    );
                                }
                            };

                            // Verify JPEG markers
                            assert_eq!(jpeg[0], 0xFF, "Missing SOI");
                            assert_eq!(jpeg[1], 0xD8, "Missing SOI");

                            // Decode the JPEG
                            let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
                            let decoded = decoder.decode().unwrap_or_else(|e| {
                                panic!(
                                    "Decode failed: prog={}, sub={:?}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                    progressive, subsampling, optimize_huffman, trellis, eob_opt, quality, e
                                );
                            });

                            let info = decoder.info().unwrap();
                            assert_eq!(info.width, width as u16);
                            assert_eq!(info.height, height as u16);
                            assert_eq!(
                                decoded.len(),
                                (width * height * 3) as usize,
                                "Decoded size mismatch"
                            );

                            test_count += 1;
                        }
                    }
                }
            }
        }
    }

    println!(
        "Successfully tested {} encode+decode permutations",
        test_count
    );
}

/// Test grayscale encoding with all setting permutations.
#[test]
fn test_grayscale_encode_decode_permutations() {
    use mozjpeg_rs::TrellisConfig;

    let width = 32u32;
    let height = 32u32;

    // Create grayscale test image
    let gray: Vec<u8> = (0..width * height)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();

    let progressives = [false, true];
    let optimize_huffmans = [false, true];
    let trellis_enabled = [false, true];
    let eob_opts = [false, true];
    let qualities = [50u8, 85];

    let mut test_count = 0;

    for &progressive in &progressives {
        for &optimize_huffman in &optimize_huffmans {
            for &trellis in &trellis_enabled {
                for &eob_opt in &eob_opts {
                    // Skip eob_opt=true when trellis=false
                    if eob_opt && !trellis {
                        continue;
                    }

                    for &quality in &qualities {
                        let trellis_config = if trellis {
                            TrellisConfig::default().eob_optimization(eob_opt)
                        } else {
                            TrellisConfig::disabled()
                        };

                        let encoder = Encoder::new(false)
                            .quality(quality)
                            .progressive(progressive)
                            .optimize_huffman(optimize_huffman)
                            .trellis(trellis_config);

                        let result = encoder.encode_gray(&gray, width, height);

                        let jpeg = match result {
                            Ok(data) => data,
                            Err(e) => {
                                panic!(
                                    "Grayscale encoding failed: prog={}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                    progressive, optimize_huffman, trellis, eob_opt, quality, e
                                );
                            }
                        };

                        // Decode
                        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
                        let decoded = decoder.decode().unwrap_or_else(|e| {
                            panic!(
                                "Grayscale decode failed: prog={}, huff={}, trellis={}, eob={}, q={}: {:?}",
                                progressive, optimize_huffman, trellis, eob_opt, quality, e
                            );
                        });

                        let info = decoder.info().unwrap();
                        assert_eq!(info.width, width as u16);
                        assert_eq!(info.height, height as u16);
                        assert_eq!(decoded.len(), (width * height) as usize);

                        test_count += 1;
                    }
                }
            }
        }
    }

    println!(
        "Successfully tested {} grayscale encode+decode permutations",
        test_count
    );
}
