//! Compare baseline vs progressive encoding paths.
//!
//! Verifies that both encoding modes produce valid, decodable JPEGs
//! with consistent quality characteristics.

use std::io::Cursor;

/// Test baseline vs progressive encoding for small images.
#[test]
fn test_baseline_vs_progressive_small() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    // Create gradient pattern
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 16) as u8;
            rgb_data[i * 3 + 1] = (y * 16) as u8;
            rgb_data[i * 3 + 2] = 128;
        }
    }

    // Encode both ways with 4:4:4 (simpler structure)
    let baseline = mozjpeg_oxide::Encoder::new()
        .quality(85)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height)
        .expect("Baseline encoding failed");

    let progressive = mozjpeg_oxide::Encoder::max_compression()
        .quality(85)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height)
        .expect("Progressive encoding failed");

    println!("\n=== Baseline vs Progressive (16x16) ===");
    println!("Baseline:    {} bytes", baseline.len());
    println!("Progressive: {} bytes", progressive.len());

    // Decode both
    let base_dec = decode_jpeg(&baseline);
    let prog_dec = decode_jpeg(&progressive);

    // Show pixel differences (first 8 pixels)
    println!("\nPixel comparison (first 8 pixels):");
    println!("Original | Baseline | Progressive");
    for i in 0..8 {
        let orig = (rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
        let base = (base_dec[i * 3], base_dec[i * 3 + 1], base_dec[i * 3 + 2]);
        let prog = (prog_dec[i * 3], prog_dec[i * 3 + 1], prog_dec[i * 3 + 2]);
        println!(
            "({:3},{:3},{:3}) | ({:3},{:3},{:3}) | ({:3},{:3},{:3})",
            orig.0, orig.1, orig.2, base.0, base.1, base.2, prog.0, prog.1, prog.2
        );
    }

    // Calculate PSNR
    let base_psnr = calculate_psnr(&rgb_data, &base_dec);
    let prog_psnr = calculate_psnr(&rgb_data, &prog_dec);
    println!("\nBaseline PSNR:    {:.2} dB", base_psnr);
    println!("Progressive PSNR: {:.2} dB", prog_psnr);

    // Both should have reasonable quality
    assert!(base_psnr > 30.0, "Baseline PSNR too low: {:.2}", base_psnr);
    assert!(prog_psnr > 30.0, "Progressive PSNR too low: {:.2}", prog_psnr);

    // Check decoded data similarity
    if base_dec == prog_dec {
        println!("\nDecoded data matches exactly!");
    } else {
        let diffs: Vec<usize> = base_dec
            .iter()
            .zip(prog_dec.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!(
            "\nDecoded data differs at {} positions (expected for different modes)",
            diffs.len()
        );

        // Show first few differences
        if !diffs.is_empty() {
            println!("First differences:");
            for &idx in diffs.iter().take(5) {
                println!(
                    "  [{}]: baseline={}, progressive={}",
                    idx, base_dec[idx], prog_dec[idx]
                );
            }
        }

        // Differences should be small
        let max_diff: u8 = base_dec
            .iter()
            .zip(prog_dec.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        println!("Max pixel difference: {}", max_diff);
        assert!(
            max_diff <= 10,
            "Max difference between modes too high: {}",
            max_diff
        );
    }
}

/// Test with larger image and 4:2:0 subsampling.
#[test]
fn test_baseline_vs_progressive_420() {
    let width = 64u32;
    let height = 64u32;

    // Create photo-like image
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let max_dist = (cx * cx + cy * cy).sqrt();

            rgb_data[i] = (255.0 * (1.0 - dist / max_dist)).clamp(0.0, 255.0) as u8;
            rgb_data[i + 1] = (200.0 * (x as f32 / width as f32)).clamp(0.0, 255.0) as u8;
            rgb_data[i + 2] = (200.0 * (y as f32 / height as f32)).clamp(0.0, 255.0) as u8;
        }
    }

    let baseline = mozjpeg_oxide::Encoder::new()
        .quality(85)
        .subsampling(mozjpeg_oxide::Subsampling::S420)
        .progressive(false)
        .encode_rgb(&rgb_data, width, height)
        .expect("Baseline failed");

    let progressive = mozjpeg_oxide::Encoder::new()
        .quality(85)
        .subsampling(mozjpeg_oxide::Subsampling::S420)
        .progressive(true)
        .encode_rgb(&rgb_data, width, height)
        .expect("Progressive failed");

    println!("\n=== Baseline vs Progressive (64x64, 4:2:0) ===");
    println!("Baseline:    {} bytes", baseline.len());
    println!("Progressive: {} bytes", progressive.len());

    let base_dec = decode_jpeg(&baseline);
    let prog_dec = decode_jpeg(&progressive);

    let base_psnr = calculate_psnr(&rgb_data, &base_dec);
    let prog_psnr = calculate_psnr(&rgb_data, &prog_dec);
    println!("Baseline PSNR:    {:.2} dB", base_psnr);
    println!("Progressive PSNR: {:.2} dB", prog_psnr);

    assert!(base_psnr > 30.0, "Baseline PSNR too low");
    assert!(prog_psnr > 30.0, "Progressive PSNR too low");

    // Count scans in progressive
    let base_scans = baseline.windows(2).filter(|w| *w == [0xFF, 0xDA]).count();
    let prog_scans = progressive
        .windows(2)
        .filter(|w| *w == [0xFF, 0xDA])
        .count();
    println!("Baseline scans:    {}", base_scans);
    println!("Progressive scans: {}", prog_scans);

    assert_eq!(base_scans, 1, "Baseline should have exactly 1 scan");
    assert!(prog_scans > 1, "Progressive should have multiple scans");
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(Cursor::new(data))
        .decode()
        .expect("Decode failed")
}

fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
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
