//! Compare block values between baseline and progressive encoding paths

use std::io::Cursor;

fn main() {
    // Create a minimal 16x16 test image
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i*3] = (x * 16) as u8;
            rgb_data[i*3+1] = (y * 16) as u8;
            rgb_data[i*3+2] = 128;
        }
    }

    // Encode both ways
    let baseline = mozjpeg::Encoder::new()
        .quality(85)
        .subsampling(mozjpeg::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height).unwrap();

    let progressive = mozjpeg::Encoder::max_compression()
        .quality(85)
        .subsampling(mozjpeg::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height).unwrap();

    println!("Baseline: {} bytes", baseline.len());
    println!("Progressive: {} bytes", progressive.len());

    // Decode both
    let base_dec = decode_jpeg(&baseline);
    let prog_dec = decode_jpeg(&progressive);

    // Show pixel differences
    println!("\nPixel comparison (first 8 pixels):");
    println!("Original | Baseline | Progressive");
    for i in 0..8 {
        let orig = (rgb_data[i*3], rgb_data[i*3+1], rgb_data[i*3+2]);
        let base = (base_dec[i*3], base_dec[i*3+1], base_dec[i*3+2]);
        let prog = (prog_dec[i*3], prog_dec[i*3+1], prog_dec[i*3+2]);
        println!("({:3},{:3},{:3}) | ({:3},{:3},{:3}) | ({:3},{:3},{:3})",
            orig.0, orig.1, orig.2,
            base.0, base.1, base.2,
            prog.0, prog.1, prog.2);
    }

    // Calculate PSNR
    let base_psnr = calculate_psnr(&rgb_data, &base_dec);
    let prog_psnr = calculate_psnr(&rgb_data, &prog_dec);
    println!("\nBaseline PSNR: {:.2} dB", base_psnr);
    println!("Progressive PSNR: {:.2} dB", prog_psnr);

    // Check if decoded data matches
    if base_dec == prog_dec {
        println!("\nDecoded data matches!");
    } else {
        let diffs: Vec<usize> = base_dec.iter()
            .zip(prog_dec.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!("\nDecoded data differs at {} positions", diffs.len());
        if !diffs.is_empty() {
            println!("First 10 differences:");
            for &idx in diffs.iter().take(10) {
                println!("  [{}]: baseline={}, progressive={}",
                    idx, base_dec[idx], prog_dec[idx]);
            }
        }
    }

    // Save files for external analysis (cross-platform temp directory)
    let temp_dir = std::env::temp_dir();
    let baseline_path = temp_dir.join("test_baseline.jpg");
    let progressive_path = temp_dir.join("test_progressive.jpg");
    std::fs::write(&baseline_path, &baseline).unwrap();
    std::fs::write(&progressive_path, &progressive).unwrap();
    println!("\nSaved to {:?} and {:?}", baseline_path, progressive_path);
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(Cursor::new(data)).decode().unwrap()
}

fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    let mse: f64 = original.iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>() / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}
