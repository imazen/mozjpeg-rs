//! Debug small image encoding issue

use mozjpeg_rs::sample;

fn main() {
    // First, test just the downsampling step
    println!("=== Testing downsample_plane for 64x64 -> 32x32 ===\n");

    let w = 64usize;
    let h = 64usize;

    // Create a simple pattern: x + y*256 so we can detect scrambling
    let mut cb_full = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            cb_full[y * w + x] = ((x + y) % 256) as u8;
        }
    }

    // Downsample h2v2
    let (out_w, out_h) = (w / 2, h / 2);
    let mut cb_sub = vec![0u8; out_w * out_h];
    sample::downsample_plane(&cb_full, w, h, 2, 2, &mut cb_sub);

    // Verify the output - each 2x2 block should average
    println!("Downsampled chroma {}x{} samples:", out_w, out_h);
    let mut errors = 0;
    for y in 0..out_h {
        for x in 0..out_w {
            let actual = cb_sub[y * out_w + x];
            // Expected average of 2x2 block
            let x0 = x * 2;
            let y0 = y * 2;
            let expected_sum = cb_full[y0 * w + x0] as u32
                + cb_full[y0 * w + x0 + 1] as u32
                + cb_full[(y0 + 1) * w + x0] as u32
                + cb_full[(y0 + 1) * w + x0 + 1] as u32;
            let expected = ((expected_sum + 1) / 4) as u8; // approx with rounding
            let diff = (actual as i16 - expected as i16).abs();
            if diff > 2 {
                // Allow small rounding differences
                errors += 1;
                if errors < 10 {
                    println!(
                        "  ERROR at ({},{}): expected ~{}, got {}",
                        x, y, expected, actual
                    );
                }
            }
        }
    }
    println!("Downsampling errors: {} / {}", errors, out_w * out_h);

    // Now test expand_to_mcu
    println!("\n=== Testing expand_to_mcu for 32x32 -> 32x32 ===\n");

    let mcu_w = 32;
    let mcu_h = 32;
    let mut cb_mcu = vec![0u8; mcu_w * mcu_h];
    sample::expand_to_mcu(&cb_sub, out_w, out_h, &mut cb_mcu, mcu_w, mcu_h);

    // Should be a simple copy since dimensions match
    errors = 0;
    for y in 0..out_h {
        for x in 0..out_w {
            if cb_mcu[y * mcu_w + x] != cb_sub[y * out_w + x] {
                errors += 1;
                if errors < 10 {
                    println!(
                        "  MCU ERROR at ({},{}): expected {}, got {}",
                        x,
                        y,
                        cb_sub[y * out_w + x],
                        cb_mcu[y * mcu_w + x]
                    );
                }
            }
        }
    }
    println!("MCU expansion errors: {} / {}", errors, mcu_w * mcu_h);

    println!("\n=== Now testing actual encode ===\n");
    fn create_photo_like(width: u32, height: u32) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 3) as usize;
                let cx = width as f32 / 2.0;
                let cy = height as f32 / 2.0;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let max_dist = (cx * cx + cy * cy).sqrt();
                let r = (255.0 * (1.0 - dist / max_dist)).clamp(0.0, 255.0) as u8;
                let g = (200.0 * (x as f32 / width as f32)).clamp(0.0, 255.0) as u8;
                let b = (200.0 * (y as f32 / height as f32)).clamp(0.0, 255.0) as u8;
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
            }
        }
        data
    }

    // Test 64x64 specifically
    let w = 64u32;
    let h = 64u32;
    let image = create_photo_like(w, h);

    // Test 4:2:0 with and without Huffman optimization
    // Without optimization uses streaming encode, with optimization uses stored blocks + SIMD encoder
    for (name, opt_huffman) in [
        ("4:2:0 (no Huffman opt, streaming)", false),
        ("4:2:0 (with Huffman opt, SIMD)", true),
        ("4:4:4 (with Huffman opt)", true),
    ] {
        let sub = if name.starts_with("4:4:4") {
            mozjpeg_rs::Subsampling::S444
        } else {
            mozjpeg_rs::Subsampling::S420
        };
        println!("\n=== Testing {} ===", name);

        let encoder = mozjpeg_rs::Encoder::baseline_optimized()
            .quality(75)
            .subsampling(sub)
            .progressive(false)
            .optimize_huffman(opt_huffman)
            .trellis(mozjpeg_rs::TrellisConfig::disabled())
            .force_baseline(true);

        let jpeg = encoder.encode_rgb(&image, w, h).expect("encoding failed");
        println!("JPEG size: {} bytes", jpeg.len());

        // Decode
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        let decoded = decoder.decode().expect("decoding failed");

        // Calculate per-block error
        let mut total_err = 0u64;
        let mut max_block_err = 0.0f64;
        for by in 0..h / 8 {
            for bx in 0..w / 8 {
                let mut block_sum = 0u64;
                for y in 0..8 {
                    for x in 0..8 {
                        let px = bx * 8 + x;
                        let py = by * 8 + y;
                        let idx = ((py * w + px) * 3) as usize;
                        for c in 0..3 {
                            let diff = (image[idx + c] as i16 - decoded[idx + c] as i16).abs();
                            block_sum += diff as u64;
                        }
                    }
                }
                let avg = block_sum as f64 / (8 * 8 * 3) as f64;
                max_block_err = max_block_err.max(avg);
                total_err += block_sum;
            }
        }
        let avg_err = total_err as f64 / (w * h * 3) as f64;
        println!(
            "Average error: {:.2}, Max block error: {:.2}",
            avg_err, max_block_err
        );
    }

    // Rest of the debug for 4:2:0
    let encoder = mozjpeg_rs::Encoder::baseline_optimized()
        .quality(75)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .progressive(false)
        .optimize_huffman(true)
        .trellis(mozjpeg_rs::TrellisConfig::disabled())
        .force_baseline(true);

    let jpeg = encoder.encode_rgb(&image, w, h).expect("encoding failed");
    println!("JPEG size: {} bytes", jpeg.len());

    // Decode
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
    let decoded = decoder.decode().expect("decoding failed");
    let info = decoder.info().unwrap();
    println!(
        "Decoded: {}x{} ({} bytes)",
        info.width,
        info.height,
        decoded.len()
    );

    // Compare first 10 pixels
    println!("\nFirst 10 pixels (R,G,B):");
    for i in 0..10 {
        let orig = &image[i * 3..i * 3 + 3];
        let dec = &decoded[i * 3..i * 3 + 3];
        println!(
            "  Pixel {}: orig({},{},{}) dec({},{},{}) diff({},{},{})",
            i,
            orig[0],
            orig[1],
            orig[2],
            dec[0],
            dec[1],
            dec[2],
            orig[0] as i16 - dec[0] as i16,
            orig[1] as i16 - dec[1] as i16,
            orig[2] as i16 - dec[2] as i16
        );
    }

    // Calculate MSE and find worst pixels
    let mut sum_sq = 0u64;
    let mut max_diff = 0i16;
    let mut worst_pixel = (0, 0i16);
    for (i, (orig, dec)) in image.iter().zip(decoded.iter()).enumerate() {
        let diff = (*orig as i16 - *dec as i16).abs();
        sum_sq += (diff as u64) * (diff as u64);
        if diff > max_diff {
            max_diff = diff;
            worst_pixel = (i, diff);
        }
    }
    let mse = sum_sq as f64 / image.len() as f64;
    println!(
        "\nMSE: {:.2}, RMSE: {:.2}, Max diff: {} at component index {}",
        mse,
        mse.sqrt(),
        max_diff,
        worst_pixel.0
    );

    // Show pixels around worst area
    let worst_pix = worst_pixel.0 / 3;
    let worst_x = worst_pix as u32 % w;
    let worst_y = worst_pix as u32 / w;
    println!(
        "Worst pixel at ({}, {}) = pixel index {}",
        worst_x, worst_y, worst_pix
    );

    // Show a 4x4 block around the worst pixel
    println!("\n4x4 block around worst pixel:");
    for dy in 0..4i32 {
        for dx in 0..4i32 {
            let px = (worst_x as i32 + dx - 2).clamp(0, w as i32 - 1) as u32;
            let py = (worst_y as i32 + dy - 2).clamp(0, h as i32 - 1) as u32;
            let idx = ((py * w + px) * 3) as usize;
            let orig_r = image[idx];
            let dec_r = decoded[idx];
            let diff = (orig_r as i16 - dec_r as i16).abs();
            print!("{:3} ", diff);
        }
        println!();
    }

    // Show per-block average error
    println!("\nPer 8x8 block average error:");
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let mut block_sum = 0u64;
            for y in 0..8 {
                for x in 0..8 {
                    let px = bx * 8 + x;
                    let py = by * 8 + y;
                    let idx = ((py * w + px) * 3) as usize;
                    for c in 0..3 {
                        let diff = (image[idx + c] as i16 - decoded[idx + c] as i16).abs();
                        block_sum += diff as u64;
                    }
                }
            }
            let avg = block_sum as f64 / (8 * 8 * 3) as f64;
            print!("{:5.1} ", avg);
        }
        println!();
    }

    // Save for inspection
    std::fs::write("/tmp/small_test.jpg", &jpeg).unwrap();
    println!("\nSaved to /tmp/small_test.jpg");
}
