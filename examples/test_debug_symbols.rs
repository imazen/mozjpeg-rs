//! Debug symbol counting vs encoding to find the mismatch

use mozjpeg_rs::huffman::FrequencyCounter;

fn main() {
    // Create minimal 16x16 gradient image
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 16) as u8;
            rgb_data[i * 3 + 1] = (y * 16) as u8;
            rgb_data[i * 3 + 2] = 128;
        }
    }

    // Encode baseline (works)
    let baseline = mozjpeg_rs::Encoder::baseline_optimized()
        .quality(85)
        .subsampling(mozjpeg_rs::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // Decode baseline
    let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
        .decode()
        .unwrap();

    // Encode progressive
    let progressive = mozjpeg_rs::Encoder::max_compression()
        .quality(85)
        .subsampling(mozjpeg_rs::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height);

    match progressive {
        Ok(data) => {
            println!("Progressive encoded: {} bytes", data.len());
            std::fs::write("/tmp/debug_prog.jpg", &data).unwrap();

            // Try to decode
            match jpeg_decoder::Decoder::new(std::io::Cursor::new(&data)).decode() {
                Ok(decoded) => {
                    let mse: f64 = rgb_data
                        .iter()
                        .zip(decoded.iter())
                        .map(|(&a, &b)| {
                            let diff = a as f64 - b as f64;
                            diff * diff
                        })
                        .sum::<f64>()
                        / rgb_data.len() as f64;
                    let psnr = if mse == 0.0 {
                        f64::INFINITY
                    } else {
                        10.0 * (255.0_f64 * 255.0 / mse).log10()
                    };
                    println!("Progressive PSNR: {:.2} dB", psnr);
                }
                Err(e) => {
                    println!("Progressive decode failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("Progressive encode failed: {:?}", e);
        }
    }

    // Baseline PSNR
    let mse: f64 = rgb_data
        .iter()
        .zip(base_dec.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / rgb_data.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    println!("Baseline PSNR: {:.2} dB", psnr);

    // Print FrequencyCounter behavior
    println!("\n=== Frequency Counter Debug ===");
    let mut freq = FrequencyCounter::new();
    freq.count(0x00); // EOB
    freq.count(0x10); // EOB1
    freq.count(0x11); // (1,1)

    match freq.generate_table() {
        Ok(table) => {
            println!(
                "Generated table with {} symbols",
                table.bits.iter().sum::<u8>()
            );
        }
        Err(e) => {
            println!("Failed to generate table: {:?}", e);
        }
    }
}
