//! Validate trellis quantization and progressive encoding.
//!
//! Compares Rust vs C mozjpeg with various encoder configurations.
//! Uses bundled test image if corpus not available.

use mozjpeg_oxide::corpus::{bundled_test_image, kodak_dir};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Get a test image path, preferring corpus but falling back to bundled.
fn get_test_image() -> PathBuf {
    kodak_dir()
        .map(|d| d.join("kodim01.png"))
        .filter(|p| p.exists())
        .or_else(|| bundled_test_image("1.png"))
        .expect("No test image found")
}

/// Test trellis + progressive encoding comparison.
#[test]
fn test_trellis_progressive_comparison() {
    let path = get_test_image();
    let (rgb_data, width, height) = load_png(&path).expect("Failed to load test image");

    // Rust with max_compression (progressive + trellis)
    let rust_progressive = mozjpeg_oxide::Encoder::max_compression()
        .quality(75)
        .encode_rgb(&rgb_data, width, height)
        .expect("Rust progressive encoding failed");

    // Rust baseline with trellis enabled (default)
    let rust_baseline = mozjpeg_oxide::Encoder::new()
        .quality(75)
        .encode_rgb(&rgb_data, width, height)
        .expect("Rust baseline encoding failed");

    // C mozjpeg with defaults
    let c_jpeg = encode_c(&rgb_data, width, height, 75);

    println!("\n=== Trellis/Progressive Comparison ===");
    println!("Image: {} ({}x{})", path.display(), width, height);
    println!("Rust (progressive+trellis): {} bytes", rust_progressive.len());
    println!("Rust (baseline+trellis):    {} bytes", rust_baseline.len());
    println!("C mozjpeg:                  {} bytes", c_jpeg.len());

    let ratio_prog = rust_progressive.len() as f64 / c_jpeg.len() as f64;
    let ratio_base = rust_baseline.len() as f64 / c_jpeg.len() as f64;
    println!("\nRatio (Rust progressive / C): {:.4}", ratio_prog);
    println!("Ratio (Rust baseline / C):    {:.4}", ratio_base);

    // Verify ratios are reasonable
    assert!(
        ratio_prog < 1.10 && ratio_prog > 0.90,
        "Progressive ratio {:.4} out of range",
        ratio_prog
    );
    assert!(
        ratio_base < 1.15 && ratio_base > 0.85,
        "Baseline ratio {:.4} out of range",
        ratio_base
    );

    // Count SOS markers (number of scans)
    let rust_scans = rust_progressive
        .windows(2)
        .filter(|w| *w == [0xFF, 0xDA])
        .count();
    let c_scans = c_jpeg.windows(2).filter(|w| *w == [0xFF, 0xDA]).count();
    println!("\nScan count: Rust={} C={}", rust_scans, c_scans);

    // Progressive should have multiple scans
    assert!(rust_scans > 1, "Progressive should have multiple scans");

    // Verify decoded quality
    let rust_prog_decoded = decode_jpeg(&rust_progressive);
    let rust_base_decoded = decode_jpeg(&rust_baseline);
    let c_decoded = decode_jpeg(&c_jpeg);

    let psnr_rust_prog = calculate_psnr(&rgb_data, &rust_prog_decoded);
    let psnr_rust_base = calculate_psnr(&rgb_data, &rust_base_decoded);
    let psnr_c = calculate_psnr(&rgb_data, &c_decoded);

    println!("\nPSNR (higher is better):");
    println!("  Rust progressive: {:.2} dB", psnr_rust_prog);
    println!("  Rust baseline:    {:.2} dB", psnr_rust_base);
    println!("  C mozjpeg:        {:.2} dB", psnr_c);

    // Quality should be reasonable (> 28 dB at Q75 for real photos)
    // Note: Synthetic patterns achieve higher PSNR, but real photos with
    // complex detail typically achieve 28-35 dB at Q75.
    assert!(psnr_rust_prog > 28.0, "Progressive PSNR too low: {:.2}", psnr_rust_prog);
    assert!(psnr_rust_base > 28.0, "Baseline PSNR too low: {:.2}", psnr_rust_base);

    // Rust should be within 2 dB of C mozjpeg
    let prog_diff = (psnr_rust_prog - psnr_c).abs();
    let base_diff = (psnr_rust_base - psnr_c).abs();
    assert!(prog_diff < 2.0, "Progressive differs too much from C: {:.2} dB", prog_diff);
    assert!(base_diff < 2.0, "Baseline differs too much from C: {:.2} dB", base_diff);
}

/// Test small image encoding.
#[test]
fn test_small_image_encoding() {
    let width = 16u32;
    let height = 16u32;
    let mut rgb = vec![128u8; (width * height * 3) as usize];

    // Create gradient pattern
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb[i * 3] = ((x * 16) % 256) as u8;
            rgb[i * 3 + 1] = ((y * 16) % 256) as u8;
            rgb[i * 3 + 2] = 128;
        }
    }

    let progressive = mozjpeg_oxide::Encoder::max_compression()
        .quality(85)
        .encode_rgb(&rgb, width, height)
        .expect("Progressive failed");

    let baseline = mozjpeg_oxide::Encoder::new()
        .quality(85)
        .encode_rgb(&rgb, width, height)
        .expect("Baseline failed");

    println!("\n=== Small Image (16x16) ===");
    println!("Progressive: {} bytes", progressive.len());
    println!("Baseline:    {} bytes", baseline.len());

    let prog_dec = decode_jpeg(&progressive);
    let base_dec = decode_jpeg(&baseline);

    let prog_psnr = calculate_psnr(&rgb, &prog_dec);
    let base_psnr = calculate_psnr(&rgb, &base_dec);

    println!("Progressive PSNR: {:.2} dB", prog_psnr);
    println!("Baseline PSNR:    {:.2} dB", base_psnr);

    // Both should decode successfully with reasonable quality
    assert!(prog_psnr > 30.0, "Progressive PSNR too low: {:.2}", prog_psnr);
    assert!(base_psnr > 30.0, "Baseline PSNR too low: {:.2}", base_psnr);
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(std::io::Cursor::new(data))
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

fn encode_c(rgb: &[u8], width: u32, height: u32, quality: i32) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::ptr;

    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_compress_struct>(),
        );

        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality, 1);
        cinfo.optimize_coding = 1;

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = (width * 3) as usize;
        let mut row_pointer: [*const u8; 1] = [ptr::null()];

        while cinfo.next_scanline < cinfo.image_height {
            let offset = cinfo.next_scanline as usize * row_stride;
            row_pointer[0] = rgb.as_ptr().add(offset);
            jpeg_write_scanlines(&mut cinfo, row_pointer.as_ptr(), 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);

        result
    }
}

/// Load a PNG image and return RGB data.
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };

    Some((rgb_data, width, height))
}
