//! Test edge cropping - encode images with dimensions not multiples of 8.
//!
//! JPEG uses 8x8 blocks, and with 4:2:0 subsampling, MCUs are 16x16.
//! This test verifies correct handling of edge padding for all 63 combinations
//! of non-aligned dimensions (width % 8 ∈ {1..7}, height % 8 ∈ {1..7}, plus mixed).
//!
//! Usage: cargo run --example test_edge_cropping

use dssim::Dssim;
use mozjpeg::{Encoder, Subsampling, TrellisConfig};
use png::ColorType;
use rgb::RGB;
use std::fs;
use std::path::Path;

fn main() {
    let input_path = Path::new("mozjpeg/tests/images/1.png");

    // Load the PNG image
    let file = fs::File::open(input_path).expect("Failed to open input image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    let full_width = info.width as usize;
    let full_height = info.height as usize;

    // Convert to RGB
    let full_rgb: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    println!("Loaded {}x{} image from {:?}\n", full_width, full_height, input_path);

    // Base size: 48 pixels = 3 MCUs (for 4:2:0), enough to show edge handling
    // We'll test sizes 49-55 (remainders 1-7 mod 8) plus 48 (remainder 0)
    let base = 48;
    let dssim_calc = Dssim::new();

    println!("Testing edge cropping: comparing Rust vs C mozjpeg");
    println!("Base size: {} pixels (3 MCUs for 4:2:0)\n", base);

    println!("{:>5} {:>5} {:>3} {:>3}  {:>10} {:>10} {:>12}  {}",
        "Width", "Height", "W%8", "H%8", "Rust Size", "C Size", "Rust vs C", "Status");
    println!("{}", "-".repeat(75));

    let mut failures = Vec::new();
    let mut total_tests = 0;

    // Test all combinations of remainders 0-7 for width and height
    for w_rem in 0..8 {
        for h_rem in 0..8 {
            // Skip 0,0 - that's the aligned case (less interesting)
            if w_rem == 0 && h_rem == 0 {
                continue;
            }

            let width = base + w_rem;
            let height = base + h_rem;

            // Skip if larger than source image
            if width > full_width || height > full_height {
                continue;
            }

            // Crop from top-left of the image
            let cropped = crop_rgb(&full_rgb, full_width, 0, 0, width, height);

            // Encode with Rust
            let rust_result = Encoder::new()
                .quality(85)
                .subsampling(Subsampling::S420)
                .progressive(false)
                .trellis(TrellisConfig::disabled())
                .optimize_huffman(true)
                .encode_rgb(&cropped, width as u32, height as u32);

            let rust_jpeg = match rust_result {
                Ok(data) => data,
                Err(e) => {
                    println!("{:>5} {:>5} {:>3} {:>3}  Rust encode FAILED: {}",
                        width, height, w_rem, h_rem, e);
                    failures.push((width, height, format!("Rust encode failed: {}", e)));
                    total_tests += 1;
                    continue;
                }
            };

            // Encode with C mozjpeg
            let c_jpeg = c_encode(&cropped, width as u32, height as u32, 85);

            // Decode both and compare with DSSIM
            let rust_decoded = decode_jpeg(&rust_jpeg);
            let c_decoded = decode_jpeg(&c_jpeg);

            if rust_decoded.is_none() {
                println!("{:>5} {:>5} {:>3} {:>3}  Rust decode FAILED",
                    width, height, w_rem, h_rem);
                failures.push((width, height, "Rust decode failed".to_string()));
                total_tests += 1;
                continue;
            }

            if c_decoded.is_none() {
                println!("{:>5} {:>5} {:>3} {:>3}  C decode FAILED",
                    width, height, w_rem, h_rem);
                failures.push((width, height, "C decode failed".to_string()));
                total_tests += 1;
                continue;
            }

            let rust_decoded = rust_decoded.unwrap();
            let c_decoded = c_decoded.unwrap();

            // Compare Rust output vs C output
            let rust_pixels: Vec<RGB<u8>> = rust_decoded.0
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let c_pixels: Vec<RGB<u8>> = c_decoded.0
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();

            let rust_img = dssim_calc
                .create_image_rgb(&rust_pixels, rust_decoded.1, rust_decoded.2)
                .expect("Failed to create Rust DSSIM image");
            let c_img = dssim_calc
                .create_image_rgb(&c_pixels, c_decoded.1, c_decoded.2)
                .expect("Failed to create C DSSIM image");

            let (dssim_val, _) = dssim_calc.compare(&rust_img, c_img);
            let dssim_f64: f64 = dssim_val.into();

            let status = if dssim_f64 < 0.001 {
                "OK"
            } else if dssim_f64 < 0.005 {
                "MARGINAL"
            } else {
                failures.push((width, height, format!("DSSIM too high: {:.6}", dssim_f64)));
                "FAIL"
            };

            println!("{:>5} {:>5} {:>3} {:>3}  {:>10} {:>10} {:>12.6}  {}",
                width, height, w_rem, h_rem,
                rust_jpeg.len(), c_jpeg.len(),
                dssim_f64, status);

            total_tests += 1;
        }
    }

    println!("\n{}", "=".repeat(75));
    println!("Total tests: {}", total_tests);
    println!("Failures: {}", failures.len());

    if !failures.is_empty() {
        println!("\nFailed cases:");
        for (w, h, reason) in &failures {
            println!("  {}x{}: {}", w, h, reason);
        }
        std::process::exit(1);
    } else {
        println!("\nAll edge cropping tests passed!");
    }
}

/// Crop a region from an RGB image.
fn crop_rgb(src: &[u8], src_width: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(w * h * 3);
    for row in y..(y + h) {
        let start = (row * src_width + x) * 3;
        let end = start + w * 3;
        result.extend_from_slice(&src[start..end]);
    }
    result
}

/// Decode a JPEG and return (pixels, width, height).
fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as usize, info.height as usize))
}

/// Encode using C mozjpeg via FFI.
fn c_encode(data: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = data.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}
