//! Debug comparison between Rust and C mozjpeg encoders.
//!
//! Find exactly where the outputs diverge.

use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
use std::fs;

fn main() {
    // Load test image and crop to non-aligned size (same as edge test)
    use png::ColorType;
    let input_path = "mozjpeg/tests/images/1.png";
    let file = std::fs::File::open(input_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let full_width = info.width as usize;
    let full_rgb: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported"),
    };

    // Crop to 48x48 (MCU-aligned for 4:2:0)
    let width = 48u32;
    let height = 48u32;
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height as usize {
        let start = y * full_width * 3;
        let end = start + (width as usize) * 3;
        rgb_data.extend_from_slice(&full_rgb[start..end]);
    }

    println!("Test image: {}x{} gradient pattern\n", width, height);

    // Encode with Rust - minimal settings
    let rust_jpeg = Encoder::new()
        .quality(85)
        .subsampling(Subsampling::S420)
        .progressive(false)
        .trellis(TrellisConfig::disabled())
        .optimize_huffman(false) // Use standard tables for comparison
        .encode_rgb(&rgb_data, width, height)
        .expect("Rust encode failed");

    // Encode with C mozjpeg - try to match settings exactly
    let c_jpeg = c_encode_minimal(&rgb_data, width, height, 85);

    println!("Rust JPEG size: {} bytes", rust_jpeg.len());
    println!("C JPEG size: {} bytes\n", c_jpeg.len());

    // Compare the JPEG files byte by byte
    println!("=== Byte comparison ===");
    let min_len = rust_jpeg.len().min(c_jpeg.len());
    let mut first_diff = None;
    let mut diff_count = 0;

    for i in 0..min_len {
        if rust_jpeg[i] != c_jpeg[i] {
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            diff_count += 1;
        }
    }

    if let Some(pos) = first_diff {
        println!("First difference at byte {:#x} ({})", pos, pos);
        println!("Total differing bytes: {}", diff_count);

        // Show context around first difference
        let start = pos.saturating_sub(8);
        let end = (pos + 16).min(min_len);

        println!("\nRust bytes around diff (offset {:#x}):", start);
        print!("  ");
        for i in start..end {
            if i == pos {
                print!("[{:02x}] ", rust_jpeg[i]);
            } else {
                print!("{:02x} ", rust_jpeg[i]);
            }
        }
        println!();

        println!("C bytes around diff (offset {:#x}):", start);
        print!("  ");
        for i in start..end {
            if i == pos {
                print!("[{:02x}] ", c_jpeg[i]);
            } else {
                print!("{:02x} ", c_jpeg[i]);
            }
        }
        println!();

        // Parse and identify what marker we're in
        identify_jpeg_section(&rust_jpeg, pos);
        identify_jpeg_section(&c_jpeg, pos);
    } else if rust_jpeg.len() != c_jpeg.len() {
        println!("Files are same up to min length but have different sizes");
    } else {
        println!("Files are IDENTICAL!");
    }

    // Save both files for manual inspection
    fs::write("/tmp/rust_debug.jpg", &rust_jpeg).unwrap();
    fs::write("/tmp/c_debug.jpg", &c_jpeg).unwrap();
    println!("\nSaved /tmp/rust_debug.jpg and /tmp/c_debug.jpg for inspection");

    // Decode and compare pixels
    println!("\n=== Decoded pixel comparison ===");
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let c_decoded = decode_jpeg(&c_jpeg);

    if let (Some((rust_pix, _, _)), Some((c_pix, _, _))) = (&rust_decoded, &c_decoded) {
        let mut max_diff = 0i16;
        let mut sum_diff = 0u64;
        let mut diff_pixels = 0usize;

        for i in 0..rust_pix.len() {
            let diff = (rust_pix[i] as i16 - c_pix[i] as i16).abs();
            if diff > 0 {
                diff_pixels += 1;
                sum_diff += diff as u64;
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        println!("Decoded pixels: {} total", rust_pix.len());
        println!("Differing components: {}", diff_pixels);
        println!("Max component diff: {}", max_diff);
        if diff_pixels > 0 {
            println!(
                "Avg diff (of differing): {:.2}",
                sum_diff as f64 / diff_pixels as f64
            );
        }
    }
}

fn c_encode_minimal(data: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // FORCE BASELINE MODE - disable progressive scan script
        cinfo.num_scans = 0;
        cinfo.scan_info = ptr::null();

        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Disable optimizations to get baseline comparison
        cinfo.optimize_coding = 0;

        // Disable trellis quantization (mozjpeg extension)
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 0);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 0);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 0);

        // Print all relevant settings
        println!("C mozjpeg settings:");
        println!("  image: {}x{}", cinfo.image_width, cinfo.image_height);
        println!("  optimize_coding: {}", cinfo.optimize_coding);
        println!("  smoothing_factor: {}", cinfo.smoothing_factor);
        println!("  num_scans: {}", cinfo.num_scans);
        println!("  dct_method: {:?}", cinfo.dct_method);

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

fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as usize, info.height as usize))
}

fn identify_jpeg_section(data: &[u8], pos: usize) {
    // Walk through JPEG markers to identify what section pos is in
    let mut i = 0;
    let mut last_marker = "SOI";
    let mut last_marker_pos = 0;

    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xD9 => "EOI",
                0xDB => "DQT",
                0xC0 => "SOF0",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xE0 => "APP0",
                0xFE => "COM",
                _ => "???",
            };

            if i > pos {
                break;
            }

            last_marker = name;
            last_marker_pos = i;

            // Skip marker and length
            if marker != 0xD8 && marker != 0xD9 && i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                i += 2 + len;
                continue;
            }
        }
        i += 1;
    }

    println!(
        "Position {:#x} is in/after {} marker (at {:#x})",
        pos, last_marker, last_marker_pos
    );
}
