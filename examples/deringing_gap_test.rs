//! Test if overshoot deringing setting mismatch explains the non-trellis gap

use mozjpeg_rs::{Encoder, TrellisConfig};
use std::fs;

fn main() {
    let source_path = "/home/lilith/work/mozjpeg-rs/corpus/kodak/10.png";
    let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    println!("=== Deringing Gap Test ===");
    println!("Image: {}x{}", width, height);
    println!();

    let quality = 85u8;

    // Test 1: Rust with deringing ON (default), C without explicit setting
    let rust_with_deringing = Encoder::baseline_optimized()
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .trellis(TrellisConfig::disabled())
        .overshoot_deringing(true)
        .encode_rgb(rgb_data, width, height)
        .unwrap();

    // Test 2: Rust with deringing OFF
    let rust_without_deringing = Encoder::baseline_optimized()
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .trellis(TrellisConfig::disabled())
        .overshoot_deringing(false)
        .encode_rgb(rgb_data, width, height)
        .unwrap();

    // Test 3: C without OVERSHOOT_DERINGING set (default)
    let c_default = encode_c(rgb_data, width, height, quality, None);

    // Test 4: C with OVERSHOOT_DERINGING = 0
    let c_off = encode_c(rgb_data, width, height, quality, Some(0));

    // Test 5: C with OVERSHOOT_DERINGING = 1
    let c_on = encode_c(rgb_data, width, height, quality, Some(1));

    println!("File sizes:");
    println!("  Rust deringing ON:  {} bytes", rust_with_deringing.len());
    println!(
        "  Rust deringing OFF: {} bytes",
        rust_without_deringing.len()
    );
    println!("  C default (unset):  {} bytes", c_default.len());
    println!("  C deringing OFF:    {} bytes", c_off.len());
    println!("  C deringing ON:     {} bytes", c_on.len());
    println!();

    println!("Comparisons:");
    println!(
        "  Rust ON vs C default:  {:+.2}%",
        (rust_with_deringing.len() as f64 / c_default.len() as f64 - 1.0) * 100.0
    );
    println!(
        "  Rust OFF vs C default: {:+.2}%",
        (rust_without_deringing.len() as f64 / c_default.len() as f64 - 1.0) * 100.0
    );
    println!(
        "  Rust ON vs C ON:       {:+.2}%",
        (rust_with_deringing.len() as f64 / c_on.len() as f64 - 1.0) * 100.0
    );
    println!(
        "  Rust OFF vs C OFF:     {:+.2}%",
        (rust_without_deringing.len() as f64 / c_off.len() as f64 - 1.0) * 100.0
    );
}

fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8, deringing: Option<i32>) -> Vec<u8> {
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
        cinfo.scan_info = ptr::null();
        cinfo.num_scans = 0;

        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;

        // Trellis OFF
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 0);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 0);

        // Optionally set deringing
        if let Some(value) = deringing {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, value);
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_ptr = rgb.as_ptr().add(cinfo.next_scanline as usize * row_stride);
            let row_array: [*const u8; 1] = [row_ptr];
            jpeg_write_scanlines(&mut cinfo, row_array.as_ptr(), 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}
