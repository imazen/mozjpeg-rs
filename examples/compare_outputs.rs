//! Compare Rust vs C mozjpeg encoder outputs and save to files.
//!
//! Run with: cargo run --example compare_outputs

use mozjpeg_rs::{Encoder, Subsampling};
use std::fs;

fn main() {
    // Create a test image (64x64 gradient)
    let width = 64u32;
    let height = 64u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 4) as u8; // R gradient
            rgb_data[i * 3 + 1] = (y * 4) as u8; // G gradient
            rgb_data[i * 3 + 2] = 128; // B constant
        }
    }

    // Also save the original as PNG for reference
    save_rgb_as_ppm(&rgb_data, width, height, "/tmp/original.ppm");
    println!("Saved: /tmp/original.ppm");

    for quality in [50, 75, 85] {
        // Encode with Rust implementation
        let rust_encoder = Encoder::new(false)
            .quality(quality)
            .subsampling(Subsampling::S420);
        let rust_jpeg = rust_encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with C mozjpeg
        let c_jpeg = unsafe { encode_with_c_mozjpeg(&rgb_data, width, height, quality) };

        // Save both
        let rust_path = format!("/tmp/rust_q{}.jpg", quality);
        let c_path = format!("/tmp/c_mozjpeg_q{}.jpg", quality);

        fs::write(&rust_path, &rust_jpeg).unwrap();
        fs::write(&c_path, &c_jpeg).unwrap();

        println!(
            "Q{}: Rust={} bytes -> {}",
            quality,
            rust_jpeg.len(),
            rust_path
        );
        println!("Q{}: C   ={} bytes -> {}", quality, c_jpeg.len(), c_path);
    }

    println!("\nView images with: eog /tmp/*.jpg /tmp/*.ppm");
    println!("Or: feh /tmp/rust_q75.jpg /tmp/c_mozjpeg_q75.jpg");
}

fn save_rgb_as_ppm(data: &[u8], width: u32, height: u32, path: &str) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut file_data = header.into_bytes();
    file_data.extend_from_slice(data);
    fs::write(path, file_data).unwrap();
}

unsafe fn encode_with_c_mozjpeg(rgb_data: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::ptr;

    let mut outbuffer: *mut u8 = ptr::null_mut();
    let mut outsize: std::ffi::c_ulong = 0;

    let mut cinfo = std::mem::zeroed::<mozjpeg_sys::jpeg_compress_struct>();
    let mut jerr = std::mem::zeroed::<mozjpeg_sys::jpeg_error_mgr>();

    cinfo.common.err = mozjpeg_sys::jpeg_std_error(&mut jerr);
    mozjpeg_sys::jpeg_CreateCompress(
        &mut cinfo,
        mozjpeg_sys::JPEG_LIB_VERSION as i32,
        std::mem::size_of::<mozjpeg_sys::jpeg_compress_struct>(),
    );

    mozjpeg_sys::jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = mozjpeg_sys::J_COLOR_SPACE::JCS_RGB;

    mozjpeg_sys::jpeg_set_defaults(&mut cinfo);
    mozjpeg_sys::jpeg_set_quality(&mut cinfo, quality as i32, 1);

    mozjpeg_sys::jpeg_start_compress(&mut cinfo, 1);

    let row_stride = (width * 3) as usize;
    while cinfo.next_scanline < cinfo.image_height {
        let row_ptr = rgb_data
            .as_ptr()
            .add(cinfo.next_scanline as usize * row_stride);
        let mut row_array = [row_ptr as *const u8];
        mozjpeg_sys::jpeg_write_scanlines(&mut cinfo, row_array.as_mut_ptr() as *mut *const u8, 1);
    }

    mozjpeg_sys::jpeg_finish_compress(&mut cinfo);
    mozjpeg_sys::jpeg_destroy_compress(&mut cinfo);

    let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
    libc::free(outbuffer as *mut std::ffi::c_void);

    result
}
