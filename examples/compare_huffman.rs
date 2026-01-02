//! Compare Huffman tables in Rust vs C output

use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::process::Command;

fn main() {
    let s = 16usize;
    let mut rgb = vec![0u8; s * s * 3];
    for y in 0..s {
        for x in 0..s {
            let idx = (y * s + x) * 3;
            rgb[idx] = ((x * 255) / s.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / s.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }

    // Encode with Rust - baseline, WITH Huffman optimization
    let rust_jpeg = Encoder::new(false)
        .quality(85)
        .subsampling(Subsampling::S444)
        .progressive(false)
        .optimize_huffman(true) // ENABLE Huffman optimization
        .trellis(TrellisConfig::disabled())
        .encode_rgb(&rgb, s as u32, s as u32)
        .unwrap();

    // Encode with C
    let c_jpeg = encode_c(&rgb, s as u32, s as u32, 85);

    std::fs::write("/tmp/rust_test.jpg", &rust_jpeg).unwrap();
    std::fs::write("/tmp/c_test.jpg", &c_jpeg).unwrap();

    println!("Rust size: {}", rust_jpeg.len());
    println!("C size: {}", c_jpeg.len());

    // Use djpeg to dump info
    let rust_info = Command::new("djpeg")
        .args(["-verbose", "-verbose", "/tmp/rust_test.jpg"])
        .output();
    let c_info = Command::new("djpeg")
        .args(["-verbose", "-verbose", "/tmp/c_test.jpg"])
        .output();

    if let Ok(output) = rust_info {
        println!("\n=== Rust JPEG info ===");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if let Ok(output) = c_info {
        println!("\n=== C JPEG info ===");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }
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
        let mut outsize: u64 = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality, 1);

        // CRITICAL: Explicitly disable progressive (clear scan_info that jpeg_set_defaults set)
        cinfo.scan_info = ptr::null();
        cinfo.num_scans = 0;

        // 4:4:4 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // ENABLE Huffman optimization
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
