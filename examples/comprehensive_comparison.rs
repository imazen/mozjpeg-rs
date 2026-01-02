//! Comprehensive comparison: Rust vs C mozjpeg with matching settings

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

    println!("=== Comprehensive Comparison ===");
    println!("Image: {}x{}", width, height);
    println!();
    println!("Settings: Baseline mode, 4:2:0 subsampling, Huffman optimization");
    println!();

    // Test 1: Trellis ON
    println!("=== TRELLIS ON ===");
    println!("{:>4} {:>8} {:>8} {:>8}", "Q", "Rust", "C", "Diff");
    for quality in [50, 75, 85, 90, 95, 97] {
        let rust = Encoder::new(false)
            .quality(quality)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .encode_rgb(rgb_data, width, height)
            .unwrap();

        let c = encode_c_baseline(&rgb_data, width, height, quality, true);
        let diff = (rust.len() as f64 / c.len() as f64 - 1.0) * 100.0;
        println!(
            "{:>4} {:>8} {:>8} {:>+7.2}%",
            quality,
            rust.len(),
            c.len(),
            diff
        );
    }

    // Test 2: Trellis OFF
    println!();
    println!("=== TRELLIS OFF ===");
    println!("{:>4} {:>8} {:>8} {:>8}", "Q", "Rust", "C", "Diff");
    for quality in [50, 75, 85, 90, 95, 97] {
        let rust = Encoder::new(false)
            .quality(quality)
            .progressive(false)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .encode_rgb(rgb_data, width, height)
            .unwrap();

        let c = encode_c_baseline(&rgb_data, width, height, quality, false);
        let diff = (rust.len() as f64 / c.len() as f64 - 1.0) * 100.0;
        println!(
            "{:>4} {:>8} {:>8} {:>+7.2}%",
            quality,
            rust.len(),
            c.len(),
            diff
        );
    }
}

fn encode_c_baseline(rgb: &[u8], width: u32, height: u32, quality: u8, trellis: bool) -> Vec<u8> {
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

        // Force baseline mode
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

        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT,
            if trellis { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT_DC,
            if trellis { 1 } else { 0 },
        );

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
