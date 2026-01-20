//! Compare image quality metrics between Rust and C mozjpeg

use dssim::Dssim;
use mozjpeg_rs::{Encoder, TrellisConfig};
use rgb::RGB8;
use std::fs;
use std::path::Path;

fn main() {
    let corpus_dir = "/home/lilith/work/mozjpeg-rs/corpus/kodak";

    let mut files: Vec<_> = fs::read_dir(corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .map(|e| e.path())
        .collect();
    files.sort();

    // Use first 6 images for quicker test
    let files = &files[..6.min(files.len())];

    println!("=== Quality Metrics Comparison (Rust vs C mozjpeg) ===");
    println!("Source: {} Kodak images", files.len());
    println!();
    println!("DSSIM: lower is better (0 = identical)");
    println!();

    println!(
        "{:>4} {:>8} {:>12} {:>12} {:>8}",
        "Q", "Size %", "Rust DSSIM", "C DSSIM", "Diff"
    );

    for quality in [50, 75, 85, 90, 95, 97] {
        let mut rust_dssim_sum = 0.0;
        let mut c_dssim_sum = 0.0;
        let mut rust_size_sum = 0usize;
        let mut c_size_sum = 0usize;
        let mut count = 0;

        for path in files {
            let (rgb_data, width, height) = load_png_rgb(path);

            // Encode with Rust
            let rust_jpeg = Encoder::baseline_optimized()
                .quality(quality)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::default())
                .encode_rgb(&rgb_data, width, height)
                .unwrap();

            // Encode with C
            let c_jpeg = encode_c_baseline(&rgb_data, width, height, quality, true);

            // Decode both
            let rust_decoded = jpeg_decode(&rust_jpeg);
            let c_decoded = jpeg_decode(&c_jpeg);

            // Calculate DSSIM (lower is better)
            let rust_dssim = calculate_dssim(&rgb_data, &rust_decoded, width, height);
            let c_dssim = calculate_dssim(&rgb_data, &c_decoded, width, height);

            rust_dssim_sum += rust_dssim;
            c_dssim_sum += c_dssim;
            rust_size_sum += rust_jpeg.len();
            c_size_sum += c_jpeg.len();
            count += 1;
        }

        let rust_dssim_avg = rust_dssim_sum / count as f64;
        let c_dssim_avg = c_dssim_sum / count as f64;
        let size_diff = (rust_size_sum as f64 / c_size_sum as f64 - 1.0) * 100.0;

        // Negative diff means Rust is better (lower DSSIM)
        let dssim_diff = ((rust_dssim_avg - c_dssim_avg) / c_dssim_avg) * 100.0;

        println!(
            "{:>4} {:>+7.2}% {:>12.6} {:>12.6} {:>+7.1}%",
            quality, size_diff, rust_dssim_avg, c_dssim_avg, dssim_diff
        );
    }

    println!();
    println!("Negative size % = Rust smaller, Negative DSSIM diff % = Rust better quality");
}

fn load_png_rgb(path: &Path) -> (Vec<u8>, u32, u32) {
    let decoder = png::Decoder::new(fs::File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    (buf[..info.buffer_size()].to_vec(), info.width, info.height)
}

fn jpeg_decode(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    decoder.decode().unwrap()
}

fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let attr = Dssim::new();

    let orig_rgb: Vec<RGB8> = original
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();

    let dec_rgb: Vec<RGB8> = decoded
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();

    let orig_img = attr
        .create_image_rgb(&orig_rgb, width as usize, height as usize)
        .expect("Failed to create original image");

    let dec_img = attr
        .create_image_rgb(&dec_rgb, width as usize, height as usize)
        .expect("Failed to create decoded image");

    let (dssim_val, _) = attr.compare(&orig_img, dec_img);
    dssim_val.into()
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
