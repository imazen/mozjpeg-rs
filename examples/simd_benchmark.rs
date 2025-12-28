use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
use std::time::Instant;

fn main() {
    // Load test image
    let source_path = "corpus/kodak/10.png";
    let file = std::fs::File::open(source_path).expect("Need corpus/kodak/10.png");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    println!("Image: {}x{} ({} pixels)", width, height, width * height);
    println!();

    let iterations = 20;

    // Baseline (no trellis)
    println!("=== Baseline (no trellis, huffman opt) ===");
    let encoder = Encoder::new()
        .quality(85)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true)
        .trellis(TrellisConfig::disabled());

    // Warmup
    let _ = encoder.encode_rgb(rgb_data, width, height);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.encode_rgb(rgb_data, width, height);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "  Rust: {:.2} ms/encode ({} iterations)",
        avg_ms, iterations
    );

    // With trellis
    println!();
    println!("=== Trellis quantization (AC + DC) ===");
    let encoder = Encoder::new()
        .quality(85)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true)
        .trellis(TrellisConfig::default());

    // Warmup
    let _ = encoder.encode_rgb(rgb_data, width, height);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.encode_rgb(rgb_data, width, height);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "  Rust: {:.2} ms/encode ({} iterations)",
        avg_ms, iterations
    );

    // Progressive + trellis
    println!();
    println!("=== Progressive + trellis ===");
    let encoder = Encoder::new()
        .quality(85)
        .subsampling(Subsampling::S420)
        .progressive(true)
        .optimize_huffman(true)
        .trellis(TrellisConfig::default());

    // Warmup
    let _ = encoder.encode_rgb(rgb_data, width, height);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.encode_rgb(rgb_data, width, height);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "  Rust: {:.2} ms/encode ({} iterations)",
        avg_ms, iterations
    );

    // C mozjpeg baseline
    println!();
    println!("=== C mozjpeg comparison ===");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_c_baseline(rgb_data, width, height, 85);
    }
    let elapsed = start.elapsed();
    let c_baseline = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  C baseline: {:.2} ms/encode", c_baseline);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_c_trellis(rgb_data, width, height, 85);
    }
    let elapsed = start.elapsed();
    let c_trellis = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  C trellis:  {:.2} ms/encode", c_trellis);
}

fn encode_c_baseline(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;

        cinfo.optimize_coding = 1;

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

fn encode_c_trellis(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;

        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);

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
