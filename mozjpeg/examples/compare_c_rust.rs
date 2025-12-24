//! Compare Rust vs C mozjpeg performance on larger workloads.
//!
//! Run with: cargo run --release --example compare_c_rust

use std::time::Instant;

fn main() {
    println!("Rust vs C mozjpeg Performance Comparison");
    println!("=========================================\n");

    // Test different image sizes
    let sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (width, height) in sizes {
        compare_at_size(width, height);
        println!();
    }
}

fn compare_at_size(width: usize, height: usize) {
    // Create test image with realistic content
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let noise = ((x * 7 + y * 13) % 50) as u8;
            rgb[idx] = ((x * 255 / width) as u8).saturating_add(noise);
            rgb[idx + 1] = ((y * 255 / height) as u8).saturating_add(noise);
            rgb[idx + 2] = (((x + y) * 255 / (width + height)) as u8).saturating_add(noise);
        }
    }

    let pixels = width * height;
    let iterations = if pixels > 1_000_000 { 10 } else if pixels > 250_000 { 20 } else { 50 };

    println!("Image size: {}x{} ({:.1} MP, {} iterations)", width, height, pixels as f64 / 1_000_000.0, iterations);
    println!("{:-<75}", "");
    println!("{:<25} {:>10} {:>10} {:>12} {:>12}", "Configuration", "Rust (ms)", "C (ms)", "Ratio", "Size R/C");
    println!("{:-<75}", "");

    // Test configurations: (name, use_fastest)
    // Compare only real mozjpeg settings: fastest vs defaults
    let configs = [
        ("Fastest (libjpeg-turbo)", true),
        ("mozjpeg defaults", false),
    ];

    for (name, use_fastest) in configs {
        let (rust_time, rust_size) = benchmark_rust(&rgb, width, height, iterations, use_fastest);
        let (c_time, c_size) = benchmark_c(&rgb, width, height, iterations, use_fastest);

        let ratio = rust_time / c_time;
        let ratio_str = if ratio < 1.0 {
            format!("{:.2}x faster", 1.0 / ratio)
        } else {
            format!("{:.2}x slower", ratio)
        };

        println!("{:<25} {:>10.2} {:>10.2} {:>12} {:>6}/{:<6}",
            name, rust_time, c_time, ratio_str, rust_size, c_size);
    }
}

fn benchmark_rust(rgb: &[u8], width: usize, height: usize, iterations: usize,
                  use_fastest: bool) -> (f64, usize) {
    let encoder = if use_fastest {
        mozjpeg::Encoder::fastest().quality(85)
    } else {
        // Use mozjpeg defaults: max_compression() which enables optimize_scans
        mozjpeg::Encoder::max_compression().quality(85)
    };

    // Warm up
    let jpeg = encoder.encode_rgb(rgb, width as u32, height as u32).unwrap();
    let size = jpeg.len();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.encode_rgb(rgb, width as u32, height as u32).unwrap();
    }
    let elapsed = start.elapsed();

    (elapsed.as_secs_f64() * 1000.0 / iterations as f64, size)
}

fn benchmark_c(rgb: &[u8], width: usize, height: usize, iterations: usize,
               use_fastest: bool) -> (f64, usize) {
    // Warm up and get size
    let size = encode_c_once(rgb, width, height, use_fastest).len();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_c_once(rgb, width, height, use_fastest);
    }
    let elapsed = start.elapsed();

    (elapsed.as_secs_f64() * 1000.0 / iterations as f64, size)
}

fn encode_c_once(rgb: &[u8], width: usize, height: usize, use_fastest: bool) -> Vec<u8> {
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

        cinfo.image_width = width as u32;
        cinfo.image_height = height as u32;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_set_quality(&mut cinfo, 85, 1);

        // Set 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        if use_fastest {
            // Disable all mozjpeg optimizations for fastest mode
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 0);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 0);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);
            cinfo.optimize_coding = 0;
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        } else {
            // Use mozjpeg defaults: trellis, progressive with optimize_scans
            // jpeg_set_defaults already enables trellis, just need to enable progressive
            jpeg_simple_progression(&mut cinfo);
            // optimize_scans is enabled by default with jpeg_simple_progression
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = rgb.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}
