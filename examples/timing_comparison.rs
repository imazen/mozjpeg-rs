//! Quick Rust vs C mozjpeg timing comparison
//!
//! Run with: cargo run --release --example timing_comparison

use mozjpeg_rs::{Encoder, QuantTableIdx, TrellisConfig};
use std::time::Instant;

fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    (0..(width * height * 3))
        .map(|i| ((i * 17 + 31) % 256) as u8)
        .collect()
}

fn encode_with_rust(rgb: &[u8], width: u32, height: u32, quality: u8, trellis: bool) -> Vec<u8> {
    let trellis_config = if trellis {
        TrellisConfig::default()
    } else {
        TrellisConfig::disabled()
    };

    // Use JPEG Annex K tables (standard libjpeg default) for fair comparison
    Encoder::fastest()
        .quality(quality)
        .quant_tables(QuantTableIdx::JpegAnnexK)
        .trellis(trellis_config)
        .encode_rgb(rgb, width, height)
        .unwrap()
}

fn encode_with_c(rgb: &[u8], width: u32, height: u32, quality: u8, trellis: bool) -> Vec<u8> {
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
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        if trellis {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_ptr =
                rgb.as_ptr().add(cinfo.next_scanline as usize * row_stride) as *const u8;
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}

fn benchmark<F: Fn() -> Vec<u8>>(f: F, iterations: u32) -> (f64, usize) {
    // Warmup
    let _ = f();

    let start = Instant::now();
    let mut size = 0;
    for _ in 0..iterations {
        size = f().len();
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    (ms, size)
}

fn main() {
    let sizes = [(512, 512), (1024, 1024), (2048, 2048)];
    let iterations = [50, 20, 10];

    println!("\n=== Rust vs C mozjpeg Performance Comparison ===\n");

    for (i, &(w, h)) in sizes.iter().enumerate() {
        let rgb = create_test_image(w, h);
        let iters = iterations[i];

        println!("### {}x{} ({} iterations)\n", w, h, iters);
        println!("| Config | Rust (ms) | C (ms) | Ratio | Rust Size | C Size |");
        println!("|--------|-----------|--------|-------|-----------|--------|");

        // Baseline (no trellis, no optimizations)
        let (rust_ms, rust_size) = benchmark(|| encode_with_rust(&rgb, w, h, 85, false), iters);
        let (c_ms, c_size) = benchmark(|| encode_with_c(&rgb, w, h, 85, false), iters);
        let ratio = rust_ms / c_ms;
        println!(
            "| Baseline (no opts) | {:.2} | {:.2} | {:.2}x | {} | {} |",
            rust_ms, c_ms, ratio, rust_size, c_size
        );

        // Trellis AC+DC (the core mozjpeg feature)
        let (rust_ms, rust_size) = benchmark(|| encode_with_rust(&rgb, w, h, 85, true), iters);
        let (c_ms, c_size) = benchmark(|| encode_with_c(&rgb, w, h, 85, true), iters);
        let ratio = rust_ms / c_ms;
        println!(
            "| Trellis AC+DC | {:.2} | {:.2} | {:.2}x | {} | {} |",
            rust_ms, c_ms, ratio, rust_size, c_size
        );

        println!();
    }
}
