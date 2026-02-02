//! Apples-to-apples timing benchmark: Rust vs C mozjpeg
//!
//! Uses identical settings for fair comparison:
//! - 4:2:0 subsampling
//! - Robidoux quant tables (mozjpeg default)
//! - Trellis quantization
//! - Huffman optimization
//! - Overshoot deringing
//!
//! Run with: cargo run --release --example timing_benchmark

use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::fs::File;
use std::path::Path;
use std::time::Instant;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn encode_rust_baseline(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::baseline_optimized()
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
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

        // Use Robidoux tables (mozjpeg default, index 3)
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Huffman optimization
        cinfo.optimize_coding = 1;

        // Trellis quantization (AC + DC)
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);

        // Overshoot deringing
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);

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

fn benchmark<F: Fn() -> Vec<u8>>(f: F, warmup: u32, iterations: u32) -> (f64, usize) {
    // Warmup
    for _ in 0..warmup {
        let _ = f();
    }

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
    let corpus_path = Path::new("corpus/CID22/CID22-512/training");
    if !corpus_path.exists() {
        eprintln!("Corpus not found at corpus/CID22/CID22-512/training");
        eprintln!("Run: ./scripts/fetch-corpus.sh");
        return;
    }

    // Load a representative image
    let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let test_image = entries.first().expect("No images in corpus").path();
    let (rgb, width, height) = load_png(&test_image).expect("Failed to load test image");

    println!("\n=== Apples-to-Apples Timing Benchmark ===");
    println!(
        "Image: {}x{} ({})",
        width,
        height,
        test_image.file_name().unwrap().to_string_lossy()
    );
    println!("Settings: 4:2:0, Robidoux tables, Trellis AC+DC, Huffman opt, Deringing");
    println!();

    let quality = 85u8;
    let warmup = 5;
    let iterations = 30;

    println!(
        "Quality {} ({} iterations, {} warmup):\n",
        quality, iterations, warmup
    );

    // Baseline mode (sequential)
    let (rust_ms, rust_size) = benchmark(
        || encode_rust_baseline(&rgb, width, height, quality),
        warmup,
        iterations,
    );
    let (c_ms, c_size) = benchmark(
        || encode_c_baseline(&rgb, width, height, quality),
        warmup,
        iterations,
    );

    println!("| Encoder | Time (ms) | Size (bytes) | Size Diff |");
    println!("|---------|-----------|--------------|-----------|");
    println!(
        "| Rust    | {:>9.2} | {:>12} | {:>+8.2}% |",
        rust_ms,
        rust_size,
        ((rust_size as f64 / c_size as f64) - 1.0) * 100.0
    );
    println!(
        "| C       | {:>9.2} | {:>12} | {:>8} |",
        c_ms, c_size, "baseline"
    );
    println!();

    let speed_ratio = rust_ms / c_ms;
    if speed_ratio < 1.0 {
        println!("Rust is {:.2}x FASTER than C mozjpeg", 1.0 / speed_ratio);
    } else {
        println!("Rust is {:.2}x SLOWER than C mozjpeg", speed_ratio);
    }

    // Also run on a larger synthetic image for more stable timing
    println!("\n--- Synthetic 2048x2048 image ---\n");

    let large_rgb: Vec<u8> = (0..(2048 * 2048 * 3))
        .map(|i| ((i * 17 + (i / 2048) * 31) % 256) as u8)
        .collect();

    let warmup = 2;
    let iterations = 10;

    println!(
        "Quality {} ({} iterations, {} warmup):\n",
        quality, iterations, warmup
    );

    let (rust_ms, rust_size) = benchmark(
        || encode_rust_baseline(&large_rgb, 2048, 2048, quality),
        warmup,
        iterations,
    );
    let (c_ms, c_size) = benchmark(
        || encode_c_baseline(&large_rgb, 2048, 2048, quality),
        warmup,
        iterations,
    );

    println!("| Encoder | Time (ms) | Size (bytes) | Size Diff |");
    println!("|---------|-----------|--------------|-----------|");
    println!(
        "| Rust    | {:>9.2} | {:>12} | {:>+8.2}% |",
        rust_ms,
        rust_size,
        ((rust_size as f64 / c_size as f64) - 1.0) * 100.0
    );
    println!(
        "| C       | {:>9.2} | {:>12} | {:>8} |",
        c_ms, c_size, "baseline"
    );
    println!();

    let speed_ratio = rust_ms / c_ms;
    if speed_ratio < 1.0 {
        println!("Rust is {:.2}x FASTER than C mozjpeg", 1.0 / speed_ratio);
    } else {
        println!("Rust is {:.2}x SLOWER than C mozjpeg", speed_ratio);
    }
}
