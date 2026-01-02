//! Comprehensive benchmark comparing Rust and C mozjpeg across all modes and quality levels
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::fs::File;
use std::path::Path;

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

// Rust encoders
fn encode_rust_baseline(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::new(false)
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

fn encode_rust_progressive(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::new(false)
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(false)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

fn encode_rust_max_compression(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::max_compression()
        .quality(quality)
        .encode_rgb(rgb, width, height)
        .expect("encoding failed")
}

// C mozjpeg encoders
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
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
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

fn encode_c_progressive(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
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

fn encode_c_max_compression(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // Match Rust max_compression settings manually
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1); // Enable optimize_scans
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
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

struct BenchmarkResult {
    quality: u8,
    rust_bytes: usize,
    c_bytes: usize,
    diff_pct: f64,
}

fn run_benchmark<F1, F2>(
    images: &[(Vec<u8>, u32, u32)],
    qualities: &[u8],
    rust_encoder: F1,
    c_encoder: F2,
) -> Vec<BenchmarkResult>
where
    F1: Fn(&[u8], u32, u32, u8) -> Vec<u8>,
    F2: Fn(&[u8], u32, u32, u8) -> Vec<u8>,
{
    let mut results = Vec::new();

    for &quality in qualities {
        let mut total_rust = 0usize;
        let mut total_c = 0usize;

        for (rgb, width, height) in images {
            let rust_jpeg = rust_encoder(rgb, *width, *height, quality);
            let c_jpeg = c_encoder(rgb, *width, *height, quality);
            total_rust += rust_jpeg.len();
            total_c += c_jpeg.len();
        }

        let diff_pct = ((total_rust as f64 / total_c as f64) - 1.0) * 100.0;
        results.push(BenchmarkResult {
            quality,
            rust_bytes: total_rust,
            c_bytes: total_c,
            diff_pct,
        });
    }

    results
}

fn main() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Corpus not found at corpus/kodak");
        eprintln!("Run: ./scripts/fetch-corpus.sh");
        return;
    }

    // Load all images
    let mut images: Vec<(Vec<u8>, u32, u32)> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        if let Some(img) = load_png(&entry.path()) {
            images.push(img);
        }
    }

    println!("Loaded {} images from Kodak corpus\n", images.len());

    let qualities = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99, 100];

    // Baseline mode
    println!("=== BASELINE MODE (sequential, trellis + Huffman opt) ===");
    println!("| Quality | Rust (bytes) | C (bytes) | Diff |");
    println!("|---------|-------------|-----------|------|");
    let baseline = run_benchmark(&images, &qualities, encode_rust_baseline, encode_c_baseline);
    for r in &baseline {
        println!(
            "| Q{:3} | {:>11} | {:>9} | {:>+.2}% |",
            r.quality, r.rust_bytes, r.c_bytes, r.diff_pct
        );
    }

    // Simple progressive mode
    println!("\n=== PROGRESSIVE MODE (simple 4-scan script) ===");
    println!("| Quality | Rust (bytes) | C (bytes) | Diff |");
    println!("|---------|-------------|-----------|------|");
    let progressive = run_benchmark(
        &images,
        &qualities,
        encode_rust_progressive,
        encode_c_progressive,
    );
    for r in &progressive {
        println!(
            "| Q{:3} | {:>11} | {:>9} | {:>+.2}% |",
            r.quality, r.rust_bytes, r.c_bytes, r.diff_pct
        );
    }

    // Max compression mode (with optimize_scans)
    println!("\n=== MAX COMPRESSION MODE (progressive + optimize_scans) ===");
    println!("| Quality | Rust (bytes) | C (bytes) | Diff |");
    println!("|---------|-------------|-----------|------|");
    let max_comp = run_benchmark(
        &images,
        &qualities,
        encode_rust_max_compression,
        encode_c_max_compression,
    );
    for r in &max_comp {
        println!(
            "| Q{:3} | {:>11} | {:>9} | {:>+.2}% |",
            r.quality, r.rust_bytes, r.c_bytes, r.diff_pct
        );
    }

    // Summary table for README
    println!("\n=== SUMMARY (for README) ===\n");
    println!(
        "### Compression Parity with C mozjpeg (Kodak corpus, {} images)\n",
        images.len()
    );
    println!("| Quality | Baseline | Progressive | Max Compression |");
    println!("|---------|----------|-------------|-----------------|");
    for i in 0..qualities.len() {
        println!(
            "| Q{:3} | {:>+.2}% | {:>+.2}% | {:>+.2}% |",
            qualities[i], baseline[i].diff_pct, progressive[i].diff_pct, max_comp[i].diff_pct
        );
    }
}
