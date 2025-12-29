//! Test the benefit of multipass (use_scans_in_trellis) in C mozjpeg
//!
//! Run with: cargo run --release --example multipass_test

use std::time::Instant;

fn decode_jpeg(jpeg_data: &[u8]) -> (Vec<u8>, u32, u32) {
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(jpeg_data);
    let pixels = decoder.decode().expect("Failed to decode JPEG");
    let info = decoder.info().unwrap();
    (pixels, info.width as u32, info.height as u32)
}

fn compute_butteraugli_score(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    // Butteraugli: lower is better (perceptual distance)
    let params = butteraugli::ButteraugliParams::default();
    let result = butteraugli::compute_butteraugli(
        original,
        decoded,
        width as usize,
        height as usize,
        &params,
    )
    .expect("butteraugli computation failed");
    result.score
}

fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    // Create a more realistic test image with gradients and edges
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            // Create a pattern with gradients and some noise
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

fn encode_c_mozjpeg(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    trellis: bool,
    progressive: bool,
    multipass: bool,
) -> Vec<u8> {
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

        // Set trellis options
        if trellis {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
        }

        // Set multipass (use_scans_in_trellis)
        if multipass {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_USE_SCANS_IN_TRELLIS, 1);
        }

        // Set progressive mode
        if progressive {
            jpeg_simple_progression(&mut cinfo);
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_ptr = rgb.as_ptr().add(cinfo.next_scanline as usize * row_stride) as *const u8;
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

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    use std::fs::File;

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

fn main() {
    println!("\n=== Multipass (use_scans_in_trellis) Benefit Test ===\n");
    println!("Testing whether multipass provides compression benefit in C mozjpeg.\n");

    // First test with synthetic image
    let sizes = [(512, 512)];
    let qualities = [75, 85, 95];
    let iterations = 10;

    for &(w, h) in &sizes {
        let rgb = create_test_image(w, h);

        println!(
            "### Synthetic {}x{} image ({} iterations)\n",
            w, h, iterations
        );

        // Test baseline (non-progressive) - multipass has no effect
        println!("**Baseline mode (multipass has no effect here):**\n");
        println!("| Quality | Trellis | Trellis+MP | Diff |");
        println!("|---------|---------|------------|------|");

        for &q in &qualities {
            let (_, size_trellis) = benchmark(
                || encode_c_mozjpeg(&rgb, w, h, q, true, false, false),
                iterations,
            );
            let (_, size_mp) = benchmark(
                || encode_c_mozjpeg(&rgb, w, h, q, true, false, true),
                iterations,
            );

            let diff_pct = (size_mp as f64 - size_trellis as f64) / size_trellis as f64 * 100.0;
            println!(
                "| Q{} | {} | {} | {:+.2}% |",
                q, size_trellis, size_mp, diff_pct
            );
        }

        // Test progressive - where multipass should have effect
        println!("\n**Progressive mode (multipass may help):**\n");
        println!("| Quality | Trellis | Trellis+MP | Diff | Time (no MP) | Time (MP) |");
        println!("|---------|---------|------------|------|--------------|-----------|");

        for &q in &qualities {
            let (time_trellis, size_trellis) = benchmark(
                || encode_c_mozjpeg(&rgb, w, h, q, true, true, false),
                iterations,
            );
            let (time_mp, size_mp) = benchmark(
                || encode_c_mozjpeg(&rgb, w, h, q, true, true, true),
                iterations,
            );

            let diff_pct = (size_mp as f64 - size_trellis as f64) / size_trellis as f64 * 100.0;
            println!(
                "| Q{} | {} | {} | {:+.2}% | {:.1}ms | {:.1}ms |",
                q, size_trellis, size_mp, diff_pct, time_trellis, time_mp
            );
        }

        println!();
    }

    // Now test with real Kodak images
    println!("\n### Real Kodak images (progressive mode, Q85)\n");

    let corpus_dir = std::env::var("CODEC_CORPUS_DIR")
        .or_else(|_| std::env::var("MOZJPEG_CORPUS_DIR"))
        .unwrap_or_else(|_| "./corpus".to_string());

    let kodak_dir = format!("{}/kodak", corpus_dir);

    if std::path::Path::new(&kodak_dir).exists() {
        println!("| Image | Trellis | Trellis+MP | Size Diff | Quality (no MP) | Quality (MP) | Q Diff |");
        println!("|-------|---------|------------|-----------|-----------------|--------------|--------|");

        let mut total_no_mp = 0usize;
        let mut total_mp = 0usize;
        let mut total_q_no_mp = 0.0f64;
        let mut total_q_mp = 0.0f64;
        let mut count = 0;

        for entry in std::fs::read_dir(&kodak_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "png") {
                if let Some((rgb, w, h)) = load_png(&path) {
                    let jpeg_no_mp = encode_c_mozjpeg(&rgb, w, h, 85, true, true, false);
                    let jpeg_mp = encode_c_mozjpeg(&rgb, w, h, 85, true, true, true);

                    let size_no_mp = jpeg_no_mp.len();
                    let size_mp = jpeg_mp.len();

                    // Decode and measure quality
                    let (decoded_no_mp, _, _) = decode_jpeg(&jpeg_no_mp);
                    let (decoded_mp, _, _) = decode_jpeg(&jpeg_mp);

                    let q_no_mp = compute_butteraugli_score(&rgb, &decoded_no_mp, w, h);
                    let q_mp = compute_butteraugli_score(&rgb, &decoded_mp, w, h);

                    let size_diff_pct =
                        (size_mp as f64 - size_no_mp as f64) / size_no_mp as f64 * 100.0;
                    let q_diff = q_mp - q_no_mp;
                    let name = path.file_name().unwrap().to_str().unwrap();
                    println!(
                        "| {} | {} | {} | {:+.2}% | {:.2} | {:.2} | {:+.3} |",
                        name, size_no_mp, size_mp, size_diff_pct, q_no_mp, q_mp, q_diff
                    );

                    total_no_mp += size_no_mp;
                    total_mp += size_mp;
                    total_q_no_mp += q_no_mp;
                    total_q_mp += q_mp;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_size_diff = (total_mp as f64 - total_no_mp as f64) / total_no_mp as f64 * 100.0;
            let avg_q_no_mp = total_q_no_mp / count as f64;
            let avg_q_mp = total_q_mp / count as f64;
            let avg_q_diff = avg_q_mp - avg_q_no_mp;
            println!("|-------|---------|------------|-----------|-----------------|--------------|--------|");
            println!(
                "| **Avg ({})** | {} | {} | **{:+.2}%** | {:.2} | {:.2} | **{:+.3}** |",
                count, total_no_mp, total_mp, avg_size_diff, avg_q_no_mp, avg_q_mp, avg_q_diff
            );
        }
    } else {
        println!(
            "Kodak corpus not found at {}. Run ./scripts/fetch-corpus.sh first.",
            kodak_dir
        );
    }

    println!("\n**Interpretation:**");
    println!("- Negative diff = multipass produces smaller files");
    println!("- Positive diff = multipass produces larger files");
    println!("- Multipass should only affect progressive mode");
    println!("- Multipass considers scan structure during trellis optimization");
}
