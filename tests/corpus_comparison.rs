//! Corpus comparison test - Rust vs C mozjpeg on Kodak images.
//!
//! This test compares file sizes between Rust and C mozjpeg encoders
//! across the full Kodak corpus at multiple quality levels.

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

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, optimize_scans: bool) -> Vec<u8> {
    Encoder::baseline_optimized()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(optimize_scans)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

#[allow(unsafe_code)]
fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8, optimize_scans: bool) -> Vec<u8> {
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

        // Use JCP_MAX_COMPRESSION profile to match Rust's default progressive script
        // This enables the 9-scan progressive script with SA for luma
        jpeg_c_set_int_param(&mut cinfo, JINT_COMPRESS_PROFILE, 0x5D083AAD_u32 as i32);

        // Use ImageMagick quant tables (index 3) to match Rust
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Set optimize_scans BEFORE calling jpeg_simple_progression
        if optimize_scans {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1);
        } else {
            // Explicitly disable optimize_scans (JCP_MAX_COMPRESSION enables it by default)
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);
        }

        // Enable progressive mode (uses JCP_MAX_COMPRESSION 9-scan script if not optimize_scans)
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Enable optimizations
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

fn count_scans(jpeg_data: &[u8]) -> usize {
    let mut count = 0;
    for i in 0..jpeg_data.len().saturating_sub(1) {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            count += 1;
        }
    }
    count
}

/// Print detailed scan info for debugging
fn print_scan_details(jpeg_data: &[u8], label: &str) {
    println!("  {} scans:", label);
    for i in 0..jpeg_data.len().saturating_sub(1) {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            let ns = jpeg_data[i + 4]; // Number of components in scan
            let ss_se_pos = i + 4 + 1 + (ns as usize * 2);
            if ss_se_pos + 2 < jpeg_data.len() {
                let ss = jpeg_data[ss_se_pos];
                let se = jpeg_data[ss_se_pos + 1];
                let ah_al = jpeg_data[ss_se_pos + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;
                let comps: Vec<u8> = (0..ns)
                    .map(|j| jpeg_data[i + 5 + (j as usize * 2)])
                    .collect();
                println!(
                    "    Ns={} comps={:?} Ss={:2} Se={:2} Ah={} Al={}",
                    ns, comps, ss, se, ah, al
                );
            }
        }
    }
}

#[test]
fn test_corpus_comparison_optimize_scans() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Skipping: corpus not available. Run ./scripts/fetch-corpus.sh");
        return;
    }

    let qualities = [50, 75, 85, 90, 95, 97];

    println!("\n{}", "=".repeat(80));
    println!("CORPUS COMPARISON: Rust vs C mozjpeg (optimize_scans=true)");
    println!("{}\n", "=".repeat(80));

    for &quality in &qualities {
        println!("Quality {}", quality);
        println!("{:-<70}", "");
        println!(
            "{:<12} {:>10} {:>10} {:>8} {:>6} {:>6}",
            "Image", "Rust", "C", "Diff%", "R.Sc", "C.Sc"
        );

        let mut total_rust = 0usize;
        let mut total_c = 0usize;
        let mut count = 0;

        let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries.iter().take(6) {
            // Test first 6 images for speed
            let path = entry.path();
            let name = path.file_stem().unwrap().to_string_lossy();

            if let Some((rgb, width, height)) = load_png(&path) {
                let rust_jpeg = encode_rust(&rgb, width, height, quality, true);
                let c_jpeg = encode_c(&rgb, width, height, quality, true);

                let rust_scans = count_scans(&rust_jpeg);
                let c_scans = count_scans(&c_jpeg);

                let diff_pct = ((rust_jpeg.len() as f64 / c_jpeg.len() as f64) - 1.0) * 100.0;

                println!(
                    "{:<12} {:>10} {:>10} {:>+7.2}% {:>6} {:>6}",
                    name,
                    rust_jpeg.len(),
                    c_jpeg.len(),
                    diff_pct,
                    rust_scans,
                    c_scans
                );

                total_rust += rust_jpeg.len();
                total_c += c_jpeg.len();
                count += 1;
            }
        }

        let avg_diff = ((total_rust as f64 / total_c as f64) - 1.0) * 100.0;
        println!("{:-<70}", "");
        println!(
            "{:<12} {:>10} {:>10} {:>+7.2}%",
            "TOTAL", total_rust, total_c, avg_diff
        );
        println!();
    }
}

#[test]
fn test_corpus_comparison_no_optimize_scans() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Skipping: corpus not available. Run ./scripts/fetch-corpus.sh");
        return;
    }

    println!("\n{}", "=".repeat(80));
    println!("CORPUS COMPARISON: Rust vs C mozjpeg (optimize_scans=false)");
    println!("{}\n", "=".repeat(80));

    let quality = 85;
    println!("Quality {}", quality);
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>6} {:>6}",
        "Image", "Rust", "C", "Diff%", "R.Sc", "C.Sc"
    );

    let mut total_rust = 0usize;
    let mut total_c = 0usize;

    let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut first = true;
    for entry in entries.iter().take(6) {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy();

        if let Some((rgb, width, height)) = load_png(&path) {
            let rust_jpeg = encode_rust(&rgb, width, height, quality, false);
            let c_jpeg = encode_c(&rgb, width, height, quality, false);

            let rust_scans = count_scans(&rust_jpeg);
            let c_scans = count_scans(&c_jpeg);

            let diff_pct = ((rust_jpeg.len() as f64 / c_jpeg.len() as f64) - 1.0) * 100.0;

            println!(
                "{:<12} {:>10} {:>10} {:>+7.2}% {:>6} {:>6}",
                name,
                rust_jpeg.len(),
                c_jpeg.len(),
                diff_pct,
                rust_scans,
                c_scans
            );

            // Print scan details for first image only
            if first {
                print_scan_details(&rust_jpeg, "Rust");
                print_scan_details(&c_jpeg, "C");
                first = false;
            }

            total_rust += rust_jpeg.len();
            total_c += c_jpeg.len();
        }
    }

    let avg_diff = ((total_rust as f64 / total_c as f64) - 1.0) * 100.0;
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>10} {:>10} {:>+7.2}%",
        "TOTAL", total_rust, total_c, avg_diff
    );
}
