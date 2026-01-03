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

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::baseline_optimized()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(false)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

fn encode_c_simple_progressive(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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

        // Use ImageMagick quant tables (index 3) to match Rust
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Disable optimize_scans to use simple progressive
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);

        // Enable progressive mode with simple script (NOT JCP_MAX_COMPRESSION)
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Enable same optimizations as Rust
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

fn main() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Corpus not found");
        return;
    }

    let qualities = [50, 75, 85, 90, 95, 97];

    println!("FULL KODAK CORPUS - Simple Progressive (optimize_scans=false)");
    println!("Both Rust and C use same settings: trellis + huffman opt + deringing + 4:2:0\n");

    for &quality in &qualities {
        let mut total_rust = 0usize;
        let mut total_c = 0usize;
        let mut count = 0;
        let mut max_diff = f64::MIN;
        let mut max_diff_img = String::new();

        let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in &entries {
            let path = entry.path();
            let name = path.file_stem().unwrap().to_string_lossy();

            if let Some((rgb, width, height)) = load_png(&path) {
                let rust_jpeg = encode_rust(&rgb, width, height, quality);
                let c_jpeg = encode_c_simple_progressive(&rgb, width, height, quality);

                let diff_pct = ((rust_jpeg.len() as f64 / c_jpeg.len() as f64) - 1.0) * 100.0;

                if diff_pct > max_diff {
                    max_diff = diff_pct;
                    max_diff_img = name.to_string();
                }

                total_rust += rust_jpeg.len();
                total_c += c_jpeg.len();
                count += 1;
            }
        }

        let avg_diff = ((total_rust as f64 / total_c as f64) - 1.0) * 100.0;
        println!(
            "Q{:2}: {:2} images | Rust: {:>9} | C: {:>9} | Avg: {:>+6.2}% | Max: {:>+6.2}% ({})",
            quality, count, total_rust, total_c, avg_diff, max_diff, max_diff_img
        );
    }
}
