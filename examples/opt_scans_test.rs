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

fn main() {
    let corpus_path = Path::new("corpus/kodak");
    if !corpus_path.exists() {
        eprintln!("Corpus not found");
        return;
    }

    let qualities = [50, 75, 85, 90, 95, 97];

    println!("KODAK CORPUS - Comparing optimize_scans=false vs =true\n");

    for &quality in &qualities {
        let mut total_simple = 0usize;
        let mut total_opt = 0usize;
        let mut total_c = 0usize;
        let mut count = 0;

        let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries.iter().take(6) {
            let path = entry.path();

            if let Some((rgb, width, height)) = load_png(&path) {
                // Simple progressive (optimize_scans=false)
                let simple = Encoder::baseline_optimized()
                    .quality(quality)
                    .progressive(true)
                    .optimize_huffman(true)
                    .trellis(TrellisConfig::default())
                    .optimize_scans(false)
                    .subsampling(Subsampling::S420)
                    .encode_rgb(&rgb, width, height)
                    .unwrap();

                // With optimize_scans
                let opt = Encoder::baseline_optimized()
                    .quality(quality)
                    .progressive(true)
                    .optimize_huffman(true)
                    .trellis(TrellisConfig::default())
                    .optimize_scans(true)
                    .subsampling(Subsampling::S420)
                    .encode_rgb(&rgb, width, height)
                    .unwrap();

                // C mozjpeg
                let c_jpeg = encode_c(&rgb, width, height, quality);

                total_simple += simple.len();
                total_opt += opt.len();
                total_c += c_jpeg.len();
                count += 1;
            }
        }

        let simple_diff = ((total_simple as f64 / total_c as f64) - 1.0) * 100.0;
        let opt_diff = ((total_opt as f64 / total_c as f64) - 1.0) * 100.0;
        let opt_vs_simple = ((total_opt as f64 / total_simple as f64) - 1.0) * 100.0;

        println!(
            "Q{:2}: Simple {:>+6.2}% | opt_scans {:>+6.2}% | opt vs simple {:>+6.2}%",
            quality, simple_diff, opt_diff, opt_vs_simple
        );
    }
}

fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
