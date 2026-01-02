//! Full Kodak corpus comparison with explicit trellis settings

use mozjpeg_rs::{Encoder, TrellisConfig};
use std::fs;
use std::path::Path;

fn main() {
    let corpus_dir = "/home/lilith/work/mozjpeg-rs/corpus/kodak";

    let mut files: Vec<_> = fs::read_dir(corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .map(|e| e.path())
        .collect();
    files.sort();

    println!(
        "=== Kodak Corpus Full Comparison ({} images) ===",
        files.len()
    );
    println!();
    println!("Settings: Baseline mode, 4:2:0, Huffman opt, explicit trellis control");
    println!();

    // Trellis ON
    println!("=== TRELLIS ON ===");
    println!(
        "{:>4} {:>10} {:>10} {:>8} {:>10}",
        "Q", "Rust", "C", "Diff", "MaxDiff"
    );
    for quality in [50, 75, 85, 90, 95, 97] {
        let (total_rust, total_c, max_diff, max_file) = run_corpus(&files, quality, true);
        let avg_diff = (total_rust as f64 / total_c as f64 - 1.0) * 100.0;
        println!(
            "{:>4} {:>10} {:>10} {:>+7.2}% {:>+7.2}% ({})",
            quality, total_rust, total_c, avg_diff, max_diff, max_file
        );
    }

    // Trellis OFF
    println!();
    println!("=== TRELLIS OFF ===");
    println!(
        "{:>4} {:>10} {:>10} {:>8} {:>10}",
        "Q", "Rust", "C", "Diff", "MaxDiff"
    );
    for quality in [50, 75, 85, 90, 95, 97] {
        let (total_rust, total_c, max_diff, max_file) = run_corpus(&files, quality, false);
        let avg_diff = (total_rust as f64 / total_c as f64 - 1.0) * 100.0;
        println!(
            "{:>4} {:>10} {:>10} {:>+7.2}% {:>+7.2}% ({})",
            quality, total_rust, total_c, avg_diff, max_diff, max_file
        );
    }
}

fn run_corpus(
    files: &[std::path::PathBuf],
    quality: u8,
    trellis: bool,
) -> (usize, usize, f64, String) {
    let mut total_rust = 0usize;
    let mut total_c = 0usize;
    let mut max_diff = 0.0f64;
    let mut max_file = String::new();

    for path in files {
        let rgb_data = load_png(path);
        let (width, height) = get_png_dimensions(path);

        let rust_jpeg = if trellis {
            Encoder::new(false)
                .quality(quality)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::default())
                .encode_rgb(&rgb_data, width, height)
                .unwrap()
        } else {
            Encoder::new(false)
                .quality(quality)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::disabled())
                .encode_rgb(&rgb_data, width, height)
                .unwrap()
        };

        let c_jpeg = encode_c_baseline(&rgb_data, width, height, quality, trellis);

        total_rust += rust_jpeg.len();
        total_c += c_jpeg.len();

        let diff = (rust_jpeg.len() as f64 / c_jpeg.len() as f64 - 1.0) * 100.0;
        if diff.abs() > max_diff.abs() {
            max_diff = diff;
            max_file = path.file_name().unwrap().to_string_lossy().to_string();
        }
    }

    (total_rust, total_c, max_diff, max_file)
}

fn load_png(path: &Path) -> Vec<u8> {
    let decoder = png::Decoder::new(fs::File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf[..info.buffer_size()].to_vec()
}

fn get_png_dimensions(path: &Path) -> (u32, u32) {
    let decoder = png::Decoder::new(fs::File::open(path).unwrap());
    let reader = decoder.read_info().unwrap();
    let info = reader.info();
    (info.width, info.height)
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
