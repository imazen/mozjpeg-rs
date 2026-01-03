//! Test SA script encoding vs simple progressive
use mozjpeg_rs::progressive::{
    generate_minimal_progressive_scans, generate_mozjpeg_max_compression_scans,
};
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
    println!("Testing SA script encoding...\n");

    // Print both scripts
    println!("=== Simple 4-scan script ===");
    for (i, scan) in generate_minimal_progressive_scans(3).iter().enumerate() {
        println!(
            "Scan {}: Ss={:2} Se={:2} Ah={} Al={}",
            i, scan.ss, scan.se, scan.ah, scan.al
        );
    }

    println!("\n=== SA 9-scan script ===");
    for (i, scan) in generate_mozjpeg_max_compression_scans(3).iter().enumerate() {
        println!(
            "Scan {}: Ss={:2} Se={:2} Ah={} Al={}",
            i, scan.ss, scan.se, scan.ah, scan.al
        );
    }

    let (rgb, width, height) = load_png(Path::new("corpus/kodak/10.png")).unwrap();

    println!("\n=== Encoding comparison (kodak/10.png) ===");

    for quality in [50, 75, 85, 90, 95, 97] {
        // Simple progressive
        let simple = Encoder::baseline_optimized()
            .quality(quality)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .subsampling(Subsampling::S420)
            .encode_rgb(&rgb, width, height)
            .unwrap();

        // Check if it decodes
        let simple_ok = jpeg_decoder::Decoder::new(&simple[..]).decode().is_ok();

        // C mozjpeg with same settings
        let c_jpeg = encode_c(&rgb, width, height, quality);
        let c_ok = jpeg_decoder::Decoder::new(&c_jpeg[..]).decode().is_ok();

        let diff = (simple.len() as f64 / c_jpeg.len() as f64 - 1.0) * 100.0;

        println!(
            "Q{:2}: Rust {:>6} bytes (valid={}) | C {:>6} bytes (valid={}) | diff {:>+6.2}%",
            quality,
            simple.len(),
            simple_ok,
            c_jpeg.len(),
            c_ok,
            diff
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
