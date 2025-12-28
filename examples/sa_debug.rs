use mozjpeg_oxide::progressive::generate_mozjpeg_max_compression_scans;
use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
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
    let (rgb, width, height) = load_png(Path::new("corpus/kodak/10.png")).unwrap();
    let quality = 85;

    // Print the scan script
    println!("=== mozjpeg_max_compression scan script ===");
    let scans = generate_mozjpeg_max_compression_scans(3);
    for (i, scan) in scans.iter().enumerate() {
        println!(
            "Scan {}: comps={} Ss={} Se={} Ah={} Al={}",
            i, scan.comps_in_scan, scan.ss, scan.se, scan.ah, scan.al
        );
    }
    println!();

    // Encode with simple progressive (our current default)
    let simple = Encoder::new()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(false)
        .subsampling(Subsampling::S420)
        .encode_rgb(&rgb, width, height)
        .unwrap();

    // Count scans
    let simple_scans = count_scans(&simple);

    // Decode and verify
    let simple_ok = jpeg_decoder::Decoder::new(&simple[..]).decode().is_ok();

    println!(
        "Simple progressive: {} bytes, {} scans, valid={}",
        simple.len(),
        simple_scans,
        simple_ok
    );

    // Now let's see what C produces
    let c_jpeg = encode_c(rgb.as_slice(), width, height, quality);
    let c_scans = count_scans(&c_jpeg);
    let c_ok = jpeg_decoder::Decoder::new(&c_jpeg[..]).decode().is_ok();

    println!(
        "C mozjpeg:          {} bytes, {} scans, valid={}",
        c_jpeg.len(),
        c_scans,
        c_ok
    );

    // Print C's scan structure
    println!("\n=== C mozjpeg scan structure ===");
    print_scan_structure(&c_jpeg);
}

fn count_scans(jpeg: &[u8]) -> usize {
    jpeg.windows(2).filter(|w| w == &[0xFF, 0xDA]).count()
}

fn print_scan_structure(jpeg: &[u8]) {
    for i in 0..jpeg.len().saturating_sub(10) {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            // SOS marker
            let ns = jpeg[i + 4];
            let ss_se_pos = i + 4 + 1 + (ns as usize * 2);
            if ss_se_pos + 2 < jpeg.len() {
                let ss = jpeg[ss_se_pos];
                let se = jpeg[ss_se_pos + 1];
                let ah_al = jpeg[ss_se_pos + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;
                let comps: Vec<u8> = (0..ns).map(|j| jpeg[i + 5 + (j as usize * 2)]).collect();
                println!(
                    "  Ns={} comps={:?} Ss={:2} Se={:2} Ah={} Al={}",
                    ns, comps, ss, se, ah, al
                );
            }
        }
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

        // Use ImageMagick quant tables (index 3)
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Disable optimize_scans to use simple progressive
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 0);

        // Enable progressive mode
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
