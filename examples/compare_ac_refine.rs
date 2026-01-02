//! Compare AC refinement encoding between Rust and C mozjpeg
//!
//! This encodes a single block through AC refinement and compares the output.

use mozjpeg_rs::{Encoder, Subsampling};
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

/// Encode with C mozjpeg using a 10-scan SA script (optimize_scans=false)
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

        // Disable optimize_scans to get 10-scan SA script
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

        // Enable Huffman optimization
        cinfo.optimize_coding = 1;

        // Disable trellis to simplify comparison
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 0);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 0);

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

/// Count SOS markers and extract scan info
fn analyze_scans(jpeg: &[u8]) -> Vec<(u8, u8, u8, u8, u8)> {
    let mut scans = Vec::new();
    let mut i = 0;
    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            // SOS marker
            if i + 5 < jpeg.len() {
                let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                if i + 2 + len <= jpeg.len() {
                    let num_components = jpeg[i + 4];
                    let ss_offset = i + 4 + 1 + (num_components as usize * 2);
                    if ss_offset + 2 < jpeg.len() {
                        let ss = jpeg[ss_offset];
                        let se = jpeg[ss_offset + 1];
                        let ahl = jpeg[ss_offset + 2];
                        let ah = ahl >> 4;
                        let al = ahl & 0x0F;
                        scans.push((num_components, ss, se, ah, al));
                    }
                }
            }
        }
        i += 1;
    }
    scans
}

fn main() {
    let path = Path::new("corpus/kodak/1.png");
    let (rgb, w, h) = match load_png(path) {
        Some(d) => d,
        None => {
            println!("Failed to load {}", path.display());
            return;
        }
    };

    println!("Image: {} ({}x{})", path.display(), w, h);

    // Encode with C mozjpeg progressive (10-scan SA script)
    let c_jpeg = encode_c_progressive(&rgb, w, h, 85);
    println!("\nC mozjpeg progressive: {} bytes", c_jpeg.len());

    let c_scans = analyze_scans(&c_jpeg);
    println!("C scans ({}):", c_scans.len());
    for (i, (nc, ss, se, ah, al)) in c_scans.iter().enumerate() {
        let scan_type = if *ss == 0 && *se == 0 {
            if *ah > 0 {
                "DC refine"
            } else {
                "DC first"
            }
        } else if *ah > 0 {
            "AC refine"
        } else {
            "AC first"
        };
        println!(
            "  {:2}: {} comps={} Ss={:2} Se={:2} Ah={} Al={}",
            i, scan_type, nc, ss, se, ah, al
        );
    }

    // Try to decode C output
    match jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg)).decode() {
        Ok(_) => println!("C JPEG decodes OK"),
        Err(e) => println!("C JPEG decode FAILED: {}", e),
    }

    // Now try Rust with equivalent settings (but we can't do SA currently)
    let rust_jpeg = Encoder::new(false)
        .quality(85)
        .progressive(true)
        .optimize_huffman(true)
        .trellis(mozjpeg_rs::TrellisConfig::disabled())
        .optimize_scans(false)
        .subsampling(Subsampling::S420)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    println!("\nRust progressive: {} bytes", rust_jpeg.len());

    let rust_scans = analyze_scans(&rust_jpeg);
    println!("Rust scans ({}):", rust_scans.len());
    for (i, (nc, ss, se, ah, al)) in rust_scans.iter().enumerate() {
        let scan_type = if *ss == 0 && *se == 0 {
            if *ah > 0 {
                "DC refine"
            } else {
                "DC first"
            }
        } else if *ah > 0 {
            "AC refine"
        } else {
            "AC first"
        };
        println!(
            "  {:2}: {} comps={} Ss={:2} Se={:2} Ah={} Al={}",
            i, scan_type, nc, ss, se, ah, al
        );
    }

    match jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg)).decode() {
        Ok(_) => println!("Rust JPEG decodes OK"),
        Err(e) => println!("Rust JPEG decode FAILED: {}", e),
    }
}
