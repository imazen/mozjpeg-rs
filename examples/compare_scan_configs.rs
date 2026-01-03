//! Compare scan configurations between Rust and C mozjpeg outputs

use mozjpeg_rs::{Encoder, TrellisConfig};
use std::fs;

#[derive(Debug, Clone)]
struct ScanSpec {
    components: Vec<u8>,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    data_size: usize,
}

fn main() {
    let source_path = "/home/lilith/work/mozjpeg-rs/corpus/kodak/10.png";
    let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    println!("=== Compare Scan Configurations ===");
    println!("Image: {}x{}", width, height);
    println!();

    let quality = 85u8;

    // Rust with optimize_scans
    let rust_jpeg = Encoder::baseline_optimized()
        .quality(quality)
        .progressive(true)
        .trellis(TrellisConfig::default())
        .optimize_huffman(true)
        .optimize_scans(true)
        .encode_rgb(rgb_data, width, height)
        .unwrap();

    // C with optimize_scans
    let c_jpeg = encode_c_max_compression(rgb_data, width, height, quality);

    let rust_scans = parse_scans(&rust_jpeg);
    let c_scans = parse_scans(&c_jpeg);

    println!("Rust scans ({}):", rust_scans.len());
    let mut rust_total = 0;
    for (i, scan) in rust_scans.iter().enumerate() {
        let comp_str = format!("{:?}", scan.components);
        println!(
            "  {}: comps={:<12} ss={:2} se={:2} ah={} al={} size={:5} bytes",
            i, comp_str, scan.ss, scan.se, scan.ah, scan.al, scan.data_size
        );
        rust_total += scan.data_size;
    }
    println!("  Total scan data: {} bytes", rust_total);

    println!();
    println!("C scans ({}):", c_scans.len());
    let mut c_total = 0;
    for (i, scan) in c_scans.iter().enumerate() {
        let comp_str = format!("{:?}", scan.components);
        println!(
            "  {}: comps={:<12} ss={:2} se={:2} ah={} al={} size={:5} bytes",
            i, comp_str, scan.ss, scan.se, scan.ah, scan.al, scan.data_size
        );
        c_total += scan.data_size;
    }
    println!("  Total scan data: {} bytes", c_total);

    println!();
    println!("File sizes:");
    println!("  Rust: {} bytes", rust_jpeg.len());
    println!("  C:    {} bytes", c_jpeg.len());
    println!(
        "  Diff: {:+.2}%",
        (rust_jpeg.len() as f64 / c_jpeg.len() as f64 - 1.0) * 100.0
    );
}

fn parse_scans(jpeg: &[u8]) -> Vec<ScanSpec> {
    let mut scans = Vec::new();
    let mut i = 0;

    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            // SOS marker
            let sos_start = i;
            i += 2;
            if i + 2 > jpeg.len() {
                break;
            }
            let length = ((jpeg[i] as usize) << 8) | (jpeg[i + 1] as usize);
            i += 2;

            let ns = jpeg[i];
            i += 1;

            let mut components = Vec::new();
            for _ in 0..ns {
                let cs = jpeg[i];
                components.push(cs);
                i += 2; // Skip Cs and Td/Ta
            }

            let ss = jpeg[i];
            let se = jpeg[i + 1];
            let ah_al = jpeg[i + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0x0F;
            i += 3;

            // Skip to end of SOS header
            let header_end = sos_start + 2 + length;
            i = header_end;

            // Find end of entropy-coded segment (next 0xFF byte not followed by 0x00)
            let data_start = i;
            while i < jpeg.len() - 1 {
                if jpeg[i] == 0xFF {
                    if jpeg[i + 1] == 0x00 {
                        // Stuffed byte, skip
                        i += 2;
                    } else if jpeg[i + 1] == 0xD9 || jpeg[i + 1] == 0xDA {
                        // EOI or next SOS
                        break;
                    } else if jpeg[i + 1] >= 0xD0 && jpeg[i + 1] <= 0xD7 {
                        // RST marker, skip
                        i += 2;
                    } else {
                        // Some other marker, end of scan
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            let data_size = i - data_start;

            scans.push(ScanSpec {
                components,
                ss,
                se,
                ah,
                al,
                data_size,
            });
        } else {
            i += 1;
        }
    }

    scans
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

        // Use JCP_MAX_COMPRESSION profile
        jpeg_c_set_int_param(
            &mut cinfo,
            JINT_COMPRESS_PROFILE,
            JCP_MAX_COMPRESSION as i32,
        );
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

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
