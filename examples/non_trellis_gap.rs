//! Investigate the non-trellis path gap (Rust 1.7-4.5% larger than C)

use mozjpeg_rs::{Encoder, TrellisConfig};
use std::collections::HashMap;
use std::fs;

fn main() {
    let source_path = "/home/lilith/work/mozjpeg-rs/corpus/kodak/10.png";
    let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb_data = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    println!("=== Non-Trellis Gap Investigation ===");
    println!("Image: {}x{}", width, height);
    println!();

    let quality = 85u8;

    // Encode without trellis
    let rust_jpeg = Encoder::new(false)
        .quality(quality)
        .progressive(false)
        .optimize_huffman(true)
        .trellis(TrellisConfig::disabled())
        .encode_rgb(rgb_data, width, height)
        .unwrap();

    let c_jpeg = encode_c_baseline(rgb_data, width, height, quality, false);

    println!("File sizes:");
    println!("  Rust: {} bytes", rust_jpeg.len());
    println!("  C:    {} bytes", c_jpeg.len());
    println!(
        "  Diff: {:+.2}%",
        (rust_jpeg.len() as f64 / c_jpeg.len() as f64 - 1.0) * 100.0
    );
    println!();

    // Compare JPEG structure
    println!("=== JPEG Structure ===");
    let rust_parts = analyze_jpeg(&rust_jpeg);
    let c_parts = analyze_jpeg(&c_jpeg);

    println!("{:20} {:>10} {:>10} {:>10}", "Segment", "Rust", "C", "Diff");
    for (name, rust_size) in &rust_parts {
        let c_size = c_parts.get(name).unwrap_or(&0);
        let diff = *rust_size as i64 - *c_size as i64;
        println!("{:20} {:>10} {:>10} {:>+10}", name, rust_size, c_size, diff);
    }

    // Decode and compare coefficients
    println!();
    println!("=== Decoded Pixel Comparison ===");
    let rust_decoded = jpeg_decode(&rust_jpeg);
    let c_decoded = jpeg_decode(&c_jpeg);

    let mut diff_histogram: HashMap<i32, usize> = HashMap::new();
    let mut max_diff = 0i32;

    for i in 0..rust_decoded.len() {
        let diff = (rust_decoded[i] as i32 - c_decoded[i] as i32).abs();
        max_diff = max_diff.max(diff);
        *diff_histogram.entry(diff).or_insert(0) += 1;
    }

    println!("  Max pixel diff: {}", max_diff);
    println!(
        "  Identical: {} ({:.1}%)",
        diff_histogram.get(&0).unwrap_or(&0),
        100.0 * *diff_histogram.get(&0).unwrap_or(&0) as f64 / rust_decoded.len() as f64
    );

    // Compare quantization tables
    println!();
    println!("=== Quantization Tables ===");
    compare_dqt(&rust_jpeg, &c_jpeg);

    // Compare Huffman tables
    println!();
    println!("=== Huffman Tables ===");
    compare_dht(&rust_jpeg, &c_jpeg);
}

fn analyze_jpeg(jpeg: &[u8]) -> HashMap<String, usize> {
    let mut parts = HashMap::new();
    let mut i = 0;
    let mut entropy_start = 0;

    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF {
            match jpeg[i + 1] {
                0xD8 => {
                    parts.insert("SOI".to_string(), 2);
                    i += 2;
                }
                0xDB => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    *parts.entry("DQT".to_string()).or_insert(0) += len + 2;
                    i += len + 2;
                }
                0xC0 | 0xC2 => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    parts.insert("SOF".to_string(), len + 2);
                    i += len + 2;
                }
                0xC4 => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    *parts.entry("DHT".to_string()).or_insert(0) += len + 2;
                    i += len + 2;
                }
                0xDA => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    parts.insert("SOS header".to_string(), len + 2);
                    entropy_start = i + len + 2;
                    // Skip to end of entropy data
                    let mut j = entropy_start;
                    while j < jpeg.len() - 1 {
                        if jpeg[j] == 0xFF
                            && jpeg[j + 1] != 0x00
                            && jpeg[j + 1] != 0xFF
                            && !(0xD0..=0xD7).contains(&jpeg[j + 1])
                        {
                            break;
                        }
                        j += 1;
                    }
                    parts.insert("Entropy data".to_string(), j - entropy_start);
                    i = j;
                }
                0xD9 => {
                    parts.insert("EOI".to_string(), 2);
                    i += 2;
                }
                _ => i += 1,
            }
        } else {
            i += 1;
        }
    }
    parts
}

fn compare_dqt(rust_jpeg: &[u8], c_jpeg: &[u8]) {
    let rust_tables = extract_quant_tables(rust_jpeg);
    let c_tables = extract_quant_tables(c_jpeg);

    for (idx, (rust_tbl, c_tbl)) in rust_tables.iter().zip(c_tables.iter()).enumerate() {
        let mut diffs = 0;
        for i in 0..rust_tbl.len().min(c_tbl.len()) {
            if rust_tbl[i] != c_tbl[i] {
                diffs += 1;
            }
        }
        if diffs > 0 {
            println!("  Table {}: {} differences", idx, diffs);
        } else {
            println!("  Table {}: IDENTICAL", idx);
        }
    }
}

fn extract_quant_tables(jpeg: &[u8]) -> Vec<Vec<u8>> {
    let mut tables = Vec::new();
    let mut i = 0;

    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDB {
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let segment = &jpeg[i + 4..i + 2 + len];
            tables.push(segment.to_vec());
            i += 2 + len;
        } else {
            i += 1;
        }
    }
    tables
}

fn compare_dht(rust_jpeg: &[u8], c_jpeg: &[u8]) {
    let rust_tables = extract_huffman_tables(rust_jpeg);
    let c_tables = extract_huffman_tables(c_jpeg);

    println!(
        "  Rust: {} DHT segments, total {} bytes",
        rust_tables.len(),
        rust_tables.iter().map(|t| t.len()).sum::<usize>()
    );
    println!(
        "  C:    {} DHT segments, total {} bytes",
        c_tables.len(),
        c_tables.iter().map(|t| t.len()).sum::<usize>()
    );

    // Compare individual tables
    if rust_tables.len() == c_tables.len() {
        for (idx, (rust_tbl, c_tbl)) in rust_tables.iter().zip(c_tables.iter()).enumerate() {
            if rust_tbl == c_tbl {
                println!("  Table {}: IDENTICAL ({} bytes)", idx, rust_tbl.len());
            } else {
                println!(
                    "  Table {}: DIFFERENT (Rust {} vs C {} bytes)",
                    idx,
                    rust_tbl.len(),
                    c_tbl.len()
                );
            }
        }
    }
}

fn extract_huffman_tables(jpeg: &[u8]) -> Vec<Vec<u8>> {
    let mut tables = Vec::new();
    let mut i = 0;

    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xC4 {
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let segment = &jpeg[i + 4..i + 2 + len];
            tables.push(segment.to_vec());
            i += 2 + len;
        } else {
            i += 1;
        }
    }
    tables
}

fn jpeg_decode(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    decoder.decode().unwrap()
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
