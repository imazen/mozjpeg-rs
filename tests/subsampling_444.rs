//! Test 4:4:4 subsampling comparison between Rust and C mozjpeg.
//!
//! This test compares file sizes at Q75 with 4:4:4 subsampling.
//! Requires the Kodak corpus - run `./scripts/fetch-corpus.sh` first.

use mozjpeg_oxide::corpus::{kodak_dir, png_files_in_dir};
use mozjpeg_oxide::Subsampling;
use std::fs::File;
use std::path::Path;

/// Test 4:4:4 subsampling comparison across Kodak corpus.
///
/// Requires external corpus. Run `./scripts/fetch-corpus.sh` first.
#[test]
#[ignore = "requires Kodak corpus - run ./scripts/fetch-corpus.sh"]
fn test_444_subsampling_corpus() {
    let corpus_dir = kodak_dir().expect("Kodak corpus not found");
    let files = png_files_in_dir(&corpus_dir);

    assert!(!files.is_empty(), "No PNG files found in Kodak corpus");

    let mut total_rust_bytes = 0u64;
    let mut total_c_bytes = 0u64;
    let mut total_images = 0;
    let mut failures = Vec::new();

    println!("\nComparing Rust vs C mozjpeg at Q75 with 4:4:4 subsampling\n");

    for path in files.iter().take(5) {
        let filename = path.file_name().unwrap().to_string_lossy();

        let (rgb_data, width, height) = match load_png(path) {
            Some(data) => data,
            None => {
                failures.push(format!("{}: failed to load", filename));
                continue;
            }
        };

        // Rust encoder with 4:4:4
        let rust_jpeg = mozjpeg_oxide::Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S444)
            .encode_rgb(&rgb_data, width, height)
            .expect("Rust encoding failed");

        // C encoder with 4:4:4
        let c_jpeg = encode_with_c_444(&rgb_data, width, height, 75);

        let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;
        println!(
            "{:<30} Rust: {:>8} C: {:>8} Ratio: {:.3}",
            &filename[..filename.len().min(30)],
            rust_jpeg.len(),
            c_jpeg.len(),
            ratio
        );

        // Verify ratio is reasonable (within 15%)
        if ratio > 1.15 || ratio < 0.85 {
            failures.push(format!("{}: ratio {:.3} out of range", filename, ratio));
        }

        total_rust_bytes += rust_jpeg.len() as u64;
        total_c_bytes += c_jpeg.len() as u64;
        total_images += 1;
    }

    if total_images > 0 {
        let avg_ratio = total_rust_bytes as f64 / total_c_bytes as f64;
        println!("\nAverage ratio: {:.4}x ({} images)", avg_ratio, total_images);

        // Overall ratio should be close to 1.0
        assert!(
            avg_ratio < 1.10 && avg_ratio > 0.90,
            "Average ratio {:.4} out of acceptable range (0.90-1.10)",
            avg_ratio
        );
    }

    if !failures.is_empty() {
        panic!("Test failures:\n  {}", failures.join("\n  "));
    }
}

fn encode_with_c_444(rgb: &[u8], width: u32, height: u32, quality: i32) -> Vec<u8> {
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
        jpeg_set_quality(&mut cinfo, quality, 1);

        // Set 4:4:4 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = (width * 3) as usize;
        let mut row_pointer: [*const u8; 1] = [ptr::null()];

        while cinfo.next_scanline < cinfo.image_height {
            let offset = cinfo.next_scanline as usize * row_stride;
            row_pointer[0] = rgb.as_ptr().add(offset);
            jpeg_write_scanlines(&mut cinfo, row_pointer.as_ptr(), 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);

        result
    }
}

/// Load a PNG image and return RGB data.
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };

    Some((rgb_data, width, height))
}
