//! Test 4:4:4 subsampling comparison

use std::fs;

fn main() {
    let corpus_dirs = [
        "/home/lilith/work/codec-comparison/codec-corpus/kodak",
    ];

    let mut total_rust_bytes = 0u64;
    let mut total_c_bytes = 0u64;
    let mut total_images = 0;

    println!("Comparing Rust vs C mozjpeg at Q75 with 4:4:4 subsampling\n");

    for corpus_dir in &corpus_dirs {
        let dir = match fs::read_dir(corpus_dir) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut entries: Vec<_> = dir
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "png")
                    .unwrap_or(false)
            })
            .collect();

        entries.sort_by_key(|e| e.path());

        for entry in entries.iter().take(5) {
            let path = entry.path();
            let filename = path.file_name().unwrap().to_string_lossy();

            match process_image(&path) {
                Ok((rust_size, c_size)) => {
                    let ratio = rust_size as f64 / c_size as f64;
                    println!("{:<30} Rust: {:>8} C: {:>8} Ratio: {:.3}",
                        &filename[..filename.len().min(30)],
                        rust_size, c_size, ratio);
                    total_rust_bytes += rust_size as u64;
                    total_c_bytes += c_size as u64;
                    total_images += 1;
                }
                Err(e) => eprintln!("Error: {}: {}", filename, e),
            }
        }
    }

    if total_images > 0 {
        let avg_ratio = total_rust_bytes as f64 / total_c_bytes as f64;
        println!("\nAverage ratio: {:.4}x ({} images)", avg_ratio, total_images);
    }
}

fn process_image(path: &std::path::Path) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in bytes.chunks(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
            rgb
        }
        _ => return Err("Unsupported color type".into()),
    };

    // Rust encoder with 4:4:4
    let encoder = mozjpeg::Encoder::new()
        .quality(75)
        .subsampling(mozjpeg::Subsampling::S444);
    let rust_jpeg = encoder.encode_rgb(&rgb_data, width, height)?;

    // C encoder with 4:4:4
    let c_jpeg = encode_with_c_444(&rgb_data, width, height, 75)?;

    Ok((rust_jpeg.len(), c_jpeg.len()))
}

fn encode_with_c_444(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: i32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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
        let mut outsize: u64 = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;

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

        Ok(result)
    }
}
