//! Benchmark against codec-corpus images.
//!
//! Compares Rust mozjpeg encoder against C mozjpeg for real-world images.
//!
//! Run `./scripts/fetch-corpus.sh` first to download test images,
//! or set CODEC_CORPUS_DIR to your codec-corpus location.

use mozjpeg_rs::corpus::all_corpus_dirs;
use std::fs;
use std::io::Cursor;
use std::path::Path;

fn main() {
    let corpus_dirs = all_corpus_dirs();

    if corpus_dirs.is_empty() {
        eprintln!("No corpus directories found. Please run:");
        eprintln!("  ./scripts/fetch-corpus.sh");
        eprintln!("Or set CODEC_CORPUS_DIR environment variable.");
        std::process::exit(1);
    }

    let mut total_rust_bytes = 0u64;
    let mut total_c_bytes = 0u64;
    let mut total_images = 0;
    let mut rust_psnr_sum = 0.0f64;
    let mut c_psnr_sum = 0.0f64;

    println!("Benchmarking Rust vs C mozjpeg at Q75\n");
    println!(
        "{:<50} {:>12} {:>12} {:>8} {:>10} {:>10}",
        "Image", "Rust", "C mozjpeg", "Ratio", "Rust PSNR", "C PSNR"
    );
    println!("{}", "-".repeat(102));

    for corpus_dir in &corpus_dirs {
        let dir = match fs::read_dir(&corpus_dir) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping {:?}: not found", corpus_dir);
                continue;
            }
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

        // Limit to first 20 images per directory
        for entry in entries.iter().take(20) {
            let path = entry.path();
            let filename = path.file_name().unwrap().to_string_lossy();

            match process_image(&path) {
                Ok((rust_size, c_size, rust_psnr, c_psnr)) => {
                    let ratio = rust_size as f64 / c_size as f64;
                    println!(
                        "{:<50} {:>12} {:>12} {:>8.3} {:>10.2} {:>10.2}",
                        &filename[..filename.len().min(50)],
                        format_bytes(rust_size),
                        format_bytes(c_size),
                        ratio,
                        rust_psnr,
                        c_psnr
                    );

                    total_rust_bytes += rust_size as u64;
                    total_c_bytes += c_size as u64;
                    rust_psnr_sum += rust_psnr;
                    c_psnr_sum += c_psnr;
                    total_images += 1;
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", filename, e);
                }
            }
        }
    }

    if total_images > 0 {
        println!("{}", "-".repeat(102));
        let avg_ratio = total_rust_bytes as f64 / total_c_bytes as f64;
        let avg_rust_psnr = rust_psnr_sum / total_images as f64;
        let avg_c_psnr = c_psnr_sum / total_images as f64;

        println!(
            "{:<50} {:>12} {:>12} {:>8.3} {:>10.2} {:>10.2}",
            format!("TOTAL ({} images)", total_images),
            format_bytes(total_rust_bytes as usize),
            format_bytes(total_c_bytes as usize),
            avg_ratio,
            avg_rust_psnr,
            avg_c_psnr
        );
        println!("\nAverage file size ratio: {:.3}x", avg_ratio);
        println!(
            "Average PSNR: Rust={:.2} dB, C={:.2} dB",
            avg_rust_psnr, avg_c_psnr
        );
    }
}

fn process_image(path: &Path) -> Result<(usize, usize, f64, f64), Box<dyn std::error::Error>> {
    // Load PNG
    let file = fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let bytes = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;

    // Convert to RGB if needed
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
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for &g in bytes.iter() {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in bytes.chunks(2) {
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
            }
            rgb
        }
        _ => return Err("Unsupported color type".into()),
    };

    // Encode with Rust mozjpeg
    let encoder = mozjpeg_rs::Encoder::new(false).quality(75);
    let rust_jpeg = encoder.encode_rgb(&rgb_data, width, height)?;
    let rust_size = rust_jpeg.len();

    // Encode with C mozjpeg
    let c_jpeg = encode_with_c_mozjpeg(&rgb_data, width, height, 75)?;
    let c_size = c_jpeg.len();

    // Calculate PSNR
    let rust_psnr = calculate_psnr(&rgb_data, &rust_jpeg)?;
    let c_psnr = calculate_psnr(&rgb_data, &c_jpeg)?;

    Ok((rust_size, c_size, rust_psnr, c_psnr))
}

fn encode_with_c_mozjpeg(
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

        // Enable mozjpeg optimizations
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

fn calculate_psnr(original: &[u8], jpeg: &[u8]) -> Result<f64, Box<dyn std::error::Error>> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(jpeg));
    let decoded = decoder.decode()?;

    if original.len() != decoded.len() {
        return Err("Size mismatch".into());
    }

    let mut mse = 0.0f64;
    for (a, b) in original.iter().zip(decoded.iter()) {
        let diff = *a as f64 - *b as f64;
        mse += diff * diff;
    }
    mse /= original.len() as f64;

    if mse == 0.0 {
        Ok(100.0)
    } else {
        Ok(10.0 * (255.0 * 255.0 / mse).log10())
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}
