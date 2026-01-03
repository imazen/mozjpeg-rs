//! Compare Rust vs C mozjpeg encoder with real images.
//!
//! Run with: cargo run --example compare_real_images
//!
//! Requires corpus images. Run `./scripts/fetch-corpus.sh --full` first,
//! or set CODEC_CORPUS_DIR to your codec-corpus location.

use mozjpeg_rs::corpus::{clic_validation_dir, png_files_in_dir};
use mozjpeg_rs::{Encoder, Subsampling};
use std::fs;
use std::path::PathBuf;

fn main() {
    let source_dir = match clic_validation_dir() {
        Some(dir) => dir,
        None => {
            eprintln!("CLIC corpus not found. Please run:");
            eprintln!("  ./scripts/fetch-corpus.sh --full");
            eprintln!("Or set CODEC_CORPUS_DIR environment variable.");
            std::process::exit(1);
        }
    };

    // Output to comparison_outputs in project root
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("comparison_outputs");
    fs::create_dir_all(&output_dir).unwrap();

    // Specific images chosen for testing (different sizes/content)
    let preferred_images = [
        "4cd6910a0b7b39365fda5df87618d091.png", // Smaller one (398KB)
        "097cb426910ba8ce2525dd8bb7fb1777.png", // Medium (2.5MB)
    ];

    // Use preferred images if they exist, otherwise fall back to first 2 in directory
    let images: Vec<_> = {
        let preferred: Vec<_> = preferred_images
            .iter()
            .map(|name| source_dir.join(name))
            .filter(|p| p.exists())
            .collect();

        if !preferred.is_empty() {
            preferred
        } else {
            // Fall back to first 2 PNG files
            png_files_in_dir(&source_dir).into_iter().take(2).collect()
        }
    };

    if images.is_empty() {
        eprintln!("No PNG files found in {:?}", source_dir);
        std::process::exit(1);
    }

    let quality = 75;

    for source_path in &images {
        let image_name = source_path.file_name().unwrap().to_string_lossy();
        let stem = source_path.file_stem().unwrap().to_string_lossy();

        println!("Processing: {}", image_name);

        // Load PNG
        let decoder = png::Decoder::new(fs::File::open(source_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let rgb_data = &buf[..info.buffer_size()];

        let width = info.width;
        let height = info.height;
        let bytes_per_pixel = info.color_type.samples();

        println!(
            "  Size: {}x{}, {} bytes/pixel",
            width, height, bytes_per_pixel
        );

        // Handle different color types
        let rgb_data: Vec<u8> = if bytes_per_pixel == 4 {
            // RGBA -> RGB
            rgb_data
                .chunks(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect()
        } else if bytes_per_pixel == 3 {
            rgb_data.to_vec()
        } else {
            println!("  Skipping: unsupported color type");
            continue;
        };

        // Encode with Rust
        let rust_encoder = Encoder::baseline_optimized()
            .quality(quality)
            .subsampling(Subsampling::S420);
        let rust_jpeg = rust_encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with C mozjpeg
        let c_jpeg = unsafe { encode_with_c_mozjpeg(&rgb_data, width, height, quality) };

        // Save outputs
        let rust_path = output_dir.join(format!("{}_rust_q{}.jpg", stem, quality));
        let c_path = output_dir.join(format!("{}_cmozjpeg_q{}.jpg", stem, quality));

        fs::write(&rust_path, &rust_jpeg).unwrap();
        fs::write(&c_path, &c_jpeg).unwrap();

        // Also copy original PNG for comparison
        let orig_path = output_dir.join(&*image_name);
        fs::copy(source_path, &orig_path).unwrap();

        println!(
            "  Rust:     {:>8} bytes -> {:?}",
            rust_jpeg.len(),
            rust_path
        );
        println!("  C mozjpeg:{:>8} bytes -> {:?}", c_jpeg.len(), c_path);
        println!("  Original copied to: {:?}", orig_path);
        println!();
    }

    println!("Done! Outputs in: {:?}", output_dir);
}

unsafe fn encode_with_c_mozjpeg(rgb_data: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::ptr;

    let mut outbuffer: *mut u8 = ptr::null_mut();
    let mut outsize: std::ffi::c_ulong = 0;

    let mut cinfo = std::mem::zeroed::<mozjpeg_sys::jpeg_compress_struct>();
    let mut jerr = std::mem::zeroed::<mozjpeg_sys::jpeg_error_mgr>();

    cinfo.common.err = mozjpeg_sys::jpeg_std_error(&mut jerr);
    mozjpeg_sys::jpeg_CreateCompress(
        &mut cinfo,
        mozjpeg_sys::JPEG_LIB_VERSION as i32,
        std::mem::size_of::<mozjpeg_sys::jpeg_compress_struct>(),
    );

    mozjpeg_sys::jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = mozjpeg_sys::J_COLOR_SPACE::JCS_RGB;

    mozjpeg_sys::jpeg_set_defaults(&mut cinfo);
    mozjpeg_sys::jpeg_set_quality(&mut cinfo, quality as i32, 1);

    mozjpeg_sys::jpeg_start_compress(&mut cinfo, 1);

    let row_stride = (width * 3) as usize;
    while cinfo.next_scanline < cinfo.image_height {
        let row_ptr = rgb_data
            .as_ptr()
            .add(cinfo.next_scanline as usize * row_stride);
        let mut row_array = [row_ptr as *const u8];
        mozjpeg_sys::jpeg_write_scanlines(&mut cinfo, row_array.as_mut_ptr() as *mut *const u8, 1);
    }

    mozjpeg_sys::jpeg_finish_compress(&mut cinfo);
    mozjpeg_sys::jpeg_destroy_compress(&mut cinfo);

    let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
    libc::free(outbuffer as *mut std::ffi::c_void);

    result
}
