//! Generate side-by-side visual comparison samples.
//!
//! Creates JPEGs at various quality levels for visual sanity checking.
//!
//! Usage:
//!   cargo run --release --example visual_compare

use mozjpeg_rs::Encoder;
use mozjpeg_sys::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::ptr;

fn main() {
    let output_dir = Path::new("comparison_outputs");
    fs::create_dir_all(output_dir).unwrap();

    // Use Kodak image 1 if available, otherwise use tests/images/1.png
    let image_path = if Path::new("corpus/kodak/1.png").exists() {
        "corpus/kodak/1.png"
    } else if Path::new("tests/images/1.png").exists() {
        "tests/images/1.png"
    } else {
        eprintln!("No test image found. Run ./scripts/fetch-corpus.sh first.");
        std::process::exit(1);
    };

    println!("Loading: {}", image_path);
    let (rgb, width, height) = load_png(Path::new(image_path)).expect("Failed to load image");

    println!("Image size: {}x{}", width, height);
    println!("Output dir: {:?}", output_dir);
    println!();

    let qualities = [50, 75, 85, 95];

    println!(
        "{:>5} {:>12} {:>12} {:>10}",
        "Q", "Rust Size", "C Size", "Ratio"
    );
    println!("{}", "-".repeat(45));

    for q in qualities {
        // Encode with Rust
        let rust_data = Encoder::new(false)
            .quality(q)
            .progressive(true)
            .optimize_huffman(true)
            .overshoot_deringing(true)
            .trellis(mozjpeg_rs::TrellisConfig::default())
            .encode_rgb(&rgb, width, height)
            .expect("Rust encode failed");

        // Encode with C
        let c_data = encode_c(&rgb, width, height, q);

        let ratio = rust_data.len() as f64 / c_data.len() as f64;
        println!(
            "{:>5} {:>12} {:>12} {:>10.3}",
            q,
            format!("{} bytes", rust_data.len()),
            format!("{} bytes", c_data.len()),
            ratio
        );

        // Save both
        let rust_path = output_dir.join(format!("kodak01_q{}_rust.jpg", q));
        let c_path = output_dir.join(format!("kodak01_q{}_c.jpg", q));

        File::create(&rust_path)
            .unwrap()
            .write_all(&rust_data)
            .unwrap();
        File::create(&c_path).unwrap().write_all(&c_data).unwrap();
    }

    println!();
    println!("Output files saved to: {:?}", output_dir);
    println!();
    println!("Compare visually:");
    for q in qualities {
        println!(
            "  feh {:?}/kodak01_q{}_rust.jpg {:?}/kodak01_q{}_c.jpg",
            output_dir, q, output_dir, q
        );
    }
}

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
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

#[allow(unsafe_code)]
fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
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
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
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
