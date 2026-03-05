//! Profile C mozjpeg vs Rust mozjpeg-rs for comparison.
//!
//! Run with:
//!   cargo build --release --example profile_compare
//!   perf record -g ./target/release/examples/profile_compare rust
//!   perf record -g ./target/release/examples/profile_compare c
//!   perf report

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::path::Path;

use png::ColorType;

fn load_rgb_image(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = File::open(path).expect("Failed to open test image");
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let width = info.width;
    let height = info.height;

    let rgb: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..width as usize * height as usize * 3].to_vec(),
        ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect(),
        ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    (rgb, width, height)
}

fn run_rust(rgb: &[u8], width: u32, height: u32, iterations: usize) {
    use mozjpeg_rs::{Encoder, Preset};
    use std::env;

    let use_trellis = env::var("TRELLIS").is_ok();

    for _ in 0..iterations {
        let encoder = if use_trellis {
            // With trellis - Rust is 1.4x faster than C
            Encoder::new(Preset::ProgressiveBalanced).quality(85)
        } else {
            // Without trellis - entropy encoding becomes the bottleneck
            Encoder::new(Preset::BaselineFastest).quality(85)
        };
        let result = encoder.encode_rgb(rgb, width, height).unwrap();
        black_box(result);
    }
}

fn run_c_mozjpeg(rgb: &[u8], width: u32, height: u32, iterations: usize) {
    use mozjpeg_sys::*;
    use std::env;
    use std::ptr;

    let use_trellis = env::var("TRELLIS").is_ok();

    for _ in 0..iterations {
        unsafe {
            let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
            let mut jerr: jpeg_error_mgr = std::mem::zeroed();

            cinfo.common.err = jpeg_std_error(&mut jerr);
            jpeg_create_compress(&mut cinfo);

            // Memory destination
            let mut outbuffer: *mut u8 = ptr::null_mut();
            let mut outsize: libc::c_ulong = 0;
            jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = 3;
            cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

            jpeg_set_defaults(&mut cinfo);
            jpeg_set_quality(&mut cinfo, 85, true as boolean);

            if !use_trellis {
                // Disable progressive and ALL optimizations for baseline comparison
                cinfo.scan_info = ptr::null();
                cinfo.num_scans = 0;
                cinfo.optimize_coding = false as boolean;

                // Disable trellis quantization
                jpeg_c_set_bool_param(
                    &mut cinfo,
                    J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT,
                    false as boolean,
                );
                jpeg_c_set_bool_param(
                    &mut cinfo,
                    J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT_DC,
                    false as boolean,
                );
                jpeg_c_set_bool_param(
                    &mut cinfo,
                    J_BOOLEAN_PARAM::JBOOLEAN_OVERSHOOT_DERINGING,
                    false as boolean,
                );
            }

            jpeg_start_compress(&mut cinfo, true as boolean);

            let row_stride = width as usize * 3;
            while cinfo.next_scanline < cinfo.image_height {
                let row_ptr =
                    rgb.as_ptr().add(cinfo.next_scanline as usize * row_stride) as *const u8;
                let row_array: [*const u8; 1] = [row_ptr];
                jpeg_write_scanlines(&mut cinfo, row_array.as_ptr(), 1);
            }

            jpeg_finish_compress(&mut cinfo);

            // Copy output and free
            let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
            libc::free(outbuffer as *mut libc::c_void);

            jpeg_destroy_compress(&mut cinfo);

            black_box(result);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("rust");

    // Load real image
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_image = Path::new(manifest_dir).join("tests/images/1.png");
    let (rgb, width, height) = load_rgb_image(&test_image);

    println!("Image: {}x{}", width, height);
    println!("Mode: {}", mode);
    println!("Running 500 iterations for profiling...");

    let iterations = 500;

    match mode {
        "rust" => run_rust(&rgb, width, height, iterations),
        "c" => run_c_mozjpeg(&rgb, width, height, iterations),
        _ => {
            eprintln!("Usage: profile_compare [rust|c]");
            std::process::exit(1);
        }
    }

    println!("Done.");
}
