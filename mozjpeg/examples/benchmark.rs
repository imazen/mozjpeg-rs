//! Benchmark Rust vs C mozjpeg encoding performance.
//!
//! Usage: cargo run --release --example benchmark

use mozjpeg::test_encoder::{encode_rust, TestEncoderConfig};
use mozjpeg::Subsampling;
use mozjpeg_sys::*;
use png::ColorType;
use std::ptr;
use std::time::{Duration, Instant};

fn main() {
    // Load test image
    let input_path = "mozjpeg/tests/images/1.png";
    let file = std::fs::File::open(input_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let width = info.width;
    let height = info.height;
    let rgb_data: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    println!("Benchmark: {}x{} image ({} pixels)\n", width, height, width * height);

    // Test configurations
    let configs = [
        ("Baseline (no opts)", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: false,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
        }),
        ("Huffman optimized", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
        }),
        ("Trellis AC", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: false,
            overshoot_deringing: false,
        }),
        ("Trellis AC+DC", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: false,
        }),
        ("Progressive", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
        }),
        ("Max compression", TestEncoderConfig {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: false,
        }),
    ];

    println!("{:<20} {:>12} {:>12} {:>10} {:>10} {:>10}",
             "Configuration", "Rust (ms)", "C (ms)", "Ratio", "Rust KB", "C KB");
    println!("{}", "-".repeat(76));

    for (name, config) in &configs {
        // Warmup
        let _ = encode_rust(&rgb_data, width, height, config);
        let _ = encode_c(&rgb_data, width, height, config);

        // Benchmark Rust
        let iterations = 20;
        let rust_start = Instant::now();
        let mut rust_size = 0;
        for _ in 0..iterations {
            let jpeg = encode_rust(&rgb_data, width, height, config);
            rust_size = jpeg.len();
        }
        let rust_time = rust_start.elapsed() / iterations;

        // Benchmark C
        let c_start = Instant::now();
        let mut c_size = 0;
        for _ in 0..iterations {
            let jpeg = encode_c(&rgb_data, width, height, config);
            c_size = jpeg.len();
        }
        let c_time = c_start.elapsed() / iterations;

        let ratio = rust_time.as_secs_f64() / c_time.as_secs_f64();

        println!("{:<20} {:>12.2} {:>12.2} {:>10.2}x {:>9.1} {:>9.1}",
                 name,
                 rust_time.as_secs_f64() * 1000.0,
                 c_time.as_secs_f64() * 1000.0,
                 ratio,
                 rust_size as f64 / 1024.0,
                 c_size as f64 / 1024.0);
    }

    println!("\nNote: Ratio > 1.0 means Rust is slower than C");
}

fn encode_c(data: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
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

        if config.progressive {
            jpeg_simple_progression(&mut cinfo);
        } else {
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        }

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => (1, 1),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT,
            if config.trellis_quant { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC,
            if config.trellis_dc { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING,
            if config.overshoot_deringing { 1 } else { 0 });

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = data.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}
