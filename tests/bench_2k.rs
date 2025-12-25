//! 2048x2048 benchmark test for accurate Rust vs C comparison

use mozjpeg_oxide::test_encoder::{encode_rust, TestEncoderConfig};
use mozjpeg_oxide::Subsampling;
use mozjpeg_sys::*;
use std::ptr;
use std::time::Instant;

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let noise = ((x * 7 + y * 13) % 50) as u8;
            rgb[idx] = ((x * 255 / width) as u8).saturating_add(noise);
            rgb[idx + 1] = ((y * 255 / height) as u8).saturating_add(noise);
            rgb[idx + 2] = (((x + y) * 255 / (width + height)) as u8).saturating_add(noise);
        }
    }
    rgb
}

fn encode_c(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();
        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(&mut cinfo, JPEG_LIB_VERSION as i32, std::mem::size_of::<jpeg_compress_struct>());
        
        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);
        
        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_set_defaults(&mut cinfo);
        
        cinfo.num_scans = 0;
        cinfo.scan_info = ptr::null();
        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);
        
        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            _ => (2, 2),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;
        
        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, if config.trellis_quant { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, if config.trellis_dc { 1 } else { 0 });
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, if config.overshoot_deringing { 1 } else { 0 });
        
        jpeg_start_compress(&mut cinfo, 1);
        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = rgb.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }
        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);
        
        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}

#[test]
fn bench_2048x2048_comparison() {
    let width = 2048u32;
    let height = 2048u32;
    let iterations = 30u32;
    
    println!("\n=== 2048x2048 Benchmark ({} iterations) ===\n", iterations);
    
    let rgb = create_test_image(width as usize, height as usize);
    println!("Image: {}x{} = {} megapixels\n", width, height, (width * height) as f64 / 1_000_000.0);
    
    // Warmup
    let baseline = TestEncoderConfig::baseline();
    for _ in 0..3 {
        let _ = encode_rust(&rgb, width, height, &baseline);
        let _ = encode_c(&rgb, width, height, &baseline);
    }
    
    // Baseline benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_rust(&rgb, width, height, &baseline);
    }
    let rust_baseline = start.elapsed() / iterations;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_c(&rgb, width, height, &baseline);
    }
    let c_baseline = start.elapsed() / iterations;
    
    // Trellis benchmark
    let trellis = TestEncoderConfig {
        optimize_huffman: true,
        trellis_quant: true,
        trellis_dc: true,
        ..TestEncoderConfig::default()
    };
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_rust(&rgb, width, height, &trellis);
    }
    let rust_trellis = start.elapsed() / iterations;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_c(&rgb, width, height, &trellis);
    }
    let c_trellis = start.elapsed() / iterations;
    
    println!("| Config   | Rust     | C        | Ratio |");
    println!("|----------|----------|----------|-------|");
    println!("| Baseline | {:>6.2} ms | {:>6.2} ms | {:.2}x  |", 
        rust_baseline.as_secs_f64() * 1000.0,
        c_baseline.as_secs_f64() * 1000.0,
        rust_baseline.as_secs_f64() / c_baseline.as_secs_f64());
    println!("| Trellis  | {:>6.2} ms | {:>6.2} ms | {:.2}x  |", 
        rust_trellis.as_secs_f64() * 1000.0,
        c_trellis.as_secs_f64() * 1000.0,
        rust_trellis.as_secs_f64() / c_trellis.as_secs_f64());
}
