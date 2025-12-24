//! Debug edge padding order difference between Rust and C.
//!
//! C mozjpeg pads BEFORE downsampling.
//! Our code downsamples first, then pads.

use png::ColorType;
use std::fs;

fn main() {
    // Load actual test image and crop to non-aligned size
    let input_path = "mozjpeg/tests/images/1.png";
    let file = fs::File::open(input_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let full_width = info.width as usize;
    let full_rgb: Vec<u8> = match info.color_type {
        ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported"),
    };

    // Crop to 49x51 (non-aligned)
    let width = 49;
    let height = 51;
    let mut rgb_data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let start = y * full_width * 3;
        let end = start + width * 3;
        rgb_data.extend_from_slice(&full_rgb[start..end]);
    }

    // Convert to Y plane (simple luma extraction for testing)
    let mut image: Vec<u8> = vec![0; width * height];
    for i in 0..width * height {
        let r = rgb_data[i * 3] as u16;
        let g = rgb_data[i * 3 + 1] as u16;
        let b = rgb_data[i * 3 + 2] as u16;
        image[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
    }

    println!("Loaded {}x{} cropped from {}", width, height, input_path);
    println!("Corner Y values: TL={}, TR={}, BL={}, BR={}",
             image[0], image[width - 1],
             image[(height - 1) * width], image[(height - 1) * width + width - 1]);

    // Method 1: Our approach - downsample then pad
    let ds_w = (width + 1) / 2;  // 25
    let ds_h = (height + 1) / 2; // 26
    let mut our_ds = vec![0u8; ds_w * ds_h];
    our_downsample_h2v2(&image, width, height, &mut our_ds, ds_w, ds_h);

    // Pad to MCU-aligned
    let mcu_w = 32;
    let mcu_h = 32;
    let mut our_result = vec![0u8; mcu_w * mcu_h];
    pad_plane(&our_ds, ds_w, ds_h, &mut our_result, mcu_w, mcu_h);

    // Method 2: C approach - pad then downsample
    let pad_w = 64;
    let pad_h = 64;
    let mut padded = vec![0u8; pad_w * pad_h];
    pad_plane(&image, width, height, &mut padded, pad_w, pad_h);

    let mut c_result = vec![0u8; mcu_w * mcu_h];
    c_downsample_h2v2(&padded, pad_w, pad_h, &mut c_result, mcu_w, mcu_h);

    // Compare
    println!("\nComparing {}x{} results:", mcu_w, mcu_h);
    let mut diff_count = 0;
    let mut max_diff = 0i16;
    for y in 0..mcu_h {
        for x in 0..mcu_w {
            let i = y * mcu_w + x;
            let d = (our_result[i] as i16 - c_result[i] as i16).abs();
            if d > 0 {
                diff_count += 1;
                max_diff = max_diff.max(d);
                if diff_count <= 10 {
                    println!("  [{},{}]: ours={}, C={}, diff={}",
                             x, y, our_result[i], c_result[i], d);
                }
            }
        }
    }
    println!("Total differences: {}, max diff: {}", diff_count, max_diff);
}

fn our_downsample_h2v2(input: &[u8], in_w: usize, in_h: usize,
                       output: &mut [u8], out_w: usize, out_h: usize) {
    // Our approach: handle edges during downsampling
    for oy in 0..out_h {
        let iy0 = oy * 2;
        let iy1 = (iy0 + 1).min(in_h - 1);
        let mut bias = 1u16;
        for ox in 0..out_w {
            let ix0 = ox * 2;
            let ix1 = (ix0 + 1).min(in_w - 1);

            let p00 = input[iy0 * in_w + ix0] as u16;
            let p01 = input[iy0 * in_w + ix1] as u16;
            let p10 = input[iy1 * in_w + ix0] as u16;
            let p11 = input[iy1 * in_w + ix1] as u16;

            output[oy * out_w + ox] = ((p00 + p01 + p10 + p11 + bias) >> 2) as u8;
            bias ^= 3;
        }
    }
}

fn c_downsample_h2v2(input: &[u8], in_w: usize, _in_h: usize,
                     output: &mut [u8], out_w: usize, out_h: usize) {
    // C approach: input is already padded, no edge handling needed
    for oy in 0..out_h {
        let iy0 = oy * 2;
        let iy1 = iy0 + 1;
        let mut bias = 1u16;
        for ox in 0..out_w {
            let ix0 = ox * 2;
            let ix1 = ix0 + 1;

            let p00 = input[iy0 * in_w + ix0] as u16;
            let p01 = input[iy0 * in_w + ix1] as u16;
            let p10 = input[iy1 * in_w + ix0] as u16;
            let p11 = input[iy1 * in_w + ix1] as u16;

            output[oy * out_w + ox] = ((p00 + p01 + p10 + p11 + bias) >> 2) as u8;
            bias ^= 3;
        }
    }
}

fn pad_plane(input: &[u8], in_w: usize, in_h: usize,
             output: &mut [u8], out_w: usize, out_h: usize) {
    for y in 0..out_h {
        let src_y = y.min(in_h - 1);
        for x in 0..out_w {
            let src_x = x.min(in_w - 1);
            output[y * out_w + x] = input[src_y * in_w + src_x];
        }
    }
}
