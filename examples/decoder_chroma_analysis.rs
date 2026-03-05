//! Detailed analysis of chroma upsampling differences at the worst-case boundary.
//!
//! Compares decoder outputs pixel-by-pixel against each other and against
//! the original lossless input to understand which decoder is "more correct".
//!
//! Run with: cargo run --release --example decoder_chroma_analysis

use mozjpeg_rs::{Encoder, Preset, Subsampling};

struct DecodedImage {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

fn decode_jpeg_decoder(jpeg: &[u8]) -> DecodedImage {
    let mut dec = jpeg_decoder::Decoder::new(jpeg);
    let pixels = dec.decode().unwrap();
    let info = dec.info().unwrap();
    DecodedImage {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
    }
}

fn decode_zune_jpeg(jpeg: &[u8]) -> DecodedImage {
    use zune_jpeg::JpegDecoder;
    let mut dec = JpegDecoder::new(std::io::Cursor::new(jpeg));
    let pixels = dec.decode().unwrap();
    let info = dec.info().unwrap();
    DecodedImage {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
    }
}

fn decode_mozjpeg_sys(jpeg: &[u8]) -> DecodedImage {
    use mozjpeg_sys::*;
    unsafe {
        let mut cinfo: jpeg_decompress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();
        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateDecompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_decompress_struct>(),
        );
        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as _);
        jpeg_read_header(&mut cinfo, 1);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let w = cinfo.output_width as usize;
        let h = cinfo.output_height as usize;
        let mut pixels = vec![0u8; h * w * 3];
        let mut row: [*mut u8; 1] = [std::ptr::null_mut()];
        while cinfo.output_scanline < cinfo.output_height {
            let off = cinfo.output_scanline as usize * w * 3;
            row[0] = pixels.as_mut_ptr().add(off);
            jpeg_read_scanlines(&mut cinfo, row.as_mut_ptr(), 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        DecodedImage {
            pixels,
            width: w,
            height: h,
        }
    }
}

/// Create a simple test pattern that shows chroma differences clearly.
fn create_test_pattern(w: usize, h: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Create a pattern with strong color variation
            // Red increases left to right
            pixels[i] = (x * 255 / w.max(1)) as u8;
            // Green increases top to bottom
            pixels[i + 1] = (y * 255 / h.max(1)) as u8;
            // Blue is constant mid-gray
            pixels[i + 2] = 128;
        }
    }
    pixels
}

/// Print a pixel grid comparison.
fn print_pixel_grid(
    name: &str,
    img: &DecodedImage,
    original: Option<&[u8]>,
    channel: usize,
    channel_name: &str,
) {
    println!("\n{} - {} channel:", name, channel_name);
    print!("     ");
    for x in 0..img.width {
        print!(" x={:<3}", x);
    }
    println!();

    for y in 0..img.height {
        print!("y={:<2} ", y);
        for x in 0..img.width {
            let idx = (y * img.width + x) * 3 + channel;
            let val = img.pixels[idx];
            if let Some(orig) = original {
                let orig_val = orig[idx];
                let diff = (val as i16 - orig_val as i16).abs();
                if diff > 10 {
                    print!(" {:3}! ", val); // Mark large diffs
                } else {
                    print!(" {:3}  ", val);
                }
            } else {
                print!(" {:3}  ", val);
            }
        }
        println!();
    }
}

/// Print difference grid between two images.
fn print_diff_grid(
    name_a: &str,
    name_b: &str,
    img_a: &DecodedImage,
    img_b: &DecodedImage,
    channel: usize,
    channel_name: &str,
) {
    println!(
        "\nDiff ({} - {}) - {} channel:",
        name_a, name_b, channel_name
    );
    print!("     ");
    for x in 0..img_a.width {
        print!(" x={:<3}", x);
    }
    println!();

    let mut max_diff = 0i16;
    let mut sum_diff = 0i32;
    let mut count = 0;

    for y in 0..img_a.height {
        print!("y={:<2} ", y);
        for x in 0..img_a.width {
            let idx = (y * img_a.width + x) * 3 + channel;
            let a = img_a.pixels[idx] as i16;
            let b = img_b.pixels[idx] as i16;
            let diff = a - b;
            max_diff = max_diff.max(diff.abs());
            sum_diff += diff.abs() as i32;
            count += 1;
            if diff == 0 {
                print!("   .  ");
            } else if diff > 0 {
                print!(" +{:<3} ", diff);
            } else {
                print!(" {:<4} ", diff);
            }
        }
        println!();
    }
    println!(
        "  Max diff: {}, Mean diff: {:.1}",
        max_diff,
        sum_diff as f64 / count as f64
    );
}

fn analyze_case(width: u32, height: u32, subsampling: Subsampling, quality: u8) {
    let sub_name = match subsampling {
        Subsampling::S444 => "4:4:4",
        Subsampling::S422 => "4:2:2",
        Subsampling::S420 => "4:2:0",
        Subsampling::S440 => "4:4:0",
        Subsampling::Gray => "Gray",
    };

    let chroma_w = match subsampling {
        Subsampling::S422 | Subsampling::S420 => (width + 1) / 2,
        _ => width,
    };
    let chroma_h = match subsampling {
        Subsampling::S420 | Subsampling::S440 => (height + 1) / 2,
        _ => height,
    };

    println!("\n{}", "=".repeat(70));
    println!(
        "Case: {}x{} {} Q{} (chroma: {}x{})",
        width, height, sub_name, quality, chroma_w, chroma_h
    );
    println!("{}", "=".repeat(70));

    // Create and encode
    let original = create_test_pattern(width as usize, height as usize);
    let jpeg = Encoder::new(Preset::BaselineBalanced)
        .quality(quality)
        .subsampling(subsampling)
        .encode_rgb(&original, width, height)
        .unwrap();

    println!("JPEG size: {} bytes", jpeg.len());

    // Decode with all three
    let jd = decode_jpeg_decoder(&jpeg);
    let zune = decode_zune_jpeg(&jpeg);
    let moz = decode_mozjpeg_sys(&jpeg);

    // Print original
    println!("\n--- ORIGINAL INPUT ---");
    let orig_img = DecodedImage {
        pixels: original.clone(),
        width: width as usize,
        height: height as usize,
    };
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_pixel_grid("Original", &orig_img, None, ch, name);
    }

    // Print each decoder's output with diff from original
    println!("\n--- JPEG-DECODER OUTPUT (vs original) ---");
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_pixel_grid("jpeg-decoder", &jd, Some(&original), ch, name);
    }

    println!("\n--- ZUNE-JPEG OUTPUT (vs original) ---");
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_pixel_grid("zune-jpeg", &zune, Some(&original), ch, name);
    }

    println!("\n--- MOZJPEG-SYS OUTPUT (vs original) ---");
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_pixel_grid("mozjpeg-sys", &moz, Some(&original), ch, name);
    }

    // Print diffs between decoders
    println!("\n--- DECODER DIFFERENCES ---");
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_diff_grid("jpeg-decoder", "zune-jpeg", &jd, &zune, ch, name);
    }

    println!();
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        print_diff_grid("jpeg-decoder", "mozjpeg-sys", &jd, &moz, ch, name);
    }

    // Summary: distance from original for each decoder
    println!("\n--- SUMMARY: DISTANCE FROM ORIGINAL ---");
    let calc_stats = |img: &DecodedImage| -> (i16, f64) {
        let mut max_diff = 0i16;
        let mut sum = 0i64;
        for (a, b) in img.pixels.iter().zip(original.iter()) {
            let diff = (*a as i16 - *b as i16).abs();
            max_diff = max_diff.max(diff);
            sum += diff as i64;
        }
        let mean = sum as f64 / img.pixels.len() as f64;
        (max_diff, mean)
    };

    let (jd_max, jd_mean) = calc_stats(&jd);
    let (zune_max, zune_mean) = calc_stats(&zune);
    let (moz_max, moz_mean) = calc_stats(&moz);

    println!(
        "jpeg-decoder vs original: max={:3}, mean={:.2}",
        jd_max, jd_mean
    );
    println!(
        "zune-jpeg    vs original: max={:3}, mean={:.2}",
        zune_max, zune_mean
    );
    println!(
        "mozjpeg-sys  vs original: max={:3}, mean={:.2}",
        moz_max, moz_mean
    );

    if jd_mean < moz_mean {
        println!(
            "\n→ Rust decoders are CLOSER to original (by {:.2})",
            moz_mean - jd_mean
        );
    } else if moz_mean < jd_mean {
        println!(
            "\n→ mozjpeg is CLOSER to original (by {:.2})",
            jd_mean - moz_mean
        );
    } else {
        println!("\n→ All decoders equally close to original");
    }
}

fn main() {
    println!("Chroma Upsampling Analysis");
    println!("==========================");
    println!();
    println!("Investigating the worst-case boundary: chroma_width == 2");
    println!("(luma width 3-4 with horizontal subsampling)");

    // The worst cases
    analyze_case(3, 4, Subsampling::S422, 95);
    analyze_case(4, 4, Subsampling::S420, 95);

    // Control case: width 5 (chroma_width = 3) should be fine
    println!("\n\n### CONTROL CASE: width=5 (chroma_width=3) ###");
    analyze_case(5, 4, Subsampling::S422, 95);

    // Another control: 4:4:4 (no subsampling)
    println!("\n\n### CONTROL CASE: 4:4:4 (no subsampling) ###");
    analyze_case(4, 4, Subsampling::S444, 95);
}
