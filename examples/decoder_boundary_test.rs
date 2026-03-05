//! Investigate exact boundaries where decoder differences become significant.
//!
//! This maps out where chroma upsampling differences between decoders cause
//! visible pixel differences, to document the exact thresholds.

use mozjpeg_rs::{Encoder, Preset, Subsampling};

fn decode_and_compare(jpeg: &[u8]) -> (u8, u8, u8) {
    // Decode with jpeg-decoder
    let jd = {
        let mut dec = jpeg_decoder::Decoder::new(jpeg);
        dec.decode().unwrap()
    };

    // Decode with zune-jpeg
    let zune = {
        use zune_jpeg::JpegDecoder;
        let mut dec = JpegDecoder::new(std::io::Cursor::new(jpeg));
        dec.decode().unwrap()
    };

    // Decode with mozjpeg-sys
    let moz = {
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
            pixels
        }
    };

    let diff = |a: &[u8], b: &[u8]| -> u8 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0)
    };

    (diff(&jd, &zune), diff(&jd, &moz), diff(&zune, &moz))
}

fn create_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut p = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            p[i] = (x * 255 / w.max(1)) as u8;
            p[i + 1] = (y * 255 / h.max(1)) as u8;
            p[i + 2] = 128;
        }
    }
    p
}

fn test_dimension(w: u32, h: u32, sub: Subsampling) -> u8 {
    let pixels = create_gradient(w as usize, h as usize);
    let jpeg = Encoder::new(Preset::BaselineBalanced)
        .quality(85)
        .subsampling(sub)
        .encode_rgb(&pixels, w, h)
        .unwrap();
    let (jd_z, jd_m, z_m) = decode_and_compare(&jpeg);
    jd_z.max(jd_m).max(z_m)
}

fn main() {
    println!("Decoder difference by dimension and subsampling");
    println!("(max pixel diff across all decoder pairs)\n");

    let dims: Vec<u32> = vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 20, 24, 32,
    ];

    println!("=== Square dimensions ===");
    println!("{:>6} {:>8} {:>8} {:>8}", "dim", "4:4:4", "4:2:2", "4:2:0");
    println!("{:-<6} {:-<8} {:-<8} {:-<8}", "", "", "", "");

    for d in &dims {
        let d444 = test_dimension(*d, *d, Subsampling::S444);
        let d422 = test_dimension(*d, *d, Subsampling::S422);
        let d420 = test_dimension(*d, *d, Subsampling::S420);
        let flag = |v: u8| if v > 16 { "!" } else { "" };
        println!(
            "{:>4}x{:<4} {:>6}{:<2} {:>6}{:<2} {:>6}{:<2}",
            d,
            d,
            d444,
            flag(d444),
            d422,
            flag(d422),
            d420,
            flag(d420)
        );
    }

    // Non-square: test width vs height separately for 4:2:2
    println!("\n=== Width variation (height=16) with 4:2:2 ===");
    println!("(4:2:2 subsamples width by 2, so chroma_width = ceil(width/2))");
    println!("{:>6} {:>8} {:>12}", "width", "diff", "chroma_w");
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32] {
        let d = test_dimension(w, 16, Subsampling::S422);
        let chroma_w = (w + 1) / 2;
        let flag = if d > 16 { " !" } else { "" };
        println!("{:>6} {:>6}{:<2} {:>10}", w, d, flag, chroma_w);
    }

    // Non-square: test height variation for 4:2:0
    println!("\n=== Height variation (width=16) with 4:2:0 ===");
    println!("(4:2:0 subsamples height by 2, so chroma_height = ceil(height/2))");
    println!("{:>6} {:>8} {:>12}", "height", "diff", "chroma_h");
    for h in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32] {
        let d = test_dimension(16, h, Subsampling::S420);
        let chroma_h = (h + 1) / 2;
        let flag = if d > 16 { " !" } else { "" };
        println!("{:>6} {:>6}{:<2} {:>10}", h, d, flag, chroma_h);
    }

    // Non-square: test width variation for 4:2:0
    println!("\n=== Width variation (height=16) with 4:2:0 ===");
    println!("{:>6} {:>8} {:>12}", "width", "diff", "chroma_w");
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32] {
        let d = test_dimension(w, 16, Subsampling::S420);
        let chroma_w = (w + 1) / 2;
        let flag = if d > 16 { " !" } else { "" };
        println!("{:>6} {:>6}{:<2} {:>10}", w, d, flag, chroma_w);
    }

    // Summary
    println!("\n=== Summary ===");
    println!("! = decoder diff exceeds 16 (normal threshold)");
    println!("\nThe boundary appears to be related to chroma plane dimensions.");
    println!("When the chroma plane is very small, different upsampling");
    println!("algorithms (bilinear vs nearest-neighbor) produce different results.");
}
