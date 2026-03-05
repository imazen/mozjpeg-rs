//! Detailed boundary investigation focusing on the chroma_width = 2 threshold.

use mozjpeg_rs::{Encoder, Preset, Subsampling};

fn decode_and_compare(jpeg: &[u8]) -> (u8, u8, u8) {
    let jd = {
        let mut dec = jpeg_decoder::Decoder::new(jpeg);
        dec.decode().unwrap()
    };
    let zune = {
        use zune_jpeg::JpegDecoder;
        let mut dec = JpegDecoder::new(std::io::Cursor::new(jpeg));
        dec.decode().unwrap()
    };
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

fn test_dimension(w: u32, h: u32, sub: Subsampling) -> (u8, u8, u8) {
    let pixels = create_gradient(w as usize, h as usize);
    let jpeg = Encoder::new(Preset::BaselineBalanced)
        .quality(85)
        .subsampling(sub)
        .encode_rgb(&pixels, w, h)
        .unwrap();
    decode_and_compare(&jpeg)
}

fn main() {
    println!("=== Detailed 4:2:2 boundary (varying width, height=32) ===");
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>8}",
        "width", "chroma_w", "jd-zune", "jd-moz", "zune-moz"
    );
    for w in 1..=10 {
        let (jz, jm, zm) = test_dimension(w, 32, Subsampling::S422);
        let cw = (w + 1) / 2;
        println!("{:>5} {:>8} {:>8} {:>8} {:>8}", w, cw, jz, jm, zm);
    }

    println!("\n=== Detailed 4:2:0 boundary (varying width, height=32) ===");
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>8}",
        "width", "chroma_w", "jd-zune", "jd-moz", "zune-moz"
    );
    for w in 1..=10 {
        let (jz, jm, zm) = test_dimension(w, 32, Subsampling::S420);
        let cw = (w + 1) / 2;
        println!("{:>5} {:>8} {:>8} {:>8} {:>8}", w, cw, jz, jm, zm);
    }

    println!("\n=== Detailed 4:2:0 boundary (varying height, width=32) ===");
    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>8}",
        "height", "chroma_h", "jd-zune", "jd-moz", "zune-moz"
    );
    for h in 1..=10 {
        let (jz, jm, zm) = test_dimension(32, h, Subsampling::S420);
        let ch = (h + 1) / 2;
        println!("{:>6} {:>8} {:>8} {:>8} {:>8}", h, ch, jz, jm, zm);
    }

    println!("\n=== 4:4:0 (subsamples height only) ===");
    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>8}",
        "height", "chroma_h", "jd-zune", "jd-moz", "zune-moz"
    );
    for h in 1..=10 {
        let (jz, jm, zm) = test_dimension(32, h, Subsampling::S440);
        let ch = (h + 1) / 2;
        println!("{:>6} {:>8} {:>8} {:>8} {:>8}", h, ch, jz, jm, zm);
    }

    println!("\n=== KEY FINDING ===");
    println!("The large differences occur when:");
    println!("- jpeg-decoder and zune-jpeg AGREE (jd-zune ≈ 0-2)");
    println!("- mozjpeg-sys DIFFERS from both (jd-moz, zune-moz > 20)");
    println!("\nThis means mozjpeg uses a different chroma upsampling algorithm");
    println!("than the Rust decoders when the chroma plane is very small.");
}
