//! CID22 benchmark: Rust vs C mozjpeg with DSSIM and Butteraugli metrics
//!
//! Usage: cargo run --release --example cid22_bench
//!
//! Set CODEC_CORPUS_DIR to point to your codec-corpus checkout.
use butteraugli::{ButteraugliParams, Img, RGB8};
use dssim::Dssim;
use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use rgb::RGB8;
use std::fs::File;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
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
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn encode_rust(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    progressive: bool,
    trellis: bool,
    optimize_scans: bool,
) -> Vec<u8> {
    // Start from BaselineFastest (no optimizations) and add what we need
    let mut enc = Encoder::fastest()
        .quality(quality)
        .subsampling(Subsampling::S420)
        .progressive(progressive)
        .optimize_huffman(true);

    if trellis {
        enc = enc
            .trellis(TrellisConfig::default())
            .overshoot_deringing(true);
    }
    if optimize_scans {
        enc = enc.optimize_scans(true);
    }

    enc.encode_rgb(rgb, w, h).unwrap()
}

fn encode_c(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    progressive: bool,
    trellis: bool,
    optimize_scans: bool,
) -> Vec<u8> {
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
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = w;
        cinfo.image_height = h;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = 1;

        // Explicitly set trellis parameters (mozjpeg may enable them by default)
        if trellis {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);
        } else {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 0);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 0);
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 0);
        }

        // Progressive settings must be set AFTER jpeg_set_defaults and quality
        // Note: JBOOLEAN_OPTIMIZE_SCANS cannot be set on non-progressive images
        if progressive {
            jpeg_simple_progression(&mut cinfo);
            if optimize_scans {
                jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1);
            }
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = w as usize * 3;
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

fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as u32, info.height as u32))
}

fn compute_dssim(original: &[u8], decoded: &[u8], w: u32, h: u32) -> f64 {
    let dssim = Dssim::new();

    let orig_rgb: Vec<RGB8> = original
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = dssim
        .create_image_rgb(&orig_rgb, w as usize, h as usize)
        .expect("create orig");

    let dec_rgb: Vec<RGB8> = decoded
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = dssim
        .create_image_rgb(&dec_rgb, w as usize, h as usize)
        .expect("create dec");

    let (val, _) = dssim.compare(&orig_img, dec_img);
    val.into()
}

fn compute_butteraugli(original: &[u8], decoded: &[u8], w: u32, h: u32) -> f64 {
    let to_pixels = |rgb: &[u8]| -> Vec<RGB8> {
        rgb.chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect()
    };
    let width = w as usize;
    let height = h as usize;
    let img1 = Img::new(to_pixels(original), width, height);
    let img2 = Img::new(to_pixels(decoded), width, height);
    butteraugli::butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())
        .map(|r| r.score)
        .unwrap_or(f64::MAX)
}

#[derive(Debug, Clone, Copy)]
struct Config {
    name: &'static str,
    progressive: bool,
    trellis: bool,
    optimize_scans: bool,
}

const CONFIGS: &[Config] = &[
    Config {
        name: "Baseline",
        progressive: false,
        trellis: false,
        optimize_scans: false,
    },
    Config {
        name: "Baseline+Trellis",
        progressive: false,
        trellis: true,
        optimize_scans: false,
    },
    Config {
        name: "Progressive",
        progressive: true,
        trellis: false,
        optimize_scans: false,
    },
    Config {
        name: "Progressive+Trellis",
        progressive: true,
        trellis: true,
        optimize_scans: false,
    },
    Config {
        name: "MaxCompression",
        progressive: true,
        trellis: true,
        optimize_scans: true,
    },
];

const QUALITIES: &[u8] = &[75, 85, 90, 95];

fn main() {
    let corpus_dir = std::env::var("CODEC_CORPUS_DIR")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".to_string());
    let training_dir = format!("{}/CID22/CID22-512/training", corpus_dir);

    let mut images: Vec<_> = std::fs::read_dir(&training_dir)
        .expect("CID22 training dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    images.sort_by_key(|e| e.path());

    let total = images.len();
    eprintln!("Processing {} images from CID22-512/training\n", total);

    // Header
    println!(
        "| Config | Q | Rust Size | C Size | Size Δ | DSSIM (R) | DSSIM (C) | Butteraugli (R) | Butteraugli (C) |"
    );
    println!(
        "|--------|---|-----------|--------|--------|-----------|-----------|-----------------|-----------------|"
    );

    for config in CONFIGS {
        for &quality in QUALITIES {
            let mut rust_total_size = 0usize;
            let mut c_total_size = 0usize;
            let mut rust_dssim_sum = 0.0;
            let mut c_dssim_sum = 0.0;
            let mut rust_butter_sum = 0.0;
            let mut c_butter_sum = 0.0;
            let mut count = 0;

            for (i, entry) in images.iter().enumerate() {
                let path = entry.path();
                let Some((rgb, w, h)) = load_png(&path) else {
                    continue;
                };

                let rust_jpeg = encode_rust(
                    &rgb,
                    w,
                    h,
                    quality,
                    config.progressive,
                    config.trellis,
                    config.optimize_scans,
                );
                let c_jpeg = encode_c(
                    &rgb,
                    w,
                    h,
                    quality,
                    config.progressive,
                    config.trellis,
                    config.optimize_scans,
                );

                rust_total_size += rust_jpeg.len();
                c_total_size += c_jpeg.len();

                // Decode and measure quality
                if let (Some((rust_dec, _, _)), Some((c_dec, _, _))) =
                    (decode_jpeg(&rust_jpeg), decode_jpeg(&c_jpeg))
                {
                    rust_dssim_sum += compute_dssim(&rgb, &rust_dec, w, h);
                    c_dssim_sum += compute_dssim(&rgb, &c_dec, w, h);
                    rust_butter_sum += compute_butteraugli(&rgb, &rust_dec, w, h);
                    c_butter_sum += compute_butteraugli(&rgb, &c_dec, w, h);
                }

                count += 1;
                if (i + 1) % 50 == 0 {
                    eprint!("\r{} Q{}: {}/{}", config.name, quality, i + 1, total);
                }
            }
            eprintln!(
                "\r{} Q{}: done ({} images)    ",
                config.name, quality, count
            );

            let size_delta = (rust_total_size as f64 / c_total_size as f64 - 1.0) * 100.0;
            let avg_rust_dssim = rust_dssim_sum / count as f64;
            let avg_c_dssim = c_dssim_sum / count as f64;
            let avg_rust_butter = rust_butter_sum / count as f64;
            let avg_c_butter = c_butter_sum / count as f64;

            println!(
                "| {} | {} | {:.1} KB | {:.1} KB | {:+.2}% | {:.6} | {:.6} | {:.3} | {:.3} |",
                config.name,
                quality,
                rust_total_size as f64 / 1024.0,
                c_total_size as f64 / 1024.0,
                size_delta,
                avg_rust_dssim,
                avg_c_dssim,
                avg_rust_butter,
                avg_c_butter,
            );
        }
    }
}
