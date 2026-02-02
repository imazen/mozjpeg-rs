//! Comprehensive parity benchmark: Rust vs C mozjpeg file sizes.
//!
//! Encodes the full Kodak corpus (24 images) across 6 encoder configurations
//! and 4 quality levels (24 rows), producing a reproducible markdown table.
//!
//! Uses raw mozjpeg-sys FFI (not the `mozjpeg` crate) because we need
//! `set_trellis_quant()`, `set_trellis_quant_dc()`, and
//! `set_overshoot_deringing()` — none of which the high-level crate exposes.
//!
//! Run with:
//! ```sh
//! cargo test --release --test parity_benchmark -- --nocapture
//! ```

use mozjpeg_rs::corpus;
use mozjpeg_rs::test_encoder::{encode_rust, TestEncoderConfig};
use std::path::Path;

// ---------------------------------------------------------------------------
// PNG loader
// ---------------------------------------------------------------------------

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
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
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

// ---------------------------------------------------------------------------
// C encoder via raw mozjpeg-sys FFI
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn encode_c_with_config(
    rgb: &[u8],
    width: u32,
    height: u32,
    config: &TestEncoderConfig,
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

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);

        // CRITICAL: set optimize_scans BEFORE jpeg_simple_progression.
        // JCP_MAX_COMPRESSION enables optimize_scans by default, which causes
        // jpeg_simple_progression() to call jpeg_search_progression() and
        // generate an optimized ~12-scan script instead of the fixed 9-scan one.
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OPTIMIZE_SCANS,
            if config.optimize_scans { 1 } else { 0 },
        );

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        // Subsampling — always 4:2:0
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Huffman optimization
        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        // Progressive mode (must come AFTER optimize_scans)
        if config.progressive {
            jpeg_simple_progression(&mut cinfo);
        } else {
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        }

        // Trellis / deringing
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT,
            if config.trellis_quant { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT_DC,
            if config.trellis_dc { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OVERSHOOT_DERINGING,
            if config.overshoot_deringing { 1 } else { 0 },
        );

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

// ---------------------------------------------------------------------------
// Config matrix
// ---------------------------------------------------------------------------

struct NamedConfig {
    name: &'static str,
    make: fn(u8) -> TestEncoderConfig,
}

const CONFIGS: &[NamedConfig] = &[
    NamedConfig {
        name: "Baseline",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Baseline + Trellis",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: false,
            overshoot_deringing: false,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Full Baseline",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: false,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: true,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Progressive",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Progressive + Trellis",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: false,
            overshoot_deringing: false,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Full Progressive",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: true,
            optimize_scans: false,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
    NamedConfig {
        name: "Max Compression",
        make: |q| TestEncoderConfig {
            quality: q,
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: true,
            optimize_scans: true,
            force_baseline: true,
            ..TestEncoderConfig::default()
        },
    },
];

const QUALITIES: &[u8] = &[55, 65, 75, 85, 90, 95];

// ---------------------------------------------------------------------------
// Main benchmark test
// ---------------------------------------------------------------------------

#[test]
fn parity_benchmark() {
    // Locate Kodak corpus
    let kodak = match corpus::kodak_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping parity_benchmark: Kodak corpus not found.");
            eprintln!("Run ./scripts/fetch-corpus.sh to download it.");
            return;
        }
    };

    let png_paths = corpus::png_files_in_dir(&kodak);
    if png_paths.is_empty() {
        eprintln!(
            "Skipping parity_benchmark: no PNG files in {}",
            kodak.display()
        );
        return;
    }

    // Load all images into memory
    let images: Vec<(String, Vec<u8>, u32, u32)> = png_paths
        .iter()
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            let (rgb, w, h) = load_png(p)?;
            Some((name, rgb, w, h))
        })
        .collect();

    let n_images = images.len();
    if n_images == 0 {
        eprintln!("Skipping parity_benchmark: could not load any images.");
        return;
    }

    let fast_yuv_status = if cfg!(feature = "fast-yuv") {
        "enabled"
    } else {
        "disabled"
    };

    // Header
    println!();
    println!("## File Size Parity: mozjpeg-rs vs C mozjpeg");
    println!();
    println!(
        "Kodak corpus ({} images), 4:2:0, fast-yuv {}.",
        n_images, fast_yuv_status
    );
    println!();
    println!(
        "| {:<24} | {:>2} | {:>10} | {:>10} | {:>7} | {:>7} |",
        "Config", "Q", "Avg Rust", "Avg C", "Delta", "Max Dev"
    );
    println!(
        "|{:-<26}|{:-<4}|{:-<12}|{:-<12}|{:-<9}|{:-<9}|",
        "", "", "", "", "", ""
    );

    let mut failures: Vec<String> = Vec::new();

    for named in CONFIGS {
        for &quality in QUALITIES {
            let config = (named.make)(quality);

            let mut total_rust: u64 = 0;
            let mut total_c: u64 = 0;
            let mut max_dev: f64 = 0.0;

            for (name, rgb, w, h) in &images {
                let rust_bytes = encode_rust(rgb, *w, *h, &config);
                let c_bytes = encode_c_with_config(rgb, *w, *h, &config);

                let rust_len = rust_bytes.len() as u64;
                let c_len = c_bytes.len() as u64;
                total_rust += rust_len;
                total_c += c_len;

                let dev = if c_len > 0 {
                    ((rust_len as f64 / c_len as f64) - 1.0).abs() * 100.0
                } else {
                    0.0
                };

                if dev > max_dev {
                    max_dev = dev;
                }

                // Check per-image deviation
                if dev > 3.0 {
                    failures.push(format!(
                        "Per-image deviation {:.2}% > 3.0% for {} @ {} Q{} \
                         (Rust={}, C={})",
                        dev, name, named.name, quality, rust_len, c_len
                    ));
                }
            }

            let avg_rust = total_rust as f64 / n_images as f64;
            let avg_c = total_c as f64 / n_images as f64;
            let delta_pct = if total_c > 0 {
                ((total_rust as f64 / total_c as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "| {:<24} | {:>2} | {:>10} | {:>10} | {:>+6.2}% | {:>6.2}% |",
                named.name,
                quality,
                format_size(avg_rust),
                format_size(avg_c),
                delta_pct,
                max_dev,
            );

            // Check average delta
            if delta_pct.abs() > 1.0 {
                failures.push(format!(
                    "Average delta {:.2}% > 1.0% for {} Q{} \
                     (Rust avg={:.0}, C avg={:.0})",
                    delta_pct, named.name, quality, avg_rust, avg_c
                ));
            }
        }
    }

    println!();

    // Report all failures at end
    if !failures.is_empty() {
        println!("## Failures ({}):", failures.len());
        for f in &failures {
            println!("  - {}", f);
        }
        println!();
        panic!(
            "{} parity assertion(s) failed. See table above for details.",
            failures.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a byte size with thousands separators for readability.
fn format_size(size: f64) -> String {
    let s = format!("{:.0}", size);
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 3 {
        return s;
    }
    let mut result = String::with_capacity(len + len / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

#[cfg(test)]
mod format_tests {
    use super::format_size;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(123.0), "123");
        assert_eq!(format_size(1234.0), "1,234");
        assert_eq!(format_size(12345.0), "12,345");
        assert_eq!(format_size(123456.0), "123,456");
        assert_eq!(format_size(1234567.0), "1,234,567");
    }
}
