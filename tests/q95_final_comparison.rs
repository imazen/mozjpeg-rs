//! Final Q95 comparison with identical settings

use mozjpeg_oxide::test_encoder::{encode_rust, TestEncoderConfig};
use mozjpeg_oxide::Subsampling;
use mozjpeg_sys::*;
use std::ptr;

fn create_photo_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let max_dist = (cx * cx + cy * cy).sqrt();
            let r = (255.0 * (1.0 - dist / max_dist)).clamp(0.0, 255.0) as u8;
            let g = (200.0 * (x as f32 / width as f32)).clamp(0.0, 255.0) as u8;
            let b = (200.0 * (y as f32 / height as f32)).clamp(0.0, 255.0) as u8;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    rgb
}

fn encode_c_with_config(
    rgb: &[u8],
    width: u32,
    height: u32,
    config: &TestEncoderConfig,
) -> Vec<u8> {
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
        cinfo.num_scans = 0;
        cinfo.scan_info = ptr::null();

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        // Set subsampling
        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => panic!("Gray not supported"),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

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

fn count_entropy_bytes(data: &[u8]) -> usize {
    // Find SOS, count bytes until EOI
    for i in 0..data.len() - 3 {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            let start = i + 2 + len;
            // Find EOI
            for j in start..data.len() - 1 {
                if data[j] == 0xFF && data[j + 1] == 0xD9 {
                    return j - start;
                }
            }
        }
    }
    0
}

#[test]
fn final_quality_sweep() {
    println!("\n=== Quality Sweep with Identical Settings ===\n");

    let sizes = [(64, 64), (128, 128), (256, 256)];
    // Full quality range from 10 to 100
    let qualities: Vec<u8> = (1..=10).map(|i| i * 10).collect();

    println!("Config: baseline + Huffman optimization (no trellis)\n");

    for (width, height) in sizes {
        let image = create_photo_image(width, height);
        println!("--- Image {}x{} ---", width, height);
        println!(
            "{:>5} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
            "Q", "Rust", "C", "Ratio", "R-Ent", "C-Ent", "E-Ratio"
        );

        for quality in qualities.iter().copied() {
            let config = TestEncoderConfig {
                quality,
                optimize_huffman: true,
                ..TestEncoderConfig::baseline()
            };

            let rust_jpeg = encode_rust(&image, width as u32, height as u32, &config);
            let c_jpeg = encode_c_with_config(&image, width as u32, height as u32, &config);

            let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;
            let rust_entropy = count_entropy_bytes(&rust_jpeg);
            let c_entropy = count_entropy_bytes(&c_jpeg);
            let entropy_ratio = rust_entropy as f64 / c_entropy.max(1) as f64;

            println!(
                "{:>5} {:>10} {:>10} {:>8.2}% {:>10} {:>10} {:>8.2}%",
                quality,
                rust_jpeg.len(),
                c_jpeg.len(),
                ratio * 100.0,
                rust_entropy,
                c_entropy,
                entropy_ratio * 100.0
            );

            // Assert parity
            if ratio > 1.05 || ratio < 0.95 {
                println!("  *** WARNING: Out of 5% tolerance! ***");
            }
        }
        println!();
    }
}

#[test]
fn compare_with_baseline_no_huffman_opt() {
    println!("\n=== Baseline WITHOUT Huffman Optimization ===\n");

    let sizes = [(128, 128)];
    let qualities = [85, 95];

    for (width, height) in sizes {
        let image = create_photo_image(width, height);
        println!("--- Image {}x{} ---", width, height);

        for &quality in qualities.iter() {
            // Pure baseline - no Huffman optimization
            let config = TestEncoderConfig::baseline().with_quality(quality);

            let rust_jpeg = encode_rust(&image, width as u32, height as u32, &config);
            let c_jpeg = encode_c_with_config(&image, width as u32, height as u32, &config);

            let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;
            let rust_entropy = count_entropy_bytes(&rust_jpeg);
            let c_entropy = count_entropy_bytes(&c_jpeg);

            println!(
                "Q{}: Rust={} C={} ({:.2}%)",
                quality,
                rust_jpeg.len(),
                c_jpeg.len(),
                ratio * 100.0
            );
            println!(
                "     Entropy: Rust={} C={} ({:.2}%)",
                rust_entropy,
                c_entropy,
                rust_entropy as f64 / c_entropy.max(1) as f64 * 100.0
            );

            // With identical settings and no Huffman opt, entropy should be identical
            if rust_entropy != c_entropy {
                println!("     *** ENTROPY DIFFERS! ***");
            }
        }
        println!();
    }
}

#[test]
fn compare_decoded_pixels_q95() {
    println!("\n=== Decoded Pixel Comparison at Q95 ===\n");

    let width = 128u32;
    let height = 128u32;
    let image = create_photo_image(width as usize, height as usize);

    let config = TestEncoderConfig {
        quality: 95,
        optimize_huffman: true,
        ..TestEncoderConfig::baseline()
    };

    let rust_jpeg = encode_rust(&image, width, height, &config);
    let c_jpeg = encode_c_with_config(&image, width, height, &config);

    // Decode both
    let mut rust_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg));
    let rust_pixels = rust_dec.decode().expect("Rust decode failed");

    let mut c_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg));
    let c_pixels = c_dec.decode().expect("C decode failed");

    // Compare pixels
    let mut max_diff = 0i32;
    let mut total_diff = 0u64;
    let mut diff_count = 0usize;

    for (&r, &c) in rust_pixels.iter().zip(c_pixels.iter()) {
        let d = (r as i32 - c as i32).abs();
        if d > 0 {
            diff_count += 1;
            total_diff += d as u64;
            if d > max_diff {
                max_diff = d;
            }
        }
    }

    println!("Rust JPEG: {} bytes", rust_jpeg.len());
    println!("C JPEG:    {} bytes", c_jpeg.len());
    println!(
        "Ratio:     {:.2}%",
        rust_jpeg.len() as f64 / c_jpeg.len() as f64 * 100.0
    );
    println!();
    println!("Decoded pixel comparison:");
    println!("  Max diff:     {}", max_diff);
    println!("  Total pixels: {}", rust_pixels.len());
    println!(
        "  Diff count:   {} ({:.2}%)",
        diff_count,
        diff_count as f64 / rust_pixels.len() as f64 * 100.0
    );
    if diff_count > 0 {
        println!(
            "  Avg diff:     {:.2}",
            total_diff as f64 / diff_count as f64
        );
    }

    // If pixels are identical, the difference is purely Huffman efficiency
    if max_diff == 0 {
        println!("\n*** Decoded pixels are IDENTICAL ***");
        println!("Difference is in Huffman encoding efficiency only.");
    } else {
        println!("\n*** Decoded pixels DIFFER ***");
        println!("Quantized coefficients are different between Rust and C.");
    }
}

/// Comprehensive quality sweep from Q1 to Q100
#[test]
fn comprehensive_quality_sweep() {
    println!("\n=== Comprehensive Quality Sweep (Q1-Q100) ===\n");

    let width = 128u32;
    let height = 128u32;
    let image = create_photo_image(width as usize, height as usize);

    println!("Image: {}x{}", width, height);
    println!("Config: baseline + Huffman optimization\n");
    println!(
        "{:>5} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "Q", "Rust", "C", "Ratio", "MaxDiff", "Status"
    );
    println!("{}", "-".repeat(60));

    let mut first_divergence = None;
    let mut worst_ratio = 1.0f64;
    let mut worst_quality = 0u8;

    // Test every quality level from 1 to 100
    for quality in 1..=100u8 {
        let config = TestEncoderConfig {
            quality,
            optimize_huffman: true,
            ..TestEncoderConfig::baseline()
        };

        let rust_jpeg = encode_rust(&image, width, height, &config);
        let c_jpeg = encode_c_with_config(&image, width, height, &config);

        let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

        // Decode and compare pixels
        let rust_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg))
            .decode()
            .expect("Rust decode failed");
        let c_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg))
            .decode()
            .expect("C decode failed");

        let max_diff: i32 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&r, &c)| (r as i32 - c as i32).abs())
            .max()
            .unwrap_or(0);

        let status = if ratio > 1.10 {
            "FAIL >10%"
        } else if ratio > 1.05 {
            "WARN >5%"
        } else if max_diff > 2 {
            "DIFF"
        } else {
            "OK"
        };

        // Only print notable results to avoid spam
        if quality <= 10 || quality >= 85 || quality % 10 == 0 || status != "OK" {
            println!(
                "{:>5} {:>10} {:>10} {:>8.2}% {:>8} {:>10}",
                quality,
                rust_jpeg.len(),
                c_jpeg.len(),
                ratio * 100.0,
                max_diff,
                status
            );
        }

        // Track first divergence
        if first_divergence.is_none() && max_diff > 0 {
            first_divergence = Some(quality);
        }

        // Track worst ratio
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_quality = quality;
        }
    }

    println!("{}", "-".repeat(60));
    println!("\nSummary:");
    if let Some(q) = first_divergence {
        println!("  First pixel difference at: Q{}", q);
    } else {
        println!("  No pixel differences found!");
    }
    println!(
        "  Worst size ratio: {:.2}% at Q{}",
        worst_ratio * 100.0,
        worst_quality
    );
}

/// Fine-grained analysis around the divergence point
#[test]
fn analyze_high_quality_divergence() {
    println!("\n=== High Quality Divergence Analysis (Q80-Q100) ===\n");

    let width = 64u32;
    let height = 64u32;
    let image = create_photo_image(width as usize, height as usize);

    println!("Image: {}x{}", width, height);
    println!(
        "{:>5} {:>10} {:>10} {:>8} {:>8} {:>8} {:>10}",
        "Q", "Rust", "C", "Ratio", "MaxDiff", "DiffPct", "Entropy%"
    );
    println!("{}", "-".repeat(75));

    for quality in 80..=100u8 {
        let config = TestEncoderConfig {
            quality,
            optimize_huffman: true,
            ..TestEncoderConfig::baseline()
        };

        let rust_jpeg = encode_rust(&image, width, height, &config);
        let c_jpeg = encode_c_with_config(&image, width, height, &config);

        let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;
        let rust_entropy = count_entropy_bytes(&rust_jpeg);
        let c_entropy = count_entropy_bytes(&c_jpeg);
        let entropy_ratio = rust_entropy as f64 / c_entropy.max(1) as f64;

        // Decode and compare
        let rust_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg))
            .decode()
            .expect("Rust decode failed");
        let c_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg))
            .decode()
            .expect("C decode failed");

        let mut max_diff = 0i32;
        let mut diff_count = 0usize;
        for (&r, &c) in rust_pixels.iter().zip(c_pixels.iter()) {
            let d = (r as i32 - c as i32).abs();
            if d > 0 {
                diff_count += 1;
            }
            if d > max_diff {
                max_diff = d;
            }
        }
        let diff_pct = diff_count as f64 / rust_pixels.len() as f64 * 100.0;

        println!(
            "{:>5} {:>10} {:>10} {:>8.2}% {:>8} {:>7.1}% {:>9.2}%",
            quality,
            rust_jpeg.len(),
            c_jpeg.len(),
            ratio * 100.0,
            max_diff,
            diff_pct,
            entropy_ratio * 100.0
        );
    }
}

/// Compare across multiple image sizes at problematic quality levels
#[test]
fn analyze_size_scaling() {
    println!("\n=== Size Scaling Analysis at Various Quality Levels ===\n");

    let sizes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ];
    let qualities = [75, 85, 90, 95, 98];

    for quality in qualities {
        println!("--- Quality {} ---", quality);
        println!(
            "{:>10} {:>10} {:>10} {:>8} {:>8}",
            "Size", "Rust", "C", "Ratio", "MaxDiff"
        );

        for (width, height) in sizes {
            let image = create_photo_image(width, height);

            let config = TestEncoderConfig {
                quality,
                optimize_huffman: true,
                ..TestEncoderConfig::baseline()
            };

            let rust_jpeg = encode_rust(&image, width as u32, height as u32, &config);
            let c_jpeg = encode_c_with_config(&image, width as u32, height as u32, &config);

            let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

            // Decode and compare
            let rust_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg))
                .decode()
                .expect("Rust decode failed");
            let c_pixels = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg))
                .decode()
                .expect("C decode failed");

            let max_diff: i32 = rust_pixels
                .iter()
                .zip(c_pixels.iter())
                .map(|(&r, &c)| (r as i32 - c as i32).abs())
                .max()
                .unwrap_or(0);

            println!(
                "{:>10} {:>10} {:>10} {:>8.2}% {:>8}",
                format!("{}x{}", width, height),
                rust_jpeg.len(),
                c_jpeg.len(),
                ratio * 100.0,
                max_diff
            );
        }
        println!();
    }
}

/// Analyze marker structure differences
#[test]
fn compare_marker_structure() {
    println!("\n=== Marker Structure Comparison ===\n");

    let width = 64u32;
    let height = 64u32;
    let image = create_photo_image(width as usize, height as usize);

    // Test at Q10 where overhead difference is most visible
    let config = TestEncoderConfig {
        quality: 10,
        optimize_huffman: true,
        ..TestEncoderConfig::baseline()
    };

    let rust_jpeg = encode_rust(&image, width, height, &config);
    let c_jpeg = encode_c_with_config(&image, width, height, &config);

    fn parse_markers(data: &[u8]) -> Vec<(u8, usize, String)> {
        let mut markers = Vec::new();
        let mut i = 0;
        while i < data.len() - 1 {
            if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
                let marker = data[i + 1];
                let name = match marker {
                    0xD8 => "SOI",
                    0xD9 => "EOI",
                    0xE0 => "APP0",
                    0xDB => "DQT",
                    0xC0 => "SOF0",
                    0xC4 => "DHT",
                    0xDA => "SOS",
                    _ => "OTHER",
                };

                if marker == 0xD8 || marker == 0xD9 {
                    markers.push((marker, 2, name.to_string()));
                    i += 2;
                } else if i + 3 < data.len() {
                    let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                    markers.push((marker, len + 2, name.to_string()));
                    i += 2 + len;
                } else {
                    break;
                }
            } else {
                i += 1;
            }
        }
        markers
    }

    let rust_markers = parse_markers(&rust_jpeg);
    let c_markers = parse_markers(&c_jpeg);

    println!("Rust JPEG: {} bytes total", rust_jpeg.len());
    println!("{:>8} {:>8}  Marker", "Size", "Cumul");
    let mut cumul = 0;
    for (marker, size, name) in &rust_markers {
        cumul += size;
        println!("{:>8} {:>8}  FF{:02X} {}", size, cumul, marker, name);
    }

    println!("\nC JPEG: {} bytes total", c_jpeg.len());
    println!("{:>8} {:>8}  Marker", "Size", "Cumul");
    cumul = 0;
    for (marker, size, name) in &c_markers {
        cumul += size;
        println!("{:>8} {:>8}  FF{:02X} {}", size, cumul, marker, name);
    }

    // Summarize differences
    println!("\n--- Size Comparison by Marker Type ---");
    let mut rust_by_type: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut c_by_type: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();

    for (marker, size, _) in &rust_markers {
        *rust_by_type.entry(*marker).or_insert(0) += size;
    }
    for (marker, size, _) in &c_markers {
        *c_by_type.entry(*marker).or_insert(0) += size;
    }

    let all_markers: std::collections::HashSet<u8> = rust_by_type
        .keys()
        .chain(c_by_type.keys())
        .copied()
        .collect();
    let mut sorted_markers: Vec<_> = all_markers.into_iter().collect();
    sorted_markers.sort();

    for marker in sorted_markers {
        let r = rust_by_type.get(&marker).unwrap_or(&0);
        let c = c_by_type.get(&marker).unwrap_or(&0);
        let diff = *r as i32 - *c as i32;
        let name = match marker {
            0xD8 => "SOI",
            0xD9 => "EOI",
            0xE0 => "APP0",
            0xDB => "DQT",
            0xC0 => "SOF0",
            0xC4 => "DHT",
            0xDA => "SOS",
            _ => "OTHER",
        };
        if diff != 0 {
            println!(
                "FF{:02X} {:5}: Rust={:>4}, C={:>4}, diff={:+}",
                marker, name, r, c, diff
            );
        }
    }
}

/// Test minimal 8x8 block encoding
#[test]
fn test_minimal_8x8_block() {
    println!("\n=== Minimal 8x8 Block Test at Q95 ===\n");

    // Create a simple 8x8 gradient
    let width = 8u32;
    let height = 8u32;
    let mut rgb = vec![0u8; 8 * 8 * 3];
    for y in 0..8 {
        for x in 0..8 {
            let i = (y * 8 + x) * 3;
            rgb[i] = ((x * 32) % 256) as u8;
            rgb[i + 1] = ((y * 32) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    let config = TestEncoderConfig {
        quality: 95,
        optimize_huffman: false,
        ..TestEncoderConfig::baseline()
    };

    let rust_jpeg = encode_rust(&rgb, width, height, &config);
    let c_jpeg = encode_c_with_config(&rgb, width, height, &config);

    println!("8x8 block at Q95:");
    println!("  Rust: {} bytes", rust_jpeg.len());
    println!("  C:    {} bytes", c_jpeg.len());
    println!(
        "  Ratio: {:.2}%",
        rust_jpeg.len() as f64 / c_jpeg.len() as f64 * 100.0
    );

    // Decode and compare
    let mut rust_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg));
    let rust_pixels = rust_dec.decode().expect("Rust decode failed");

    let mut c_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&c_jpeg));
    let c_pixels = c_dec.decode().expect("C decode failed");

    println!("\nDecoded first row:");
    println!("  Rust: {:?}", &rust_pixels[..24]); // First 8 RGB pixels
    println!("  C:    {:?}", &c_pixels[..24]);

    let mut max_diff = 0i32;
    for (&r, &c) in rust_pixels.iter().zip(c_pixels.iter()) {
        let d = (r as i32 - c as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("\nMax pixel diff: {}", max_diff);

    if max_diff == 0 {
        println!("*** IDENTICAL at 8x8 ***");
    } else {
        println!("*** DIFFERS at 8x8 (max diff {}) ***", max_diff);
    }
}
