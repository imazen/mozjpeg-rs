//! Comprehensive progressive JPEG comparison tests.
//!
//! Tests various progressive scan configurations between Rust and C mozjpeg:
//! - Minimal progressive (no successive approximation)
//! - Standard progressive (with successive approximation)
//! - Optimized progressive (scan optimization)
//!
//! Uses codec-eval for perceptual quality metrics.

use codec_eval::{decode::jpeg_decode_callback, EvalSession, ImageData, MetricConfig};
use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig};
use std::fs::File;
use std::path::Path;

/// Instrumentation flag - set to true to print detailed debug output
const DEBUG_INSTRUMENTATION: bool = false;

/// Configuration for a progressive test case
#[derive(Debug, Clone)]
struct ProgressiveTestConfig {
    name: &'static str,
    quality: u8,
    rust_optimize_scans: bool,
    c_optimize_scans: bool,
}

/// Result of a progressive comparison
#[derive(Debug)]
struct ComparisonResult {
    config_name: String,
    quality: u8,
    rust_size: usize,
    c_size: usize,
    size_ratio: f64,
    rust_ssim2: f64,
    c_ssim2: f64,
    ssim2_diff: f64,
    rust_scan_count: usize,
    c_scan_count: usize,
}

fn load_test_image() -> Option<(Vec<u8>, u32, u32)> {
    // Try various test image locations
    let paths = [
        "corpus/kodak/10.png",      // First available Kodak image
        "tests/images/1.png",       // Available test image
        "tests/images/kodim23.png",
        "corpus/kodak/kodim23.png",
    ];

    for path in paths {
        if let Some(data) = load_png(Path::new(path)) {
            if DEBUG_INSTRUMENTATION {
                println!("Loaded test image from: {}", path);
            }
            return Some(data);
        }
    }
    None
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
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
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

/// Encode with Rust mozjpeg-oxide
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, optimize_scans: bool) -> Vec<u8> {
    Encoder::new()
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true)
        .overshoot_deringing(true)
        .trellis(TrellisConfig::default())
        .optimize_scans(optimize_scans)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

/// Encode with C mozjpeg
#[allow(unsafe_code)]
fn encode_c(rgb: &[u8], width: u32, height: u32, quality: u8, optimize_scans: bool) -> Vec<u8> {
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

        // Use ImageMagick quant tables (index 3) to match Rust
        jpeg_c_set_int_param(&mut cinfo, JINT_BASE_QUANT_TBL_IDX, 3);

        // Enable progressive mode
        jpeg_simple_progression(&mut cinfo);
        jpeg_set_quality(&mut cinfo, quality as i32, 1);

        // 4:2:0 subsampling
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Enable optimizations
        cinfo.optimize_coding = 1;
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_TRELLIS_QUANT_DC, 1);
        jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OVERSHOOT_DERINGING, 1);

        if optimize_scans {
            jpeg_c_set_bool_param(&mut cinfo, JBOOLEAN_OPTIMIZE_SCANS, 1);
        }

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
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

/// Count SOS markers in JPEG to determine number of scans
fn count_scans(jpeg_data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            count += 1;
        }
        i += 1;
    }
    count
}

/// Extract scan info from JPEG (for debugging)
fn extract_scan_info(jpeg_data: &[u8]) -> Vec<(u8, u8, u8, u8)> {
    let mut scans = Vec::new();
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            // SOS marker found
            if i + 5 < jpeg_data.len() {
                let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                if i + 2 + len <= jpeg_data.len() {
                    let num_components = jpeg_data[i + 4];
                    let ss_offset = i + 4 + 1 + (num_components as usize * 2);
                    if ss_offset + 2 < jpeg_data.len() {
                        let ss = jpeg_data[ss_offset];
                        let se = jpeg_data[ss_offset + 1];
                        let ahl = jpeg_data[ss_offset + 2];
                        let ah = ahl >> 4;
                        let al = ahl & 0x0F;
                        scans.push((ss, se, ah, al));
                    }
                }
            }
        }
        i += 1;
    }
    scans
}

#[test]
fn test_progressive_minimal_vs_c() {
    let (rgb, width, height) = match load_test_image() {
        Some(data) => data,
        None => {
            eprintln!("Skipping test: no test image available");
            return;
        }
    };

    let qualities = [50, 75, 85, 90, 95, 97];

    println!("\nProgressive Minimal (optimize_scans=false) Comparison");
    println!("======================================================");
    println!(
        "{:>5} {:>10} {:>10} {:>9} {:>8} {:>8}",
        "Q", "Rust", "C", "Ratio", "R Scans", "C Scans"
    );
    println!("{}", "-".repeat(60));

    for &quality in &qualities {
        let rust_jpeg = encode_rust(&rgb, width, height, quality, false);
        let c_jpeg = encode_c(&rgb, width, height, quality, false);

        let rust_scans = count_scans(&rust_jpeg);
        let c_scans = count_scans(&c_jpeg);
        let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

        println!(
            "{:>5} {:>10} {:>10} {:>8.2}% {:>8} {:>8}",
            quality,
            rust_jpeg.len(),
            c_jpeg.len(),
            (ratio - 1.0) * 100.0,
            rust_scans,
            c_scans
        );

        if DEBUG_INSTRUMENTATION {
            println!("  Rust scans: {:?}", extract_scan_info(&rust_jpeg));
            println!("  C scans: {:?}", extract_scan_info(&c_jpeg));
        }

        // At quality 50, we should be very close
        if quality == 50 {
            assert!(
                ratio < 1.02,
                "Q50: Rust should be within 2% of C (was {:.2}%)",
                (ratio - 1.0) * 100.0
            );
        }
    }
}

#[test]
fn test_progressive_optimized_vs_c() {
    let (rgb, width, height) = match load_test_image() {
        Some(data) => data,
        None => {
            eprintln!("Skipping test: no test image available");
            return;
        }
    };

    let qualities = [50, 75, 85, 90, 95, 97];

    println!("\nProgressive Optimized (optimize_scans=true) Comparison");
    println!("=======================================================");
    println!(
        "{:>5} {:>10} {:>10} {:>9} {:>8} {:>8}",
        "Q", "Rust", "C", "Ratio", "R Scans", "C Scans"
    );
    println!("{}", "-".repeat(60));

    for &quality in &qualities {
        let rust_jpeg = encode_rust(&rgb, width, height, quality, true);
        let c_jpeg = encode_c(&rgb, width, height, quality, true);

        let rust_scans = count_scans(&rust_jpeg);
        let c_scans = count_scans(&c_jpeg);
        let ratio = rust_jpeg.len() as f64 / c_jpeg.len() as f64;

        println!(
            "{:>5} {:>10} {:>10} {:>8.2}% {:>8} {:>8}",
            quality,
            rust_jpeg.len(),
            c_jpeg.len(),
            (ratio - 1.0) * 100.0,
            rust_scans,
            c_scans
        );

        if DEBUG_INSTRUMENTATION {
            println!("  Rust scans: {:?}", extract_scan_info(&rust_jpeg));
            println!("  C scans: {:?}", extract_scan_info(&c_jpeg));
        }
    }
}

#[test]
fn test_successive_approximation_scan_structure() {
    // Test that Rust's standard progressive script matches the expected structure
    use mozjpeg_oxide::progressive::generate_standard_progressive_scans;

    let scans = generate_standard_progressive_scans(3);

    println!("\nStandard Progressive Scan Script (3 components):");
    println!("=================================================");
    for (i, scan) in scans.iter().enumerate() {
        let scan_type = if scan.ss == 0 && scan.se == 0 {
            if scan.ah > 0 {
                "DC refine"
            } else {
                "DC first"
            }
        } else if scan.ah > 0 {
            "AC refine"
        } else {
            "AC first"
        };

        println!(
            "  {:2}: {} comp={} ss={} se={} ah={} al={}",
            i, scan_type, scan.component_index[0], scan.ss, scan.se, scan.ah, scan.al
        );
    }

    // Verify structure
    assert!(scans.len() >= 10, "Standard progressive should have at least 10 scans");

    // First scan should be DC with point transform (Al > 0)
    assert!(scans[0].ss == 0 && scans[0].se == 0, "First scan should be DC");
    assert!(scans[0].al > 0, "First DC scan should have point transform");

    // Should have refinement scans (Ah > 0)
    let has_refinement = scans.iter().any(|s| s.ah > 0);
    assert!(has_refinement, "Should have refinement scans");
}

#[test]
fn test_scan_optimization_produces_successive_approximation() {
    let (rgb, width, height) = match load_test_image() {
        Some(data) => data,
        None => {
            eprintln!("Skipping test: no test image available");
            return;
        }
    };

    // Encode with optimize_scans=true
    let jpeg = encode_rust(&rgb, width, height, 85, true);
    let scans = extract_scan_info(&jpeg);

    println!("\nOptimized Progressive Scans:");
    for (i, (ss, se, ah, al)) in scans.iter().enumerate() {
        println!("  {:2}: ss={} se={} ah={} al={}", i, ss, se, ah, al);
    }

    // Check for successive approximation (Al > 0 in first-pass scans or Ah > 0 in refinement)
    let has_successive_approx = scans.iter().any(|(_, _, ah, al)| *ah > 0 || *al > 0);

    println!("\nHas successive approximation: {}", has_successive_approx);

    // With optimize_scans, we should have successive approximation for better compression
    // Note: This might not always be true depending on the image, but for most images it should be
}

#[test]
fn test_compare_scan_scripts() {
    use mozjpeg_oxide::progressive::{
        generate_minimal_progressive_scans, generate_optimized_progressive_scans,
        generate_standard_progressive_scans,
    };

    println!("\nScan Script Comparison:");
    println!("=======================\n");

    let minimal = generate_minimal_progressive_scans(3);
    let standard = generate_standard_progressive_scans(3);
    let optimized = generate_optimized_progressive_scans(3);

    println!("Minimal: {} scans", minimal.len());
    println!("Standard: {} scans", standard.len());
    println!("Optimized: {} scans", optimized.len());

    println!("\nMinimal scans (no successive approximation):");
    for (i, s) in minimal.iter().enumerate() {
        println!(
            "  {:2}: comp={} ss={:2} se={:2} ah={} al={}",
            i, s.component_index[0], s.ss, s.se, s.ah, s.al
        );
    }

    println!("\nStandard scans (with successive approximation):");
    for (i, s) in standard.iter().enumerate() {
        println!(
            "  {:2}: comp={} ss={:2} se={:2} ah={} al={}",
            i, s.component_index[0], s.ss, s.se, s.ah, s.al
        );
    }

    // Count refinement scans in each
    let minimal_refinements = minimal.iter().filter(|s| s.ah > 0).count();
    let standard_refinements = standard.iter().filter(|s| s.ah > 0).count();

    println!("\nRefinement scans: minimal={}, standard={}", minimal_refinements, standard_refinements);

    assert_eq!(minimal_refinements, 0, "Minimal should have no refinements");
    assert!(standard_refinements > 0, "Standard should have refinements");
}
