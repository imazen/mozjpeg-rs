//! Preset parity tests - compares each Preset against mozjpeg crate at multiple sizes.
//!
//! Tests 3 image sizes against the mozjpeg crate (wraps mozjpeg-sys) to verify
//! output parity for each preset.
//!
//! ## mozjpeg crate settings order (CRITICAL):
//!
//! The mozjpeg crate internally calls `jpeg_set_defaults()` in `Compress::new()`.
//! Settings must be applied in this EXACT order:
//!
//! 1. `new(ColorSpace)` - creates compressor, calls jpeg_set_defaults (JCP_MAX_COMPRESSION)
//! 2. `set_fastest_defaults()` - OPTIONAL, switches to JCP_FASTEST profile (must be early!)
//! 3. `set_size()` - image dimensions
//! 4. `set_color_space()` - output colorspace
//! 5. `set_quality()` - sets quant tables based on quality
//! 6. `set_chroma_sampling_pixel_sizes()` - subsampling
//! 7. `set_optimize_coding()` - Huffman optimization
//! 8. **`set_optimize_scans()`** - MUST come BEFORE set_progressive_mode!
//! 9. `set_progressive_mode()` - enables progressive (calls jpeg_simple_progression)
//!    ⚠️ jpeg_simple_progression uses optimize_scans flag to choose scan script
//! 10. `start_compress()` - locks settings, begins encoding
//!
//! ## mozjpeg crate limitations:
//!
//! The high-level crate does NOT expose:
//! - Direct trellis control (JBOOLEAN_TRELLIS_QUANT, JBOOLEAN_TRELLIS_QUANT_DC)
//! - Deringing control (JBOOLEAN_OVERSHOOT_DERINGING)
//! - Quant table selection (JINT_BASE_QUANT_TBL_IDX)
//!
//! Therefore:
//! - JCP_MAX_COMPRESSION (default): trellis=ON, deringing=ON, ImageMagick quant tables
//! - JCP_FASTEST (set_fastest_defaults): trellis=OFF, deringing=OFF, **Annex K quant tables**
//!
//! **IMPORTANT**: JCP_FASTEST uses JPEG Annex K quant tables (index 0), while our
//! presets consistently use ImageMagick tables (index 3). This causes ~15-25% file
//! size difference for BaselineFastest vs JCP_FASTEST. This is NOT a bug - we
//! intentionally use better tables. The mozjpeg crate doesn't expose table selection.

use mozjpeg_rs::{Encoder, Preset, Subsampling};

/// Image sizes to test (small, medium, large)
const TEST_SIZES: [(u32, u32); 3] = [
    (64, 64),   // Small: 4KB
    (256, 256), // Medium: 196KB
    (512, 512), // Large: 768KB
];

/// Configuration that can be matched between Rust and mozjpeg crate
#[derive(Debug, Clone, Copy)]
struct MatchableConfig {
    /// Use JCP_FASTEST profile (no trellis, no deringing)
    use_fastest_profile: bool,
    /// Progressive mode
    progressive: bool,
    /// Huffman optimization
    optimize_coding: bool,
    /// Scan optimization (progressive only)
    optimize_scans: bool,
}

/// Error when a preset cannot be matched by the mozjpeg crate
#[derive(Debug)]
struct PresetMatchError {
    preset: &'static str,
    reason: &'static str,
}

impl std::fmt::Display for PresetMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot match preset '{}': {}", self.preset, self.reason)
    }
}

/// Convert a Preset to a matchable mozjpeg crate configuration.
///
/// Returns an error if the preset uses settings not exposed by the mozjpeg crate.
fn preset_to_matchable_config(preset: Preset) -> Result<MatchableConfig, PresetMatchError> {
    match preset {
        Preset::BaselineFastest => {
            // BaselineFastest: no trellis, no huffman opt, no progressive, no deringing
            // Uses ImageMagick quant tables (index 3)
            //
            // JCP_FASTEST uses Annex K tables (index 0), which produce ~20% larger files.
            // The mozjpeg crate doesn't expose quant table selection, so we can't match.
            Err(PresetMatchError {
                preset: "BaselineFastest",
                reason: "JCP_FASTEST uses Annex K quant tables; mozjpeg crate doesn't expose table selection",
            })
        }
        Preset::BaselineBalanced => {
            // BaselineBalanced: trellis=ON, huffman=ON, progressive=OFF, deringing=ON
            // mozjpeg crate: JCP_MAX_COMPRESSION has trellis+deringing ON
            // But we can't do baseline (non-progressive) with trellis in mozjpeg crate
            // because set_progressive_mode() is the only way to NOT use progressive,
            // but JCP_MAX_COMPRESSION defaults to progressive.
            //
            // Actually, JCP_MAX_COMPRESSION just sets parameters - progressive mode
            // is only enabled when set_progressive_mode() is called.
            Ok(MatchableConfig {
                use_fastest_profile: false, // Use JCP_MAX_COMPRESSION for trellis
                progressive: false,
                optimize_coding: true,
                optimize_scans: false,
            })
        }
        Preset::ProgressiveBalanced => {
            // ProgressiveBalanced: trellis=ON, huffman=ON, progressive=ON, optimize_scans=OFF
            Ok(MatchableConfig {
                use_fastest_profile: false,
                progressive: true,
                optimize_coding: true,
                optimize_scans: false,
            })
        }
        Preset::ProgressiveSmallest => {
            // ProgressiveSmallest: trellis=ON, huffman=ON, progressive=ON, optimize_scans=ON
            Ok(MatchableConfig {
                use_fastest_profile: false,
                progressive: true,
                optimize_coding: true,
                optimize_scans: true,
            })
        }
    }
}

/// Encode with mozjpeg crate using matchable configuration
fn encode_with_mozjpeg_crate(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    config: &MatchableConfig,
) -> Vec<u8> {
    use mozjpeg::ColorSpace;

    let mut comp = mozjpeg::Compress::new(ColorSpace::JCS_RGB);

    // Step 1: Profile selection (must be early!)
    if config.use_fastest_profile {
        comp.set_fastest_defaults();
    }
    // else: keep JCP_MAX_COMPRESSION defaults (trellis enabled)

    // Step 2: Image setup
    comp.set_size(width as usize, height as usize);
    comp.set_color_space(ColorSpace::JCS_YCbCr);

    // Step 3: Quality (sets quant tables)
    comp.set_quality(quality as f32);

    // Step 4: Subsampling (4:2:0)
    comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2));

    // Step 5: Huffman optimization
    comp.set_optimize_coding(config.optimize_coding);

    // Step 6: Scan optimization
    // MUST set optimize_scans BEFORE set_progressive_mode!
    // JCP_MAX_COMPRESSION enables optimize_scans by default, and
    // jpeg_simple_progression() uses this flag to choose the scan script.
    comp.set_optimize_scans(config.optimize_scans);

    // Step 7: Progressive mode (must come AFTER optimize_scans)
    if config.progressive {
        comp.set_progressive_mode();
    }

    // Step 8: Start compression
    let mut comp = comp
        .start_compress(Vec::new())
        .expect("start_compress failed");

    // Write all scanlines
    comp.write_scanlines(rgb).expect("write_scanlines failed");

    // Finish and return
    comp.finish().expect("finish failed")
}

/// Count SOS markers (scans) in a JPEG
fn count_scans(data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < data.len().saturating_sub(1) {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            count += 1;
        }
        i += 1;
    }
    count
}

/// Create a gradient test image
fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let noise = ((x * 7 + y * 13) % 50) as u8;
            rgb[idx] = ((x * 255 / width.max(1)) as u8).saturating_add(noise);
            rgb[idx + 1] = ((y * 255 / height.max(1)) as u8).saturating_add(noise);
            rgb[idx + 2] = (((x + y) * 255 / (width + height).max(1)) as u8).saturating_add(noise);
        }
    }
    rgb
}

/// Test all presets against mozjpeg crate at multiple sizes
#[test]
fn test_preset_parity_all_sizes() {
    println!("\n=== Preset Parity Test (vs mozjpeg crate) ===\n");

    let presets = [
        ("BaselineFastest", Preset::BaselineFastest),
        ("BaselineBalanced", Preset::BaselineBalanced),
        ("ProgressiveBalanced", Preset::ProgressiveBalanced),
        ("ProgressiveSmallest", Preset::ProgressiveSmallest),
    ];

    println!(
        "| {:^20} | {:^10} | {:^8} | {:^10} | {:^10} | {:^8} |",
        "Preset", "Size", "Quality", "Rust (B)", "C (B)", "Diff %"
    );
    println!(
        "|{:-<22}|{:-<12}|{:-<10}|{:-<12}|{:-<12}|{:-<10}|",
        "", "", "", "", "", ""
    );

    for (name, preset) in &presets {
        let config = match preset_to_matchable_config(*preset) {
            Ok(c) => c,
            Err(e) => {
                println!("| {:^20} | SKIPPED: {} |", name, e.reason);
                continue;
            }
        };

        for &(width, height) in &TEST_SIZES {
            let rgb = create_test_image(width, height);
            let quality = 75u8;

            // Encode with Rust
            let rust_jpeg = Encoder::new(*preset)
                .quality(quality)
                .subsampling(Subsampling::S420)
                .encode_rgb(&rgb, width, height)
                .unwrap();

            // Encode with mozjpeg crate
            let c_jpeg = encode_with_mozjpeg_crate(&rgb, width, height, quality, &config);

            let diff_pct =
                ((rust_jpeg.len() as f64 - c_jpeg.len() as f64) / c_jpeg.len() as f64) * 100.0;

            let rust_scans = count_scans(&rust_jpeg);
            let c_scans = count_scans(&c_jpeg);

            println!(
                "| {:^20} | {:>4}x{:<4} | {:^8} | {:>10} | {:>10} | {:>+7.2}% | R:{} C:{} |",
                name,
                width,
                height,
                quality,
                rust_jpeg.len(),
                c_jpeg.len(),
                diff_pct,
                rust_scans,
                c_scans
            );

            // Both should produce valid JPEGs
            let mut decoder = jpeg_decoder::Decoder::new(&rust_jpeg[..]);
            decoder.decode().expect("Rust JPEG should decode");

            let mut decoder = jpeg_decoder::Decoder::new(&c_jpeg[..]);
            decoder.decode().expect("C JPEG should decode");

            // Assert parity thresholds based on preset
            // - BaselineBalanced & ProgressiveBalanced: <2% (exact config match)
            // - ProgressiveSmallest: <5% for large images (optimize_scans algorithm differs)
            //   Small images (64x64) have higher variance due to different scan selection
            let threshold = match (*preset, width * height) {
                (Preset::ProgressiveSmallest, pixels) if pixels < 10000 => 20.0, // Small image + optimize_scans
                (Preset::ProgressiveSmallest, _) => 5.0,
                _ => 2.0, // BaselineBalanced, ProgressiveBalanced
            };

            assert!(
                diff_pct.abs() < threshold,
                "{} at {}x{}: {:.2}% diff exceeds {:.0}% threshold (scans: R={}, C={})",
                name,
                width,
                height,
                diff_pct,
                threshold,
                rust_scans,
                c_scans
            );
        }
    }

    println!();
}

/// Test that ProgressiveSmallest closely matches mozjpeg crate with optimize_scans
#[test]
fn test_progressive_smallest_parity() {
    let config = preset_to_matchable_config(Preset::ProgressiveSmallest).unwrap();

    for &(width, height) in &TEST_SIZES {
        let rgb = create_test_image(width, height);

        for quality in [50, 75, 85, 95] {
            let rust_jpeg = Encoder::new(Preset::ProgressiveSmallest)
                .quality(quality)
                .subsampling(Subsampling::S420)
                .encode_rgb(&rgb, width, height)
                .unwrap();

            let c_jpeg = encode_with_mozjpeg_crate(&rgb, width, height, quality, &config);

            let diff_pct =
                ((rust_jpeg.len() as f64 - c_jpeg.len() as f64) / c_jpeg.len() as f64) * 100.0;

            // optimize_scans algorithms differ, so allow wider tolerance
            // Small images have more variance due to scan selection overhead
            let threshold = if width * height < 10000 { 20.0 } else { 5.0 };
            assert!(
                diff_pct.abs() < threshold,
                "ProgressiveSmallest at {}x{} Q{}: {:.2}% exceeds {:.0}%",
                width,
                height,
                quality,
                diff_pct,
                threshold
            );
        }
    }
}

/// Test that all presets produce decodable output at all sizes
#[test]
fn test_all_presets_decodable() {
    let presets = [
        Preset::BaselineFastest,
        Preset::BaselineBalanced,
        Preset::ProgressiveBalanced,
        Preset::ProgressiveSmallest,
    ];

    for preset in &presets {
        for &(width, height) in &TEST_SIZES {
            let rgb = create_test_image(width, height);

            let jpeg = Encoder::new(*preset)
                .quality(75)
                .encode_rgb(&rgb, width, height)
                .unwrap();

            let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
            decoder.decode().unwrap_or_else(|e| {
                panic!(
                    "{:?} at {}x{} failed to decode: {:?}",
                    preset, width, height, e
                )
            });
        }
    }
}
