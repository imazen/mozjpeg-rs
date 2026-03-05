//! Decoder round-trip validation test.
//!
//! Tests that JPEGs encoded by mozjpeg-rs can be decoded by all major decoders
//! and produce consistent results.
//!
//! Run with: cargo run --release --example decoder_roundtrip
//!
//! Decoders tested:
//! - jpeg-decoder (pure Rust)
//! - zune-jpeg (pure Rust, SIMD)
//! - mozjpeg-sys (C mozjpeg)
//!
//! Image dimensions tested:
//! - Around DCT block size (8): 7, 8, 9
//! - Around 4:2:2/4:2:0 MCU (16): 15, 16, 17
//! - Larger boundaries: 31, 32, 33, 63, 64, 65
//! - Non-square: 7x17, 15x9, 17x31, etc.
//! - Prime dimensions: 73, 97, 127

use mozjpeg_rs::{Encoder, Preset, Subsampling};
use std::time::Instant;

/// Decoded image result from a decoder.
struct DecodedImage {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

/// Decode using jpeg-decoder crate.
fn decode_jpeg_decoder(jpeg: &[u8]) -> Result<DecodedImage, String> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    let pixels = decoder
        .decode()
        .map_err(|e| format!("jpeg-decoder: {}", e))?;
    let info = decoder.info().ok_or("jpeg-decoder: no info")?;
    Ok(DecodedImage {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
    })
}

/// Decode using zune-jpeg crate.
fn decode_zune_jpeg(jpeg: &[u8]) -> Result<DecodedImage, String> {
    use zune_jpeg::JpegDecoder;

    let mut decoder = JpegDecoder::new(std::io::Cursor::new(jpeg));
    let pixels = decoder
        .decode()
        .map_err(|e| format!("zune-jpeg: {:?}", e))?;
    let info = decoder.info().ok_or("zune-jpeg: no info")?;
    Ok(DecodedImage {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
    })
}

/// Decode using mozjpeg-sys (C mozjpeg).
fn decode_mozjpeg_sys(jpeg: &[u8]) -> Result<DecodedImage, String> {
    use mozjpeg_sys::*;
    use std::ptr;

    unsafe {
        let mut cinfo: jpeg_decompress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateDecompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_decompress_struct>(),
        );

        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as libc::c_ulong);

        if jpeg_read_header(&mut cinfo, 1) != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg-sys: failed to read header".to_string());
        }

        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;

        if jpeg_start_decompress(&mut cinfo) != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg-sys: failed to start decompress".to_string());
        }

        let width = cinfo.output_width as usize;
        let height = cinfo.output_height as usize;
        let row_stride = width * 3;
        let mut pixels = vec![0u8; height * row_stride];

        let mut row_pointer: [*mut u8; 1] = [ptr::null_mut()];
        while cinfo.output_scanline < cinfo.output_height {
            let offset = cinfo.output_scanline as usize * row_stride;
            row_pointer[0] = pixels.as_mut_ptr().add(offset);
            jpeg_read_scanlines(&mut cinfo, row_pointer.as_mut_ptr(), 1);
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);

        Ok(DecodedImage {
            pixels,
            width,
            height,
        })
    }
}

/// Compare two decoded images, returning max pixel difference.
fn compare_images(a: &DecodedImage, b: &DecodedImage) -> Result<u8, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "dimension mismatch: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        ));
    }
    if a.pixels.len() != b.pixels.len() {
        return Err(format!(
            "pixel count mismatch: {} vs {}",
            a.pixels.len(),
            b.pixels.len()
        ));
    }

    let max_diff = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    Ok(max_diff)
}

/// Test configuration.
#[derive(Debug, Clone)]
struct TestConfig {
    preset: Preset,
    subsampling: Subsampling,
    quality: u8,
}

impl std::fmt::Display for TestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let preset_name = match self.preset {
            Preset::BaselineFastest => "BaselineFastest",
            Preset::BaselineBalanced => "BaselineBalanced",
            Preset::ProgressiveBalanced => "ProgressiveBalanced",
            Preset::ProgressiveSmallest => "ProgressiveSmallest",
        };
        let subsampling_name = match self.subsampling {
            Subsampling::S444 => "4:4:4",
            Subsampling::S422 => "4:2:2",
            Subsampling::S420 => "4:2:0",
            Subsampling::S440 => "4:4:0",
            Subsampling::Gray => "Gray",
        };
        write!(f, "{} {} Q{}", preset_name, subsampling_name, self.quality)
    }
}

/// Generate all test configurations.
fn generate_configs() -> Vec<TestConfig> {
    let presets = [
        Preset::BaselineFastest,
        Preset::BaselineBalanced,
        Preset::ProgressiveBalanced,
        Preset::ProgressiveSmallest,
    ];
    let subsamplings = [Subsampling::S444, Subsampling::S422, Subsampling::S420];
    let qualities = [50, 60, 70, 75, 80, 85, 90, 95];

    let mut configs = Vec::new();
    for &preset in &presets {
        for &subsampling in &subsamplings {
            for &quality in &qualities {
                configs.push(TestConfig {
                    preset,
                    subsampling,
                    quality,
                });
            }
        }
    }
    configs
}

/// Test dimension pair (width, height).
#[derive(Debug, Clone, Copy)]
struct TestDimension {
    width: u32,
    height: u32,
    description: &'static str,
}

impl std::fmt::Display for TestDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} ({})", self.width, self.height, self.description)
    }
}

/// Generate test dimensions covering edge cases around DCT/MCU boundaries.
///
/// Key boundaries:
/// - DCT block size: 8x8
/// - 4:4:4 MCU: 8x8
/// - 4:2:2 MCU: 16x8
/// - 4:2:0 MCU: 16x16
/// - 4:4:0 MCU: 8x16
fn generate_test_dimensions() -> Vec<TestDimension> {
    vec![
        // Around DCT block size (8)
        TestDimension {
            width: 7,
            height: 7,
            description: "below DCT",
        },
        TestDimension {
            width: 8,
            height: 8,
            description: "exact DCT",
        },
        TestDimension {
            width: 9,
            height: 9,
            description: "above DCT",
        },
        // Around 4:2:2/4:2:0 MCU width (16)
        TestDimension {
            width: 15,
            height: 15,
            description: "below MCU",
        },
        TestDimension {
            width: 16,
            height: 16,
            description: "exact MCU",
        },
        TestDimension {
            width: 17,
            height: 17,
            description: "above MCU",
        },
        // Larger boundaries
        TestDimension {
            width: 31,
            height: 31,
            description: "2MCU-1",
        },
        TestDimension {
            width: 32,
            height: 32,
            description: "2MCU",
        },
        TestDimension {
            width: 33,
            height: 33,
            description: "2MCU+1",
        },
        TestDimension {
            width: 63,
            height: 63,
            description: "4MCU-1",
        },
        TestDimension {
            width: 64,
            height: 64,
            description: "4MCU",
        },
        TestDimension {
            width: 65,
            height: 65,
            description: "4MCU+1",
        },
        // Non-square: stress different MCU alignments per axis
        TestDimension {
            width: 7,
            height: 17,
            description: "non-square 7x17",
        },
        TestDimension {
            width: 17,
            height: 7,
            description: "non-square 17x7",
        },
        TestDimension {
            width: 15,
            height: 9,
            description: "non-square 15x9",
        },
        TestDimension {
            width: 9,
            height: 15,
            description: "non-square 9x15",
        },
        TestDimension {
            width: 17,
            height: 31,
            description: "non-square 17x31",
        },
        TestDimension {
            width: 31,
            height: 17,
            description: "non-square 31x17",
        },
        // Prime dimensions (no nice alignment)
        TestDimension {
            width: 73,
            height: 73,
            description: "prime 73",
        },
        TestDimension {
            width: 97,
            height: 97,
            description: "prime 97",
        },
        TestDimension {
            width: 127,
            height: 127,
            description: "prime 127",
        },
        TestDimension {
            width: 73,
            height: 97,
            description: "prime non-square",
        },
        // Original test dimension (for regression)
        TestDimension {
            width: 256,
            height: 256,
            description: "original 256",
        },
        // Very small (edge cases)
        TestDimension {
            width: 1,
            height: 1,
            description: "minimum 1x1",
        },
        TestDimension {
            width: 2,
            height: 2,
            description: "tiny 2x2",
        },
        TestDimension {
            width: 3,
            height: 5,
            description: "tiny non-square",
        },
    ]
}

/// Create a test image with gradients and patterns.
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Red: horizontal gradient
            pixels[idx] = (x * 255 / width) as u8;
            // Green: vertical gradient
            pixels[idx + 1] = (y * 255 / height) as u8;
            // Blue: checkerboard pattern
            pixels[idx + 2] = if (x / 8 + y / 8) % 2 == 0 { 200 } else { 50 };
        }
    }
    pixels
}

/// Test result with details about what was tested.
enum TestResult {
    /// All decoders passed
    AllDecoders,
    /// Test failed
    Failed(String),
}

/// Determine max allowed decoder difference for given image/config.
///
/// For very small images with chroma subsampling, decoders use different
/// upsampling algorithms which causes larger pixel differences. This is
/// expected behavior, not an encoder bug.
///
/// Investigation (see decoder_boundary_test.rs) found the exact threshold:
///
/// **The issue occurs when chroma_width == 2 (luma width 3 or 4)**
///
/// | Width | Chroma Width | 4:2:2 diff | 4:2:0 diff |
/// |-------|--------------|------------|------------|
/// | 1-2   | 1            | ≤2         | ≤4         |
/// | 3-4   | 2            | **24-28**  | **27-28**  |
/// | 5+    | 3+           | ≤3         | ≤5         |
///
/// Height subsampling (4:4:0) does NOT show this issue.
///
/// Root cause: mozjpeg uses "fancy" chroma upsampling (triangle filter)
/// while jpeg-decoder and zune-jpeg use simpler replication. When
/// chroma_width == 2, the two algorithms produce visibly different results.
fn max_allowed_diff(width: u32, _height: u32, subsampling: Subsampling) -> u8 {
    // Calculate chroma width for horizontal subsampling modes
    let chroma_width = match subsampling {
        Subsampling::S422 | Subsampling::S420 => (width + 1) / 2,
        _ => width, // No horizontal subsampling
    };

    if chroma_width == 2 {
        // Exact boundary: chroma_width == 2 causes large decoder differences
        // due to different upsampling algorithms (fancy vs simple)
        50
    } else {
        // Normal case: IDCT differences are typically ≤4, but progressive
        // with optimize_scans can show up to ~16 due to different coefficient
        // ordering interpretations. Windows ARM64 shows up to 18 due to
        // platform-specific decoder rounding differences.
        20
    }
}

/// Run a single test configuration.
fn run_test(config: &TestConfig, pixels: &[u8], width: u32, height: u32) -> TestResult {
    // Encode
    let jpeg = match Encoder::new(config.preset)
        .quality(config.quality)
        .subsampling(config.subsampling)
        .encode_rgb(pixels, width, height)
    {
        Ok(j) => j,
        Err(e) => return TestResult::Failed(format!("encode failed: {}", e)),
    };

    // Decode with all decoders
    let jpeg_decoder_result = match decode_jpeg_decoder(&jpeg) {
        Ok(r) => r,
        Err(e) => return TestResult::Failed(e),
    };
    let zune_result = match decode_zune_jpeg(&jpeg) {
        Ok(r) => r,
        Err(e) => return TestResult::Failed(e),
    };
    let mozjpeg_result = match decode_mozjpeg_sys(&jpeg) {
        Ok(r) => r,
        Err(e) => return TestResult::Failed(e),
    };

    // Compare decoder outputs (they should be identical or nearly identical)
    let diff_jd_zune = match compare_images(&jpeg_decoder_result, &zune_result) {
        Ok(d) => d,
        Err(e) => return TestResult::Failed(e),
    };
    let diff_jd_moz = match compare_images(&jpeg_decoder_result, &mozjpeg_result) {
        Ok(d) => d,
        Err(e) => return TestResult::Failed(e),
    };
    let diff_zune_moz = match compare_images(&zune_result, &mozjpeg_result) {
        Ok(d) => d,
        Err(e) => return TestResult::Failed(e),
    };

    let allowed = max_allowed_diff(width, height, config.subsampling);
    let max_diff = diff_jd_zune.max(diff_jd_moz).max(diff_zune_moz);
    if max_diff > allowed {
        return TestResult::Failed(format!(
            "decoder diff {} exceeds {} (jd-zune:{}, jd-moz:{}, zune-moz:{})",
            max_diff, allowed, diff_jd_zune, diff_jd_moz, diff_zune_moz
        ));
    }

    TestResult::AllDecoders
}

/// A failure record including dimension and config info.
struct FailureInfo {
    dimension: TestDimension,
    config: TestConfig,
    error: String,
}

fn main() {
    println!("=== Decoder Round-Trip Validation ===\n");
    println!("Testing that mozjpeg-rs output can be decoded by:");
    println!("  - jpeg-decoder (pure Rust)");
    println!("  - zune-jpeg (pure Rust, SIMD)");
    println!("  - mozjpeg-sys (C mozjpeg)\n");

    // Generate test dimensions and configurations
    let dimensions = generate_test_dimensions();
    let configs = generate_configs();
    let total_tests = dimensions.len() * configs.len();

    println!(
        "Test matrix: {} dimensions × {} configs = {} total tests\n",
        dimensions.len(),
        configs.len(),
        total_tests
    );

    println!("Dimensions:");
    for dim in &dimensions {
        println!("  - {}", dim);
    }
    println!();

    let start = Instant::now();
    let mut passed = 0;
    let mut failed = 0;
    let mut failures: Vec<FailureInfo> = Vec::new();

    // Use I/O for progress
    use std::io::Write;

    for (dim_idx, dim) in dimensions.iter().enumerate() {
        // Create test image for this dimension
        let pixels = create_test_image(dim.width as usize, dim.height as usize);

        print!("\n[{}/{}] {} ", dim_idx + 1, dimensions.len(), dim);
        std::io::stdout().flush().unwrap();

        for config in &configs {
            match run_test(config, &pixels, dim.width, dim.height) {
                TestResult::AllDecoders => {
                    passed += 1;
                    print!(".");
                }
                TestResult::Failed(e) => {
                    failed += 1;
                    print!("F");
                    failures.push(FailureInfo {
                        dimension: *dim,
                        config: config.clone(),
                        error: e,
                    });
                }
            }
            std::io::stdout().flush().unwrap();
        }
    }

    let elapsed = start.elapsed();
    println!("\n");

    // Print results
    if failures.is_empty() {
        println!(
            "PASSED: All {} tests decoded successfully with all 3 decoders",
            passed
        );
    } else {
        println!("FAILED: {} passed, {} failed\n", passed, failed);
        println!("Failures:");
        for f in &failures {
            println!("  {} @ {} - {}", f.config, f.dimension, f.error);
        }
    }

    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());

    // Exit with error if any failures
    if !failures.is_empty() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to run a test with given dimensions and config.
    fn assert_roundtrip(width: u32, height: u32, config: &TestConfig) {
        let pixels = create_test_image(width as usize, height as usize);
        match run_test(config, &pixels, width, height) {
            TestResult::Failed(e) => {
                panic!("{}x{} {} failed: {}", width, height, config, e)
            }
            TestResult::AllDecoders => {}
        }
    }

    #[test]
    fn test_decoder_roundtrip_baseline() {
        let config = TestConfig {
            preset: Preset::BaselineBalanced,
            subsampling: Subsampling::S420,
            quality: 75,
        };
        assert_roundtrip(64, 64, &config);
    }

    #[test]
    fn test_decoder_roundtrip_progressive() {
        let config = TestConfig {
            preset: Preset::ProgressiveSmallest,
            subsampling: Subsampling::S444,
            quality: 90,
        };
        assert_roundtrip(64, 64, &config);
    }

    /// Tests that ProgressiveBalanced Q90+ works with all decoders.
    /// This was previously a known issue (GitHub #2) - now fixed.
    #[test]
    fn test_progressive_balanced_high_quality() {
        let config = TestConfig {
            preset: Preset::ProgressiveBalanced,
            subsampling: Subsampling::S420,
            quality: 95,
        };
        assert_roundtrip(64, 64, &config);
        assert_roundtrip(256, 256, &config);
    }

    /// Test minimum dimension (1x1).
    #[test]
    fn test_minimum_dimension() {
        let config = TestConfig {
            preset: Preset::BaselineBalanced,
            subsampling: Subsampling::S444,
            quality: 75,
        };
        assert_roundtrip(1, 1, &config);
    }

    /// Test dimensions just below DCT block size (8).
    #[test]
    fn test_below_dct_block() {
        let config = TestConfig {
            preset: Preset::ProgressiveBalanced,
            subsampling: Subsampling::S420,
            quality: 85,
        };
        assert_roundtrip(7, 7, &config);
        assert_roundtrip(7, 9, &config); // non-square
    }

    /// Test dimensions just below MCU size (16 for 4:2:0).
    #[test]
    fn test_below_mcu_420() {
        let config = TestConfig {
            preset: Preset::ProgressiveBalanced,
            subsampling: Subsampling::S420,
            quality: 85,
        };
        assert_roundtrip(15, 15, &config);
        assert_roundtrip(17, 15, &config); // non-square across MCU
    }

    /// Test prime dimensions (no nice alignment).
    #[test]
    fn test_prime_dimensions() {
        let config = TestConfig {
            preset: Preset::ProgressiveSmallest,
            subsampling: Subsampling::S422,
            quality: 80,
        };
        assert_roundtrip(73, 97, &config);
    }

    /// Run all presets with a challenging non-aligned dimension.
    #[test]
    fn test_all_presets_non_aligned() {
        let presets = [
            Preset::BaselineFastest,
            Preset::BaselineBalanced,
            Preset::ProgressiveBalanced,
            Preset::ProgressiveSmallest,
        ];
        for preset in presets {
            let config = TestConfig {
                preset,
                subsampling: Subsampling::S420,
                quality: 90,
            };
            assert_roundtrip(17, 31, &config);
        }
    }
}
