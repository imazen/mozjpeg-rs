//! Unified test encoder interface for comparing Rust vs C implementations.
//!
//! This module provides identical interfaces for both implementations,
//! ensuring parameter parity for apples-to-apples comparison.
//!
//! # Usage
//!
//! ```ignore
//! use mozjpeg_oxide::test_encoder::{TestEncoderConfig, encode_rust, encode_c};
//!
//! let config = TestEncoderConfig::baseline();
//! let rust_jpeg = encode_rust(&rgb, width, height, &config);
//! let c_jpeg = encode_c(&rgb, width, height, &config);
//!
//! // Compare quality
//! let result = compare_quality(&rust_jpeg, &c_jpeg, &rgb, width, height);
//! assert!(result.dssim_rust_vs_c < 0.001);
//! ```
//!
//! # Important
//!
//! All tests comparing Rust to C should use this API to ensure identical settings.
//! Do NOT create ad-hoc encoder wrappers in tests.

use crate::Subsampling;

/// Configuration for test encoding - shared between Rust and C.
#[derive(Clone, Debug)]
pub struct TestEncoderConfig {
    pub quality: u8,
    pub subsampling: Subsampling,
    pub progressive: bool,
    pub optimize_huffman: bool,
    pub trellis_quant: bool,
    pub trellis_dc: bool,
    pub overshoot_deringing: bool,
    /// C mozjpeg's optimize_scans feature (tries multiple scan configurations).
    /// Rust doesn't implement this yet, so set to false for fair comparison.
    pub optimize_scans: bool,
}

impl Default for TestEncoderConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: false,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
            optimize_scans: false,
        }
    }
}

impl TestEncoderConfig {
    /// Baseline JPEG with no optimizations (for strict parity comparison).
    /// Both Rust and C should produce nearly identical output with this config.
    pub fn baseline() -> Self {
        Self::default()
    }

    /// Baseline with Huffman optimization only.
    /// This is what C mozjpeg uses with just optimize_coding=1.
    pub fn baseline_huffman_opt() -> Self {
        Self {
            optimize_huffman: true,
            ..Self::default()
        }
    }

    /// Rust's default settings (trellis + deringing + Huffman opt).
    /// Use this to test Rust's typical output quality.
    pub fn rust_defaults() -> Self {
        Self {
            optimize_huffman: true,
            trellis_quant: true,
            overshoot_deringing: true,
            ..Self::default()
        }
    }

    /// Maximum compression settings (progressive + all optimizations).
    pub fn max_compression() -> Self {
        Self {
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            overshoot_deringing: true,
            ..Self::default()
        }
    }

    /// Builder: set quality
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Builder: set subsampling
    pub fn with_subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Builder: set progressive mode
    pub fn with_progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }
}

/// Encode using Rust implementation.
pub fn encode_rust(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
    use crate::{Encoder, TrellisConfig};

    let trellis = if config.trellis_quant || config.trellis_dc {
        TrellisConfig {
            enabled: config.trellis_quant,
            dc_enabled: config.trellis_dc,
            ..TrellisConfig::default()
        }
    } else {
        TrellisConfig::disabled()
    };

    Encoder::new()
        .quality(config.quality)
        .subsampling(config.subsampling)
        .progressive(config.progressive)
        .optimize_huffman(config.optimize_huffman)
        .trellis(trellis)
        .overshoot_deringing(config.overshoot_deringing)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

/// Encode using C mozjpeg implementation via FFI.
/// Only available in unit tests (uses mozjpeg-sys dev-dependency).
/// For integration tests, use the encode_c_impl function template below.
#[cfg(test)]
pub fn encode_c(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
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

        // Set progressive mode
        if config.progressive {
            jpeg_simple_progression(&mut cinfo);
        } else {
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        }

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        // Set subsampling
        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => panic!("Gray subsampling not supported in encode_c test helper"),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Set optimization flags
        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        // Set trellis options
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

/// Compare two JPEGs by decoding and checking pixel differences.
#[cfg(test)]
pub fn compare_decoded(rust_jpeg: &[u8], c_jpeg: &[u8]) -> CompareResult {
    let rust_decoded = decode_jpeg(rust_jpeg);
    let c_decoded = decode_jpeg(c_jpeg);

    match (rust_decoded, c_decoded) {
        (Some((rust_pix, rw, rh)), Some((c_pix, cw, ch))) => {
            if (rw, rh) != (cw, ch) {
                return CompareResult {
                    dimensions_match: false,
                    ..Default::default()
                };
            }

            let mut max_diff = 0u8;
            let mut sum_diff = 0u64;
            let mut diff_count = 0usize;

            for (r, c) in rust_pix.iter().zip(c_pix.iter()) {
                let d = (*r as i16 - *c as i16).unsigned_abs() as u8;
                if d > 0 {
                    diff_count += 1;
                    sum_diff += d as u64;
                    max_diff = max_diff.max(d);
                }
            }

            let avg_diff = if diff_count > 0 {
                sum_diff as f64 / diff_count as f64
            } else {
                0.0
            };

            CompareResult {
                decode_failed: false,
                dimensions_match: true,
                total_components: rust_pix.len(),
                differing_components: diff_count,
                max_diff,
                avg_diff,
            }
        }
        _ => CompareResult {
            decode_failed: true,
            ..Default::default()
        },
    }
}

/// Decode JPEG using external decoder.
/// Note: jpeg_decoder is a dev-dependency, so this won't work in the main crate.
/// For test usage, import jpeg_decoder in your test and use this helper.
pub fn decode_with_decoder<D: JpegDecoder>(data: &[u8], decoder: &D) -> Option<(Vec<u8>, usize, usize)> {
    decoder.decode(data)
}

/// Trait for JPEG decoders (allows injecting different decoders in tests)
pub trait JpegDecoder {
    fn decode(&self, data: &[u8]) -> Option<(Vec<u8>, usize, usize)>;
}

/// Default implementation that calls jpeg_decoder (for tests)
#[cfg(test)]
fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as usize, info.height as usize))
}

#[derive(Debug, Default)]
pub struct CompareResult {
    pub decode_failed: bool,
    pub dimensions_match: bool,
    pub total_components: usize,
    pub differing_components: usize,
    pub max_diff: u8,
    pub avg_diff: f64,
}

impl CompareResult {
    /// Check if results are within acceptable tolerance for identical parameters.
    /// With truly identical parameters, we expect max_diff <= 1 (rounding only).
    pub fn is_acceptable(&self) -> bool {
        !self.decode_failed && self.dimensions_match && self.max_diff <= 1
    }
}

/// Quality comparison result with DSSIM metrics.
#[derive(Debug, Default)]
pub struct QualityResult {
    /// DSSIM of Rust output vs original (lower is better, 0 = identical)
    pub dssim_rust: f64,
    /// DSSIM of C output vs original
    pub dssim_c: f64,
    /// DSSIM between Rust and C outputs (measures encoder difference)
    pub dssim_rust_vs_c: f64,
    /// Rust JPEG file size in bytes
    pub rust_size: usize,
    /// C JPEG file size in bytes
    pub c_size: usize,
    /// Size ratio (Rust / C)
    pub size_ratio: f64,
    /// Pixel comparison details
    pub pixel_compare: CompareResult,
}

impl QualityResult {
    /// Check if quality is acceptable for identical encoder settings.
    /// Criteria:
    /// - DSSIM between outputs < 0.001 (imperceptible difference)
    /// - Size ratio within 5% (0.95 - 1.05)
    pub fn is_parity_acceptable(&self) -> bool {
        self.dssim_rust_vs_c < 0.001 && self.size_ratio > 0.95 && self.size_ratio < 1.05
    }

    /// Check if quality is acceptable with different settings (e.g., Rust with trellis).
    /// Criteria:
    /// - DSSIM between outputs < 0.001
    /// - Size ratio within 15% (0.85 - 1.15)
    pub fn is_quality_acceptable(&self) -> bool {
        self.dssim_rust_vs_c < 0.001 && self.size_ratio > 0.85 && self.size_ratio < 1.15
    }
}

/// Compare quality of two JPEGs using DSSIM.
/// Returns comprehensive quality metrics for both encoders.
#[cfg(test)]
pub fn compare_quality(
    rust_jpeg: &[u8],
    c_jpeg: &[u8],
    original_rgb: &[u8],
    width: u32,
    height: u32,
) -> QualityResult {
    use dssim::Dssim;
    use rgb::RGB8;

    let rust_decoded = decode_jpeg(rust_jpeg);
    let c_decoded = decode_jpeg(c_jpeg);

    let (rust_pix, c_pix) = match (&rust_decoded, &c_decoded) {
        (Some((r, _, _)), Some((c, _, _))) => (r, c),
        _ => {
            return QualityResult {
                pixel_compare: CompareResult {
                    decode_failed: true,
                    ..Default::default()
                },
                ..Default::default()
            }
        }
    };

    // Calculate DSSIM values
    let attr = Dssim::new();

    let orig_rgb: Vec<RGB8> = original_rgb
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = attr
        .create_image_rgb(&orig_rgb, width as usize, height as usize)
        .expect("Failed to create original image");

    let rust_rgb: Vec<RGB8> = rust_pix
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let rust_img = attr
        .create_image_rgb(&rust_rgb, width as usize, height as usize)
        .expect("Failed to create Rust image");

    let c_rgb: Vec<RGB8> = c_pix
        .chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let c_img = attr
        .create_image_rgb(&c_rgb, width as usize, height as usize)
        .expect("Failed to create C image");

    let (dssim_rust, _) = attr.compare(&orig_img, rust_img.clone());
    let (dssim_c, _) = attr.compare(&orig_img, c_img.clone());
    let (dssim_rust_vs_c, _) = attr.compare(&rust_img, c_img);

    // Pixel comparison
    let pixel_compare = compare_decoded(rust_jpeg, c_jpeg);

    QualityResult {
        dssim_rust: dssim_rust.into(),
        dssim_c: dssim_c.into(),
        dssim_rust_vs_c: dssim_rust_vs_c.into(),
        rust_size: rust_jpeg.len(),
        c_size: c_jpeg.len(),
        size_ratio: rust_jpeg.len() as f64 / c_jpeg.len() as f64,
        pixel_compare,
    }
}

/// Encode both Rust and C with identical settings and compare.
/// This is the primary function tests should use for Rust vs C comparison.
#[cfg(test)]
pub fn encode_and_compare(
    rgb: &[u8],
    width: u32,
    height: u32,
    config: &TestEncoderConfig,
) -> QualityResult {
    let rust_jpeg = encode_rust(rgb, width, height, config);
    let c_jpeg = encode_c(rgb, width, height, config);
    compare_quality(&rust_jpeg, &c_jpeg, rgb, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = ((x * 255) / width.max(1)) as u8;
                rgb[i + 1] = ((y * 255) / height.max(1)) as u8;
                rgb[i + 2] = 128;
            }
        }
        rgb
    }

    #[test]
    fn test_encode_rust_baseline() {
        let width = 16;
        let height = 16;
        let rgb = create_test_image(width, height);

        let config = TestEncoderConfig::baseline();
        let jpeg = encode_rust(&rgb, width as u32, height as u32, &config);
        assert!(!jpeg.is_empty());
        assert_eq!(jpeg[0..2], [0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_rust_vs_c_baseline_parity() {
        let width = 64;
        let height = 64;
        let rgb = create_test_image(width, height);

        // With identical baseline settings, outputs should be nearly identical
        let config = TestEncoderConfig::baseline_huffman_opt().with_quality(85);
        let result = encode_and_compare(&rgb, width as u32, height as u32, &config);

        println!("Baseline parity test:");
        println!("  Rust size: {} bytes", result.rust_size);
        println!("  C size: {} bytes", result.c_size);
        println!("  Size ratio: {:.4}", result.size_ratio);
        println!("  DSSIM Rust vs C: {:.6}", result.dssim_rust_vs_c);
        println!("  Max pixel diff: {}", result.pixel_compare.max_diff);

        assert!(
            result.is_parity_acceptable(),
            "Baseline parity failed: ratio={:.4}, dssim={:.6}",
            result.size_ratio,
            result.dssim_rust_vs_c
        );
    }

    #[test]
    fn test_config_builders() {
        let config = TestEncoderConfig::baseline()
            .with_quality(90)
            .with_subsampling(Subsampling::S444)
            .with_progressive(true);

        assert_eq!(config.quality, 90);
        assert!(matches!(config.subsampling, Subsampling::S444));
        assert!(config.progressive);
    }

    #[test]
    fn test_all_presets_encode() {
        let width = 32;
        let height = 32;
        let rgb = create_test_image(width, height);

        let presets = [
            ("baseline", TestEncoderConfig::baseline()),
            ("baseline_huffman", TestEncoderConfig::baseline_huffman_opt()),
            ("rust_defaults", TestEncoderConfig::rust_defaults()),
            ("max_compression", TestEncoderConfig::max_compression()),
        ];

        for (name, config) in presets {
            let rust_jpeg = encode_rust(&rgb, width as u32, height as u32, &config);
            let c_jpeg = encode_c(&rgb, width as u32, height as u32, &config);

            assert!(!rust_jpeg.is_empty(), "{}: Rust encoding failed", name);
            assert!(!c_jpeg.is_empty(), "{}: C encoding failed", name);

            println!("{}: Rust={} bytes, C={} bytes", name, rust_jpeg.len(), c_jpeg.len());
        }
    }
}
