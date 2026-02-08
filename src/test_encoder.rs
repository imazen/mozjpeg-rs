//! Unified test encoder interface for comparing Rust vs C implementations.
//!
//! This module provides identical interfaces for both implementations,
//! ensuring parameter parity for apples-to-apples comparison.
//!
//! # Usage
//!
//! ```ignore
//! use mozjpeg_rs::test_encoder::{TestEncoderConfig, encode_rust, encode_c};
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
    /// Force baseline compatibility (clamp quant values to 255, use 8-bit precision).
    /// C mozjpeg's jpeg_set_quality passes TRUE for this by default.
    pub force_baseline: bool,
    /// Use C mozjpeg-compatible color conversion for exact baseline parity.
    /// When enabled, uses scalar conversion matching C mozjpeg's jccolor.c.
    pub c_compat_color: bool,
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
            force_baseline: true, // Match C mozjpeg's default (jpeg_set_quality passes TRUE)
            c_compat_color: true, // C-compatible for exact parity (default)
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
#[allow(deprecated)] // Uses c_compat_color for test compatibility
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

    Encoder::baseline_optimized()
        .quality(config.quality)
        .subsampling(config.subsampling)
        .progressive(config.progressive)
        .optimize_huffman(config.optimize_huffman)
        .optimize_scans(config.optimize_scans)
        .trellis(trellis)
        .overshoot_deringing(config.overshoot_deringing)
        .force_baseline(config.force_baseline)
        .c_compat_color(config.c_compat_color)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

// encode_c() has been removed from src/ to achieve #![forbid(unsafe_code)].
// Tests and benchmarks that need C comparison implement their own encode_c().
// See tests/test_encoder.rs for the reference implementation.

// Comparison utilities removed - each test implements what it needs.
// Kept decode_with_decoder and JpegDecoder trait for flexibility.

/// Decode JPEG using external decoder.
/// Note: jpeg_decoder is a dev-dependency, so this won't work in the main crate.
/// For test usage, import jpeg_decoder in your test and use this helper.
pub fn decode_with_decoder<D: JpegDecoder>(
    data: &[u8],
    decoder: &D,
) -> Option<(Vec<u8>, usize, usize)> {
    decoder.decode(data)
}

/// Trait for JPEG decoders (allows injecting different decoders in tests)
pub trait JpegDecoder {
    fn decode(&self, data: &[u8]) -> Option<(Vec<u8>, usize, usize)>;
}
