//! # mozjpeg-oxide
//!
//! Pure Rust JPEG encoder based on Mozilla's [mozjpeg](https://github.com/mozilla/mozjpeg).
//!
//! This library provides a high-quality JPEG encoder with mozjpeg's advanced compression features:
//!
//! - **Trellis quantization** - Rate-distortion optimized coefficient selection
//! - **Progressive JPEG** - Multi-scan encoding with DC-first, AC-band progression
//! - **Huffman optimization** - 2-pass encoding for optimal entropy coding
//! - **Overshoot deringing** - Reduces ringing artifacts near hard edges
//!
//! ## Quick Start
//!
//! The [`Encoder`] struct is the main entry point for encoding images:
//!
//! ```no_run
//! use mozjpeg_oxide::{Encoder, Subsampling};
//!
//! # fn main() -> Result<(), mozjpeg_oxide::Error> {
//! // RGB pixel data (3 bytes per pixel, row-major order)
//! let rgb_pixels: Vec<u8> = vec![0; 640 * 480 * 3];
//! let width = 640;
//! let height = 480;
//!
//! // Default encoding with trellis quantization and Huffman optimization
//! let jpeg_data = Encoder::new()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, width, height)?;
//!
//! // Maximum compression (progressive + trellis + scan optimization)
//! let jpeg_data = Encoder::max_compression()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, width, height)?;
//!
//! // Fastest encoding (no optimizations, libjpeg-turbo compatible)
//! let jpeg_data = Encoder::fastest()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, width, height)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Encoder Presets
//!
//! | Preset | Progressive | Trellis | Huffman Opt | Use Case |
//! |--------|------------|---------|-------------|----------|
//! | [`Encoder::new()`] | No | Yes | Yes | Default, good balance |
//! | [`Encoder::max_compression()`] | Yes | Yes | Yes | Smallest files |
//! | [`Encoder::fastest()`] | No | No | No | Maximum speed |
//!
//! ## Advanced Configuration
//!
//! ```no_run
//! use mozjpeg_oxide::{Encoder, Subsampling, TrellisConfig, QuantTableIdx};
//!
//! # fn main() -> Result<(), mozjpeg_oxide::Error> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! let encoder = Encoder::new()
//!     .quality(75)
//!     .progressive(true)
//!     .subsampling(Subsampling::S420)      // 4:2:0 chroma subsampling
//!     .quant_tables(QuantTableIdx::Flat)   // Flat quantization tables
//!     .trellis(TrellisConfig::default())   // Enable trellis quantization
//!     .optimize_huffman(true)              // 2-pass Huffman optimization
//!     .overshoot_deringing(true);          // Reduce ringing at edges
//!
//! let jpeg_data = encoder.encode_rgb(&rgb_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Low-Level API
//!
//! For advanced use cases (custom encoding pipelines, research, etc.), this crate
//! also exposes the underlying components: DCT transforms, quantization, color
//! conversion, Huffman encoding, and more. See the individual module documentation
//! for details.
//!
//! Most users should use the [`Encoder`] API instead.

// Enforce no unsafe code in this crate, except where explicitly allowed.
// SIMD intrinsics in dct.rs and test FFI code are the only exceptions.
#![deny(unsafe_code)]

// Public modules - available for advanced use cases
pub mod bitstream;
pub mod color;
pub mod consts;
pub mod dct;
pub mod deringing;
pub mod encode;
pub mod entropy;
pub mod error;
pub mod huffman;
pub mod marker;
pub mod progressive;
pub mod quant;
pub mod sample;
pub mod scan_optimize;
pub mod simd;
pub mod trellis;
pub mod types;

/// Test encoder module for comparing Rust vs C implementations.
/// Hidden from public API but available for tests.
#[doc(hidden)]
pub mod test_encoder;

/// Corpus utilities for locating test images.
/// Hidden from public API but available for tests and examples.
#[doc(hidden)]
pub mod corpus;

// ============================================================================
// Primary API - What most users need
// ============================================================================

/// The main JPEG encoder. See [module documentation](crate) for usage examples.
pub use encode::Encoder;

/// Error types for encoding operations.
pub use error::{Error, Result};

/// Image configuration types.
pub use types::{Subsampling, TrellisConfig};

/// Quantization table variants (ImageMagick, Flat, JPEG Annex K, etc.)
pub use consts::QuantTableIdx;

// ============================================================================
// Secondary API - Additional types that may be useful
// ============================================================================

pub use types::{
    ColorSpace, ComponentInfo, CompressionProfile, DctBlock, DctMethod, FloatBlock, HuffmanTable,
    QuantTable, SampleBlock, ScanInfo,
};

pub use consts::{
    DCTSIZE, DCTSIZE2, JPEG_DHT, JPEG_DQT, JPEG_EOI, JPEG_SOF0, JPEG_SOF2, JPEG_SOI, JPEG_SOS,
    STD_CHROMINANCE_QUANT_TBL, STD_LUMINANCE_QUANT_TBL,
};

// ============================================================================
// Low-Level API - For advanced/custom encoding pipelines
// ============================================================================

pub use quant::{
    create_quant_table, create_quant_tables, dequantize_block, dequantize_coef,
    get_chrominance_quant_table, get_luminance_quant_table, quality_to_scale_factor,
    quality_to_scale_factor_f32, quantize_block, quantize_coef,
};

pub use dct::{
    forward_dct, forward_dct_8x8, forward_dct_8x8_simd, forward_dct_8x8_transpose,
    forward_dct_with_deringing, level_shift,
};

pub use color::{
    cmyk_to_ycck, convert_block_rgb_to_ycbcr, convert_rgb_to_gray, convert_rgb_to_ycbcr,
    rgb_to_gray, rgb_to_ycbcr,
};

pub use huffman::{
    generate_optimal_table, DerivedTable, FrequencyCounter, HuffTable, MAX_CODE_LENGTH,
    NUM_HUFF_TBLS,
};

pub use bitstream::{BitWriter, VecBitWriter};

pub use entropy::{
    encode_block_standalone, jpeg_nbits, EntropyEncoder, ProgressiveEncoder,
    ProgressiveSymbolCounter, SymbolCounter,
};

pub use sample::{
    downsample_h2v1_row, downsample_h2v2_rows, downsample_plane, expand_to_mcu,
    mcu_aligned_dimensions, subsampled_dimensions,
};

pub use trellis::{
    compute_block_eob_info, optimize_eob_runs, simple_quantize_block, trellis_quantize_block,
    trellis_quantize_block_with_eob_info, BlockEobInfo,
};

pub use deringing::preprocess_deringing;

pub use progressive::{
    generate_baseline_scan, generate_optimized_progressive_scans,
    generate_simple_progressive_scans, generate_standard_progressive_scans, is_progressive_script,
    validate_scan_script,
};

pub use marker::MarkerWriter;
