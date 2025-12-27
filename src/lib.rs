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

// ============================================================================
// Module visibility
// ============================================================================
// Core public modules - stable API
pub mod consts;
pub mod dct;
pub mod encode;
pub mod error;
pub mod quant;
pub mod simd;
pub mod types;

// Implementation modules - hidden from docs but accessible for tests/advanced use
// These are not part of the stable API and may change between versions.
#[doc(hidden)]
pub mod bitstream;
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod deringing;
#[doc(hidden)]
pub mod entropy;
#[doc(hidden)]
pub mod huffman;
#[doc(hidden)]
pub mod marker;
#[doc(hidden)]
pub mod progressive;
#[doc(hidden)]
pub mod sample;
#[doc(hidden)]
pub mod scan_optimize;
#[doc(hidden)]
pub mod trellis;

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

/// Streaming JPEG encoder. See [`StreamingEncoder`] for details.
pub use encode::StreamingEncoder;

/// Active streaming encoding session.
pub use encode::EncodingStream;

/// Trait for batch JPEG encoding, implemented by both [`Encoder`] and [`StreamingEncoder`].
pub use encode::Encode;

/// Error types for encoding operations.
pub use error::{Error, Result};

/// Image configuration types.
pub use types::{PixelDensity, Subsampling, TrellisConfig};

/// Quantization table variants (ImageMagick, Flat, JPEG Annex K, etc.)
pub use consts::QuantTableIdx;

// ============================================================================
// Secondary API - Additional types that may be useful
// ============================================================================

/// Type aliases for block-level operations.
pub use types::{DctBlock, SampleBlock};

/// Types for custom scan scripts and advanced configuration.
pub use types::{ColorSpace, QuantTable, ScanInfo};

/// Core constants for JPEG encoding.
pub use consts::{DCTSIZE, DCTSIZE2};

// ============================================================================
// Low-Level API - For advanced/custom encoding pipelines
// ============================================================================

/// Quantization table creation and quality scaling.
pub use quant::{create_quant_table, quality_to_scale_factor};

/// Forward DCT and level shifting for custom pipelines.
pub use dct::{forward_dct_8x8, level_shift};

/// Single-pixel color conversion utilities.
pub use color::{rgb_to_gray, rgb_to_ycbcr};
