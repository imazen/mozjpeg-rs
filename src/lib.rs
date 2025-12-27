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
//! use mozjpeg_oxide::Encoder;
//!
//! # fn main() -> Result<(), mozjpeg_oxide::Error> {
//! // RGB pixel data (3 bytes per pixel, row-major order)
//! let rgb_pixels: Vec<u8> = vec![0; 640 * 480 * 3];
//!
//! // Default encoding (trellis + Huffman optimization)
//! let jpeg_data = Encoder::new()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
//!
//! // Maximum compression (progressive + trellis + scan optimization)
//! let jpeg_data = Encoder::max_compression()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
//!
//! // Fastest encoding (no optimizations)
//! let jpeg_data = Encoder::fastest()
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
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
//! let jpeg_data = Encoder::new()
//!     .quality(75)
//!     .progressive(true)
//!     .subsampling(Subsampling::S420)      // 4:2:0 chroma subsampling
//!     .quant_tables(QuantTableIdx::Flat)   // Flat quantization tables
//!     .trellis(TrellisConfig::default())   // Enable trellis quantization
//!     .optimize_huffman(true)              // 2-pass Huffman optimization
//!     .overshoot_deringing(true)           // Reduce ringing at edges
//!     .encode_rgb(&rgb_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Grayscale Encoding
//!
//! ```no_run
//! use mozjpeg_oxide::Encoder;
//!
//! # fn main() -> Result<(), mozjpeg_oxide::Error> {
//! let gray_pixels: Vec<u8> = vec![128; 100 * 100]; // 1 byte per pixel
//!
//! let jpeg_data = Encoder::new()
//!     .quality(85)
//!     .encode_gray(&gray_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Writing to a File or Stream
//!
//! ```no_run
//! use mozjpeg_oxide::Encoder;
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! let mut file = File::create("output.jpg")?;
//!
//! Encoder::new()
//!     .quality(85)
//!     .encode_rgb_to_writer(&rgb_pixels, 100, 100, &mut file)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Metadata
//!
//! ```no_run
//! use mozjpeg_oxide::{Encoder, PixelDensity};
//!
//! # fn main() -> Result<(), mozjpeg_oxide::Error> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! # let exif_bytes: Vec<u8> = vec![];
//! # let icc_profile: Vec<u8> = vec![];
//! let jpeg_data = Encoder::new()
//!     .quality(85)
//!     .pixel_density(PixelDensity::dpi(300, 300))  // 300 DPI
//!     .exif_data(exif_bytes)                        // EXIF metadata
//!     .icc_profile(icc_profile)                     // Color profile
//!     .encode_rgb(&rgb_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```

// Enforce no unsafe code in this crate, except where explicitly allowed.
// SIMD intrinsics in dct.rs and test FFI code are the only exceptions.
#![deny(unsafe_code)]
#![warn(missing_docs)]

// ============================================================================
// Internal modules - hidden from public docs but accessible for tests
// ============================================================================
// These modules contain internal implementation details. They are exposed
// for testing and advanced use cases but are not part of the stable API.

/// Bitstream writing utilities (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod bitstream;

/// Color conversion utilities (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod color;

/// Constants and quantization tables (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod consts;

/// DCT transform (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod dct;

/// Overshoot deringing (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod deringing;

/// Entropy encoding (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod entropy;

/// Huffman table utilities (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod huffman;

/// JPEG marker writing (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod marker;

/// Progressive scan generation (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod progressive;

/// Quantization utilities (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod quant;

/// Chroma subsampling (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod sample;

/// Scan optimization (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod scan_optimize;

/// SIMD acceleration (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod simd;

/// Trellis quantization (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod trellis;

/// Type definitions (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod types;

// Main encoder module (not hidden)
mod encode;
mod error;

// ============================================================================
// Test support modules - hidden from public API
// ============================================================================

/// Test encoder module for comparing Rust vs C implementations.
#[doc(hidden)]
pub mod test_encoder;

/// Corpus utilities for locating test images.
#[doc(hidden)]
pub mod corpus;

// ============================================================================
// Public API
// ============================================================================

/// The main JPEG encoder.
///
/// Use the builder pattern to configure encoding options, then call
/// [`encode_rgb()`](Encoder::encode_rgb) or [`encode_gray()`](Encoder::encode_gray)
/// to produce JPEG data.
///
/// # Presets
///
/// - [`Encoder::new()`] - Default settings with trellis quantization and Huffman optimization
/// - [`Encoder::max_compression()`] - Progressive mode with all optimizations for smallest files
/// - [`Encoder::fastest()`] - No optimizations, maximum encoding speed
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::Encoder;
///
/// # fn main() -> Result<(), mozjpeg_oxide::Error> {
/// let pixels: Vec<u8> = vec![0; 640 * 480 * 3];
///
/// let jpeg = Encoder::new()
///     .quality(85)
///     .encode_rgb(&pixels, 640, 480)?;
/// # Ok(())
/// # }
/// ```
pub use encode::Encoder;

/// Error type for encoding operations.
///
/// All encoding errors are represented by this type. Use the [`Error`] variants
/// to determine the specific failure mode.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::{Encoder, Error};
///
/// # fn example() {
/// let result = Encoder::new().encode_rgb(&[], 0, 0);
/// match result {
///     Ok(data) => println!("Encoded {} bytes", data.len()),
///     Err(Error::InvalidDimensions { width, height }) => {
///         eprintln!("Invalid dimensions: {}x{}", width, height);
///     }
///     Err(e) => eprintln!("Encoding failed: {}", e),
/// }
/// # }
/// ```
pub use error::Error;

/// Result type alias for encoding operations.
///
/// Equivalent to `std::result::Result<T, mozjpeg_oxide::Error>`.
pub use error::Result;

/// Chroma subsampling mode.
///
/// Controls how color information is stored relative to luminance.
/// Lower subsampling ratios reduce file size but may cause color artifacts.
///
/// | Mode | Ratio | Description |
/// |------|-------|-------------|
/// | [`S444`](Subsampling::S444) | 4:4:4 | No subsampling (highest quality) |
/// | [`S422`](Subsampling::S422) | 4:2:2 | Horizontal subsampling |
/// | [`S420`](Subsampling::S420) | 4:2:0 | Both directions (most common) |
/// | [`S440`](Subsampling::S440) | 4:4:0 | Vertical subsampling only |
/// | [`Gray`](Subsampling::Gray) | N/A | Grayscale (1 component) |
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::{Encoder, Subsampling};
///
/// # fn main() -> Result<(), mozjpeg_oxide::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // High quality - no color subsampling
/// let jpeg = Encoder::new()
///     .subsampling(Subsampling::S444)
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Smaller files - standard 4:2:0 subsampling
/// let jpeg = Encoder::new()
///     .subsampling(Subsampling::S420)
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::Subsampling;

/// Pixel density for JFIF metadata.
///
/// Specifies the physical resolution or aspect ratio of the image.
/// Most software ignores JFIF density in favor of EXIF metadata.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::{Encoder, PixelDensity};
///
/// # fn main() -> Result<(), mozjpeg_oxide::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // 300 DPI for print
/// let jpeg = Encoder::new()
///     .pixel_density(PixelDensity::dpi(300, 300))
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // 2:1 pixel aspect ratio
/// let jpeg = Encoder::new()
///     .pixel_density(PixelDensity::aspect_ratio(2, 1))
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::PixelDensity;

/// Pixel density unit for JFIF metadata.
///
/// Used with [`PixelDensity`] to specify the unit of measurement.
pub use types::DensityUnit;

/// Configuration for trellis quantization.
///
/// Trellis quantization is mozjpeg's signature feature - it uses dynamic
/// programming to find the optimal quantized coefficients that minimize
/// a rate-distortion cost function.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::{Encoder, TrellisConfig};
///
/// # fn main() -> Result<(), mozjpeg_oxide::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // Default trellis settings (enabled by default with Encoder::new())
/// let jpeg = Encoder::new()
///     .trellis(TrellisConfig::default())
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Disable trellis for faster encoding
/// let jpeg = Encoder::new()
///     .trellis(TrellisConfig::disabled())
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::TrellisConfig;

/// Quantization table preset.
///
/// mozjpeg provides 9 different quantization tables optimized for different
/// use cases. The default is [`ImageMagick`](QuantTableIdx::ImageMagick).
///
/// | Preset | Description |
/// |--------|-------------|
/// | [`JpegAnnexK`](QuantTableIdx::JpegAnnexK) | JPEG standard (Annex K) |
/// | [`Flat`](QuantTableIdx::Flat) | Uniform quantization |
/// | [`MssimTuned`](QuantTableIdx::MssimTuned) | Optimized for MS-SSIM metric |
/// | [`ImageMagick`](QuantTableIdx::ImageMagick) | ImageMagick default (mozjpeg default) |
/// | [`PsnrHvsM`](QuantTableIdx::PsnrHvsM) | PSNR-HVS-M tuned |
/// | [`Klein`](QuantTableIdx::Klein) | Klein, Silverstein, Carney (1992) |
/// | [`Watson`](QuantTableIdx::Watson) | Watson, Taylor, Borthwick (1997) |
/// | [`Ahumada`](QuantTableIdx::Ahumada) | Ahumada, Watson, Peterson (1993) |
/// | [`Peterson`](QuantTableIdx::Peterson) | Peterson, Ahumada, Watson (1993) |
///
/// # Example
///
/// ```no_run
/// use mozjpeg_oxide::{Encoder, QuantTableIdx};
///
/// # fn main() -> Result<(), mozjpeg_oxide::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// let jpeg = Encoder::new()
///     .quant_tables(QuantTableIdx::Flat)
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use consts::QuantTableIdx;

/// Number of coefficients in an 8x8 DCT block (64).
///
/// Used when providing custom quantization tables via
/// [`Encoder::custom_luma_qtable()`] or [`Encoder::custom_chroma_qtable()`].
pub use consts::DCTSIZE2;
