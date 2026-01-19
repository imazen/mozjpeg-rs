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
//! use mozjpeg_rs::{Encoder, Preset};
//!
//! # fn main() -> Result<(), mozjpeg_rs::Error> {
//! // RGB pixel data (3 bytes per pixel, row-major order)
//! let rgb_pixels: Vec<u8> = vec![0; 640 * 480 * 3];
//!
//! // Default: progressive with all optimizations (recommended)
//! let jpeg_data = Encoder::new(Preset::default())
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
//!
//! // Fastest encoding (no optimizations)
//! let jpeg_data = Encoder::new(Preset::BaselineFastest)
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
//!
//! // Maximum compression (matches C mozjpeg)
//! let jpeg_data = Encoder::new(Preset::ProgressiveSmallest)
//!     .quality(85)
//!     .encode_rgb(&rgb_pixels, 640, 480)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Encoder Presets
//!
//! | Preset | Time | Size | Use Case |
//! |--------|------|------|----------|
//! | [`Preset::BaselineFastest`] | ~2ms | baseline | Real-time, thumbnails |
//! | [`Preset::BaselineBalanced`] | ~7ms | -13% | Sequential playback |
//! | [`Preset::ProgressiveBalanced`] | ~9ms | -13% | Web images (default) |
//! | [`Preset::ProgressiveSmallest`] | ~21ms | -14% | Storage, archival |
//!
//! *Benchmarks: 512×512 Q75 image*
//!
//! ## Advanced Configuration
//!
//! ```no_run
//! use mozjpeg_rs::{Encoder, Preset, Subsampling, TrellisConfig, QuantTableIdx};
//!
//! # fn main() -> Result<(), mozjpeg_rs::Error> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! let jpeg_data = Encoder::new(Preset::BaselineBalanced)
//!     .quality(75)
//!     .progressive(true)                    // Override to progressive
//!     .subsampling(Subsampling::S420)       // 4:2:0 chroma subsampling
//!     .quant_tables(QuantTableIdx::Flat)    // Flat quantization tables
//!     .trellis(TrellisConfig::default())    // Enable trellis quantization
//!     .optimize_huffman(true)               // 2-pass Huffman optimization
//!     .overshoot_deringing(true)            // Reduce ringing at edges
//!     .encode_rgb(&rgb_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Grayscale Encoding
//!
//! ```no_run
//! use mozjpeg_rs::{Encoder, Preset};
//!
//! # fn main() -> Result<(), mozjpeg_rs::Error> {
//! let gray_pixels: Vec<u8> = vec![128; 100 * 100]; // 1 byte per pixel
//!
//! let jpeg_data = Encoder::new(Preset::default())
//!     .quality(85)
//!     .encode_gray(&gray_pixels, 100, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Writing to a File or Stream
//!
//! ```no_run
//! use mozjpeg_rs::{Encoder, Preset};
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! let mut file = File::create("output.jpg")?;
//!
//! Encoder::new(Preset::default())
//!     .quality(85)
//!     .encode_rgb_to_writer(&rgb_pixels, 100, 100, &mut file)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Metadata
//!
//! ```no_run
//! use mozjpeg_rs::{Encoder, Preset, PixelDensity};
//!
//! # fn main() -> Result<(), mozjpeg_rs::Error> {
//! # let rgb_pixels: Vec<u8> = vec![0; 100 * 100 * 3];
//! # let exif_bytes: Vec<u8> = vec![];
//! # let icc_profile: Vec<u8> = vec![];
//! let jpeg_data = Encoder::new(Preset::default())
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

/// AVX2-optimized color conversion (internal).
#[doc(hidden)]
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod color_avx2;

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

/// Fast entropy encoding - jpegli-rs style (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod fast_entropy;

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

/// Input smoothing filter (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod smooth;

/// Scan optimization (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod scan_optimize;

/// Sequential scan trial encoding (internal).
#[doc(hidden)]
#[allow(dead_code)]
pub mod scan_trial;

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

// Optional mozjpeg-sys configuration layer
#[cfg(feature = "mozjpeg-sys-config")]
pub mod compat;

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
/// Use [`Encoder::new(preset)`](Encoder::new) with a [`Preset`] to choose your settings:
///
/// - [`Preset::ProgressiveBalanced`] - Progressive with all optimizations (default)
/// - [`Preset::BaselineBalanced`] - Baseline with all optimizations
/// - [`Preset::BaselineFastest`] - No optimizations, maximum speed
/// - [`Preset::ProgressiveSmallest`] - Maximum compression (matches C mozjpeg)
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// let pixels: Vec<u8> = vec![0; 640 * 480 * 3];
///
/// let jpeg = Encoder::new(Preset::default())
///     .quality(85)
///     .encode_rgb(&pixels, 640, 480)?;
/// # Ok(())
/// # }
/// ```
pub use encode::{Encode, Encoder, EncodingStream, StreamingEncoder};

/// Error type for encoding operations.
///
/// All encoding errors are represented by this type. Use the [`Error`] variants
/// to determine the specific failure mode.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Error, Preset};
///
/// # fn example() {
/// let result = Encoder::new(Preset::default()).encode_rgb(&[], 0, 0);
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
/// Equivalent to `std::result::Result<T, mozjpeg_rs::Error>`.
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
/// use mozjpeg_rs::{Encoder, Preset, Subsampling};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // High quality - no color subsampling
/// let jpeg = Encoder::new(Preset::default())
///     .subsampling(Subsampling::S444)
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Smaller files - standard 4:2:0 subsampling
/// let jpeg = Encoder::new(Preset::default())
///     .subsampling(Subsampling::S420)
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::Subsampling;

/// Encoder preset controlling compression mode and optimization level.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // Default: progressive with good balance
/// let jpeg = Encoder::new(Preset::default())
///     .quality(85)
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Fastest encoding
/// let jpeg = Encoder::new(Preset::BaselineFastest)
///     .quality(80)
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Maximum compression (matches C mozjpeg)
/// let jpeg = Encoder::new(Preset::ProgressiveSmallest)
///     .quality(85)
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::Preset;

/// Pixel density for JFIF metadata.
///
/// Specifies the physical resolution or aspect ratio of the image.
/// Most software ignores JFIF density in favor of EXIF metadata.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset, PixelDensity};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // 300 DPI for print
/// let jpeg = Encoder::new(Preset::default())
///     .pixel_density(PixelDensity::dpi(300, 300))
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // 2:1 pixel aspect ratio
/// let jpeg = Encoder::new(Preset::default())
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
/// use mozjpeg_rs::{Encoder, Preset, TrellisConfig};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// // Default trellis settings (enabled by default with most presets)
/// let jpeg = Encoder::new(Preset::default())
///     .trellis(TrellisConfig::default())
///     .encode_rgb(&pixels, 100, 100)?;
///
/// // Disable trellis for faster encoding
/// let jpeg = Encoder::new(Preset::default())
///     .trellis(TrellisConfig::disabled())
///     .encode_rgb(&pixels, 100, 100)?;
/// # Ok(())
/// # }
/// ```
pub use types::TrellisConfig;

/// Estimated resource usage for an encoding operation.
///
/// Use [`Encoder::estimate_resources()`] to predict memory and CPU requirements
/// before encoding. Useful for scheduling, resource limits, or progress feedback.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset};
///
/// let encoder = Encoder::new(Preset::ProgressiveBalanced).quality(85);
/// let estimate = encoder.estimate_resources(1920, 1080);
///
/// println!("Peak memory: {} MB", estimate.peak_memory_bytes / 1_000_000);
/// println!("CPU cost: {:.1}x baseline", estimate.cpu_cost_multiplier);
/// println!("Blocks to process: {}", estimate.block_count);
/// ```
pub use types::ResourceEstimate;

/// Resource limits for the encoder.
///
/// Use this to restrict encoding operations by dimensions, memory usage,
/// or metadata size. All limits default to 0 (disabled).
///
/// # Example
///
/// ```
/// use mozjpeg_rs::{Encoder, Preset, Limits};
///
/// // Create limits for a thumbnail service
/// let limits = Limits::default()
///     .max_width(4096)
///     .max_height(4096)
///     .max_pixel_count(16_000_000)  // 16 megapixels
///     .max_alloc_bytes(100 * 1024 * 1024);  // 100 MB
///
/// let encoder = Encoder::new(Preset::BaselineFastest)
///     .limits(limits);
/// ```
pub use types::Limits;

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
/// use mozjpeg_rs::{Encoder, Preset, QuantTableIdx};
///
/// # fn main() -> Result<(), mozjpeg_rs::Error> {
/// # let pixels: Vec<u8> = vec![0; 100 * 100 * 3];
/// let jpeg = Encoder::new(Preset::default())
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

// ============================================================================
// mozjpeg-sys compatibility (optional feature)
// ============================================================================

/// Warnings from configuring a C mozjpeg encoder.
///
/// Some settings cannot be applied to `jpeg_compress_struct` directly
/// and must be handled separately after `jpeg_start_compress`.
#[cfg(feature = "mozjpeg-sys-config")]
pub use compat::ConfigWarnings;

/// Error configuring a C mozjpeg encoder.
#[cfg(feature = "mozjpeg-sys-config")]
pub use compat::ConfigError;

/// C mozjpeg encoder with settings from a Rust [`Encoder`].
///
/// Created via [`Encoder::to_c_mozjpeg()`]. Provides methods for encoding
/// images using the C mozjpeg library.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset};
///
/// let pixels: Vec<u8> = vec![128; 64 * 64 * 3];
/// let jpeg = Encoder::new(Preset::ProgressiveBalanced)
///     .quality(85)
///     .to_c_mozjpeg()
///     .encode_rgb(&pixels, 64, 64)?;
/// # Ok::<(), mozjpeg_rs::Error>(())
/// ```
#[cfg(feature = "mozjpeg-sys-config")]
pub use compat::CMozjpeg;
