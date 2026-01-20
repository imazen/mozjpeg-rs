//! JPEG encoder pipeline.
//!
//! This module provides two encoder types:
//!
//! - [`Encoder`]: Full-featured encoder with trellis quantization, progressive mode,
//!   and Huffman optimization. Batch encoding only.
//! - [`StreamingEncoder`]: Streaming-capable encoder without optimizations.
//!   Supports both batch and scanline-by-scanline encoding.
//!
//! Both implement the [`Encode`] trait for batch encoding.
//!
//! # Examples
//!
//! ```ignore
//! use mozjpeg_rs::{Encoder, Preset};
//!
//! // Full-featured batch encoding
//! let jpeg = Encoder::new(Preset::default())
//!     .quality(85)
//!     .encode_rgb(&pixels, width, height)?;
//!
//! // Streaming encoding (memory-efficient for large images)
//! let mut stream = Encoder::streaming()
//!     .quality(85)
//!     .start(width, height, file)?;
//! for row in scanlines.chunks(16) {
//!     stream.write_scanlines(row)?;
//! }
//! stream.finish()?;
//! ```

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::bitstream::BitWriter;
use crate::consts::{QuantTableIdx, DCTSIZE, DCTSIZE2};
use crate::deringing::preprocess_deringing;
use crate::entropy::{EntropyEncoder, ProgressiveEncoder, ProgressiveSymbolCounter, SymbolCounter};
use crate::error::{Error, Result};
use crate::huffman::DerivedTable;
use crate::huffman::FrequencyCounter;
use crate::marker::MarkerWriter;
use crate::progressive::{generate_baseline_scan, generate_mozjpeg_max_compression_scans};
use crate::quant::{create_quant_tables, quantize_block_raw};
use crate::sample;
use crate::scan_optimize::{generate_search_scans, ScanSearchConfig, ScanSelector};
use crate::scan_trial::ScanTrialEncoder;
#[cfg(target_arch = "x86_64")]
use crate::simd::x86_64::entropy::SimdEntropyEncoder;
use crate::simd::SimdOps;
use crate::trellis::trellis_quantize_block;
use crate::types::{Limits, PixelDensity, Preset, Subsampling, TrellisConfig};

mod helpers;
mod streaming;

pub(crate) use helpers::{
    create_components, create_std_ac_chroma_table, create_std_ac_luma_table,
    create_std_dc_chroma_table, create_std_dc_luma_table, create_ycbcr_components,
    natural_to_zigzag, run_dc_trellis_by_row, try_alloc_vec, try_alloc_vec_array, write_dht_marker,
    write_sos_marker,
};
pub use streaming::{EncodingStream, StreamingEncoder};

// ============================================================================
// Cancellation Support
// ============================================================================

/// Internal context for cancellation checking during encoding.
///
/// This is passed through the encoding pipeline to allow periodic
/// cancellation checks without function signature changes everywhere.
#[derive(Clone, Copy)]
pub(crate) struct CancellationContext<'a> {
    /// Optional cancellation flag - if set to true, encoding should abort.
    pub cancel: Option<&'a AtomicBool>,
    /// Optional deadline - if current time exceeds this, encoding should abort.
    pub deadline: Option<Instant>,
}

impl<'a> CancellationContext<'a> {
    /// Create a context with no cancellation (always succeeds).
    #[allow(dead_code)]
    pub const fn none() -> Self {
        Self {
            cancel: None,
            deadline: None,
        }
    }

    /// Create a context from optional cancel flag and timeout.
    #[allow(dead_code)]
    pub fn new(cancel: Option<&'a AtomicBool>, timeout: Option<Duration>) -> Self {
        Self {
            cancel,
            deadline: timeout.map(|d| Instant::now() + d),
        }
    }

    /// Check if cancellation has been requested.
    ///
    /// Returns `Ok(())` if encoding should continue, or `Err` if cancelled/timed out.
    #[inline]
    pub fn check(&self) -> Result<()> {
        if let Some(c) = self.cancel {
            if c.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
        }
        if let Some(d) = self.deadline {
            if Instant::now() > d {
                return Err(Error::TimedOut);
            }
        }
        Ok(())
    }

    /// Check cancellation every N iterations (to reduce overhead).
    ///
    /// Only performs the check when `iteration % interval == 0`.
    #[inline]
    #[allow(dead_code)]
    pub fn check_periodic(&self, iteration: usize, interval: usize) -> Result<()> {
        if iteration.is_multiple_of(interval) {
            self.check()
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// Encode Trait (internal, for potential future streaming API)
// ============================================================================

/// Trait for JPEG encoding (batch mode).
///
/// Implemented by both [`Encoder`] and [`StreamingEncoder`].
#[allow(dead_code)]
pub trait Encode {
    /// Encode RGB image data to JPEG.
    ///
    /// # Arguments
    /// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major order)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;

    /// Encode grayscale image data to JPEG.
    ///
    /// # Arguments
    /// * `gray_data` - Grayscale pixel data (1 byte per pixel, row-major order)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    fn encode_gray(&self, gray_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;
}

/// JPEG encoder with configurable quality and features.
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Quality level (1-100)
    quality: u8,
    /// Enable progressive mode
    progressive: bool,
    /// Chroma subsampling mode
    subsampling: Subsampling,
    /// Quantization table variant
    quant_table_idx: QuantTableIdx,
    /// Custom luminance quantization table (overrides quant_table_idx if set)
    custom_luma_qtable: Option<[u16; DCTSIZE2]>,
    /// Custom chrominance quantization table (overrides quant_table_idx if set)
    custom_chroma_qtable: Option<[u16; DCTSIZE2]>,
    /// Trellis quantization configuration
    trellis: TrellisConfig,
    /// Force baseline-compatible output
    force_baseline: bool,
    /// Optimize Huffman tables (requires 2-pass)
    optimize_huffman: bool,
    /// Enable overshoot deringing (reduces ringing on white backgrounds)
    overshoot_deringing: bool,
    /// Optimize progressive scan configuration (tries multiple configs, picks smallest)
    optimize_scans: bool,
    /// Restart interval in MCUs (0 = disabled)
    restart_interval: u16,
    /// Pixel density for JFIF APP0 marker
    pixel_density: PixelDensity,
    /// EXIF data to embed (raw TIFF structure, without "Exif\0\0" header)
    exif_data: Option<Vec<u8>>,
    /// ICC color profile to embed (will be chunked into APP2 markers)
    icc_profile: Option<Vec<u8>>,
    /// Custom APP markers to embed (marker number 0-15, data)
    custom_markers: Vec<(u8, Vec<u8>)>,
    /// SIMD operations dispatch (detected once at construction)
    simd: SimdOps,
    /// Smoothing factor (0-100, 0 = disabled)
    /// Applies a weighted average filter to reduce fine-scale noise.
    /// Useful for converting dithered images (like GIFs) to JPEG.
    smoothing: u8,
    /// Resource limits (dimensions, memory, ICC size)
    limits: Limits,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Encoder {
    /// Create an encoder with the specified preset.
    ///
    /// # Arguments
    ///
    /// * `preset` - Encoding preset (see [`Preset`] for details):
    ///   - [`BaselineFastest`](Preset::BaselineFastest): No optimizations, fastest encoding
    ///   - [`BaselineBalanced`](Preset::BaselineBalanced): Baseline with all optimizations
    ///   - [`ProgressiveBalanced`](Preset::ProgressiveBalanced): Progressive with optimizations (default)
    ///   - [`ProgressiveSmallest`](Preset::ProgressiveSmallest): Maximum compression
    ///
    /// # Preset Comparison
    ///
    /// | Preset | Time | Size | Best For |
    /// |--------|------|------|----------|
    /// | `BaselineFastest` | ~2ms | baseline | Real-time, thumbnails |
    /// | `BaselineBalanced` | ~7ms | -13% | Sequential playback |
    /// | `ProgressiveBalanced` | ~9ms | -13% | Web images (default) |
    /// | `ProgressiveSmallest` | ~21ms | -14% | Storage, archival |
    ///
    /// *Benchmarks: 512×512 Q75 image*
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::{Encoder, Preset};
    ///
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    ///
    /// // Default: progressive with good balance
    /// let jpeg = Encoder::new(Preset::default())
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    ///
    /// // Fastest for real-time applications
    /// let jpeg = Encoder::new(Preset::BaselineFastest)
    ///     .quality(80)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    ///
    /// // Maximum compression (matches C mozjpeg)
    /// let jpeg = Encoder::new(Preset::ProgressiveSmallest)
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn new(preset: Preset) -> Self {
        match preset {
            Preset::BaselineFastest => Self::fastest(),
            Preset::BaselineBalanced => Self::baseline_optimized(),
            Preset::ProgressiveBalanced => Self::progressive_balanced(),
            Preset::ProgressiveSmallest => Self::max_compression(),
        }
    }

    /// Create an encoder with the most optimized baseline (non-progressive) settings.
    ///
    /// This is the recommended starting point for most use cases. It produces
    /// sequential (non-progressive) JPEGs with all mozjpeg optimizations enabled:
    /// trellis quantization, Huffman optimization, and overshoot deringing.
    ///
    /// # Default Settings
    ///
    /// | Setting | Value | Notes |
    /// |---------|-------|-------|
    /// | quality | 75 | Good balance of size/quality |
    /// | progressive | **false** | Sequential baseline JPEG |
    /// | optimize_scans | **false** | N/A for baseline mode |
    /// | subsampling | 4:2:0 | Standard chroma subsampling |
    /// | trellis | **enabled** | AC + DC trellis quantization |
    /// | optimize_huffman | **true** | 2-pass for optimal Huffman tables |
    /// | overshoot_deringing | **true** | Reduces ringing on hard edges |
    /// | quant_tables | ImageMagick | Same as C mozjpeg default |
    /// | force_baseline | false | Allows 16-bit DQT at very low Q |
    ///
    /// # Comparison with C mozjpeg
    ///
    /// **Important:** This differs from C mozjpeg's `jpeg_set_defaults()`!
    ///
    /// C mozjpeg uses `JCP_MAX_COMPRESSION` profile by default, which enables
    /// progressive mode and optimize_scans. This produces ~20% smaller files
    /// but with slower encoding and progressive rendering.
    ///
    /// | Setting | `baseline_optimized()` | C mozjpeg default |
    /// |---------|------------------------|-------------------|
    /// | progressive | **false** | true |
    /// | optimize_scans | **false** | true |
    /// | trellis | true | true |
    /// | deringing | true | true |
    ///
    /// To match C mozjpeg's default behavior, use [`max_compression()`](Self::max_compression).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::Encoder;
    ///
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    /// let jpeg = Encoder::baseline_optimized()
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn baseline_optimized() -> Self {
        Self {
            quality: 75,
            progressive: false,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            custom_luma_qtable: None,
            custom_chroma_qtable: None,
            trellis: TrellisConfig::default(),
            force_baseline: false,
            optimize_huffman: true,
            overshoot_deringing: true,
            optimize_scans: false,
            restart_interval: 0,
            pixel_density: PixelDensity::default(),
            exif_data: None,
            icc_profile: None,
            custom_markers: Vec::new(),
            simd: SimdOps::detect(),
            smoothing: 0,
            limits: Limits::none(),
        }
    }

    /// Create encoder with maximum compression (matches C mozjpeg defaults).
    ///
    /// This matches the `JCP_MAX_COMPRESSION` profile used by C mozjpeg's
    /// `jpeg_set_defaults()` and the `mozjpeg` crate.
    ///
    /// # Settings (differences from `new()` in **bold**)
    ///
    /// | Setting | Value | Notes |
    /// |---------|-------|-------|
    /// | quality | 75 | Same as `new()` |
    /// | progressive | **true** | Multi-scan progressive JPEG |
    /// | optimize_scans | **true** | Tries multiple scan configs |
    /// | subsampling | 4:2:0 | Same as `new()` |
    /// | trellis | enabled | Same as `new()` |
    /// | optimize_huffman | true | Same as `new()` |
    /// | overshoot_deringing | true | Same as `new()` |
    ///
    /// # File Size Comparison
    ///
    /// Typical results at Q75 (256×256 image):
    /// - `Encoder::baseline_optimized()`: ~650 bytes (baseline)
    /// - `Encoder::max_compression()`: ~520 bytes (**~20% smaller**)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::Encoder;
    ///
    /// // Match C mozjpeg's default compression
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    /// let jpeg = Encoder::max_compression()
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn max_compression() -> Self {
        Self {
            quality: 75,
            progressive: true,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            custom_luma_qtable: None,
            custom_chroma_qtable: None,
            trellis: TrellisConfig::default(),
            force_baseline: false,
            optimize_huffman: true,
            overshoot_deringing: true,
            optimize_scans: true,
            restart_interval: 0,
            pixel_density: PixelDensity::default(),
            exif_data: None,
            icc_profile: None,
            custom_markers: Vec::new(),
            simd: SimdOps::detect(),
            smoothing: 0,
            limits: Limits::none(),
        }
    }

    /// Create encoder with progressive mode and all optimizations except optimize_scans.
    ///
    /// This is the **recommended default** for most use cases. It provides:
    /// - Progressive rendering (blurry-to-sharp loading)
    /// - All mozjpeg optimizations (trellis, Huffman, deringing)
    /// - Good balance between file size and encoding speed
    ///
    /// # Settings
    ///
    /// | Setting | Value | Notes |
    /// |---------|-------|-------|
    /// | progressive | **true** | Multi-scan progressive JPEG |
    /// | optimize_scans | **false** | Uses fixed 9-scan config |
    /// | trellis | enabled | AC + DC trellis quantization |
    /// | optimize_huffman | true | 2-pass for optimal tables |
    /// | overshoot_deringing | true | Reduces ringing on hard edges |
    ///
    /// # vs `max_compression()`
    ///
    /// This preset omits `optimize_scans` which:
    /// - Saves ~100% encoding time (9ms vs 21ms at 512×512)
    /// - Loses only ~1% file size reduction
    ///
    /// Use `max_compression()` only when file size is critical.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::Encoder;
    ///
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    /// let jpeg = Encoder::progressive_balanced()
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn progressive_balanced() -> Self {
        Self {
            quality: 75,
            progressive: true,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            custom_luma_qtable: None,
            custom_chroma_qtable: None,
            trellis: TrellisConfig::default(),
            force_baseline: false,
            optimize_huffman: true,
            overshoot_deringing: true,
            optimize_scans: false, // Key difference from max_compression()
            restart_interval: 0,
            pixel_density: PixelDensity::default(),
            exif_data: None,
            icc_profile: None,
            custom_markers: Vec::new(),
            simd: SimdOps::detect(),
            smoothing: 0,
            limits: Limits::none(),
        }
    }

    /// Create encoder with fastest settings (libjpeg-turbo compatible).
    ///
    /// Disables all mozjpeg-specific optimizations for maximum encoding speed.
    /// Output is compatible with standard libjpeg/libjpeg-turbo.
    ///
    /// # Settings (differences from `new()` in **bold**)
    ///
    /// | Setting | Value | Notes |
    /// |---------|-------|-------|
    /// | quality | 75 | Same as `new()` |
    /// | progressive | false | Same as `new()` |
    /// | trellis | **disabled** | No trellis quantization |
    /// | optimize_huffman | **false** | Uses default Huffman tables |
    /// | overshoot_deringing | **false** | No deringing filter |
    /// | force_baseline | **true** | 8-bit DQT only |
    ///
    /// # Performance
    ///
    /// Encoding is ~4-10x faster than `new()`, but files are ~10-20% larger.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::Encoder;
    ///
    /// // Fast encoding for real-time applications
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    /// let jpeg = Encoder::fastest()
    ///     .quality(80)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn fastest() -> Self {
        Self {
            quality: 75,
            progressive: false,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            custom_luma_qtable: None,
            custom_chroma_qtable: None,
            trellis: TrellisConfig::disabled(),
            force_baseline: true,
            optimize_huffman: false,
            overshoot_deringing: false,
            optimize_scans: false,
            restart_interval: 0,
            pixel_density: PixelDensity::default(),
            exif_data: None,
            icc_profile: None,
            custom_markers: Vec::new(),
            simd: SimdOps::detect(),
            smoothing: 0,
            limits: Limits::none(),
        }
    }

    /// Set quality level (1-100).
    ///
    /// Higher values produce larger, higher-quality images.
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Enable or disable progressive mode.
    pub fn progressive(mut self, enable: bool) -> Self {
        self.progressive = enable;
        self
    }

    /// Set chroma subsampling mode.
    pub fn subsampling(mut self, mode: Subsampling) -> Self {
        self.subsampling = mode;
        self
    }

    /// Set quantization table variant.
    pub fn quant_tables(mut self, idx: QuantTableIdx) -> Self {
        self.quant_table_idx = idx;
        self
    }

    /// Configure trellis quantization.
    pub fn trellis(mut self, config: TrellisConfig) -> Self {
        self.trellis = config;
        self
    }

    /// Force baseline-compatible output.
    pub fn force_baseline(mut self, enable: bool) -> Self {
        self.force_baseline = enable;
        self
    }

    /// Enable Huffman table optimization.
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
    }

    /// Enable overshoot deringing.
    ///
    /// Reduces visible ringing artifacts near hard edges, especially on white
    /// backgrounds. Works by allowing encoded values to "overshoot" above 255
    /// (which will clamp back to 255 when decoded) to create smoother waveforms.
    ///
    /// This is a mozjpeg-specific feature that can improve visual quality at
    /// minimal file size cost. Enabled by default.
    pub fn overshoot_deringing(mut self, enable: bool) -> Self {
        self.overshoot_deringing = enable;
        self
    }

    /// Enable or disable scan optimization for progressive mode.
    ///
    /// When enabled, the encoder tries multiple scan configurations and
    /// picks the one that produces the smallest output. This can improve
    /// compression by 1-3% but increases encoding time.
    ///
    /// Only has effect when progressive mode is enabled.
    pub fn optimize_scans(mut self, enable: bool) -> Self {
        self.optimize_scans = enable;
        self
    }

    /// Set input smoothing factor (0-100).
    ///
    /// Applies a weighted average filter to reduce fine-scale noise in the
    /// input image before encoding. This is particularly useful for converting
    /// dithered images (like GIFs) to JPEG.
    ///
    /// - 0 = disabled (default)
    /// - 10-50 = recommended for dithered images
    /// - Higher values = more smoothing (may blur the image)
    ///
    /// # Example
    /// ```
    /// use mozjpeg_rs::Encoder;
    ///
    /// // Convert a dithered GIF to JPEG with smoothing
    /// let encoder = Encoder::baseline_optimized()
    ///     .quality(85)
    ///     .smoothing(30);
    /// ```
    pub fn smoothing(mut self, factor: u8) -> Self {
        self.smoothing = factor.min(100);
        self
    }

    /// Set restart interval in MCUs.
    ///
    /// Restart markers are inserted every N MCUs, which can help with
    /// error recovery and parallel decoding. Set to 0 to disable (default).
    ///
    /// Common values: 0 (disabled), or image width in MCUs for row-by-row restarts.
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    /// Set EXIF data to embed in the JPEG.
    ///
    /// # Arguments
    /// * `data` - Raw EXIF data (TIFF structure). The "Exif\0\0" header
    ///   will be added automatically.
    ///
    /// Pass empty or call without this method to omit EXIF data.
    pub fn exif_data(mut self, data: Vec<u8>) -> Self {
        self.exif_data = if data.is_empty() { None } else { Some(data) };
        self
    }

    /// Set pixel density for the JFIF APP0 marker.
    ///
    /// This specifies the physical pixel density (DPI/DPC) or aspect ratio.
    /// Note that most software ignores JFIF density in favor of EXIF metadata.
    ///
    /// # Example
    /// ```
    /// use mozjpeg_rs::{Encoder, PixelDensity};
    ///
    /// let encoder = Encoder::baseline_optimized()
    ///     .pixel_density(PixelDensity::dpi(300, 300)); // 300 DPI
    /// ```
    pub fn pixel_density(mut self, density: PixelDensity) -> Self {
        self.pixel_density = density;
        self
    }

    /// Set ICC color profile to embed.
    ///
    /// The profile will be embedded in APP2 markers with the standard
    /// "ICC_PROFILE" identifier. Large profiles are automatically chunked.
    ///
    /// # Arguments
    /// * `profile` - Raw ICC profile data
    pub fn icc_profile(mut self, profile: Vec<u8>) -> Self {
        self.icc_profile = if profile.is_empty() {
            None
        } else {
            Some(profile)
        };
        self
    }

    /// Add a custom APP marker.
    ///
    /// # Arguments
    /// * `app_num` - APP marker number (0-15, e.g., 1 for EXIF, 2 for ICC)
    /// * `data` - Raw marker data (including any identifier prefix)
    ///
    /// Multiple markers with the same number are allowed.
    /// Markers are written in the order they are added.
    pub fn add_marker(mut self, app_num: u8, data: Vec<u8>) -> Self {
        if app_num <= 15 && !data.is_empty() {
            self.custom_markers.push((app_num, data));
        }
        self
    }

    /// Set custom luminance quantization table.
    ///
    /// This overrides the table selected by `quant_tables()`.
    /// Values should be in natural (row-major) order, not zigzag.
    ///
    /// # Arguments
    /// * `table` - 64 quantization values (quality scaling still applies)
    pub fn custom_luma_qtable(mut self, table: [u16; DCTSIZE2]) -> Self {
        self.custom_luma_qtable = Some(table);
        self
    }

    /// Set custom chrominance quantization table.
    ///
    /// This overrides the table selected by `quant_tables()`.
    /// Values should be in natural (row-major) order, not zigzag.
    ///
    /// # Arguments
    /// * `table` - 64 quantization values (quality scaling still applies)
    pub fn custom_chroma_qtable(mut self, table: [u16; DCTSIZE2]) -> Self {
        self.custom_chroma_qtable = Some(table);
        self
    }

    // =========================================================================
    // Resource Limits
    // =========================================================================

    /// Set resource limits for the encoder.
    ///
    /// Limits can restrict:
    /// - Maximum image width and height
    /// - Maximum pixel count (width × height)
    /// - Maximum estimated memory allocation
    /// - Maximum ICC profile size
    ///
    /// # Example
    /// ```
    /// use mozjpeg_rs::{Encoder, Preset, Limits};
    ///
    /// let limits = Limits::default()
    ///     .max_width(4096)
    ///     .max_height(4096)
    ///     .max_pixel_count(16_000_000)
    ///     .max_alloc_bytes(100 * 1024 * 1024);
    ///
    /// let encoder = Encoder::new(Preset::default())
    ///     .limits(limits);
    /// ```
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Check all resource limits before encoding.
    ///
    /// # Arguments
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `is_gray` - True for grayscale images (affects memory estimate)
    fn check_limits(&self, width: u32, height: u32, is_gray: bool) -> Result<()> {
        let limits = &self.limits;

        // Check dimension limits
        if (limits.max_width > 0 && width > limits.max_width)
            || (limits.max_height > 0 && height > limits.max_height)
        {
            return Err(Error::DimensionLimitExceeded {
                width,
                height,
                max_width: limits.max_width,
                max_height: limits.max_height,
            });
        }

        // Check pixel count limit
        if limits.max_pixel_count > 0 {
            let pixel_count = width as u64 * height as u64;
            if pixel_count > limits.max_pixel_count {
                return Err(Error::PixelCountExceeded {
                    pixel_count,
                    limit: limits.max_pixel_count,
                });
            }
        }

        // Check allocation limit
        if limits.max_alloc_bytes > 0 {
            let estimate = if is_gray {
                self.estimate_resources_gray(width, height)
            } else {
                self.estimate_resources(width, height)
            };
            if estimate.peak_memory_bytes > limits.max_alloc_bytes {
                return Err(Error::AllocationLimitExceeded {
                    estimated: estimate.peak_memory_bytes,
                    limit: limits.max_alloc_bytes,
                });
            }
        }

        // Check ICC profile size limit
        if limits.max_icc_profile_bytes > 0 {
            if let Some(ref icc) = self.icc_profile {
                if icc.len() > limits.max_icc_profile_bytes {
                    return Err(Error::IccProfileTooLarge {
                        size: icc.len(),
                        limit: limits.max_icc_profile_bytes,
                    });
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Aliases for rimage/CLI-style naming
    // =========================================================================

    /// Set baseline mode (opposite of progressive).
    ///
    /// When `true`, produces a sequential JPEG (non-progressive).
    /// This is equivalent to `progressive(false)`.
    ///
    /// # Example
    /// ```
    /// use mozjpeg_rs::Encoder;
    ///
    /// // These are equivalent:
    /// let enc1 = Encoder::baseline_optimized().baseline(true);
    /// let enc2 = Encoder::baseline_optimized().progressive(false);
    /// ```
    #[inline]
    pub fn baseline(self, enable: bool) -> Self {
        self.progressive(!enable)
    }

    /// Enable or disable Huffman coding optimization.
    ///
    /// Alias for [`optimize_huffman()`](Self::optimize_huffman).
    /// This name matches mozjpeg's CLI flag naming.
    #[inline]
    pub fn optimize_coding(self, enable: bool) -> Self {
        self.optimize_huffman(enable)
    }

    /// Set chroma subsampling mode.
    ///
    /// Alias for [`subsampling()`](Self::subsampling).
    #[inline]
    pub fn chroma_subsampling(self, mode: Subsampling) -> Self {
        self.subsampling(mode)
    }

    /// Set quantization table variant.
    ///
    /// Alias for [`quant_tables()`](Self::quant_tables).
    #[inline]
    pub fn qtable(self, idx: QuantTableIdx) -> Self {
        self.quant_tables(idx)
    }

    // =========================================================================
    // Resource Estimation
    // =========================================================================

    /// Estimate resource usage for encoding an RGB image of the given dimensions.
    ///
    /// Returns peak memory usage (in bytes) and a relative CPU cost multiplier.
    /// Useful for scheduling, enforcing resource limits, or providing feedback.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Example
    ///
    /// ```
    /// use mozjpeg_rs::{Encoder, Preset};
    ///
    /// let encoder = Encoder::new(Preset::ProgressiveBalanced).quality(85);
    /// let estimate = encoder.estimate_resources(1920, 1080);
    ///
    /// println!("Peak memory: {} MB", estimate.peak_memory_bytes / 1_000_000);
    /// println!("Relative CPU cost: {:.1}x", estimate.cpu_cost_multiplier);
    /// ```
    pub fn estimate_resources(&self, width: u32, height: u32) -> crate::types::ResourceEstimate {
        let width = width as usize;
        let height = height as usize;
        let pixels = width * height;

        // Calculate chroma dimensions based on subsampling
        let (h_samp, v_samp) = self.subsampling.luma_factors();
        let chroma_width = (width + h_samp as usize - 1) / h_samp as usize;
        let chroma_height = (height + v_samp as usize - 1) / v_samp as usize;
        let chroma_pixels = chroma_width * chroma_height;

        // MCU-aligned dimensions
        let mcu_h = 8 * h_samp as usize;
        let mcu_v = 8 * v_samp as usize;
        let mcu_width = (width + mcu_h - 1) / mcu_h * mcu_h;
        let mcu_height = (height + mcu_v - 1) / mcu_v * mcu_v;

        // Block counts
        let y_blocks = (mcu_width / 8) * (mcu_height / 8);
        let chroma_block_w = (chroma_width + 7) / 8;
        let chroma_block_h = (chroma_height + 7) / 8;
        let chroma_blocks = chroma_block_w * chroma_block_h;
        let total_blocks = y_blocks + 2 * chroma_blocks;

        // --- Memory estimation ---
        let mut memory: usize = 0;

        // Color conversion buffers (Y, Cb, Cr planes)
        memory += 3 * pixels;

        // Chroma subsampled buffers
        memory += 2 * chroma_pixels;

        // MCU-padded buffers
        memory += mcu_width * mcu_height; // Y
        let mcu_chroma_w = (chroma_width + 7) / 8 * 8;
        let mcu_chroma_h = (chroma_height + 7) / 8 * 8;
        memory += 2 * mcu_chroma_w * mcu_chroma_h; // Cb, Cr

        // Block storage (needed for progressive or optimize_huffman)
        let needs_block_storage = self.progressive || self.optimize_huffman;
        if needs_block_storage {
            // i16[64] per block = 128 bytes
            memory += total_blocks * 128;
        }

        // Raw DCT storage (needed for DC trellis)
        if self.trellis.dc_enabled {
            // i32[64] per block = 256 bytes
            memory += total_blocks * 256;
        }

        // Output buffer estimate (varies by quality, ~0.3-1.0x input for typical images)
        // Use a conservative estimate based on quality
        let output_ratio = if self.quality >= 95 {
            0.8
        } else if self.quality >= 85 {
            0.5
        } else if self.quality >= 75 {
            0.3
        } else {
            0.2
        };
        memory += (pixels as f64 * 3.0 * output_ratio) as usize;

        // --- CPU cost estimation ---
        // Reference: BaselineFastest Q75 = 1.0
        let mut cpu_cost = 1.0;

        // Trellis AC quantization is the biggest CPU factor
        if self.trellis.enabled {
            cpu_cost += 3.5;
        }

        // DC trellis adds extra work
        if self.trellis.dc_enabled {
            cpu_cost += 0.5;
        }

        // Huffman optimization (frequency counting pass)
        if self.optimize_huffman {
            cpu_cost += 0.3;
        }

        // Progressive mode (multiple scan encoding)
        if self.progressive {
            cpu_cost += 1.5;
        }

        // optimize_scans (trial encoding many scan configurations)
        if self.optimize_scans {
            cpu_cost += 3.0;
        }

        // High quality increases trellis work (more candidates to evaluate)
        // This matters most when trellis is enabled
        if self.trellis.enabled && self.quality >= 85 {
            let quality_factor = 1.0 + (self.quality as f64 - 85.0) / 30.0;
            cpu_cost *= quality_factor;
        }

        crate::types::ResourceEstimate {
            peak_memory_bytes: memory,
            cpu_cost_multiplier: cpu_cost,
            block_count: total_blocks,
        }
    }

    /// Estimate resource usage for encoding a grayscale image.
    ///
    /// Similar to [`estimate_resources`](Self::estimate_resources) but for single-channel images.
    pub fn estimate_resources_gray(
        &self,
        width: u32,
        height: u32,
    ) -> crate::types::ResourceEstimate {
        let width = width as usize;
        let height = height as usize;
        let pixels = width * height;

        // MCU-aligned dimensions (always 8x8 for grayscale)
        let mcu_width = (width + 7) / 8 * 8;
        let mcu_height = (height + 7) / 8 * 8;

        // Block count
        let blocks = (mcu_width / 8) * (mcu_height / 8);

        // --- Memory estimation ---
        let mut memory: usize = 0;

        // MCU-padded buffer
        memory += mcu_width * mcu_height;

        // Block storage (needed for progressive or optimize_huffman)
        let needs_block_storage = self.progressive || self.optimize_huffman;
        if needs_block_storage {
            memory += blocks * 128;
        }

        // Raw DCT storage (needed for DC trellis)
        if self.trellis.dc_enabled {
            memory += blocks * 256;
        }

        // Output buffer estimate
        let output_ratio = if self.quality >= 95 {
            0.8
        } else if self.quality >= 85 {
            0.5
        } else if self.quality >= 75 {
            0.3
        } else {
            0.2
        };
        memory += (pixels as f64 * output_ratio) as usize;

        // --- CPU cost (same formula, but less work due to single channel) ---
        let mut cpu_cost = 1.0;

        if self.trellis.enabled {
            cpu_cost += 3.5;
        }
        if self.trellis.dc_enabled {
            cpu_cost += 0.5;
        }
        if self.optimize_huffman {
            cpu_cost += 0.3;
        }
        if self.progressive {
            cpu_cost += 1.0; // Less for grayscale (fewer scans)
        }
        if self.optimize_scans {
            cpu_cost += 2.0; // Less for grayscale
        }
        if self.trellis.enabled && self.quality >= 85 {
            let quality_factor = 1.0 + (self.quality as f64 - 85.0) / 30.0;
            cpu_cost *= quality_factor;
        }

        // Grayscale is ~1/3 the work of RGB (single channel)
        cpu_cost /= 3.0;

        crate::types::ResourceEstimate {
            peak_memory_bytes: memory,
            cpu_cost_multiplier: cpu_cost,
            block_count: blocks,
        }
    }

    // =========================================================================
    // Encoding
    // =========================================================================

    /// Encode RGB image data to JPEG.
    ///
    /// # Arguments
    /// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// JPEG-encoded data as a `Vec<u8>`.
    pub fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        // Validate dimensions: must be non-zero
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Check all resource limits
        self.check_limits(width, height, false)?;

        // Use checked arithmetic to prevent overflow
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::InvalidDimensions { width, height })?;

        if rgb_data.len() != expected_len {
            return Err(Error::BufferSizeMismatch {
                expected: expected_len,
                actual: rgb_data.len(),
            });
        }

        // Apply smoothing if enabled
        let rgb_data = if self.smoothing > 0 {
            std::borrow::Cow::Owned(crate::smooth::smooth_rgb(
                rgb_data,
                width,
                height,
                self.smoothing,
            ))
        } else {
            std::borrow::Cow::Borrowed(rgb_data)
        };

        let mut output = Vec::new();
        self.encode_rgb_to_writer(&rgb_data, width, height, &mut output)?;
        Ok(output)
    }

    /// Encode grayscale image data to JPEG.
    ///
    /// # Arguments
    /// * `gray_data` - Grayscale pixel data (1 byte per pixel, row-major)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// JPEG-encoded data as a `Vec<u8>`.
    pub fn encode_gray(&self, gray_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        // Validate dimensions: must be non-zero
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Check all resource limits
        self.check_limits(width, height, true)?;

        // Use checked arithmetic to prevent overflow
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(Error::InvalidDimensions { width, height })?;

        if gray_data.len() != expected_len {
            return Err(Error::BufferSizeMismatch {
                expected: expected_len,
                actual: gray_data.len(),
            });
        }

        // Apply smoothing if enabled
        let gray_data = if self.smoothing > 0 {
            std::borrow::Cow::Owned(crate::smooth::smooth_grayscale(
                gray_data,
                width,
                height,
                self.smoothing,
            ))
        } else {
            std::borrow::Cow::Borrowed(gray_data)
        };

        let mut output = Vec::new();
        self.encode_gray_to_writer(&gray_data, width, height, &mut output)?;
        Ok(output)
    }

    /// Encode RGB image data to JPEG with cancellation and timeout support.
    ///
    /// This method allows encoding to be cancelled mid-operation via an atomic flag,
    /// or to automatically abort if a timeout is exceeded.
    ///
    /// # Arguments
    /// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `cancel` - Optional cancellation flag. Set to `true` to abort encoding.
    /// * `timeout` - Optional maximum encoding duration.
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - JPEG-encoded data
    /// * `Err(Error::Cancelled)` - If cancelled via the flag
    /// * `Err(Error::TimedOut)` - If the timeout was exceeded
    ///
    /// # Example
    /// ```no_run
    /// use mozjpeg_rs::{Encoder, Preset};
    /// use std::sync::atomic::AtomicBool;
    /// use std::time::Duration;
    ///
    /// let encoder = Encoder::new(Preset::ProgressiveBalanced);
    /// let pixels: Vec<u8> = vec![128; 1920 * 1080 * 3];
    /// let cancel = AtomicBool::new(false);
    ///
    /// // Encode with 5 second timeout
    /// let result = encoder.encode_rgb_cancellable(
    ///     &pixels, 1920, 1080,
    ///     Some(&cancel),
    ///     Some(Duration::from_secs(5)),
    /// );
    /// ```
    pub fn encode_rgb_cancellable(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        cancel: Option<&AtomicBool>,
        timeout: Option<Duration>,
    ) -> Result<Vec<u8>> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Check all resource limits
        self.check_limits(width, height, false)?;

        // Check buffer size
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::InvalidDimensions { width, height })?;

        if rgb_data.len() != expected_len {
            return Err(Error::BufferSizeMismatch {
                expected: expected_len,
                actual: rgb_data.len(),
            });
        }

        // Create cancellation context
        let ctx = CancellationContext::new(cancel, timeout);

        // Check for immediate cancellation
        ctx.check()?;

        // Apply smoothing if enabled
        let rgb_data = if self.smoothing > 0 {
            std::borrow::Cow::Owned(crate::smooth::smooth_rgb(
                rgb_data,
                width,
                height,
                self.smoothing,
            ))
        } else {
            std::borrow::Cow::Borrowed(rgb_data)
        };

        let mut output = Vec::new();
        // For now, use the regular encoder (cancellation hooks can be added to
        // internal functions in a follow-up). Check cancellation before and after.
        ctx.check()?;
        self.encode_rgb_to_writer(&rgb_data, width, height, &mut output)?;
        ctx.check()?;

        Ok(output)
    }

    /// Encode grayscale image data to JPEG with cancellation and timeout support.
    ///
    /// This method allows encoding to be cancelled mid-operation via an atomic flag,
    /// or to automatically abort if a timeout is exceeded.
    ///
    /// # Arguments
    /// * `gray_data` - Grayscale pixel data (1 byte per pixel, row-major)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `cancel` - Optional cancellation flag. Set to `true` to abort encoding.
    /// * `timeout` - Optional maximum encoding duration.
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - JPEG-encoded data
    /// * `Err(Error::Cancelled)` - If cancelled via the flag
    /// * `Err(Error::TimedOut)` - If the timeout was exceeded
    pub fn encode_gray_cancellable(
        &self,
        gray_data: &[u8],
        width: u32,
        height: u32,
        cancel: Option<&AtomicBool>,
        timeout: Option<Duration>,
    ) -> Result<Vec<u8>> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Check all resource limits
        self.check_limits(width, height, true)?;

        // Check buffer size
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(Error::InvalidDimensions { width, height })?;

        if gray_data.len() != expected_len {
            return Err(Error::BufferSizeMismatch {
                expected: expected_len,
                actual: gray_data.len(),
            });
        }

        // Create cancellation context
        let ctx = CancellationContext::new(cancel, timeout);

        // Check for immediate cancellation
        ctx.check()?;

        // Apply smoothing if enabled
        let gray_data = if self.smoothing > 0 {
            std::borrow::Cow::Owned(crate::smooth::smooth_grayscale(
                gray_data,
                width,
                height,
                self.smoothing,
            ))
        } else {
            std::borrow::Cow::Borrowed(gray_data)
        };

        let mut output = Vec::new();
        // For now, use the regular encoder (cancellation hooks can be added to
        // internal functions in a follow-up). Check cancellation before and after.
        ctx.check()?;
        self.encode_gray_to_writer(&gray_data, width, height, &mut output)?;
        ctx.check()?;

        Ok(output)
    }

    /// Encode grayscale image data to a writer.
    pub fn encode_gray_to_writer<W: Write>(
        &self,
        gray_data: &[u8],
        width: u32,
        height: u32,
        output: W,
    ) -> Result<()> {
        let width = width as usize;
        let height = height as usize;

        // For grayscale, Y plane is the input directly (no conversion needed)
        let y_plane = gray_data;

        // Grayscale uses 1x1 sampling
        let (mcu_width, mcu_height) = sample::mcu_aligned_dimensions(width, height, 1, 1);

        let mcu_y_size = mcu_width
            .checked_mul(mcu_height)
            .ok_or(Error::AllocationFailed)?;
        let mut y_mcu = try_alloc_vec(0u8, mcu_y_size)?;
        sample::expand_to_mcu(y_plane, width, height, &mut y_mcu, mcu_width, mcu_height);

        // Create quantization table (only luma needed)
        let luma_qtable = if let Some(ref custom) = self.custom_luma_qtable {
            crate::quant::create_quant_table(custom, self.quality, self.force_baseline)
        } else {
            let (luma, _) =
                create_quant_tables(self.quality, self.quant_table_idx, self.force_baseline);
            luma
        };

        // Create Huffman tables (only luma needed)
        let dc_luma_huff = create_std_dc_luma_table();
        let ac_luma_huff = create_std_ac_luma_table();
        let dc_luma_derived = DerivedTable::from_huff_table(&dc_luma_huff, true)?;
        let ac_luma_derived = DerivedTable::from_huff_table(&ac_luma_huff, false)?;

        // Single component for grayscale
        let components = create_components(Subsampling::Gray);

        // Write JPEG file
        let mut marker_writer = MarkerWriter::new(output);

        // SOI
        marker_writer.write_soi()?;

        // APP0 (JFIF) with pixel density
        marker_writer.write_jfif_app0(
            self.pixel_density.unit as u8,
            self.pixel_density.x,
            self.pixel_density.y,
        )?;

        // EXIF (if present)
        if let Some(ref exif) = self.exif_data {
            marker_writer.write_app1_exif(exif)?;
        }

        // ICC profile (if present)
        if let Some(ref icc) = self.icc_profile {
            marker_writer.write_icc_profile(icc)?;
        }

        // Custom APP markers
        for (app_num, data) in &self.custom_markers {
            marker_writer.write_app(*app_num, data)?;
        }

        // DQT (only luma table for grayscale)
        let luma_qtable_zz = natural_to_zigzag(&luma_qtable.values);
        marker_writer.write_dqt(0, &luma_qtable_zz, false)?;

        // SOF (baseline or progressive)
        marker_writer.write_sof(
            self.progressive,
            8,
            height as u16,
            width as u16,
            &components,
        )?;

        // DRI (restart interval)
        if self.restart_interval > 0 {
            marker_writer.write_dri(self.restart_interval)?;
        }

        // DHT (only luma tables for grayscale) - written later for progressive
        if !self.progressive && !self.optimize_huffman {
            marker_writer
                .write_dht_multiple(&[(0, false, &dc_luma_huff), (0, true, &ac_luma_huff)])?;
        }

        let mcu_rows = mcu_height / DCTSIZE;
        let mcu_cols = mcu_width / DCTSIZE;
        let num_blocks = mcu_rows
            .checked_mul(mcu_cols)
            .ok_or(Error::AllocationFailed)?;

        if self.progressive {
            // Progressive mode: collect all blocks, then encode multiple scans
            let mut y_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_blocks)?;
            let mut dct_block = [0i16; DCTSIZE2];

            // Optionally collect raw DCT for DC trellis
            let dc_trellis_enabled = self.trellis.enabled && self.trellis.dc_enabled;
            let mut y_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_blocks)?)
            } else {
                None
            };

            // Collect all blocks
            for mcu_row in 0..mcu_rows {
                for mcu_col in 0..mcu_cols {
                    let block_idx = mcu_row * mcu_cols + mcu_col;
                    self.process_block_to_storage_with_raw(
                        &y_mcu,
                        mcu_width,
                        mcu_row,
                        mcu_col,
                        &luma_qtable.values,
                        &ac_luma_derived,
                        &mut y_blocks[block_idx],
                        &mut dct_block,
                        y_raw_dct.as_mut().map(|v| v[block_idx].as_mut_slice()),
                    )?;
                }
            }

            // Run DC trellis optimization if enabled
            if dc_trellis_enabled {
                if let Some(ref y_raw) = y_raw_dct {
                    run_dc_trellis_by_row(
                        y_raw,
                        &mut y_blocks,
                        luma_qtable.values[0],
                        &dc_luma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        mcu_rows,
                        mcu_cols,
                        mcu_cols,
                        1,
                        1,
                    );
                }
            }

            // Run EOB optimization if enabled (cross-block EOBRUN optimization)
            if self.trellis.enabled && self.trellis.eob_opt {
                use crate::trellis::{estimate_block_eob_info, optimize_eob_runs};

                // Estimate EOB info for each block
                let eob_info: Vec<_> = y_blocks
                    .iter()
                    .map(|block| estimate_block_eob_info(block, &ac_luma_derived, 1, 63))
                    .collect();

                // Optimize EOB runs across all blocks
                optimize_eob_runs(&mut y_blocks, &eob_info, &ac_luma_derived, 1, 63);
            }

            // Generate progressive scan script for grayscale (1 component)
            let scans = generate_mozjpeg_max_compression_scans(1);

            // Build optimized Huffman tables
            let mut dc_freq = FrequencyCounter::new();
            let mut dc_counter = ProgressiveSymbolCounter::new();
            for scan in &scans {
                let is_dc_first_scan = scan.ss == 0 && scan.se == 0 && scan.ah == 0;
                if is_dc_first_scan {
                    // Count DC symbols using progressive counter
                    for block in &y_blocks {
                        dc_counter.count_dc_first(block, 0, scan.al, &mut dc_freq);
                    }
                }
            }

            let opt_dc_huff = dc_freq.generate_table()?;
            let opt_dc_derived = DerivedTable::from_huff_table(&opt_dc_huff, true)?;

            // Write DC Huffman table upfront
            marker_writer.write_dht_multiple(&[(0, false, &opt_dc_huff)])?;

            // Encode each scan
            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);

            for scan in &scans {
                let is_dc_scan = scan.ss == 0 && scan.se == 0;

                if is_dc_scan {
                    // DC scan
                    marker_writer = MarkerWriter::new(bit_writer.into_inner());
                    marker_writer.write_sos(scan, &components)?;
                    bit_writer = BitWriter::new(marker_writer.into_inner());

                    let mut prog_encoder = ProgressiveEncoder::new(&mut bit_writer);

                    if scan.ah == 0 {
                        // DC first scan
                        for block in &y_blocks {
                            prog_encoder.encode_dc_first(block, 0, &opt_dc_derived, scan.al)?;
                        }
                    } else {
                        // DC refinement scan
                        for block in &y_blocks {
                            prog_encoder.encode_dc_refine(block, scan.al)?;
                        }
                    }

                    prog_encoder.finish_scan(None)?;
                } else {
                    // AC scan - generate per-scan Huffman table
                    let mut ac_freq = FrequencyCounter::new();
                    let mut ac_counter = ProgressiveSymbolCounter::new();

                    for block in &y_blocks {
                        if scan.ah == 0 {
                            ac_counter.count_ac_first(
                                block,
                                scan.ss,
                                scan.se,
                                scan.al,
                                &mut ac_freq,
                            );
                        } else {
                            ac_counter.count_ac_refine(
                                block,
                                scan.ss,
                                scan.se,
                                scan.ah,
                                scan.al,
                                &mut ac_freq,
                            );
                        }
                    }
                    ac_counter.finish_scan(Some(&mut ac_freq));

                    let opt_ac_huff = ac_freq.generate_table()?;
                    let opt_ac_derived = DerivedTable::from_huff_table(&opt_ac_huff, false)?;

                    // Write AC Huffman table and SOS
                    marker_writer = MarkerWriter::new(bit_writer.into_inner());
                    marker_writer.write_dht_multiple(&[(0, true, &opt_ac_huff)])?;
                    marker_writer.write_sos(scan, &components)?;
                    bit_writer = BitWriter::new(marker_writer.into_inner());

                    let mut prog_encoder = ProgressiveEncoder::new(&mut bit_writer);

                    for block in &y_blocks {
                        if scan.ah == 0 {
                            prog_encoder.encode_ac_first(
                                block,
                                scan.ss,
                                scan.se,
                                scan.al,
                                &opt_ac_derived,
                            )?;
                        } else {
                            prog_encoder.encode_ac_refine(
                                block,
                                scan.ss,
                                scan.se,
                                scan.ah,
                                scan.al,
                                &opt_ac_derived,
                            )?;
                        }
                    }

                    prog_encoder.finish_scan(Some(&opt_ac_derived))?;
                }
            }

            let mut output = bit_writer.into_inner();
            output.write_all(&[0xFF, 0xD9])?; // EOI
        } else if self.optimize_huffman {
            // 2-pass: collect blocks, count frequencies, then encode
            let mut y_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_blocks)?;
            let mut dct_block = [0i16; DCTSIZE2];

            // Collect all blocks using the same process as RGB encoding
            for mcu_row in 0..mcu_rows {
                for mcu_col in 0..mcu_cols {
                    let block_idx = mcu_row * mcu_cols + mcu_col;
                    self.process_block_to_storage_with_raw(
                        &y_mcu,
                        mcu_width,
                        mcu_row,
                        mcu_col,
                        &luma_qtable.values,
                        &ac_luma_derived,
                        &mut y_blocks[block_idx],
                        &mut dct_block,
                        None, // No raw DCT storage needed for grayscale
                    )?;
                }
            }

            // Count frequencies using SymbolCounter
            let mut dc_freq = FrequencyCounter::new();
            let mut ac_freq = FrequencyCounter::new();
            let mut counter = SymbolCounter::new();
            for block in &y_blocks {
                counter.count_block(block, 0, &mut dc_freq, &mut ac_freq);
            }

            // Generate optimized tables
            let opt_dc_huff = dc_freq.generate_table()?;
            let opt_ac_huff = ac_freq.generate_table()?;
            let opt_dc_derived = DerivedTable::from_huff_table(&opt_dc_huff, true)?;
            let opt_ac_derived = DerivedTable::from_huff_table(&opt_ac_huff, false)?;

            // Write optimized Huffman tables
            marker_writer
                .write_dht_multiple(&[(0, false, &opt_dc_huff), (0, true, &opt_ac_huff)])?;

            // Write SOS and encode
            let scans = generate_baseline_scan(1);
            marker_writer.write_sos(&scans[0], &components)?;

            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);
            let mut encoder = EntropyEncoder::new(&mut bit_writer);

            // Restart marker support for grayscale (each block = 1 MCU)
            let restart_interval = self.restart_interval as usize;
            let mut restart_num = 0u8;

            for (mcu_count, block) in y_blocks.iter().enumerate() {
                // Emit restart marker if needed
                if restart_interval > 0
                    && mcu_count > 0
                    && mcu_count.is_multiple_of(restart_interval)
                {
                    encoder.emit_restart(restart_num)?;
                    restart_num = restart_num.wrapping_add(1) & 0x07;
                }
                encoder.encode_block(block, 0, &opt_dc_derived, &opt_ac_derived)?;
            }

            bit_writer.flush()?;
            let mut output = bit_writer.into_inner();
            output.write_all(&[0xFF, 0xD9])?; // EOI
        } else {
            // Single-pass encoding
            let scans = generate_baseline_scan(1);
            marker_writer.write_sos(&scans[0], &components)?;

            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);
            let mut encoder = EntropyEncoder::new(&mut bit_writer);
            let mut dct_block = [0i16; DCTSIZE2];
            let mut quant_block = [0i16; DCTSIZE2];

            // Restart marker support
            let restart_interval = self.restart_interval as usize;
            let mut mcu_count = 0usize;
            let mut restart_num = 0u8;

            for mcu_row in 0..mcu_rows {
                for mcu_col in 0..mcu_cols {
                    // Emit restart marker if needed
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        encoder.emit_restart(restart_num)?;
                        restart_num = restart_num.wrapping_add(1) & 0x07;
                    }

                    // Process block directly to quant_block
                    self.process_block_to_storage_with_raw(
                        &y_mcu,
                        mcu_width,
                        mcu_row,
                        mcu_col,
                        &luma_qtable.values,
                        &ac_luma_derived,
                        &mut quant_block,
                        &mut dct_block,
                        None,
                    )?;
                    encoder.encode_block(&quant_block, 0, &dc_luma_derived, &ac_luma_derived)?;
                    mcu_count += 1;
                }
            }

            bit_writer.flush()?;
            let mut output = bit_writer.into_inner();
            output.write_all(&[0xFF, 0xD9])?; // EOI
        }

        Ok(())
    }

    /// Encode pre-converted planar YCbCr image data to JPEG.
    ///
    /// This method accepts tightly packed YCbCr data (no row padding).
    /// For strided data, use [`encode_ycbcr_planar_strided`](Self::encode_ycbcr_planar_strided).
    ///
    /// # Arguments
    /// * `y` - Luma plane (width × height bytes, tightly packed)
    /// * `cb` - Cb chroma plane (chroma_width × chroma_height bytes)
    /// * `cr` - Cr chroma plane (chroma_width × chroma_height bytes)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// The chroma plane dimensions depend on the subsampling mode:
    /// - 4:4:4: chroma_width = width, chroma_height = height
    /// - 4:2:2: chroma_width = ceil(width/2), chroma_height = height
    /// - 4:2:0: chroma_width = ceil(width/2), chroma_height = ceil(height/2)
    ///
    /// # Returns
    /// JPEG-encoded data as a `Vec<u8>`.
    ///
    /// # Errors
    /// Returns an error if plane sizes don't match expected dimensions.
    pub fn encode_ycbcr_planar(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        // For packed data, stride equals width
        let (luma_h, luma_v) = self.subsampling.luma_factors();
        let (chroma_width, _) = sample::subsampled_dimensions(
            width as usize,
            height as usize,
            luma_h as usize,
            luma_v as usize,
        );
        self.encode_ycbcr_planar_strided(
            y,
            width as usize,
            cb,
            chroma_width,
            cr,
            chroma_width,
            width,
            height,
        )
    }

    /// Encode pre-converted planar YCbCr image data to a writer.
    ///
    /// See [`encode_ycbcr_planar`](Self::encode_ycbcr_planar) for details.
    pub fn encode_ycbcr_planar_to_writer<W: Write>(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: u32,
        height: u32,
        output: W,
    ) -> Result<()> {
        // For packed data, stride equals width
        let (luma_h, luma_v) = self.subsampling.luma_factors();
        let (chroma_width, _) = sample::subsampled_dimensions(
            width as usize,
            height as usize,
            luma_h as usize,
            luma_v as usize,
        );
        self.encode_ycbcr_planar_strided_to_writer(
            y,
            width as usize,
            cb,
            chroma_width,
            cr,
            chroma_width,
            width,
            height,
            output,
        )
    }

    /// Encode pre-converted planar YCbCr image data with arbitrary strides.
    ///
    /// This method accepts YCbCr data that has already been:
    /// 1. Converted from RGB to YCbCr color space
    /// 2. Downsampled according to the encoder's subsampling mode
    ///
    /// Use this when you have YCbCr data from video decoders or other sources
    /// that may have row padding (stride > width).
    ///
    /// # Arguments
    /// * `y` - Luma plane data
    /// * `y_stride` - Bytes per row in luma plane (must be >= width)
    /// * `cb` - Cb chroma plane data
    /// * `cb_stride` - Bytes per row in Cb plane (must be >= chroma_width)
    /// * `cr` - Cr chroma plane data
    /// * `cr_stride` - Bytes per row in Cr plane (must be >= chroma_width)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// The chroma plane dimensions depend on the subsampling mode:
    /// - 4:4:4: chroma_width = width, chroma_height = height
    /// - 4:2:2: chroma_width = ceil(width/2), chroma_height = height
    /// - 4:2:0: chroma_width = ceil(width/2), chroma_height = ceil(height/2)
    ///
    /// # Returns
    /// JPEG-encoded data as a `Vec<u8>`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Strides are less than the required width
    /// - Plane sizes don't match stride × height
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ycbcr_planar_strided(
        &self,
        y: &[u8],
        y_stride: usize,
        cb: &[u8],
        cb_stride: usize,
        cr: &[u8],
        cr_stride: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.encode_ycbcr_planar_strided_to_writer(
            y,
            y_stride,
            cb,
            cb_stride,
            cr,
            cr_stride,
            width,
            height,
            &mut output,
        )?;
        Ok(output)
    }

    /// Encode pre-converted planar YCbCr image data with arbitrary strides to a writer.
    ///
    /// See [`encode_ycbcr_planar_strided`](Self::encode_ycbcr_planar_strided) for details.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ycbcr_planar_strided_to_writer<W: Write>(
        &self,
        y: &[u8],
        y_stride: usize,
        cb: &[u8],
        cb_stride: usize,
        cr: &[u8],
        cr_stride: usize,
        width: u32,
        height: u32,
        output: W,
    ) -> Result<()> {
        let width = width as usize;
        let height = height as usize;

        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            });
        }

        // Validate Y stride
        if y_stride < width {
            return Err(Error::InvalidSamplingFactor {
                h: y_stride as u8,
                v: width as u8,
            });
        }

        let (luma_h, luma_v) = self.subsampling.luma_factors();
        let (chroma_width, chroma_height) =
            sample::subsampled_dimensions(width, height, luma_h as usize, luma_v as usize);

        // Validate chroma strides
        if cb_stride < chroma_width {
            return Err(Error::InvalidSamplingFactor {
                h: cb_stride as u8,
                v: chroma_width as u8,
            });
        }
        if cr_stride < chroma_width {
            return Err(Error::InvalidSamplingFactor {
                h: cr_stride as u8,
                v: chroma_width as u8,
            });
        }

        // Calculate expected plane sizes (stride × height)
        let y_size = y_stride
            .checked_mul(height)
            .ok_or(Error::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            })?;
        let cb_size = cb_stride
            .checked_mul(chroma_height)
            .ok_or(Error::AllocationFailed)?;
        let cr_size = cr_stride
            .checked_mul(chroma_height)
            .ok_or(Error::AllocationFailed)?;

        // Validate Y plane size
        if y.len() < y_size {
            return Err(Error::BufferSizeMismatch {
                expected: y_size,
                actual: y.len(),
            });
        }

        // Validate Cb plane size
        if cb.len() < cb_size {
            return Err(Error::BufferSizeMismatch {
                expected: cb_size,
                actual: cb.len(),
            });
        }

        // Validate Cr plane size
        if cr.len() < cr_size {
            return Err(Error::BufferSizeMismatch {
                expected: cr_size,
                actual: cr.len(),
            });
        }

        // Expand planes to MCU-aligned dimensions
        let (mcu_width, mcu_height) =
            sample::mcu_aligned_dimensions(width, height, luma_h as usize, luma_v as usize);
        let (mcu_chroma_w, mcu_chroma_h) =
            (mcu_width / luma_h as usize, mcu_height / luma_v as usize);

        let mcu_y_size = mcu_width
            .checked_mul(mcu_height)
            .ok_or(Error::AllocationFailed)?;
        let mcu_chroma_size = mcu_chroma_w
            .checked_mul(mcu_chroma_h)
            .ok_or(Error::AllocationFailed)?;
        let mut y_mcu = try_alloc_vec(0u8, mcu_y_size)?;
        let mut cb_mcu = try_alloc_vec(0u8, mcu_chroma_size)?;
        let mut cr_mcu = try_alloc_vec(0u8, mcu_chroma_size)?;

        sample::expand_to_mcu_strided(
            y, width, y_stride, height, &mut y_mcu, mcu_width, mcu_height,
        );
        sample::expand_to_mcu_strided(
            cb,
            chroma_width,
            cb_stride,
            chroma_height,
            &mut cb_mcu,
            mcu_chroma_w,
            mcu_chroma_h,
        );
        sample::expand_to_mcu_strided(
            cr,
            chroma_width,
            cr_stride,
            chroma_height,
            &mut cr_mcu,
            mcu_chroma_w,
            mcu_chroma_h,
        );

        // Encode using shared helper
        self.encode_ycbcr_mcu_to_writer(
            &y_mcu,
            &cb_mcu,
            &cr_mcu,
            width,
            height,
            mcu_width,
            mcu_height,
            chroma_width,
            chroma_height,
            mcu_chroma_w,
            mcu_chroma_h,
            output,
        )
    }

    /// Encode RGB image data to a writer.
    pub fn encode_rgb_to_writer<W: Write>(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        output: W,
    ) -> Result<()> {
        let width = width as usize;
        let height = height as usize;

        // Step 1: Convert RGB to YCbCr
        // Use checked arithmetic for num_pixels calculation
        let num_pixels = width.checked_mul(height).ok_or(Error::InvalidDimensions {
            width: width as u32,
            height: height as u32,
        })?;

        let mut y_plane = try_alloc_vec(0u8, num_pixels)?;
        let mut cb_plane = try_alloc_vec(0u8, num_pixels)?;
        let mut cr_plane = try_alloc_vec(0u8, num_pixels)?;

        (self.simd.color_convert_rgb_to_ycbcr)(
            rgb_data,
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            num_pixels,
        );

        // Step 2: Downsample chroma if needed
        let (luma_h, luma_v) = self.subsampling.luma_factors();
        let (chroma_width, chroma_height) =
            sample::subsampled_dimensions(width, height, luma_h as usize, luma_v as usize);

        let chroma_size = chroma_width
            .checked_mul(chroma_height)
            .ok_or(Error::AllocationFailed)?;
        let mut cb_subsampled = try_alloc_vec(0u8, chroma_size)?;
        let mut cr_subsampled = try_alloc_vec(0u8, chroma_size)?;

        sample::downsample_plane(
            &cb_plane,
            width,
            height,
            luma_h as usize,
            luma_v as usize,
            &mut cb_subsampled,
        );
        sample::downsample_plane(
            &cr_plane,
            width,
            height,
            luma_h as usize,
            luma_v as usize,
            &mut cr_subsampled,
        );

        // Step 3: Expand planes to MCU-aligned dimensions
        let (mcu_width, mcu_height) =
            sample::mcu_aligned_dimensions(width, height, luma_h as usize, luma_v as usize);
        let (mcu_chroma_w, mcu_chroma_h) =
            (mcu_width / luma_h as usize, mcu_height / luma_v as usize);

        let mcu_y_size = mcu_width
            .checked_mul(mcu_height)
            .ok_or(Error::AllocationFailed)?;
        let mcu_chroma_size = mcu_chroma_w
            .checked_mul(mcu_chroma_h)
            .ok_or(Error::AllocationFailed)?;
        let mut y_mcu = try_alloc_vec(0u8, mcu_y_size)?;
        let mut cb_mcu = try_alloc_vec(0u8, mcu_chroma_size)?;
        let mut cr_mcu = try_alloc_vec(0u8, mcu_chroma_size)?;

        sample::expand_to_mcu(&y_plane, width, height, &mut y_mcu, mcu_width, mcu_height);
        sample::expand_to_mcu(
            &cb_subsampled,
            chroma_width,
            chroma_height,
            &mut cb_mcu,
            mcu_chroma_w,
            mcu_chroma_h,
        );
        sample::expand_to_mcu(
            &cr_subsampled,
            chroma_width,
            chroma_height,
            &mut cr_mcu,
            mcu_chroma_w,
            mcu_chroma_h,
        );

        // Encode using shared helper
        self.encode_ycbcr_mcu_to_writer(
            &y_mcu,
            &cb_mcu,
            &cr_mcu,
            width,
            height,
            mcu_width,
            mcu_height,
            chroma_width,
            chroma_height,
            mcu_chroma_w,
            mcu_chroma_h,
            output,
        )
    }

    /// Internal helper: Encode MCU-aligned YCbCr planes to JPEG.
    ///
    /// This is the shared encoding logic used by both `encode_rgb_to_writer`
    /// and `encode_ycbcr_planar_to_writer`.
    #[allow(clippy::too_many_arguments)]
    fn encode_ycbcr_mcu_to_writer<W: Write>(
        &self,
        y_mcu: &[u8],
        cb_mcu: &[u8],
        cr_mcu: &[u8],
        width: usize,
        height: usize,
        mcu_width: usize,
        mcu_height: usize,
        chroma_width: usize,
        chroma_height: usize,
        mcu_chroma_w: usize,
        mcu_chroma_h: usize,
        output: W,
    ) -> Result<()> {
        let (luma_h, luma_v) = self.subsampling.luma_factors();

        // Step 4: Create quantization tables
        let (luma_qtable, chroma_qtable) = {
            let (default_luma, default_chroma) =
                create_quant_tables(self.quality, self.quant_table_idx, self.force_baseline);
            let luma = if let Some(ref custom) = self.custom_luma_qtable {
                crate::quant::create_quant_table(custom, self.quality, self.force_baseline)
            } else {
                default_luma
            };
            let chroma = if let Some(ref custom) = self.custom_chroma_qtable {
                crate::quant::create_quant_table(custom, self.quality, self.force_baseline)
            } else {
                default_chroma
            };
            (luma, chroma)
        };

        // Step 5: Create Huffman tables (standard tables)
        let dc_luma_huff = create_std_dc_luma_table();
        let dc_chroma_huff = create_std_dc_chroma_table();
        let ac_luma_huff = create_std_ac_luma_table();
        let ac_chroma_huff = create_std_ac_chroma_table();

        let dc_luma_derived = DerivedTable::from_huff_table(&dc_luma_huff, true)?;
        let dc_chroma_derived = DerivedTable::from_huff_table(&dc_chroma_huff, true)?;
        let ac_luma_derived = DerivedTable::from_huff_table(&ac_luma_huff, false)?;
        let ac_chroma_derived = DerivedTable::from_huff_table(&ac_chroma_huff, false)?;

        // Step 6: Set up components
        let components = create_ycbcr_components(self.subsampling);

        // Step 7: Write JPEG file
        let mut marker_writer = MarkerWriter::new(output);

        // SOI
        marker_writer.write_soi()?;

        // APP0 (JFIF) with pixel density
        marker_writer.write_jfif_app0(
            self.pixel_density.unit as u8,
            self.pixel_density.x,
            self.pixel_density.y,
        )?;

        // APP1 (EXIF) - if present
        if let Some(ref exif) = self.exif_data {
            marker_writer.write_app1_exif(exif)?;
        }

        // ICC profile (if present)
        if let Some(ref icc) = self.icc_profile {
            marker_writer.write_icc_profile(icc)?;
        }

        // Custom APP markers
        for (app_num, data) in &self.custom_markers {
            marker_writer.write_app(*app_num, data)?;
        }

        // DQT (quantization tables in zigzag order) - combined into single marker
        let luma_qtable_zz = natural_to_zigzag(&luma_qtable.values);
        let chroma_qtable_zz = natural_to_zigzag(&chroma_qtable.values);
        marker_writer
            .write_dqt_multiple(&[(0, &luma_qtable_zz, false), (1, &chroma_qtable_zz, false)])?;

        // SOF
        marker_writer.write_sof(
            self.progressive,
            8,
            height as u16,
            width as u16,
            &components,
        )?;

        // DRI (restart interval) - if enabled
        if self.restart_interval > 0 {
            marker_writer.write_dri(self.restart_interval)?;
        }

        // DHT (Huffman tables) - written here for non-optimized modes,
        // or later after frequency counting for optimized modes
        if !self.optimize_huffman {
            // Combine all tables into single DHT marker for smaller file size
            marker_writer.write_dht_multiple(&[
                (0, false, &dc_luma_huff),
                (1, false, &dc_chroma_huff),
                (0, true, &ac_luma_huff),
                (1, true, &ac_chroma_huff),
            ])?;
        }

        if self.progressive {
            // Progressive mode: Store all blocks, then encode multiple scans
            let mcu_rows = mcu_height / (DCTSIZE * luma_v as usize);
            let mcu_cols = mcu_width / (DCTSIZE * luma_h as usize);
            let num_y_blocks = mcu_rows
                .checked_mul(mcu_cols)
                .and_then(|n| n.checked_mul(luma_h as usize))
                .and_then(|n| n.checked_mul(luma_v as usize))
                .ok_or(Error::AllocationFailed)?;
            let num_chroma_blocks = mcu_rows
                .checked_mul(mcu_cols)
                .ok_or(Error::AllocationFailed)?;

            // Collect all quantized blocks
            let mut y_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_y_blocks)?;
            let mut cb_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_chroma_blocks)?;
            let mut cr_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_chroma_blocks)?;

            // Optionally collect raw DCT for DC trellis
            let dc_trellis_enabled = self.trellis.enabled && self.trellis.dc_enabled;
            let mut y_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_y_blocks)?)
            } else {
                None
            };
            let mut cb_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_chroma_blocks)?)
            } else {
                None
            };
            let mut cr_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_chroma_blocks)?)
            } else {
                None
            };

            self.collect_blocks(
                y_mcu,
                mcu_width,
                mcu_height,
                cb_mcu,
                cr_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
                &luma_qtable.values,
                &chroma_qtable.values,
                &ac_luma_derived,
                &ac_chroma_derived,
                &mut y_blocks,
                &mut cb_blocks,
                &mut cr_blocks,
                y_raw_dct.as_deref_mut(),
                cb_raw_dct.as_deref_mut(),
                cr_raw_dct.as_deref_mut(),
                luma_h,
                luma_v,
            )?;

            // Run DC trellis optimization if enabled
            // C mozjpeg processes DC trellis row by row (each row is an independent chain)
            if dc_trellis_enabled {
                let h = luma_h as usize;
                let v = luma_v as usize;
                let y_block_cols = mcu_cols * h;
                let y_block_rows = mcu_rows * v;

                if let Some(ref y_raw) = y_raw_dct {
                    run_dc_trellis_by_row(
                        y_raw,
                        &mut y_blocks,
                        luma_qtable.values[0],
                        &dc_luma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        y_block_rows,
                        y_block_cols,
                        mcu_cols,
                        h,
                        v,
                    );
                }
                // Chroma has 1x1 per MCU, so MCU order = row order
                if let Some(ref cb_raw) = cb_raw_dct {
                    run_dc_trellis_by_row(
                        cb_raw,
                        &mut cb_blocks,
                        chroma_qtable.values[0],
                        &dc_chroma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        mcu_rows,
                        mcu_cols,
                        mcu_cols,
                        1,
                        1,
                    );
                }
                if let Some(ref cr_raw) = cr_raw_dct {
                    run_dc_trellis_by_row(
                        cr_raw,
                        &mut cr_blocks,
                        chroma_qtable.values[0],
                        &dc_chroma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        mcu_rows,
                        mcu_cols,
                        mcu_cols,
                        1,
                        1,
                    );
                }
            }

            // Run EOB optimization if enabled (cross-block EOBRUN optimization)
            if self.trellis.enabled && self.trellis.eob_opt {
                use crate::trellis::{estimate_block_eob_info, optimize_eob_runs};

                // Y component
                let y_eob_info: Vec<_> = y_blocks
                    .iter()
                    .map(|block| estimate_block_eob_info(block, &ac_luma_derived, 1, 63))
                    .collect();
                optimize_eob_runs(&mut y_blocks, &y_eob_info, &ac_luma_derived, 1, 63);

                // Cb component
                let cb_eob_info: Vec<_> = cb_blocks
                    .iter()
                    .map(|block| estimate_block_eob_info(block, &ac_chroma_derived, 1, 63))
                    .collect();
                optimize_eob_runs(&mut cb_blocks, &cb_eob_info, &ac_chroma_derived, 1, 63);

                // Cr component
                let cr_eob_info: Vec<_> = cr_blocks
                    .iter()
                    .map(|block| estimate_block_eob_info(block, &ac_chroma_derived, 1, 63))
                    .collect();
                optimize_eob_runs(&mut cr_blocks, &cr_eob_info, &ac_chroma_derived, 1, 63);
            }

            // Generate progressive scan script
            //
            // TEMPORARY: Always use 4-scan minimal script to avoid refinement scan bugs.
            // Our AC refinement encoding has bugs causing "failed to decode huffman code".
            // TODO: Fix AC refinement encoding and re-enable optimize_scans.
            let scans = if self.optimize_scans {
                // When optimize_scans is enabled, use the scan optimizer to find
                // the best frequency split and Al levels. However, SA refinement
                // (Ah > 0) is currently disabled due to encoding bugs.
                self.optimize_progressive_scans(
                    3, // num_components
                    &y_blocks,
                    &cb_blocks,
                    &cr_blocks,
                    mcu_rows,
                    mcu_cols,
                    luma_h,
                    luma_v,
                    width,
                    height,
                    chroma_width,
                    chroma_height,
                    &dc_luma_derived,
                    &dc_chroma_derived,
                    &ac_luma_derived,
                    &ac_chroma_derived,
                )?
            } else {
                // Use C mozjpeg's 9-scan JCP_MAX_COMPRESSION script.
                // This matches jcparam.c lines 932-947 (the JCP_MAX_COMPRESSION branch).
                // mozjpeg-sys defaults to JCP_MAX_COMPRESSION profile, which uses:
                // - DC with no successive approximation (Al=0)
                // - 8/9 frequency split for luma with successive approximation
                // - No successive approximation for chroma
                generate_mozjpeg_max_compression_scans(3)
            };

            // Build Huffman tables and encode scans
            //
            // When optimize_scans=true, each AC scan gets its own optimal Huffman table
            // written immediately before the scan. This matches C mozjpeg behavior and
            // ensures the trial encoder's size estimates match actual encoded sizes.
            //
            // When optimize_huffman=true, use per-scan AC tables (matching C mozjpeg).
            // C automatically enables optimize_coding for progressive mode and does
            // 2 passes per scan: gather statistics, then output with optimal tables.

            if self.optimize_huffman {
                // Per-scan AC tables mode: DC tables global, AC tables per-scan
                // This matches C mozjpeg's progressive behavior

                // Count DC frequencies for first-pass DC scans only (Ah == 0)
                // DC refinement scans (Ah > 0) don't use Huffman coding - they output raw bits
                let mut dc_luma_freq = FrequencyCounter::new();
                let mut dc_chroma_freq = FrequencyCounter::new();

                for scan in &scans {
                    let is_dc_first_scan = scan.ss == 0 && scan.se == 0 && scan.ah == 0;
                    if is_dc_first_scan {
                        self.count_dc_scan_symbols(
                            scan,
                            &y_blocks,
                            &cb_blocks,
                            &cr_blocks,
                            mcu_rows,
                            mcu_cols,
                            luma_h,
                            luma_v,
                            &mut dc_luma_freq,
                            &mut dc_chroma_freq,
                        );
                    }
                }

                // Generate and write DC tables upfront
                let opt_dc_luma_huff = dc_luma_freq.generate_table()?;
                let opt_dc_chroma_huff = dc_chroma_freq.generate_table()?;
                marker_writer.write_dht_multiple(&[
                    (0, false, &opt_dc_luma_huff),
                    (1, false, &opt_dc_chroma_huff),
                ])?;

                let opt_dc_luma = DerivedTable::from_huff_table(&opt_dc_luma_huff, true)?;
                let opt_dc_chroma = DerivedTable::from_huff_table(&opt_dc_chroma_huff, true)?;

                // Get output writer from marker_writer
                let output = marker_writer.into_inner();
                let mut bit_writer = BitWriter::new(output);

                // Encode each scan with per-scan AC tables
                for scan in &scans {
                    bit_writer.flush()?;
                    let mut inner = bit_writer.into_inner();

                    let is_dc_scan = scan.ss == 0 && scan.se == 0;

                    if !is_dc_scan {
                        // AC scan: build per-scan optimal Huffman table
                        let comp_idx = scan.component_index[0] as usize;
                        let blocks = match comp_idx {
                            0 => &y_blocks,
                            1 => &cb_blocks,
                            2 => &cr_blocks,
                            _ => &y_blocks,
                        };
                        let (block_cols, block_rows) = if comp_idx == 0 {
                            (width.div_ceil(DCTSIZE), height.div_ceil(DCTSIZE))
                        } else {
                            (
                                chroma_width.div_ceil(DCTSIZE),
                                chroma_height.div_ceil(DCTSIZE),
                            )
                        };

                        // Count frequencies for this scan only
                        let mut ac_freq = FrequencyCounter::new();
                        self.count_ac_scan_symbols(
                            scan,
                            blocks,
                            mcu_rows,
                            mcu_cols,
                            luma_h,
                            luma_v,
                            comp_idx,
                            block_cols,
                            block_rows,
                            &mut ac_freq,
                        );

                        // Build optimal table and write DHT
                        let ac_huff = ac_freq.generate_table()?;
                        let table_idx = if comp_idx == 0 { 0 } else { 1 };
                        write_dht_marker(&mut inner, table_idx, true, &ac_huff)?;

                        // Write SOS and encode
                        write_sos_marker(&mut inner, scan, &components)?;
                        bit_writer = BitWriter::new(inner);

                        let ac_derived = DerivedTable::from_huff_table(&ac_huff, false)?;
                        let mut prog_encoder = ProgressiveEncoder::new(&mut bit_writer);

                        self.encode_progressive_scan(
                            scan,
                            &y_blocks,
                            &cb_blocks,
                            &cr_blocks,
                            mcu_rows,
                            mcu_cols,
                            luma_h,
                            luma_v,
                            width,
                            height,
                            chroma_width,
                            chroma_height,
                            &opt_dc_luma,
                            &opt_dc_chroma,
                            &ac_derived,
                            &ac_derived, // Not used for AC scans, but needed for signature
                            &mut prog_encoder,
                        )?;
                        prog_encoder.finish_scan(Some(&ac_derived))?;
                    } else {
                        // DC scan: use global DC tables
                        write_sos_marker(&mut inner, scan, &components)?;
                        bit_writer = BitWriter::new(inner);

                        let mut prog_encoder = ProgressiveEncoder::new(&mut bit_writer);
                        self.encode_progressive_scan(
                            scan,
                            &y_blocks,
                            &cb_blocks,
                            &cr_blocks,
                            mcu_rows,
                            mcu_cols,
                            luma_h,
                            luma_v,
                            width,
                            height,
                            chroma_width,
                            chroma_height,
                            &opt_dc_luma,
                            &opt_dc_chroma,
                            &ac_luma_derived, // Not used for DC scans
                            &ac_chroma_derived,
                            &mut prog_encoder,
                        )?;
                        prog_encoder.finish_scan(None)?;
                    }
                }

                // Flush and write EOI
                bit_writer.flush()?;
                let mut output = bit_writer.into_inner();
                output.write_all(&[0xFF, 0xD9])?;
            } else {
                // Standard tables mode (no optimization)
                let output = marker_writer.into_inner();
                let mut bit_writer = BitWriter::new(output);

                for scan in &scans {
                    bit_writer.flush()?;
                    let mut inner = bit_writer.into_inner();
                    write_sos_marker(&mut inner, scan, &components)?;

                    bit_writer = BitWriter::new(inner);
                    let mut prog_encoder = ProgressiveEncoder::new_standard_tables(&mut bit_writer);

                    self.encode_progressive_scan(
                        scan,
                        &y_blocks,
                        &cb_blocks,
                        &cr_blocks,
                        mcu_rows,
                        mcu_cols,
                        luma_h,
                        luma_v,
                        width,
                        height,
                        chroma_width,
                        chroma_height,
                        &dc_luma_derived,
                        &dc_chroma_derived,
                        &ac_luma_derived,
                        &ac_chroma_derived,
                        &mut prog_encoder,
                    )?;

                    let ac_table = if scan.ss > 0 {
                        if scan.component_index[0] == 0 {
                            Some(&ac_luma_derived)
                        } else {
                            Some(&ac_chroma_derived)
                        }
                    } else {
                        None
                    };
                    prog_encoder.finish_scan(ac_table)?;
                }

                bit_writer.flush()?;
                let mut output = bit_writer.into_inner();
                output.write_all(&[0xFF, 0xD9])?;
            }
        } else if self.optimize_huffman {
            // Baseline mode with Huffman optimization (2-pass)
            // Pass 1: Collect blocks and count frequencies
            let mcu_rows = mcu_height / (DCTSIZE * luma_v as usize);
            let mcu_cols = mcu_width / (DCTSIZE * luma_h as usize);
            let num_y_blocks = mcu_rows
                .checked_mul(mcu_cols)
                .and_then(|n| n.checked_mul(luma_h as usize))
                .and_then(|n| n.checked_mul(luma_v as usize))
                .ok_or(Error::AllocationFailed)?;
            let num_chroma_blocks = mcu_rows
                .checked_mul(mcu_cols)
                .ok_or(Error::AllocationFailed)?;

            let mut y_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_y_blocks)?;
            let mut cb_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_chroma_blocks)?;
            let mut cr_blocks = try_alloc_vec_array::<i16, DCTSIZE2>(num_chroma_blocks)?;

            // Optionally collect raw DCT for DC trellis
            let dc_trellis_enabled = self.trellis.enabled && self.trellis.dc_enabled;
            let mut y_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_y_blocks)?)
            } else {
                None
            };
            let mut cb_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_chroma_blocks)?)
            } else {
                None
            };
            let mut cr_raw_dct = if dc_trellis_enabled {
                Some(try_alloc_vec_array::<i32, DCTSIZE2>(num_chroma_blocks)?)
            } else {
                None
            };

            self.collect_blocks(
                y_mcu,
                mcu_width,
                mcu_height,
                cb_mcu,
                cr_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
                &luma_qtable.values,
                &chroma_qtable.values,
                &ac_luma_derived,
                &ac_chroma_derived,
                &mut y_blocks,
                &mut cb_blocks,
                &mut cr_blocks,
                y_raw_dct.as_deref_mut(),
                cb_raw_dct.as_deref_mut(),
                cr_raw_dct.as_deref_mut(),
                luma_h,
                luma_v,
            )?;

            // Run DC trellis optimization if enabled
            // C mozjpeg processes DC trellis row by row (each row is an independent chain)
            if dc_trellis_enabled {
                let h = luma_h as usize;
                let v = luma_v as usize;
                let y_block_cols = mcu_cols * h;
                let y_block_rows = mcu_rows * v;

                if let Some(ref y_raw) = y_raw_dct {
                    run_dc_trellis_by_row(
                        y_raw,
                        &mut y_blocks,
                        luma_qtable.values[0],
                        &dc_luma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        y_block_rows,
                        y_block_cols,
                        mcu_cols,
                        h,
                        v,
                    );
                }
                // Chroma has 1x1 per MCU, so MCU order = row order
                if let Some(ref cb_raw) = cb_raw_dct {
                    run_dc_trellis_by_row(
                        cb_raw,
                        &mut cb_blocks,
                        chroma_qtable.values[0],
                        &dc_chroma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        mcu_rows,
                        mcu_cols,
                        mcu_cols,
                        1,
                        1,
                    );
                }
                if let Some(ref cr_raw) = cr_raw_dct {
                    run_dc_trellis_by_row(
                        cr_raw,
                        &mut cr_blocks,
                        chroma_qtable.values[0],
                        &dc_chroma_derived,
                        self.trellis.lambda_log_scale1,
                        self.trellis.lambda_log_scale2,
                        mcu_rows,
                        mcu_cols,
                        mcu_cols,
                        1,
                        1,
                    );
                }
            }

            // Count symbol frequencies
            let mut dc_luma_freq = FrequencyCounter::new();
            let mut dc_chroma_freq = FrequencyCounter::new();
            let mut ac_luma_freq = FrequencyCounter::new();
            let mut ac_chroma_freq = FrequencyCounter::new();

            let mut counter = SymbolCounter::new();
            let blocks_per_mcu_y = (luma_h * luma_v) as usize;
            let mut y_idx = 0;
            let mut c_idx = 0;

            for _mcu_row in 0..mcu_rows {
                for _mcu_col in 0..mcu_cols {
                    // Y blocks
                    for _ in 0..blocks_per_mcu_y {
                        counter.count_block(
                            &y_blocks[y_idx],
                            0,
                            &mut dc_luma_freq,
                            &mut ac_luma_freq,
                        );
                        y_idx += 1;
                    }
                    // Cb block
                    counter.count_block(
                        &cb_blocks[c_idx],
                        1,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    // Cr block
                    counter.count_block(
                        &cr_blocks[c_idx],
                        2,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    c_idx += 1;
                }
            }

            // Generate optimized Huffman tables
            let opt_dc_luma_huff = dc_luma_freq.generate_table()?;
            let opt_dc_chroma_huff = dc_chroma_freq.generate_table()?;
            let opt_ac_luma_huff = ac_luma_freq.generate_table()?;
            let opt_ac_chroma_huff = ac_chroma_freq.generate_table()?;

            let opt_dc_luma = DerivedTable::from_huff_table(&opt_dc_luma_huff, true)?;
            let opt_dc_chroma = DerivedTable::from_huff_table(&opt_dc_chroma_huff, true)?;
            let opt_ac_luma = DerivedTable::from_huff_table(&opt_ac_luma_huff, false)?;
            let opt_ac_chroma = DerivedTable::from_huff_table(&opt_ac_chroma_huff, false)?;

            // Write DHT with optimized tables - combined into single marker
            marker_writer.write_dht_multiple(&[
                (0, false, &opt_dc_luma_huff),
                (1, false, &opt_dc_chroma_huff),
                (0, true, &opt_ac_luma_huff),
                (1, true, &opt_ac_chroma_huff),
            ])?;

            // Write SOS and encode
            let scans = generate_baseline_scan(3);
            let scan = &scans[0];
            marker_writer.write_sos(scan, &components)?;

            let mut output = marker_writer.into_inner();

            // Use SIMD entropy encoder on x86_64 for ~2x faster encoding
            #[cfg(target_arch = "x86_64")]
            {
                let mut simd_entropy = SimdEntropyEncoder::new();

                // Encode from stored blocks with restart marker support
                y_idx = 0;
                c_idx = 0;
                let restart_interval = self.restart_interval as usize;
                let mut mcu_count = 0usize;
                let mut restart_num = 0u8;

                for _mcu_row in 0..mcu_rows {
                    for _mcu_col in 0..mcu_cols {
                        // Emit restart marker if needed (before this MCU, not first)
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count.is_multiple_of(restart_interval)
                        {
                            simd_entropy.emit_restart(restart_num);
                            restart_num = restart_num.wrapping_add(1) & 0x07;
                        }

                        // Y blocks
                        for _ in 0..blocks_per_mcu_y {
                            simd_entropy.encode_block(
                                &y_blocks[y_idx],
                                0,
                                &opt_dc_luma,
                                &opt_ac_luma,
                            );
                            y_idx += 1;
                        }
                        // Cb block
                        simd_entropy.encode_block(
                            &cb_blocks[c_idx],
                            1,
                            &opt_dc_chroma,
                            &opt_ac_chroma,
                        );
                        // Cr block
                        simd_entropy.encode_block(
                            &cr_blocks[c_idx],
                            2,
                            &opt_dc_chroma,
                            &opt_ac_chroma,
                        );
                        c_idx += 1;
                        mcu_count += 1;
                    }
                }

                simd_entropy.flush();
                output.write_all(simd_entropy.get_buffer())?;
            }

            // Fallback for non-x86_64 platforms
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut bit_writer = BitWriter::new(output);
                let mut entropy = EntropyEncoder::new(&mut bit_writer);

                // Encode from stored blocks with restart marker support
                y_idx = 0;
                c_idx = 0;
                let restart_interval = self.restart_interval as usize;
                let mut mcu_count = 0usize;
                let mut restart_num = 0u8;

                for _mcu_row in 0..mcu_rows {
                    for _mcu_col in 0..mcu_cols {
                        // Emit restart marker if needed (before this MCU, not first)
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count.is_multiple_of(restart_interval)
                        {
                            entropy.emit_restart(restart_num)?;
                            restart_num = restart_num.wrapping_add(1) & 0x07;
                        }

                        // Y blocks
                        for _ in 0..blocks_per_mcu_y {
                            entropy.encode_block(
                                &y_blocks[y_idx],
                                0,
                                &opt_dc_luma,
                                &opt_ac_luma,
                            )?;
                            y_idx += 1;
                        }
                        // Cb block
                        entropy.encode_block(
                            &cb_blocks[c_idx],
                            1,
                            &opt_dc_chroma,
                            &opt_ac_chroma,
                        )?;
                        // Cr block
                        entropy.encode_block(
                            &cr_blocks[c_idx],
                            2,
                            &opt_dc_chroma,
                            &opt_ac_chroma,
                        )?;
                        c_idx += 1;
                        mcu_count += 1;
                    }
                }

                bit_writer.flush()?;
                output = bit_writer.into_inner();
            }

            output.write_all(&[0xFF, 0xD9])?;
        } else {
            // Baseline mode: Encode directly (streaming)
            let scans = generate_baseline_scan(3);
            let scan = &scans[0]; // Baseline has only one scan
            marker_writer.write_sos(scan, &components)?;

            // Encode MCU data
            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);
            let mut entropy = EntropyEncoder::new(&mut bit_writer);

            self.encode_mcus(
                y_mcu,
                mcu_width,
                mcu_height,
                cb_mcu,
                cr_mcu,
                mcu_chroma_w,
                mcu_chroma_h,
                &luma_qtable.values,
                &chroma_qtable.values,
                &dc_luma_derived,
                &dc_chroma_derived,
                &ac_luma_derived,
                &ac_chroma_derived,
                &mut entropy,
                luma_h,
                luma_v,
            )?;

            // Flush bits and get output back
            bit_writer.flush()?;
            let mut output = bit_writer.into_inner();

            // EOI
            output.write_all(&[0xFF, 0xD9])?;
        }

        Ok(())
    }

    /// Encode all MCUs (Minimum Coded Units).
    #[allow(clippy::too_many_arguments)]
    fn encode_mcus<W: Write>(
        &self,
        y_plane: &[u8],
        y_width: usize,
        y_height: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        chroma_width: usize,
        _chroma_height: usize,
        luma_qtable: &[u16; DCTSIZE2],
        chroma_qtable: &[u16; DCTSIZE2],
        dc_luma: &DerivedTable,
        dc_chroma: &DerivedTable,
        ac_luma: &DerivedTable,
        ac_chroma: &DerivedTable,
        entropy: &mut EntropyEncoder<W>,
        h_samp: u8,
        v_samp: u8,
    ) -> Result<()> {
        let mcu_rows = y_height / (DCTSIZE * v_samp as usize);
        let mcu_cols = y_width / (DCTSIZE * h_samp as usize);
        let total_mcus = mcu_rows * mcu_cols;

        let mut dct_block = [0i16; DCTSIZE2];
        let mut quant_block = [0i16; DCTSIZE2];

        // Restart marker tracking
        let restart_interval = self.restart_interval as usize;
        let mut mcu_count = 0usize;
        let mut restart_num = 0u8;

        for mcu_row in 0..mcu_rows {
            for mcu_col in 0..mcu_cols {
                // Check if we need to emit a restart marker BEFORE this MCU
                // (except for the first MCU)
                if restart_interval > 0
                    && mcu_count > 0
                    && mcu_count.is_multiple_of(restart_interval)
                {
                    entropy.emit_restart(restart_num)?;
                    restart_num = restart_num.wrapping_add(1) & 0x07;
                }

                // Encode Y blocks (may be multiple per MCU for subsampling)
                for v in 0..v_samp as usize {
                    for h in 0..h_samp as usize {
                        let block_row = mcu_row * v_samp as usize + v;
                        let block_col = mcu_col * h_samp as usize + h;

                        self.encode_block(
                            y_plane,
                            y_width,
                            block_row,
                            block_col,
                            luma_qtable,
                            dc_luma,
                            ac_luma,
                            0, // Y component
                            entropy,
                            &mut dct_block,
                            &mut quant_block,
                        )?;
                    }
                }

                // Encode Cb block
                self.encode_block(
                    cb_plane,
                    chroma_width,
                    mcu_row,
                    mcu_col,
                    chroma_qtable,
                    dc_chroma,
                    ac_chroma,
                    1, // Cb component
                    entropy,
                    &mut dct_block,
                    &mut quant_block,
                )?;

                // Encode Cr block
                self.encode_block(
                    cr_plane,
                    chroma_width,
                    mcu_row,
                    mcu_col,
                    chroma_qtable,
                    dc_chroma,
                    ac_chroma,
                    2, // Cr component
                    entropy,
                    &mut dct_block,
                    &mut quant_block,
                )?;

                mcu_count += 1;
            }
        }

        // Suppress unused variable warning
        let _ = total_mcus;

        Ok(())
    }

    /// Encode a single 8x8 block.
    #[allow(clippy::too_many_arguments)]
    fn encode_block<W: Write>(
        &self,
        plane: &[u8],
        plane_width: usize,
        block_row: usize,
        block_col: usize,
        qtable: &[u16; DCTSIZE2],
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
        component: usize,
        entropy: &mut EntropyEncoder<W>,
        dct_block: &mut [i16; DCTSIZE2],
        quant_block: &mut [i16; DCTSIZE2],
    ) -> Result<()> {
        // Extract 8x8 block from plane
        let mut samples = [0u8; DCTSIZE2];
        let base_y = block_row * DCTSIZE;
        let base_x = block_col * DCTSIZE;

        for row in 0..DCTSIZE {
            let src_offset = (base_y + row) * plane_width + base_x;
            let dst_offset = row * DCTSIZE;
            samples[dst_offset..dst_offset + DCTSIZE]
                .copy_from_slice(&plane[src_offset..src_offset + DCTSIZE]);
        }

        // Level shift (center around 0 for DCT)
        let mut shifted = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            shifted[i] = (samples[i] as i16) - 128;
        }

        // Apply overshoot deringing if enabled (reduces ringing on white backgrounds)
        if self.overshoot_deringing {
            preprocess_deringing(&mut shifted, qtable[0]);
        }

        // Forward DCT (output scaled by factor of 8)
        self.simd.do_forward_dct(&shifted, dct_block);

        // Convert to i32 for quantization
        let mut dct_i32 = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            dct_i32[i] = dct_block[i] as i32;
        }

        // Use trellis quantization if enabled
        // Both paths expect raw DCT (scaled by 8) and handle the scaling internally
        if self.trellis.enabled {
            trellis_quantize_block(&dct_i32, quant_block, qtable, ac_table, &self.trellis);
        } else {
            // Non-trellis path: use single-step quantization matching C mozjpeg
            // This takes raw DCT (scaled by 8) and uses q_scaled = 8 * qtable[i]
            quantize_block_raw(&dct_i32, qtable, quant_block);
        }

        // Entropy encode
        entropy.encode_block(quant_block, component, dc_table, ac_table)?;

        Ok(())
    }

    /// Collect all quantized DCT blocks for progressive encoding.
    /// Also collects raw DCT blocks if DC trellis is enabled.
    #[allow(clippy::too_many_arguments)]
    fn collect_blocks(
        &self,
        y_plane: &[u8],
        y_width: usize,
        y_height: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        chroma_width: usize,
        _chroma_height: usize,
        luma_qtable: &[u16; DCTSIZE2],
        chroma_qtable: &[u16; DCTSIZE2],
        ac_luma: &DerivedTable,
        ac_chroma: &DerivedTable,
        y_blocks: &mut [[i16; DCTSIZE2]],
        cb_blocks: &mut [[i16; DCTSIZE2]],
        cr_blocks: &mut [[i16; DCTSIZE2]],
        mut y_raw_dct: Option<&mut [[i32; DCTSIZE2]]>,
        mut cb_raw_dct: Option<&mut [[i32; DCTSIZE2]]>,
        mut cr_raw_dct: Option<&mut [[i32; DCTSIZE2]]>,
        h_samp: u8,
        v_samp: u8,
    ) -> Result<()> {
        let mcu_rows = y_height / (DCTSIZE * v_samp as usize);
        let mcu_cols = y_width / (DCTSIZE * h_samp as usize);

        let mut y_idx = 0;
        let mut c_idx = 0;
        let mut dct_block = [0i16; DCTSIZE2];

        for mcu_row in 0..mcu_rows {
            for mcu_col in 0..mcu_cols {
                // Collect Y blocks (may be multiple per MCU for subsampling)
                for v in 0..v_samp as usize {
                    for h in 0..h_samp as usize {
                        let block_row = mcu_row * v_samp as usize + v;
                        let block_col = mcu_col * h_samp as usize + h;

                        // Get mutable reference to raw DCT output if collecting
                        let raw_dct_out = y_raw_dct.as_mut().map(|arr| &mut arr[y_idx][..]);
                        self.process_block_to_storage_with_raw(
                            y_plane,
                            y_width,
                            block_row,
                            block_col,
                            luma_qtable,
                            ac_luma,
                            &mut y_blocks[y_idx],
                            &mut dct_block,
                            raw_dct_out,
                        )?;
                        y_idx += 1;
                    }
                }

                // Collect Cb block
                let raw_dct_out = cb_raw_dct.as_mut().map(|arr| &mut arr[c_idx][..]);
                self.process_block_to_storage_with_raw(
                    cb_plane,
                    chroma_width,
                    mcu_row,
                    mcu_col,
                    chroma_qtable,
                    ac_chroma,
                    &mut cb_blocks[c_idx],
                    &mut dct_block,
                    raw_dct_out,
                )?;

                // Collect Cr block
                let raw_dct_out = cr_raw_dct.as_mut().map(|arr| &mut arr[c_idx][..]);
                self.process_block_to_storage_with_raw(
                    cr_plane,
                    chroma_width,
                    mcu_row,
                    mcu_col,
                    chroma_qtable,
                    ac_chroma,
                    &mut cr_blocks[c_idx],
                    &mut dct_block,
                    raw_dct_out,
                )?;

                c_idx += 1;
            }
        }

        Ok(())
    }

    /// Process a block: DCT + quantize, storing the result.
    /// Optionally stores raw DCT coefficients for DC trellis.
    #[allow(clippy::too_many_arguments)]
    fn process_block_to_storage_with_raw(
        &self,
        plane: &[u8],
        plane_width: usize,
        block_row: usize,
        block_col: usize,
        qtable: &[u16; DCTSIZE2],
        ac_table: &DerivedTable,
        out_block: &mut [i16; DCTSIZE2],
        dct_block: &mut [i16; DCTSIZE2],
        raw_dct_out: Option<&mut [i32]>,
    ) -> Result<()> {
        // Extract 8x8 block from plane
        let mut samples = [0u8; DCTSIZE2];
        let base_y = block_row * DCTSIZE;
        let base_x = block_col * DCTSIZE;

        for row in 0..DCTSIZE {
            let src_offset = (base_y + row) * plane_width + base_x;
            let dst_offset = row * DCTSIZE;
            samples[dst_offset..dst_offset + DCTSIZE]
                .copy_from_slice(&plane[src_offset..src_offset + DCTSIZE]);
        }

        // Level shift (center around 0 for DCT)
        let mut shifted = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            shifted[i] = (samples[i] as i16) - 128;
        }

        // Apply overshoot deringing if enabled (reduces ringing on white backgrounds)
        if self.overshoot_deringing {
            preprocess_deringing(&mut shifted, qtable[0]);
        }

        // Forward DCT (output scaled by factor of 8)
        self.simd.do_forward_dct(&shifted, dct_block);

        // Convert to i32 for quantization
        let mut dct_i32 = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            dct_i32[i] = dct_block[i] as i32;
        }

        // Store raw DCT if requested (for DC trellis)
        if let Some(raw_out) = raw_dct_out {
            raw_out.copy_from_slice(&dct_i32);
        }

        // Use trellis quantization if enabled
        // Both paths expect raw DCT (scaled by 8) and handle the scaling internally
        if self.trellis.enabled {
            trellis_quantize_block(&dct_i32, out_block, qtable, ac_table, &self.trellis);
        } else {
            // Non-trellis path: use single-step quantization matching C mozjpeg
            // This takes raw DCT (scaled by 8) and uses q_scaled = 8 * qtable[i]
            quantize_block_raw(&dct_i32, qtable, out_block);
        }

        Ok(())
    }

    /// Optimize progressive scan configuration (C mozjpeg-compatible).
    ///
    /// This implements the optimize_scans feature from C mozjpeg:
    /// 1. Generate 64 individual candidate scans
    /// 2. Trial-encode scans SEQUENTIALLY to get accurate sizes
    /// 3. Use ScanSelector to find optimal Al levels and frequency splits
    /// 4. Build the final scan script from the selection
    ///
    /// IMPORTANT: Scans must be encoded sequentially (not independently) because
    /// refinement scans (Ah > 0) need context from previous scans to produce
    /// correct output sizes.
    #[allow(clippy::too_many_arguments)]
    fn optimize_progressive_scans(
        &self,
        num_components: u8,
        y_blocks: &[[i16; DCTSIZE2]],
        cb_blocks: &[[i16; DCTSIZE2]],
        cr_blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        actual_width: usize,
        actual_height: usize,
        chroma_width: usize,
        chroma_height: usize,
        dc_luma: &DerivedTable,
        dc_chroma: &DerivedTable,
        ac_luma: &DerivedTable,
        ac_chroma: &DerivedTable,
    ) -> Result<Vec<crate::types::ScanInfo>> {
        let config = ScanSearchConfig::default();
        let candidate_scans = generate_search_scans(num_components, &config);

        // Use ScanTrialEncoder for sequential trial encoding with proper state tracking
        let mut trial_encoder = ScanTrialEncoder::new(
            y_blocks,
            cb_blocks,
            cr_blocks,
            dc_luma,
            dc_chroma,
            ac_luma,
            ac_chroma,
            mcu_rows,
            mcu_cols,
            h_samp,
            v_samp,
            actual_width,
            actual_height,
            chroma_width,
            chroma_height,
        );

        // Trial-encode all scans sequentially to get accurate sizes
        let scan_sizes = trial_encoder.encode_all_scans(&candidate_scans)?;

        // Use ScanSelector to find the optimal configuration
        let selector = ScanSelector::new(num_components, config.clone());
        let result = selector.select_best(&scan_sizes);

        // Build the final scan script from the selection
        Ok(result.build_final_scans(num_components, &config))
    }

    /// Encode a single progressive scan.
    #[allow(clippy::too_many_arguments)]
    fn encode_progressive_scan<W: Write>(
        &self,
        scan: &crate::types::ScanInfo,
        y_blocks: &[[i16; DCTSIZE2]],
        cb_blocks: &[[i16; DCTSIZE2]],
        cr_blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        actual_width: usize,
        actual_height: usize,
        chroma_width: usize,
        chroma_height: usize,
        dc_luma: &DerivedTable,
        dc_chroma: &DerivedTable,
        ac_luma: &DerivedTable,
        ac_chroma: &DerivedTable,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        let is_dc_scan = scan.ss == 0 && scan.se == 0;
        let is_refinement = scan.ah != 0;

        if is_dc_scan {
            // DC scan - can be interleaved (multiple components)
            self.encode_dc_scan(
                scan,
                y_blocks,
                cb_blocks,
                cr_blocks,
                mcu_rows,
                mcu_cols,
                h_samp,
                v_samp,
                dc_luma,
                dc_chroma,
                is_refinement,
                encoder,
            )?;
        } else {
            // AC scan - single component only (non-interleaved)
            // For non-interleaved scans, use actual component block dimensions
            let comp_idx = scan.component_index[0] as usize;
            let blocks = match comp_idx {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => return Err(Error::InvalidComponentIndex(comp_idx)),
            };
            let ac_table = if comp_idx == 0 { ac_luma } else { ac_chroma };

            // Calculate actual block dimensions for this component.
            // Non-interleaved AC scans encode only the actual image blocks, not MCU padding.
            // This differs from interleaved DC scans which encode all MCU blocks.
            // Reference: ITU-T T.81 Section F.2.3
            let (block_cols, block_rows) = if comp_idx == 0 {
                // Y component: full resolution
                (
                    actual_width.div_ceil(DCTSIZE),
                    actual_height.div_ceil(DCTSIZE),
                )
            } else {
                // Chroma components: subsampled resolution
                (
                    chroma_width.div_ceil(DCTSIZE),
                    chroma_height.div_ceil(DCTSIZE),
                )
            };

            self.encode_ac_scan(
                scan,
                blocks,
                mcu_rows,
                mcu_cols,
                h_samp,
                v_samp,
                comp_idx,
                block_cols,
                block_rows,
                ac_table,
                is_refinement,
                encoder,
            )?;
        }

        Ok(())
    }

    /// Encode a DC scan (Ss=Se=0).
    #[allow(clippy::too_many_arguments)]
    fn encode_dc_scan<W: Write>(
        &self,
        scan: &crate::types::ScanInfo,
        y_blocks: &[[i16; DCTSIZE2]],
        cb_blocks: &[[i16; DCTSIZE2]],
        cr_blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        dc_luma: &DerivedTable,
        dc_chroma: &DerivedTable,
        is_refinement: bool,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        let blocks_per_mcu_y = (h_samp * v_samp) as usize;
        let mut y_idx = 0;
        let mut c_idx = 0;

        for _mcu_row in 0..mcu_rows {
            for _mcu_col in 0..mcu_cols {
                // Encode Y blocks
                for _ in 0..blocks_per_mcu_y {
                    if is_refinement {
                        encoder.encode_dc_refine(&y_blocks[y_idx], scan.al)?;
                    } else {
                        encoder.encode_dc_first(&y_blocks[y_idx], 0, dc_luma, scan.al)?;
                    }
                    y_idx += 1;
                }

                // Encode Cb
                if is_refinement {
                    encoder.encode_dc_refine(&cb_blocks[c_idx], scan.al)?;
                } else {
                    encoder.encode_dc_first(&cb_blocks[c_idx], 1, dc_chroma, scan.al)?;
                }

                // Encode Cr
                if is_refinement {
                    encoder.encode_dc_refine(&cr_blocks[c_idx], scan.al)?;
                } else {
                    encoder.encode_dc_first(&cr_blocks[c_idx], 2, dc_chroma, scan.al)?;
                }

                c_idx += 1;
            }
        }

        Ok(())
    }

    /// Encode an AC scan (Ss > 0).
    ///
    /// **IMPORTANT**: Progressive AC scans are always non-interleaved, meaning blocks
    /// must be encoded in component raster order (row-major within the component's
    /// block grid), NOT in MCU-interleaved order.
    ///
    /// For non-interleaved scans, the number of blocks is determined by the actual
    /// component dimensions (ceil(width/8) × ceil(height/8)), NOT the MCU-padded
    /// dimensions. This is different from interleaved DC scans which use MCU order.
    /// The padding blocks (beyond actual image dimensions) have DC coefficients but
    /// no AC coefficients - the decoder only outputs the actual image dimensions.
    ///
    /// Reference: ITU-T T.81 Section F.2.3 - "The scan data for a non-interleaved
    /// scan shall consist of a sequence of entropy-coded segments... The data units
    /// are processed in the order defined by the scan component."
    #[allow(clippy::too_many_arguments)]
    fn encode_ac_scan<W: Write>(
        &self,
        scan: &crate::types::ScanInfo,
        blocks: &[[i16; DCTSIZE2]],
        _mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        comp_idx: usize,
        block_cols: usize,
        block_rows: usize,
        ac_table: &DerivedTable,
        is_refinement: bool,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        // For Y component with subsampling, blocks are stored in MCU-interleaved order
        // but AC scans must encode them in component raster order.
        // For chroma components (1 block per MCU), the orders are identical.
        //
        // For non-interleaved scans, encode only the actual image blocks (block_rows × block_cols),
        // not all MCU-padded blocks. Padding blocks have DC coefficients but no AC coefficients.

        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            let total_blocks = block_rows * block_cols;
            for block in blocks.iter().take(total_blocks) {
                if is_refinement {
                    encoder
                        .encode_ac_refine(block, scan.ss, scan.se, scan.ah, scan.al, ac_table)?;
                } else {
                    encoder.encode_ac_first(block, scan.ss, scan.se, scan.al, ac_table)?;
                }
            }
        } else {
            // Y component with subsampling (h_samp > 1 or v_samp > 1)
            // Convert from MCU-interleaved storage to component raster order
            let h = h_samp as usize;
            let v = v_samp as usize;

            for block_row in 0..block_rows {
                for block_col in 0..block_cols {
                    // Convert raster position to MCU-interleaved storage index
                    let mcu_row = block_row / v;
                    let mcu_col = block_col / h;
                    let v_idx = block_row % v;
                    let h_idx = block_col % h;
                    let storage_idx = mcu_row * (mcu_cols * blocks_per_mcu)
                        + mcu_col * blocks_per_mcu
                        + v_idx * h
                        + h_idx;

                    if is_refinement {
                        encoder.encode_ac_refine(
                            &blocks[storage_idx],
                            scan.ss,
                            scan.se,
                            scan.ah,
                            scan.al,
                            ac_table,
                        )?;
                    } else {
                        encoder.encode_ac_first(
                            &blocks[storage_idx],
                            scan.ss,
                            scan.se,
                            scan.al,
                            ac_table,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Count DC symbols for a progressive DC scan.
    #[allow(clippy::too_many_arguments)]
    fn count_dc_scan_symbols(
        &self,
        scan: &crate::types::ScanInfo,
        y_blocks: &[[i16; DCTSIZE2]],
        cb_blocks: &[[i16; DCTSIZE2]],
        cr_blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        dc_luma_freq: &mut FrequencyCounter,
        dc_chroma_freq: &mut FrequencyCounter,
    ) {
        let blocks_per_mcu_y = (h_samp * v_samp) as usize;
        let mut y_idx = 0;
        let mut c_idx = 0;
        let mut counter = ProgressiveSymbolCounter::new();

        for _mcu_row in 0..mcu_rows {
            for _mcu_col in 0..mcu_cols {
                // Y blocks
                for _ in 0..blocks_per_mcu_y {
                    counter.count_dc_first(&y_blocks[y_idx], 0, scan.al, dc_luma_freq);
                    y_idx += 1;
                }
                // Cb block
                counter.count_dc_first(&cb_blocks[c_idx], 1, scan.al, dc_chroma_freq);
                // Cr block
                counter.count_dc_first(&cr_blocks[c_idx], 2, scan.al, dc_chroma_freq);
                c_idx += 1;
            }
        }
    }

    /// Count AC symbols for a progressive AC scan.
    ///
    /// Must iterate blocks in the same order as `encode_ac_scan` (component raster order)
    /// to ensure EOBRUN counts match and Huffman tables are correct.
    ///
    /// Uses actual block dimensions (not MCU-padded) for non-interleaved scans.
    #[allow(clippy::too_many_arguments)]
    fn count_ac_scan_symbols(
        &self,
        scan: &crate::types::ScanInfo,
        blocks: &[[i16; DCTSIZE2]],
        _mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        comp_idx: usize,
        block_cols: usize,
        block_rows: usize,
        ac_freq: &mut FrequencyCounter,
    ) {
        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        let mut counter = ProgressiveSymbolCounter::new();
        let is_refinement = scan.ah != 0;

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            let total_blocks = block_rows * block_cols;
            for block in blocks.iter().take(total_blocks) {
                if is_refinement {
                    counter.count_ac_refine(block, scan.ss, scan.se, scan.ah, scan.al, ac_freq);
                } else {
                    counter.count_ac_first(block, scan.ss, scan.se, scan.al, ac_freq);
                }
            }
        } else {
            // Y component with subsampling - iterate in raster order (matching encode_ac_scan)
            let h = h_samp as usize;
            let v = v_samp as usize;

            for block_row in 0..block_rows {
                for block_col in 0..block_cols {
                    // Convert raster position to MCU-interleaved storage index
                    let mcu_row = block_row / v;
                    let mcu_col = block_col / h;
                    let v_idx = block_row % v;
                    let h_idx = block_col % h;
                    let storage_idx = mcu_row * (mcu_cols * blocks_per_mcu)
                        + mcu_col * blocks_per_mcu
                        + v_idx * h
                        + h_idx;

                    if is_refinement {
                        counter.count_ac_refine(
                            &blocks[storage_idx],
                            scan.ss,
                            scan.se,
                            scan.ah,
                            scan.al,
                            ac_freq,
                        );
                    } else {
                        counter.count_ac_first(
                            &blocks[storage_idx],
                            scan.ss,
                            scan.se,
                            scan.al,
                            ac_freq,
                        );
                    }
                }
            }
        }

        // Flush any pending EOBRUN
        counter.finish_scan(Some(ac_freq));
    }
}

// ============================================================================
// Encode Trait Implementation
// ============================================================================

impl Encode for Encoder {
    fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        self.encode_rgb(rgb_data, width, height)
    }

    fn encode_gray(&self, gray_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        self.encode_gray(gray_data, width, height)
    }
}

// Note: StreamingEncoder and EncodingStream are in the `streaming` module.

// Add streaming() method to Encoder
impl Encoder {
    /// Create a streaming encoder.
    ///
    /// Returns a [`StreamingEncoder`] which supports scanline-by-scanline encoding.
    /// Note that streaming mode does NOT support trellis quantization, progressive
    /// mode, or Huffman optimization (these require buffering the entire image).
    ///
    /// For full-featured encoding with all mozjpeg optimizations, use [`Encoder::new(Preset)`]
    /// with [`encode_rgb()`](Encoder::encode_rgb) or [`encode_gray()`](Encoder::encode_gray).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use mozjpeg_rs::Encoder;
    /// use std::fs::File;
    ///
    /// let file = File::create("output.jpg")?;
    /// let mut stream = Encoder::streaming()
    ///     .quality(85)
    ///     .start_rgb(1920, 1080, file)?;
    ///
    /// // Write scanlines...
    /// stream.finish()?;
    /// ```
    pub fn streaming() -> StreamingEncoder {
        StreamingEncoder::baseline_fastest()
    }
}

// ============================================================================
// C mozjpeg encoding (optional feature)
// ============================================================================

#[cfg(feature = "mozjpeg-sys-config")]
impl Encoder {
    /// Convert this encoder to a C mozjpeg encoder.
    ///
    /// Returns a [`CMozjpeg`](crate::CMozjpeg) that can encode images using
    /// the C mozjpeg library with settings matching this Rust encoder.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::{Encoder, Preset};
    ///
    /// let pixels: Vec<u8> = vec![128; 64 * 64 * 3];
    /// let encoder = Encoder::new(Preset::ProgressiveBalanced).quality(85);
    ///
    /// // Encode with C mozjpeg
    /// let c_jpeg = encoder.to_c_mozjpeg().encode_rgb(&pixels, 64, 64)?;
    ///
    /// // Compare with Rust encoder
    /// let rust_jpeg = encoder.encode_rgb(&pixels, 64, 64)?;
    /// # Ok::<(), mozjpeg_rs::Error>(())
    /// ```
    pub fn to_c_mozjpeg(&self) -> crate::compat::CMozjpeg {
        crate::compat::CMozjpeg {
            quality: self.quality,
            force_baseline: self.force_baseline,
            subsampling: self.subsampling,
            progressive: self.progressive,
            optimize_huffman: self.optimize_huffman,
            optimize_scans: self.optimize_scans,
            trellis: self.trellis,
            overshoot_deringing: self.overshoot_deringing,
            smoothing: self.smoothing,
            restart_interval: self.restart_interval,
            quant_table_idx: self.quant_table_idx,
            has_custom_qtables: self.custom_luma_qtable.is_some()
                || self.custom_chroma_qtable.is_some(),
            exif_data: self.exif_data.clone(),
            icc_profile: self.icc_profile.clone(),
            custom_markers: self.custom_markers.clone(),
        }
    }
}

/// Unit tests for private encoder internals.
/// Public API tests are in tests/encode_tests.rs.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_defaults() {
        // Default preset is ProgressiveBalanced
        let enc = Encoder::default();
        assert_eq!(enc.quality, 75);
        assert!(enc.progressive); // ProgressiveBalanced is progressive
        assert_eq!(enc.subsampling, Subsampling::S420);
        assert!(enc.trellis.enabled);
        assert!(enc.optimize_huffman);
        assert!(!enc.optimize_scans); // ProgressiveBalanced does NOT include optimize_scans
    }

    #[test]
    fn test_encoder_presets() {
        let fastest = Encoder::new(Preset::BaselineFastest);
        assert!(!fastest.progressive);
        assert!(!fastest.trellis.enabled);
        assert!(!fastest.optimize_huffman);

        let baseline = Encoder::new(Preset::BaselineBalanced);
        assert!(!baseline.progressive);
        assert!(baseline.trellis.enabled);
        assert!(baseline.optimize_huffman);

        let prog_balanced = Encoder::new(Preset::ProgressiveBalanced);
        assert!(prog_balanced.progressive);
        assert!(prog_balanced.trellis.enabled);
        assert!(!prog_balanced.optimize_scans);

        let prog_smallest = Encoder::new(Preset::ProgressiveSmallest);
        assert!(prog_smallest.progressive);
        assert!(prog_smallest.optimize_scans);
    }

    #[test]
    fn test_encoder_builder_fields() {
        let enc = Encoder::baseline_optimized()
            .quality(90)
            .progressive(true)
            .subsampling(Subsampling::S444);

        assert_eq!(enc.quality, 90);
        assert!(enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S444);
    }

    #[test]
    fn test_quality_clamping() {
        let enc = Encoder::baseline_optimized().quality(0);
        assert_eq!(enc.quality, 1);

        let enc = Encoder::baseline_optimized().quality(150);
        assert_eq!(enc.quality, 100);
    }

    #[test]
    fn test_natural_to_zigzag() {
        let mut natural = [0u16; 64];
        for i in 0..64 {
            natural[i] = i as u16;
        }
        let zigzag = natural_to_zigzag(&natural);

        assert_eq!(zigzag[0], 0);
        assert_eq!(zigzag[1], 1);
    }

    #[test]
    fn test_max_compression_uses_all_optimizations() {
        let encoder = Encoder::max_compression();
        assert!(encoder.trellis.enabled);
        assert!(encoder.progressive);
        assert!(encoder.optimize_huffman);
        assert!(encoder.optimize_scans);
    }

    #[test]
    fn test_encode_ycbcr_planar_444() {
        let width = 32u32;
        let height = 32u32;

        // Create test image with gradient pattern
        let y_plane: Vec<u8> = (0..width * height)
            .map(|i| ((i % width) * 255 / width) as u8)
            .collect();
        let cb_plane: Vec<u8> = (0..width * height)
            .map(|i| ((i / width) * 255 / height) as u8)
            .collect();
        let cr_plane: Vec<u8> = vec![128u8; (width * height) as usize];

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S444);

        let jpeg_data = encoder
            .encode_ycbcr_planar(&y_plane, &cb_plane, &cr_plane, width, height)
            .expect("encode_ycbcr_planar should succeed");

        // Verify it's a valid JPEG
        assert!(jpeg_data.starts_with(&[0xFF, 0xD8, 0xFF])); // SOI + marker
        assert!(jpeg_data.ends_with(&[0xFF, 0xD9])); // EOI
        assert!(jpeg_data.len() > 200); // Reasonable size for 32x32
    }

    #[test]
    fn test_encode_ycbcr_planar_420() {
        let width = 32u32;
        let height = 32u32;

        // For 4:2:0, chroma planes are half resolution in each dimension
        let chroma_w = (width + 1) / 2;
        let chroma_h = (height + 1) / 2;

        let y_plane: Vec<u8> = vec![128u8; (width * height) as usize];
        let cb_plane: Vec<u8> = vec![100u8; (chroma_w * chroma_h) as usize];
        let cr_plane: Vec<u8> = vec![150u8; (chroma_w * chroma_h) as usize];

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_ycbcr_planar(&y_plane, &cb_plane, &cr_plane, width, height)
            .expect("encode_ycbcr_planar with 4:2:0 should succeed");

        // Verify it's a valid JPEG
        assert!(jpeg_data.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpeg_data.ends_with(&[0xFF, 0xD9]));
    }

    #[test]
    fn test_encode_ycbcr_planar_422() {
        let width = 32u32;
        let height = 32u32;

        // For 4:2:2, chroma is half width, full height
        let chroma_w = (width + 1) / 2;

        let y_plane: Vec<u8> = vec![128u8; (width * height) as usize];
        let cb_plane: Vec<u8> = vec![100u8; (chroma_w * height) as usize];
        let cr_plane: Vec<u8> = vec![150u8; (chroma_w * height) as usize];

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S422);

        let jpeg_data = encoder
            .encode_ycbcr_planar(&y_plane, &cb_plane, &cr_plane, width, height)
            .expect("encode_ycbcr_planar with 4:2:2 should succeed");

        assert!(jpeg_data.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpeg_data.ends_with(&[0xFF, 0xD9]));
    }

    #[test]
    fn test_encode_ycbcr_planar_wrong_size() {
        let width = 32u32;
        let height = 32u32;

        // Correct Y plane but wrong chroma plane sizes for 4:2:0
        let y_plane: Vec<u8> = vec![128u8; (width * height) as usize];
        let cb_plane: Vec<u8> = vec![100u8; 10]; // Too small!
        let cr_plane: Vec<u8> = vec![150u8; 10]; // Too small!

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S420);

        let result = encoder.encode_ycbcr_planar(&y_plane, &cb_plane, &cr_plane, width, height);

        assert!(result.is_err());
    }

    #[test]
    fn test_encode_ycbcr_planar_strided() {
        let width = 30u32; // Not a multiple of stride
        let height = 20u32;
        let y_stride = 32usize; // Stride with 2 bytes padding per row

        // For 4:2:0, chroma is half resolution
        let chroma_width = 15usize;
        let chroma_height = 10usize;
        let cb_stride = 16usize; // Stride with 1 byte padding per row

        // Create Y plane with stride (fill with gradient, padding with zeros)
        let mut y_plane = vec![0u8; y_stride * height as usize];
        for row in 0..height as usize {
            for col in 0..width as usize {
                y_plane[row * y_stride + col] = ((col * 255) / width as usize) as u8;
            }
        }

        // Create chroma planes with stride
        let mut cb_plane = vec![0u8; cb_stride * chroma_height];
        let mut cr_plane = vec![0u8; cb_stride * chroma_height];
        for row in 0..chroma_height {
            for col in 0..chroma_width {
                cb_plane[row * cb_stride + col] = 100;
                cr_plane[row * cb_stride + col] = 150;
            }
        }

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder
            .encode_ycbcr_planar_strided(
                &y_plane, y_stride, &cb_plane, cb_stride, &cr_plane, cb_stride, width, height,
            )
            .expect("strided encoding should succeed");

        // Verify it's a valid JPEG
        assert!(jpeg_data.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpeg_data.ends_with(&[0xFF, 0xD9]));
    }

    #[test]
    fn test_encode_ycbcr_planar_strided_matches_packed() {
        let width = 32u32;
        let height = 32u32;

        // Create packed plane data
        let y_packed: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let chroma_w = (width + 1) / 2;
        let chroma_h = (height + 1) / 2;
        let cb_packed: Vec<u8> = vec![100u8; (chroma_w * chroma_h) as usize];
        let cr_packed: Vec<u8> = vec![150u8; (chroma_w * chroma_h) as usize];

        let encoder = Encoder::new(Preset::BaselineBalanced)
            .quality(85)
            .subsampling(Subsampling::S420);

        // Encode with packed API
        let jpeg_packed = encoder
            .encode_ycbcr_planar(&y_packed, &cb_packed, &cr_packed, width, height)
            .expect("packed encoding should succeed");

        // Encode with strided API (stride == width means packed)
        let jpeg_strided = encoder
            .encode_ycbcr_planar_strided(
                &y_packed,
                width as usize,
                &cb_packed,
                chroma_w as usize,
                &cr_packed,
                chroma_w as usize,
                width,
                height,
            )
            .expect("strided encoding should succeed");

        // Both should produce identical output
        assert_eq!(jpeg_packed, jpeg_strided);
    }

    // =========================================================================
    // Resource Estimation Tests
    // =========================================================================

    #[test]
    fn test_estimate_resources_basic() {
        let encoder = Encoder::new(Preset::BaselineBalanced);
        let estimate = encoder.estimate_resources(1920, 1080);

        // Should have reasonable memory estimate (> input size)
        let input_size = 1920 * 1080 * 3;
        assert!(
            estimate.peak_memory_bytes > input_size,
            "Peak memory {} should exceed input size {}",
            estimate.peak_memory_bytes,
            input_size
        );

        // Should have reasonable CPU cost (> 1.0 due to trellis)
        assert!(
            estimate.cpu_cost_multiplier > 1.0,
            "CPU cost {} should be > 1.0 for BaselineBalanced",
            estimate.cpu_cost_multiplier
        );

        // Block count should match expected
        assert!(estimate.block_count > 0, "Block count should be > 0");
    }

    #[test]
    fn test_estimate_resources_fastest_has_lower_cpu() {
        let fastest = Encoder::new(Preset::BaselineFastest);
        let balanced = Encoder::new(Preset::BaselineBalanced);

        let est_fast = fastest.estimate_resources(512, 512);
        let est_balanced = balanced.estimate_resources(512, 512);

        // Fastest should have lower CPU cost (no trellis)
        assert!(
            est_fast.cpu_cost_multiplier < est_balanced.cpu_cost_multiplier,
            "Fastest ({:.2}) should have lower CPU cost than Balanced ({:.2})",
            est_fast.cpu_cost_multiplier,
            est_balanced.cpu_cost_multiplier
        );
    }

    #[test]
    fn test_estimate_resources_progressive_has_higher_cpu() {
        let baseline = Encoder::new(Preset::BaselineBalanced);
        let progressive = Encoder::new(Preset::ProgressiveBalanced);

        let est_baseline = baseline.estimate_resources(512, 512);
        let est_prog = progressive.estimate_resources(512, 512);

        // Progressive should have higher CPU cost (multiple scans)
        assert!(
            est_prog.cpu_cost_multiplier > est_baseline.cpu_cost_multiplier,
            "Progressive ({:.2}) should have higher CPU cost than Baseline ({:.2})",
            est_prog.cpu_cost_multiplier,
            est_baseline.cpu_cost_multiplier
        );
    }

    #[test]
    fn test_estimate_resources_gray() {
        let encoder = Encoder::new(Preset::BaselineBalanced);
        let rgb_estimate = encoder.estimate_resources(512, 512);
        let gray_estimate = encoder.estimate_resources_gray(512, 512);

        // Grayscale should use less memory (1 channel vs 3)
        assert!(
            gray_estimate.peak_memory_bytes < rgb_estimate.peak_memory_bytes,
            "Grayscale memory {} should be less than RGB {}",
            gray_estimate.peak_memory_bytes,
            rgb_estimate.peak_memory_bytes
        );

        // Grayscale should have lower CPU cost
        assert!(
            gray_estimate.cpu_cost_multiplier < rgb_estimate.cpu_cost_multiplier,
            "Grayscale CPU {:.2} should be less than RGB {:.2}",
            gray_estimate.cpu_cost_multiplier,
            rgb_estimate.cpu_cost_multiplier
        );
    }

    // =========================================================================
    // Resource Limit Tests
    // =========================================================================

    #[test]
    fn test_dimension_limit_width() {
        let limits = Limits::default().max_width(100).max_height(100);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 200 * 50 * 3];
        let result = encoder.encode_rgb(&pixels, 200, 50);

        assert!(matches!(result, Err(Error::DimensionLimitExceeded { .. })));
    }

    #[test]
    fn test_dimension_limit_height() {
        let limits = Limits::default().max_width(100).max_height(100);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 50 * 200 * 3];
        let result = encoder.encode_rgb(&pixels, 50, 200);

        assert!(matches!(result, Err(Error::DimensionLimitExceeded { .. })));
    }

    #[test]
    fn test_dimension_limit_passes_when_within() {
        let limits = Limits::default().max_width(100).max_height(100);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_allocation_limit() {
        let limits = Limits::default().max_alloc_bytes(1000); // Very small limit
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 256 * 256 * 3];
        let result = encoder.encode_rgb(&pixels, 256, 256);

        assert!(matches!(result, Err(Error::AllocationLimitExceeded { .. })));
    }

    #[test]
    fn test_allocation_limit_passes_when_within() {
        let limits = Limits::default().max_alloc_bytes(10_000_000); // 10 MB limit
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_pixel_count_limit() {
        let limits = Limits::default().max_pixel_count(1000); // Very small limit
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3]; // 4096 pixels
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(matches!(result, Err(Error::PixelCountExceeded { .. })));
    }

    #[test]
    fn test_pixel_count_limit_passes_when_within() {
        let limits = Limits::default().max_pixel_count(10000); // 10000 pixels
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3]; // 4096 pixels
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_icc_profile_size_limit() {
        let limits = Limits::default().max_icc_profile_bytes(100);
        let encoder = Encoder::new(Preset::BaselineFastest)
            .limits(limits)
            .icc_profile(vec![0u8; 1000]); // 1000 byte ICC profile

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(matches!(result, Err(Error::IccProfileTooLarge { .. })));
    }

    #[test]
    fn test_icc_profile_size_limit_passes_when_within() {
        let limits = Limits::default().max_icc_profile_bytes(2000);
        let encoder = Encoder::new(Preset::BaselineFastest)
            .limits(limits)
            .icc_profile(vec![0u8; 1000]); // 1000 byte ICC profile

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_limits_disabled_by_default() {
        let encoder = Encoder::new(Preset::BaselineFastest);
        assert_eq!(encoder.limits, Limits::none());
    }

    #[test]
    fn test_limits_has_limits() {
        assert!(!Limits::none().has_limits());
        assert!(Limits::default().max_width(100).has_limits());
        assert!(Limits::default().max_height(100).has_limits());
        assert!(Limits::default().max_pixel_count(1000).has_limits());
        assert!(Limits::default().max_alloc_bytes(1000).has_limits());
        assert!(Limits::default().max_icc_profile_bytes(1000).has_limits());
    }

    // =========================================================================
    // Cancellation Tests
    // =========================================================================

    #[test]
    fn test_cancellable_with_no_cancellation() {
        let encoder = Encoder::new(Preset::BaselineFastest);
        let pixels = vec![128u8; 64 * 64 * 3];

        let result = encoder.encode_rgb_cancellable(&pixels, 64, 64, None, None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_cancellable_immediate_cancel() {
        let encoder = Encoder::new(Preset::BaselineFastest);
        let pixels = vec![128u8; 64 * 64 * 3];
        let cancel = AtomicBool::new(true); // Already cancelled

        let result = encoder.encode_rgb_cancellable(&pixels, 64, 64, Some(&cancel), None);

        assert!(matches!(result, Err(Error::Cancelled)));
    }

    #[test]
    fn test_cancellable_with_timeout() {
        let encoder = Encoder::new(Preset::BaselineFastest);
        let pixels = vec![128u8; 64 * 64 * 3];

        // 10 second timeout - should complete well within this
        let result =
            encoder.encode_rgb_cancellable(&pixels, 64, 64, None, Some(Duration::from_secs(10)));

        assert!(result.is_ok());
    }

    #[test]
    fn test_cancellable_gray() {
        let encoder = Encoder::new(Preset::BaselineFastest);
        let pixels = vec![128u8; 64 * 64];

        let result = encoder.encode_gray_cancellable(&pixels, 64, 64, None, None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_cancellable_with_limits() {
        // Test that limits work in cancellable method too
        let limits = Limits::default().max_width(32);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb_cancellable(&pixels, 64, 64, None, None);

        assert!(matches!(result, Err(Error::DimensionLimitExceeded { .. })));
    }

    #[test]
    fn test_cancellation_context_none() {
        let ctx = CancellationContext::none();
        assert!(ctx.check().is_ok());
    }

    #[test]
    fn test_cancellation_context_with_cancel_flag() {
        use std::sync::atomic::Ordering;

        let cancel = AtomicBool::new(false);
        let ctx = CancellationContext::new(Some(&cancel), None);
        assert!(ctx.check().is_ok());

        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(ctx.check(), Err(Error::Cancelled)));
    }

    #[test]
    fn test_cancellation_context_with_expired_deadline() {
        // Create a deadline that's already passed
        let ctx = CancellationContext {
            cancel: None,
            deadline: Some(Instant::now() - Duration::from_secs(1)),
        };

        assert!(matches!(ctx.check(), Err(Error::TimedOut)));
    }

    #[test]
    fn test_dimension_exact_at_limit_passes() {
        // Dimensions exactly at limit should pass
        let limits = Limits::default().max_width(64).max_height(64);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_pixel_count_exact_at_limit_passes() {
        // Pixel count exactly at limit should pass
        let limits = Limits::default().max_pixel_count(4096); // Exactly 64*64
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64 * 3];
        let result = encoder.encode_rgb(&pixels, 64, 64);

        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_limits_all_checked() {
        // Test that all limits are checked, not just the first
        let limits = Limits::default()
            .max_width(1000)
            .max_height(1000)
            .max_pixel_count(100); // This should fail

        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);
        let pixels = vec![128u8; 64 * 64 * 3]; // 4096 pixels

        let result = encoder.encode_rgb(&pixels, 64, 64);
        assert!(matches!(result, Err(Error::PixelCountExceeded { .. })));
    }

    #[test]
    fn test_limits_with_grayscale() {
        let limits = Limits::default().max_pixel_count(100);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64]; // Grayscale, 4096 pixels
        let result = encoder.encode_gray(&pixels, 64, 64);

        assert!(matches!(result, Err(Error::PixelCountExceeded { .. })));
    }

    #[test]
    fn test_estimate_resources_with_subsampling() {
        let encoder_444 = Encoder::new(Preset::BaselineBalanced).subsampling(Subsampling::S444);
        let encoder_420 = Encoder::new(Preset::BaselineBalanced).subsampling(Subsampling::S420);

        let est_444 = encoder_444.estimate_resources(512, 512);
        let est_420 = encoder_420.estimate_resources(512, 512);

        // 4:4:4 should use more memory than 4:2:0 (no chroma downsampling)
        assert!(
            est_444.peak_memory_bytes > est_420.peak_memory_bytes,
            "4:4:4 memory {} should exceed 4:2:0 memory {}",
            est_444.peak_memory_bytes,
            est_420.peak_memory_bytes
        );
    }

    #[test]
    fn test_estimate_resources_block_count() {
        // With 4:2:0 subsampling (default): Y gets full blocks, chroma gets 1/4
        let encoder = Encoder::new(Preset::BaselineFastest);

        // 64x64 image with 4:2:0:
        // Y blocks: 8x8 = 64
        // Chroma: 32x32 pixels, 4x4 blocks each = 16 per component
        // Total: 64 + 16 + 16 = 96
        let estimate = encoder.estimate_resources(64, 64);
        assert_eq!(estimate.block_count, 96);

        // With 4:4:4 subsampling: all components get full blocks
        let encoder_444 = Encoder::new(Preset::BaselineFastest).subsampling(Subsampling::S444);
        let estimate_444 = encoder_444.estimate_resources(64, 64);
        // 64 blocks * 3 components = 192
        assert_eq!(estimate_444.block_count, 192);
    }

    #[test]
    fn test_cancellable_gray_with_limits() {
        let limits = Limits::default().max_width(32);
        let encoder = Encoder::new(Preset::BaselineFastest).limits(limits);

        let pixels = vec![128u8; 64 * 64];
        let result = encoder.encode_gray_cancellable(&pixels, 64, 64, None, None);

        assert!(matches!(result, Err(Error::DimensionLimitExceeded { .. })));
    }
}
