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
//! use mozjpeg_rs::Encoder;
//!
//! // Full-featured batch encoding
//! let jpeg = Encoder::new(false)
//!     .quality(85)
//!     .progressive(true)
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
use crate::simd::SimdOps;
use crate::trellis::trellis_quantize_block;
use crate::types::{PixelDensity, Subsampling, TrellisConfig};

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
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Encoder {
    /// Create an encoder with optimal settings for the chosen mode.
    ///
    /// # Arguments
    ///
    /// * `progressive` - Encoding mode:
    ///   - `false`: **Baseline** (sequential) JPEG - faster decoding, wider compatibility
    ///   - `true`: **Progressive** JPEG - smaller files (~20%), progressive rendering
    ///
    /// Both modes enable all mozjpeg optimizations (trellis, Huffman, deringing).
    /// Progressive mode additionally enables `optimize_scans` to match C mozjpeg.
    ///
    /// # Settings by Mode
    ///
    /// | Setting | `new(false)` | `new(true)` |
    /// |---------|--------------|-------------|
    /// | progressive | false | **true** |
    /// | optimize_scans | false | **true** |
    /// | trellis | enabled | enabled |
    /// | optimize_huffman | true | true |
    /// | overshoot_deringing | true | true |
    /// | quality | 75 | 75 |
    /// | subsampling | 4:2:0 | 4:2:0 |
    ///
    /// # C mozjpeg Compatibility
    ///
    /// - `new(true)` matches C mozjpeg's `jpeg_set_defaults()` (JCP_MAX_COMPRESSION)
    /// - `new(false)` uses baseline mode with all optimizations
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::Encoder;
    ///
    /// let pixels: Vec<u8> = vec![128; 256 * 256 * 3];
    ///
    /// // Baseline - fast decode, good compatibility
    /// let baseline = Encoder::new(false)
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    ///
    /// // Progressive - smaller files, matches C mozjpeg
    /// let progressive = Encoder::new(true)
    ///     .quality(85)
    ///     .encode_rgb(&pixels, 256, 256)
    ///     .unwrap();
    /// ```
    pub fn new(progressive: bool) -> Self {
        if progressive {
            Self::max_compression()
        } else {
            Self::baseline_optimized()
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
    /// - `Encoder::new(false)`: ~650 bytes (baseline)
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
    /// let encoder = Encoder::new(false)
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
    /// let encoder = Encoder::new(false)
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
    /// let enc1 = Encoder::new(false).baseline(true);
    /// let enc2 = Encoder::new(false).progressive(false);
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
                &y_mcu,
                mcu_width,
                mcu_height,
                &cb_mcu,
                &cr_mcu,
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
                &y_mcu,
                mcu_width,
                mcu_height,
                &cb_mcu,
                &cr_mcu,
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

            let output = marker_writer.into_inner();
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
                        entropy.encode_block(&y_blocks[y_idx], 0, &opt_dc_luma, &opt_ac_luma)?;
                        y_idx += 1;
                    }
                    // Cb block
                    entropy.encode_block(&cb_blocks[c_idx], 1, &opt_dc_chroma, &opt_ac_chroma)?;
                    // Cr block
                    entropy.encode_block(&cr_blocks[c_idx], 2, &opt_dc_chroma, &opt_ac_chroma)?;
                    c_idx += 1;
                    mcu_count += 1;
                }
            }

            bit_writer.flush()?;
            let mut output = bit_writer.into_inner();
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
                &y_mcu,
                mcu_width,
                mcu_height,
                &cb_mcu,
                &cr_mcu,
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
        (self.simd.forward_dct)(&shifted, dct_block);

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
        (self.simd.forward_dct)(&shifted, dct_block);

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
    /// For full-featured encoding with all mozjpeg optimizations, use [`Encoder::new(false)`]
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
        StreamingEncoder::new()
    }
}

/// Unit tests for private encoder internals.
/// Public API tests are in tests/encode_tests.rs.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_defaults() {
        let enc = Encoder::new(false);
        assert_eq!(enc.quality, 75);
        assert!(!enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S420);
        assert!(enc.trellis.enabled);
        assert!(enc.optimize_huffman);
    }

    #[test]
    fn test_encoder_builder_fields() {
        let enc = Encoder::new(false)
            .quality(90)
            .progressive(true)
            .subsampling(Subsampling::S444);

        assert_eq!(enc.quality, 90);
        assert!(enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S444);
    }

    #[test]
    fn test_quality_clamping() {
        let enc = Encoder::new(false).quality(0);
        assert_eq!(enc.quality, 1);

        let enc = Encoder::new(false).quality(150);
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
}
