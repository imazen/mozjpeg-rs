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
//! use mozjpeg_oxide::Encoder;
//!
//! // Full-featured batch encoding
//! let jpeg = Encoder::new()
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
use crate::consts::{
    QuantTableIdx, AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS,
    AC_LUMINANCE_VALUES, DCTSIZE, DCTSIZE2, DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES,
    DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES, JPEG_NATURAL_ORDER,
};
use crate::deringing::preprocess_deringing;
use crate::entropy::{EntropyEncoder, ProgressiveEncoder, ProgressiveSymbolCounter, SymbolCounter};
use crate::error::{Error, Result};
use crate::huffman::FrequencyCounter;
use crate::huffman::{DerivedTable, HuffTable};
use crate::marker::MarkerWriter;
use crate::progressive::{generate_baseline_scan, generate_minimal_progressive_scans};
use crate::quant::{create_quant_tables, quantize_block};
use crate::sample;
use crate::scan_optimize::{generate_search_scans, ScanSearchConfig, ScanSelector};
use crate::simd::SimdOps;
use crate::trellis::{dc_trellis_optimize_indexed, trellis_quantize_block};
use crate::types::{ComponentInfo, PixelDensity, QuantTable, Subsampling, TrellisConfig};

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

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper to allocate a Vec with fallible allocation.
/// Returns Error::AllocationFailed if allocation fails.
#[inline]
fn try_alloc_vec<T: Clone>(value: T, len: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    v.resize(len, value);
    Ok(v)
}

/// Helper to allocate a Vec of arrays with fallible allocation.
#[inline]
fn try_alloc_vec_array<T: Copy + Default, const N: usize>(len: usize) -> Result<Vec<[T; N]>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    v.resize(len, [T::default(); N]);
    Ok(v)
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
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Create a new encoder with default settings (mozjpeg defaults).
    ///
    /// Default configuration:
    /// - Quality: 75
    /// - Progressive: false
    /// - Subsampling: 4:2:0
    /// - Quant tables: ImageMagick (mozjpeg default)
    /// - Trellis: enabled (core mozjpeg optimization)
    /// - Huffman optimization: enabled (2-pass for optimal tables)
    /// - Overshoot deringing: enabled (reduces ringing on edges)
    pub fn new() -> Self {
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
        }
    }

    /// Create encoder with max compression settings (mozjpeg defaults).
    ///
    /// Enables progressive mode, trellis quantization, Huffman optimization,
    /// and overshoot deringing.
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
        }
    }

    /// Create encoder with fastest settings (libjpeg-turbo compatible).
    ///
    /// Disables all mozjpeg optimizations (trellis, Huffman optimization, deringing).
    /// Uses ImageMagick quant tables (same as C mozjpeg defaults from jpeg_set_defaults).
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
    /// use mozjpeg_oxide::{Encoder, PixelDensity};
    ///
    /// let encoder = Encoder::new()
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

        let mut output = Vec::new();
        self.encode_rgb_to_writer(rgb_data, width, height, &mut output)?;
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

        let mut output = Vec::new();
        self.encode_gray_to_writer(gray_data, width, height, &mut output)?;
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

        // SOF (baseline for grayscale - progressive would need multi-scan support)
        marker_writer.write_sof(
            false, // Use baseline for grayscale (simpler)
            8,
            height as u16,
            width as u16,
            &components,
        )?;

        // DRI (restart interval)
        if self.restart_interval > 0 {
            marker_writer.write_dri(self.restart_interval)?;
        }

        // DHT (only luma tables for grayscale)
        if !self.optimize_huffman {
            marker_writer
                .write_dht_multiple(&[(0, false, &dc_luma_huff), (0, true, &ac_luma_huff)])?;
        }

        // Grayscale uses baseline encoding
        let mcu_rows = mcu_height / DCTSIZE;
        let mcu_cols = mcu_width / DCTSIZE;
        let num_blocks = mcu_rows
            .checked_mul(mcu_cols)
            .ok_or(Error::AllocationFailed)?;

        if self.optimize_huffman {
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

            // Generate progressive scan script
            let scans = if self.optimize_scans {
                // Use C mozjpeg-compatible optimize_scans:
                // 1. Generate 64 individual candidate scans
                // 2. Trial-encode each to get sizes
                // 3. Use ScanSelector to find optimal Al levels and frequency splits
                // 4. Build final scan script from the selection
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
                // Use minimal progressive (no successive approximation) for simpler encoding
                // This matches C mozjpeg's jpeg_simple_progression()
                // Successive approximation is only beneficial at high quality with optimize_scans
                generate_minimal_progressive_scans(3)
            };

            // Count symbol frequencies for optimized Huffman tables
            let (opt_dc_luma, opt_dc_chroma, opt_ac_luma, opt_ac_chroma) = if self.optimize_huffman
            {
                let mut dc_luma_freq = FrequencyCounter::new();
                let mut dc_chroma_freq = FrequencyCounter::new();
                let mut ac_luma_freq = FrequencyCounter::new();
                let mut ac_chroma_freq = FrequencyCounter::new();

                // Count frequencies for each scan
                for scan in &scans {
                    let is_dc_scan = scan.ss == 0 && scan.se == 0;
                    if is_dc_scan {
                        // DC scan - count DC symbols for all components
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
                    } else {
                        // AC scan - count AC symbols for single component
                        let comp_idx = scan.component_index[0] as usize;
                        let blocks = match comp_idx {
                            0 => &y_blocks,
                            1 => &cb_blocks,
                            2 => &cr_blocks,
                            _ => &y_blocks,
                        };
                        let ac_freq = if comp_idx == 0 {
                            &mut ac_luma_freq
                        } else {
                            &mut ac_chroma_freq
                        };
                        // Calculate actual block dimensions for this component
                        let (actual_block_cols, actual_block_rows) = if comp_idx == 0 {
                            (width.div_ceil(DCTSIZE), height.div_ceil(DCTSIZE))
                        } else {
                            (
                                chroma_width.div_ceil(DCTSIZE),
                                chroma_height.div_ceil(DCTSIZE),
                            )
                        };
                        self.count_ac_scan_symbols(
                            scan,
                            blocks,
                            mcu_rows,
                            mcu_cols,
                            luma_h,
                            luma_v,
                            comp_idx,
                            actual_block_cols,
                            actual_block_rows,
                            ac_freq,
                        );
                    }
                }

                // Generate optimized Huffman tables
                let opt_dc_luma_huff = dc_luma_freq.generate_table()?;
                let opt_dc_chroma_huff = dc_chroma_freq.generate_table()?;
                let opt_ac_luma_huff = ac_luma_freq.generate_table()?;
                let opt_ac_chroma_huff = ac_chroma_freq.generate_table()?;

                // Write DHT with optimized tables - combined into single marker
                marker_writer.write_dht_multiple(&[
                    (0, false, &opt_dc_luma_huff),
                    (1, false, &opt_dc_chroma_huff),
                    (0, true, &opt_ac_luma_huff),
                    (1, true, &opt_ac_chroma_huff),
                ])?;

                // Create derived tables for encoding
                (
                    DerivedTable::from_huff_table(&opt_dc_luma_huff, true)?,
                    DerivedTable::from_huff_table(&opt_dc_chroma_huff, true)?,
                    DerivedTable::from_huff_table(&opt_ac_luma_huff, false)?,
                    DerivedTable::from_huff_table(&opt_ac_chroma_huff, false)?,
                )
            } else {
                // Use standard tables (already written)
                (
                    dc_luma_derived.clone(),
                    dc_chroma_derived.clone(),
                    ac_luma_derived.clone(),
                    ac_chroma_derived.clone(),
                )
            };

            // Get output writer from marker_writer
            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);

            // Encode each scan
            for scan in &scans {
                // Write SOS for this scan
                // We need to temporarily get the output to write SOS
                bit_writer.flush()?;
                let mut inner = bit_writer.into_inner();

                // Write SOS marker manually
                write_sos_marker(&mut inner, scan, &components)?;

                bit_writer = BitWriter::new(inner);
                // Use standard tables mode when not optimizing (no extended EOBRUN)
                let mut prog_encoder = if self.optimize_huffman {
                    ProgressiveEncoder::new(&mut bit_writer)
                } else {
                    ProgressiveEncoder::new_standard_tables(&mut bit_writer)
                };

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
                    &opt_ac_luma,
                    &opt_ac_chroma,
                    &mut prog_encoder,
                )?;

                // Finish this scan
                let ac_table = if scan.ss > 0 {
                    if scan.component_index[0] == 0 {
                        Some(&opt_ac_luma)
                    } else {
                        Some(&opt_ac_chroma)
                    }
                } else {
                    None
                };
                prog_encoder.finish_scan(ac_table)?;
            }

            // Flush and get output back
            bit_writer.flush()?;
            let mut output = bit_writer.into_inner();

            // EOI
            output.write_all(&[0xFF, 0xD9])?;
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
        // Trellis expects raw DCT (scaled by 8) and handles the scaling internally
        if self.trellis.enabled {
            trellis_quantize_block(&dct_i32, quant_block, qtable, ac_table, &self.trellis);
        } else {
            // Non-trellis path: descale first, then quantize
            for coeff in dct_i32.iter_mut() {
                *coeff = (*coeff + 4) >> 3;
            }
            quantize_block(&dct_i32, qtable, quant_block);
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
        // Trellis expects raw DCT (scaled by 8) and handles the scaling internally
        if self.trellis.enabled {
            trellis_quantize_block(&dct_i32, out_block, qtable, ac_table, &self.trellis);
        } else {
            // Non-trellis path: descale first, then quantize
            for coeff in dct_i32.iter_mut() {
                *coeff = (*coeff + 4) >> 3;
            }
            quantize_block(&dct_i32, qtable, out_block);
        }

        Ok(())
    }

    /// Optimize progressive scan configuration (C mozjpeg-compatible).
    ///
    /// This implements the optimize_scans feature from C mozjpeg:
    /// 1. Generate 64 individual candidate scans
    /// 2. Trial-encode each scan to get its size
    /// 3. Use ScanSelector to find optimal Al levels and frequency splits
    /// 4. Build the final scan script from the selection
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

        // Trial-encode each candidate scan to get its size
        let mut scan_sizes = Vec::with_capacity(candidate_scans.len());

        for scan in &candidate_scans {
            let size = self.trial_encode_scan(
                scan,
                y_blocks,
                cb_blocks,
                cr_blocks,
                mcu_rows,
                mcu_cols,
                h_samp,
                v_samp,
                actual_width,
                actual_height,
                chroma_width,
                chroma_height,
                dc_luma,
                dc_chroma,
                ac_luma,
                ac_chroma,
            )?;
            scan_sizes.push(size);
        }

        // Use ScanSelector to find the optimal configuration
        let selector = ScanSelector::new(num_components, config.clone());
        let result = selector.select_best(&scan_sizes);

        // Build the final scan script from the selection
        Ok(result.build_final_scans(num_components, &config))
    }

    /// Trial-encode a single scan and return its size in bytes.
    #[allow(clippy::too_many_arguments)]
    fn trial_encode_scan(
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
    ) -> Result<usize> {
        let mut buffer = Vec::new();
        let mut bit_writer = BitWriter::new(&mut buffer);
        let mut prog_encoder = ProgressiveEncoder::new(&mut bit_writer);

        // Encode the scan
        self.encode_progressive_scan(
            scan,
            y_blocks,
            cb_blocks,
            cr_blocks,
            mcu_rows,
            mcu_cols,
            h_samp,
            v_samp,
            actual_width,
            actual_height,
            chroma_width,
            chroma_height,
            dc_luma,
            dc_chroma,
            ac_luma,
            ac_chroma,
            &mut prog_encoder,
        )?;

        // Finish the scan (flush any pending EOBRUN)
        let ac_table = if scan.ss > 0 {
            if scan.component_index[0] == 0 {
                Some(ac_luma)
            } else {
                Some(ac_chroma)
            }
        } else {
            None
        };
        prog_encoder.finish_scan(ac_table)?;
        bit_writer.flush()?;

        Ok(buffer.len())
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

            // Calculate actual block dimensions for this component
            let (actual_block_cols, actual_block_rows) = if comp_idx == 0 {
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
                actual_block_cols,
                actual_block_rows,
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
        actual_block_cols: usize,
        actual_block_rows: usize,
        ac_table: &DerivedTable,
        is_refinement: bool,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        // For Y component with subsampling, blocks are stored in MCU-interleaved order
        // but AC scans must encode them in component raster order.
        // For chroma components (1 block per MCU), the orders are identical.

        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            // Only encode actual_block_rows × actual_block_cols blocks
            let total_actual = actual_block_rows * actual_block_cols;
            for block in blocks.iter().take(total_actual) {
                if is_refinement {
                    encoder.encode_ac_refine(block, scan.ss, scan.se, scan.al, ac_table)?;
                } else {
                    encoder.encode_ac_first(block, scan.ss, scan.se, scan.al, ac_table)?;
                }
            }
        } else {
            // Y component with subsampling (h_samp > 1 or v_samp > 1)
            // Convert from MCU-interleaved storage to component raster order
            // Use actual block dimensions, not MCU-padded dimensions
            let h = h_samp as usize;
            let v = v_samp as usize;

            for block_row in 0..actual_block_rows {
                for block_col in 0..actual_block_cols {
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
        actual_block_cols: usize,
        actual_block_rows: usize,
        ac_freq: &mut FrequencyCounter,
    ) {
        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        let mut counter = ProgressiveSymbolCounter::new();

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            // Only count actual_block_rows × actual_block_cols blocks
            let total_actual = actual_block_rows * actual_block_cols;
            for block in blocks.iter().take(total_actual) {
                counter.count_ac_first(block, scan.ss, scan.se, scan.al, ac_freq);
            }
        } else {
            // Y component with subsampling - iterate in raster order (matching encode_ac_scan)
            // Use actual block dimensions, not MCU-padded dimensions
            let h = h_samp as usize;
            let v = v_samp as usize;

            for block_row in 0..actual_block_rows {
                for block_col in 0..actual_block_cols {
                    // Convert raster position to MCU-interleaved storage index
                    let mcu_row = block_row / v;
                    let mcu_col = block_col / h;
                    let v_idx = block_row % v;
                    let h_idx = block_col % h;
                    let storage_idx = mcu_row * (mcu_cols * blocks_per_mcu)
                        + mcu_col * blocks_per_mcu
                        + v_idx * h
                        + h_idx;

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

        // Flush any pending EOBRUN
        counter.finish_scan(Some(ac_freq));
    }
}

// Helper functions

/// Get MCU-order index for a block at (block_row, block_col) in image coordinates.
///
/// For multi-block MCUs (e.g., 4:2:0 luma with 2x2 blocks per MCU), blocks are stored
/// in MCU order: all blocks for MCU (0,0), then all for MCU (0,1), etc.
/// Within each MCU, blocks are stored row-by-row: (0,0), (0,1), (1,0), (1,1) for 2x2.
fn block_to_mcu_index(
    block_row: usize,
    block_col: usize,
    mcu_cols: usize,
    h_samp: usize,
    v_samp: usize,
) -> usize {
    let mcu_row = block_row / v_samp;
    let mcu_col = block_col / h_samp;
    let v = block_row % v_samp;
    let h = block_col % h_samp;
    (mcu_row * mcu_cols + mcu_col) * (h_samp * v_samp) + v * h_samp + h
}

/// Get indices for one row of blocks in row order, mapped to MCU-order storage.
fn get_row_indices(
    block_row: usize,
    block_cols: usize,
    mcu_cols: usize,
    h_samp: usize,
    v_samp: usize,
) -> Vec<usize> {
    (0..block_cols)
        .map(|block_col| block_to_mcu_index(block_row, block_col, mcu_cols, h_samp, v_samp))
        .collect()
}

/// Run DC trellis optimization row by row (matching C mozjpeg behavior).
///
/// C mozjpeg processes DC trellis one block row at a time, with each row
/// forming an independent chain. This is different from processing all blocks
/// as one giant chain.
#[allow(clippy::too_many_arguments)]
fn run_dc_trellis_by_row(
    raw_blocks: &[[i32; DCTSIZE2]],
    quantized_blocks: &mut [[i16; DCTSIZE2]],
    dc_quantval: u16,
    dc_table: &DerivedTable,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
    block_rows: usize,
    block_cols: usize,
    mcu_cols: usize,
    h_samp: usize,
    v_samp: usize,
) {
    // Process each block row independently
    for block_row in 0..block_rows {
        let indices = get_row_indices(block_row, block_cols, mcu_cols, h_samp, v_samp);

        // Each row starts with last_dc = 0 (C mozjpeg behavior for trellis pass)
        dc_trellis_optimize_indexed(
            raw_blocks,
            quantized_blocks,
            &indices,
            dc_quantval,
            dc_table,
            0, // last_dc = 0 for each row
            lambda_log_scale1,
            lambda_log_scale2,
        );
    }
}

/// Write an SOS (Start of Scan) marker.
fn write_sos_marker<W: Write>(
    output: &mut W,
    scan: &crate::types::ScanInfo,
    components: &[ComponentInfo],
) -> std::io::Result<()> {
    use crate::consts::JPEG_SOS;

    // SOS marker
    output.write_all(&[0xFF, JPEG_SOS])?;

    // Length (2 bytes): 6 + 2*Ns
    let ns = scan.comps_in_scan as usize;
    let length = 6 + 2 * ns;
    output.write_all(&[(length >> 8) as u8, (length & 0xFF) as u8])?;

    // Number of components in scan
    output.write_all(&[scan.comps_in_scan])?;

    // Component selector + Huffman table selectors for each component
    // Per JPEG spec (ITU-T T.81):
    // - For DC scans (Ss=0, Se=0): Ta (AC table) must be 0
    // - For AC scans (Ss>0): Td (DC table) must be 0
    let is_dc_scan = scan.ss == 0 && scan.se == 0;

    for i in 0..scan.comps_in_scan as usize {
        let comp_idx = scan.component_index[i] as usize;
        let comp = &components[comp_idx];
        let table_selector = if is_dc_scan {
            // DC scan: Td = dc_tbl_no, Ta = 0
            comp.dc_tbl_no << 4
        } else {
            // AC scan: Td = 0, Ta = ac_tbl_no
            comp.ac_tbl_no
        };
        output.write_all(&[comp.component_id, table_selector])?;
    }

    // Spectral selection start (Ss), end (Se), successive approximation (Ah, Al)
    output.write_all(&[scan.ss, scan.se, (scan.ah << 4) | scan.al])?;

    Ok(())
}

/// Create component info for the given subsampling mode.
/// Returns 1 component for grayscale, 3 for color modes.
fn create_components(subsampling: Subsampling) -> Vec<ComponentInfo> {
    if subsampling == Subsampling::Gray {
        // Grayscale: single Y component
        vec![ComponentInfo {
            component_id: 1,
            component_index: 0,
            h_samp_factor: 1,
            v_samp_factor: 1,
            quant_tbl_no: 0,
            dc_tbl_no: 0,
            ac_tbl_no: 0,
        }]
    } else {
        // Color: Y, Cb, Cr components
        let (h_samp, v_samp) = subsampling.luma_factors();
        vec![
            ComponentInfo {
                component_id: 1, // Y
                component_index: 0,
                h_samp_factor: h_samp,
                v_samp_factor: v_samp,
                quant_tbl_no: 0,
                dc_tbl_no: 0,
                ac_tbl_no: 0,
            },
            ComponentInfo {
                component_id: 2, // Cb
                component_index: 1,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_tbl_no: 1,
                dc_tbl_no: 1,
                ac_tbl_no: 1,
            },
            ComponentInfo {
                component_id: 3, // Cr
                component_index: 2,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_tbl_no: 1,
                dc_tbl_no: 1,
                ac_tbl_no: 1,
            },
        ]
    }
}

// Keep for backwards compatibility
fn create_ycbcr_components(subsampling: Subsampling) -> Vec<ComponentInfo> {
    create_components(subsampling)
}

#[allow(clippy::needless_range_loop)]
fn natural_to_zigzag(natural: &[u16; DCTSIZE2]) -> [u16; DCTSIZE2] {
    let mut zigzag = [0u16; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        zigzag[i] = natural[JPEG_NATURAL_ORDER[i]];
    }
    zigzag
}

fn create_std_dc_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    htbl.huffval[..DC_LUMINANCE_VALUES.len()].copy_from_slice(&DC_LUMINANCE_VALUES);
    htbl
}

fn create_std_dc_chroma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_CHROMINANCE_BITS);
    htbl.huffval[..DC_CHROMINANCE_VALUES.len()].copy_from_slice(&DC_CHROMINANCE_VALUES);
    htbl
}

fn create_std_ac_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    htbl.huffval[..AC_LUMINANCE_VALUES.len()].copy_from_slice(&AC_LUMINANCE_VALUES);
    htbl
}

fn create_std_ac_chroma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
    htbl.huffval[..AC_CHROMINANCE_VALUES.len()].copy_from_slice(&AC_CHROMINANCE_VALUES);
    htbl
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

// ============================================================================
// StreamingEncoder
// ============================================================================

/// Streaming JPEG encoder configuration.
///
/// This encoder supports scanline-by-scanline encoding, which is memory-efficient
/// for large images. It does NOT support trellis quantization, progressive mode,
/// or Huffman optimization (these require buffering the entire image).
///
/// Use [`Encoder`] for full-featured batch encoding with optimizations.
///
/// # Example
///
/// ```ignore
/// use mozjpeg_oxide::Encoder;
///
/// // Create streaming encoder
/// let mut stream = Encoder::streaming()
///     .quality(85)
///     .start_rgb(1920, 1080, output_file)?;
///
/// // Write scanlines (must be in multiples of 8 or 16 depending on subsampling)
/// for chunk in rgb_scanlines.chunks(16 * 1920 * 3) {
///     stream.write_scanlines(chunk)?;
/// }
///
/// // Finalize the JPEG
/// stream.finish()?;
/// ```
#[derive(Debug, Clone)]
pub struct StreamingEncoder {
    /// Quality level (1-100)
    quality: u8,
    /// Chroma subsampling mode
    subsampling: Subsampling,
    /// Quantization table variant
    quant_table_idx: QuantTableIdx,
    /// Custom luminance quantization table
    custom_luma_qtable: Option<[u16; DCTSIZE2]>,
    /// Custom chrominance quantization table
    custom_chroma_qtable: Option<[u16; DCTSIZE2]>,
    /// Force baseline-compatible output
    force_baseline: bool,
    /// Restart interval in MCUs (0 = disabled)
    restart_interval: u16,
    /// Pixel density for JFIF APP0 marker
    pixel_density: PixelDensity,
    /// EXIF data to embed
    exif_data: Option<Vec<u8>>,
    /// ICC color profile to embed
    icc_profile: Option<Vec<u8>>,
    /// Custom APP markers to embed
    custom_markers: Vec<(u8, Vec<u8>)>,
    /// SIMD operations dispatch
    simd: SimdOps,
}

impl Default for StreamingEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingEncoder {
    /// Create a new streaming encoder with default settings.
    ///
    /// Unlike [`Encoder::new()`], this uses settings optimized for streaming:
    /// - No trellis quantization (requires global context)
    /// - No progressive mode (requires buffering entire image)
    /// - No Huffman optimization (requires 2-pass)
    pub fn new() -> Self {
        Self {
            quality: 75,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            custom_luma_qtable: None,
            custom_chroma_qtable: None,
            force_baseline: true,
            restart_interval: 0,
            pixel_density: PixelDensity::default(),
            exif_data: None,
            icc_profile: None,
            custom_markers: Vec::new(),
            simd: SimdOps::detect(),
        }
    }

    /// Set quality level (1-100).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
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

    /// Force baseline-compatible output.
    pub fn force_baseline(mut self, enable: bool) -> Self {
        self.force_baseline = enable;
        self
    }

    /// Set restart interval in MCUs.
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    /// Set pixel density for the JFIF APP0 marker.
    pub fn pixel_density(mut self, density: PixelDensity) -> Self {
        self.pixel_density = density;
        self
    }

    /// Set EXIF data to embed.
    pub fn exif_data(mut self, data: Vec<u8>) -> Self {
        self.exif_data = if data.is_empty() { None } else { Some(data) };
        self
    }

    /// Set ICC color profile to embed.
    pub fn icc_profile(mut self, profile: Vec<u8>) -> Self {
        self.icc_profile = if profile.is_empty() {
            None
        } else {
            Some(profile)
        };
        self
    }

    /// Add a custom APP marker.
    pub fn add_marker(mut self, app_num: u8, data: Vec<u8>) -> Self {
        if app_num <= 15 && !data.is_empty() {
            self.custom_markers.push((app_num, data));
        }
        self
    }

    /// Set custom luminance quantization table.
    pub fn custom_luma_qtable(mut self, table: [u16; DCTSIZE2]) -> Self {
        self.custom_luma_qtable = Some(table);
        self
    }

    /// Set custom chrominance quantization table.
    pub fn custom_chroma_qtable(mut self, table: [u16; DCTSIZE2]) -> Self {
        self.custom_chroma_qtable = Some(table);
        self
    }

    /// Start streaming RGB encoding to a writer.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `writer` - Output writer
    ///
    /// # Returns
    /// An [`EncodingStream`] that accepts scanlines.
    pub fn start_rgb<W: Write>(
        self,
        width: u32,
        height: u32,
        writer: W,
    ) -> Result<EncodingStream<W>> {
        EncodingStream::new_rgb(self, width, height, writer)
    }

    /// Start streaming grayscale encoding to a writer.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `writer` - Output writer
    ///
    /// # Returns
    /// An [`EncodingStream`] that accepts scanlines.
    pub fn start_gray<W: Write>(
        self,
        width: u32,
        height: u32,
        writer: W,
    ) -> Result<EncodingStream<W>> {
        EncodingStream::new_gray(self, width, height, writer)
    }
}

/// Implement batch encoding for StreamingEncoder (without optimizations).
impl Encode for StreamingEncoder {
    fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut stream = self.clone().start_rgb(width, height, &mut output)?;
        stream.write_scanlines(rgb_data)?;
        stream.finish()?;
        Ok(output)
    }

    fn encode_gray(&self, gray_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut stream = self.clone().start_gray(width, height, &mut output)?;
        stream.write_scanlines(gray_data)?;
        stream.finish()?;
        Ok(output)
    }
}

// ============================================================================
// EncodingStream - Active Streaming Session
// ============================================================================

/// Active streaming encoding session.
///
/// Created by [`StreamingEncoder::start_rgb()`] or [`StreamingEncoder::start_gray()`].
/// Accepts scanlines via [`write_scanlines()`](Self::write_scanlines) and must be
/// finalized with [`finish()`](Self::finish).
pub struct EncodingStream<W: Write> {
    /// Output writer wrapped in marker writer
    writer: MarkerWriter<W>,
    /// Image width
    width: u32,
    /// Number of color components (1 for gray, 3 for RGB/YCbCr)
    num_components: u8,
    /// Bytes per input pixel
    bytes_per_pixel: u8,
    /// Chroma subsampling mode
    subsampling: Subsampling,
    /// MCU height in pixels (8 or 16 depending on subsampling)
    mcu_height: u32,
    /// MCU width in pixels (8 or 16 depending on subsampling)
    mcu_width: u32,
    /// Number of MCUs per row
    mcus_per_row: u32,
    /// Luminance quantization table (zigzag order)
    luma_qtable: QuantTable,
    /// Chrominance quantization table (zigzag order)
    chroma_qtable: QuantTable,
    /// DC Huffman table for luminance
    dc_luma_table: DerivedTable,
    /// AC Huffman table for luminance
    ac_luma_table: DerivedTable,
    /// DC Huffman table for chrominance
    dc_chroma_table: DerivedTable,
    /// AC Huffman table for chrominance
    ac_chroma_table: DerivedTable,
    /// Previous DC values for differential encoding
    prev_dc: [i32; 4],
    /// Scanlines accumulated for current MCU row
    scanline_buffer: Vec<u8>,
    /// Lines accumulated in buffer
    lines_in_buffer: u32,
    /// Bit buffer for entropy encoding
    bit_buffer: u64,
    /// Bits used in bit_buffer
    bits_in_buffer: u8,
    /// SIMD operations dispatch
    simd: SimdOps,
    /// Restart interval tracking
    restart_interval: u16,
    /// MCUs since last restart marker
    mcus_since_restart: u16,
    /// Next restart marker number (0-7)
    next_restart_num: u8,
}

impl<W: Write> EncodingStream<W> {
    /// Create a new RGB encoding stream.
    fn new_rgb(config: StreamingEncoder, width: u32, height: u32, writer: W) -> Result<Self> {
        Self::new(config, width, height, 3, writer)
    }

    /// Create a new grayscale encoding stream.
    fn new_gray(config: StreamingEncoder, width: u32, height: u32, writer: W) -> Result<Self> {
        Self::new(config, width, height, 1, writer)
    }

    /// Create a new encoding stream.
    fn new(
        config: StreamingEncoder,
        width: u32,
        height: u32,
        num_components: u8,
        writer: W,
    ) -> Result<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Determine MCU dimensions based on subsampling
        let (mcu_width, mcu_height) = if num_components == 1 {
            (DCTSIZE as u32, DCTSIZE as u32)
        } else {
            match config.subsampling {
                Subsampling::S444 | Subsampling::Gray => (DCTSIZE as u32, DCTSIZE as u32),
                Subsampling::S422 => (DCTSIZE as u32 * 2, DCTSIZE as u32),
                Subsampling::S420 => (DCTSIZE as u32 * 2, DCTSIZE as u32 * 2),
                Subsampling::S440 => (DCTSIZE as u32, DCTSIZE as u32 * 2),
            }
        };

        let mcus_per_row = (width + mcu_width - 1) / mcu_width;

        // Create quantization tables
        let (luma_qtable, chroma_qtable) = create_quant_tables(
            config.quality,
            config.quant_table_idx,
            config.force_baseline,
        );

        // Apply custom tables if specified
        let luma_qtable = if let Some(custom) = config.custom_luma_qtable {
            let mut values = luma_qtable.values;
            for (i, &val) in custom.iter().enumerate() {
                values[JPEG_NATURAL_ORDER[i] as usize] = val;
            }
            QuantTable::new(values)
        } else {
            luma_qtable
        };

        let chroma_qtable = if let Some(custom) = config.custom_chroma_qtable {
            let mut values = chroma_qtable.values;
            for (i, &val) in custom.iter().enumerate() {
                values[JPEG_NATURAL_ORDER[i] as usize] = val;
            }
            QuantTable::new(values)
        } else {
            chroma_qtable
        };

        // Create standard Huffman tables (no optimization in streaming mode)
        let dc_luma_htable = create_std_dc_luma_table();
        let ac_luma_htable = create_std_ac_luma_table();
        let dc_chroma_htable = create_std_dc_chroma_table();
        let ac_chroma_htable = create_std_ac_chroma_table();

        // Derive encoding tables
        let dc_luma_table = DerivedTable::from_huff_table(&dc_luma_htable, true)?;
        let ac_luma_table = DerivedTable::from_huff_table(&ac_luma_htable, false)?;
        let dc_chroma_table = DerivedTable::from_huff_table(&dc_chroma_htable, true)?;
        let ac_chroma_table = DerivedTable::from_huff_table(&ac_chroma_htable, false)?;

        // Allocate scanline buffer for one MCU row
        let buffer_size = (mcu_height as usize) * (width as usize) * (num_components as usize);
        let scanline_buffer = try_alloc_vec(0u8, buffer_size)?;

        let mut marker_writer = MarkerWriter::new(writer);

        // Write JPEG headers
        marker_writer.write_soi()?;
        marker_writer.write_jfif_app0(
            config.pixel_density.unit as u8,
            config.pixel_density.x,
            config.pixel_density.y,
        )?;

        // Write EXIF data if provided
        if let Some(ref exif) = config.exif_data {
            marker_writer.write_app1_exif(exif)?;
        }

        // Write ICC profile if provided
        if let Some(ref icc) = config.icc_profile {
            marker_writer.write_icc_profile(icc)?;
        }

        // Write custom markers
        for (app_num, data) in &config.custom_markers {
            marker_writer.write_app(*app_num, data)?;
        }

        // Write quantization tables
        let use_16bit = !config.force_baseline;
        if num_components == 1 {
            marker_writer.write_dqt(0, &luma_qtable.values, use_16bit)?;
        } else {
            marker_writer.write_dqt_multiple(&[
                (0, &luma_qtable.values, use_16bit),
                (1, &chroma_qtable.values, use_16bit),
            ])?;
        }

        // Write frame header
        let components: Vec<ComponentInfo> = if num_components == 1 {
            vec![ComponentInfo {
                component_id: 1,
                component_index: 0,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_tbl_no: 0,
                dc_tbl_no: 0,
                ac_tbl_no: 0,
            }]
        } else {
            let (h_samp, v_samp) = match config.subsampling {
                Subsampling::S444 | Subsampling::Gray => (1, 1),
                Subsampling::S422 => (2, 1),
                Subsampling::S420 => (2, 2),
                Subsampling::S440 => (1, 2),
            };
            vec![
                ComponentInfo {
                    component_id: 1,
                    component_index: 0,
                    h_samp_factor: h_samp,
                    v_samp_factor: v_samp,
                    quant_tbl_no: 0,
                    dc_tbl_no: 0,
                    ac_tbl_no: 0,
                },
                ComponentInfo {
                    component_id: 2,
                    component_index: 1,
                    h_samp_factor: 1,
                    v_samp_factor: 1,
                    quant_tbl_no: 1,
                    dc_tbl_no: 1,
                    ac_tbl_no: 1,
                },
                ComponentInfo {
                    component_id: 3,
                    component_index: 2,
                    h_samp_factor: 1,
                    v_samp_factor: 1,
                    quant_tbl_no: 1,
                    dc_tbl_no: 1,
                    ac_tbl_no: 1,
                },
            ]
        };

        // Always baseline (not progressive) for streaming
        marker_writer.write_sof(false, 8, height as u16, width as u16, &components)?;

        // Write Huffman tables
        if num_components == 1 {
            marker_writer
                .write_dht_multiple(&[(0, false, &dc_luma_htable), (0, true, &ac_luma_htable)])?;
        } else {
            marker_writer.write_dht_multiple(&[
                (0, false, &dc_luma_htable),
                (0, true, &ac_luma_htable),
                (1, false, &dc_chroma_htable),
                (1, true, &ac_chroma_htable),
            ])?;
        }

        // Write restart interval if specified
        if config.restart_interval > 0 {
            marker_writer.write_dri(config.restart_interval)?;
        }

        // Write SOS marker
        let scans = generate_baseline_scan(num_components);
        marker_writer.write_sos(&scans[0], &components)?;

        Ok(Self {
            writer: marker_writer,
            width,
            num_components,
            bytes_per_pixel: num_components,
            subsampling: config.subsampling,
            mcu_height,
            mcu_width,
            mcus_per_row,
            luma_qtable,
            chroma_qtable,
            dc_luma_table,
            ac_luma_table,
            dc_chroma_table,
            ac_chroma_table,
            prev_dc: [0; 4],
            scanline_buffer,
            lines_in_buffer: 0,
            bit_buffer: 0,
            bits_in_buffer: 0,
            simd: config.simd,
            restart_interval: config.restart_interval,
            mcus_since_restart: 0,
            next_restart_num: 0,
        })
    }

    /// Write scanlines to the encoder.
    ///
    /// Scanlines are buffered until a complete MCU row is available, then encoded.
    /// The number of bytes should be `num_lines * width * bytes_per_pixel`.
    ///
    /// For best performance, write in multiples of the MCU height (8 or 16 lines).
    pub fn write_scanlines(&mut self, data: &[u8]) -> Result<()> {
        let bytes_per_line = self.width as usize * self.bytes_per_pixel as usize;
        let lines_in_data = data.len() / bytes_per_line;

        if data.len() != lines_in_data * bytes_per_line {
            return Err(Error::BufferSizeMismatch {
                expected: lines_in_data * bytes_per_line,
                actual: data.len(),
            });
        }

        let mut data_offset = 0;
        let mut lines_remaining = lines_in_data as u32;

        while lines_remaining > 0 {
            // How many lines can we fit in the buffer?
            let lines_to_copy =
                (self.mcu_height - self.lines_in_buffer).min(lines_remaining) as usize;

            // Copy lines to buffer
            let buffer_offset = self.lines_in_buffer as usize * bytes_per_line;
            let src_bytes = lines_to_copy * bytes_per_line;
            self.scanline_buffer[buffer_offset..buffer_offset + src_bytes]
                .copy_from_slice(&data[data_offset..data_offset + src_bytes]);

            self.lines_in_buffer += lines_to_copy as u32;
            data_offset += src_bytes;
            lines_remaining -= lines_to_copy as u32;

            // If buffer is full, encode the MCU row
            if self.lines_in_buffer == self.mcu_height {
                self.encode_mcu_row()?;
                self.lines_in_buffer = 0;
            }
        }

        Ok(())
    }

    /// Encode one complete MCU row from the scanline buffer.
    fn encode_mcu_row(&mut self) -> Result<()> {
        let width = self.width as usize;
        let mcu_height = self.mcu_height as usize;

        if self.num_components == 1 {
            self.encode_gray_mcu_row(width, mcu_height)?;
        } else {
            self.encode_color_mcu_row(width, mcu_height)?;
        }

        Ok(())
    }

    /// Encode a grayscale MCU row.
    fn encode_gray_mcu_row(&mut self, width: usize, mcu_height: usize) -> Result<()> {
        let mcus_per_row = self.mcus_per_row as usize;

        for mcu_x in 0..mcus_per_row {
            // Handle restart markers
            if self.restart_interval > 0 && self.mcus_since_restart == self.restart_interval {
                self.write_restart_marker()?;
            }

            // Extract 8x8 block
            let mut block = [0i16; DCTSIZE2];
            let x_start = mcu_x * DCTSIZE;

            for y in 0..DCTSIZE {
                let src_y = y.min(mcu_height - 1);
                for x in 0..DCTSIZE {
                    let src_x = (x_start + x).min(width - 1);
                    let pixel = self.scanline_buffer[src_y * width + src_x];
                    // Level shift: 0..255 -> -128..127
                    block[y * DCTSIZE + x] = pixel as i16 - 128;
                }
            }

            // Forward DCT
            let mut dct_block = [0i16; DCTSIZE2];
            (self.simd.forward_dct)(&block, &mut dct_block);

            // Convert to i32 for quantization
            let mut dct_i32 = [0i32; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                dct_i32[i] = dct_block[i] as i32;
            }

            // Quantize
            let mut quantized = [0i16; DCTSIZE2];
            quantize_block(&dct_i32, &self.luma_qtable.values, &mut quantized);

            // Encode DC coefficient (differential)
            let dc = quantized[0] as i32;
            let dc_diff = dc - self.prev_dc[0];
            self.prev_dc[0] = dc;

            // Clone tables to avoid borrow conflicts
            let dc_table = self.dc_luma_table.clone();
            let ac_table = self.ac_luma_table.clone();
            self.encode_dc(dc_diff, &dc_table)?;
            self.encode_ac(&quantized, &ac_table)?;

            if self.restart_interval > 0 {
                self.mcus_since_restart += 1;
            }
        }

        Ok(())
    }

    /// Encode a color (YCbCr) MCU row.
    fn encode_color_mcu_row(&mut self, width: usize, mcu_height: usize) -> Result<()> {
        let mcus_per_row = self.mcus_per_row as usize;

        // Temporary storage for Y, Cb, Cr planes
        let mcu_width = self.mcu_width as usize;
        let mut y_plane = vec![0i16; mcu_width * mcu_height];
        let mut cb_plane = vec![0i16; mcu_width * mcu_height];
        let mut cr_plane = vec![0i16; mcu_width * mcu_height];

        for mcu_x in 0..mcus_per_row {
            // Handle restart markers
            if self.restart_interval > 0 && self.mcus_since_restart == self.restart_interval {
                self.write_restart_marker()?;
            }

            let x_start = mcu_x * mcu_width;

            // Convert RGB to YCbCr for this MCU
            for y in 0..mcu_height {
                let src_y = y.min(mcu_height - 1);
                for x in 0..mcu_width {
                    let src_x = (x_start + x).min(width - 1);
                    let pixel_idx = (src_y * width + src_x) * 3;

                    let r = self.scanline_buffer[pixel_idx] as i32;
                    let g = self.scanline_buffer[pixel_idx + 1] as i32;
                    let b = self.scanline_buffer[pixel_idx + 2] as i32;

                    // RGB to YCbCr conversion (BT.601)
                    let y_val = ((77 * r + 150 * g + 29 * b + 128) >> 8) - 128;
                    let cb_val = (-43 * r - 85 * g + 128 * b + 128) >> 8;
                    let cr_val = (128 * r - 107 * g - 21 * b + 128) >> 8;

                    let idx = y * mcu_width + x;
                    y_plane[idx] = y_val as i16;
                    cb_plane[idx] = cb_val as i16;
                    cr_plane[idx] = cr_val as i16;
                }
            }

            // Encode Y blocks
            match self.subsampling {
                Subsampling::S444 | Subsampling::Gray => {
                    // One 8x8 Y block
                    self.encode_luma_block(&y_plane, mcu_width, 0, 0)?;
                }
                Subsampling::S422 => {
                    // Two 8x8 Y blocks horizontally
                    self.encode_luma_block(&y_plane, mcu_width, 0, 0)?;
                    self.encode_luma_block(&y_plane, mcu_width, DCTSIZE, 0)?;
                }
                Subsampling::S420 => {
                    // Four 8x8 Y blocks (2x2)
                    self.encode_luma_block(&y_plane, mcu_width, 0, 0)?;
                    self.encode_luma_block(&y_plane, mcu_width, DCTSIZE, 0)?;
                    self.encode_luma_block(&y_plane, mcu_width, 0, DCTSIZE)?;
                    self.encode_luma_block(&y_plane, mcu_width, DCTSIZE, DCTSIZE)?;
                }
                Subsampling::S440 => {
                    // Two 8x8 Y blocks vertically
                    self.encode_luma_block(&y_plane, mcu_width, 0, 0)?;
                    self.encode_luma_block(&y_plane, mcu_width, 0, DCTSIZE)?;
                }
            }

            // Downsample and encode Cb, Cr
            self.encode_chroma_block(&cb_plane, mcu_width, 1)?;
            self.encode_chroma_block(&cr_plane, mcu_width, 2)?;

            if self.restart_interval > 0 {
                self.mcus_since_restart += 1;
            }
        }

        Ok(())
    }

    /// Encode a single 8x8 luma block from the Y plane.
    fn encode_luma_block(
        &mut self,
        y_plane: &[i16],
        plane_width: usize,
        x_off: usize,
        y_off: usize,
    ) -> Result<()> {
        let mut block = [0i16; DCTSIZE2];

        for y in 0..DCTSIZE {
            for x in 0..DCTSIZE {
                block[y * DCTSIZE + x] = y_plane[(y_off + y) * plane_width + x_off + x];
            }
        }

        let mut dct_block = [0i16; DCTSIZE2];
        (self.simd.forward_dct)(&block, &mut dct_block);

        // Convert to i32 for quantization
        let mut dct_i32 = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            dct_i32[i] = dct_block[i] as i32;
        }

        let mut quantized = [0i16; DCTSIZE2];
        quantize_block(&dct_i32, &self.luma_qtable.values, &mut quantized);

        let dc = quantized[0] as i32;
        let dc_diff = dc - self.prev_dc[0];
        self.prev_dc[0] = dc;

        // Clone tables to avoid borrow conflicts
        let dc_table = self.dc_luma_table.clone();
        let ac_table = self.ac_luma_table.clone();
        self.encode_dc(dc_diff, &dc_table)?;
        self.encode_ac(&quantized, &ac_table)?;

        Ok(())
    }

    /// Encode a chroma block with downsampling.
    fn encode_chroma_block(
        &mut self,
        chroma_plane: &[i16],
        plane_width: usize,
        comp_idx: usize,
    ) -> Result<()> {
        let mut block = [0i16; DCTSIZE2];

        // Downsample based on subsampling mode
        match self.subsampling {
            Subsampling::S444 | Subsampling::Gray => {
                // No downsampling
                for y in 0..DCTSIZE {
                    for x in 0..DCTSIZE {
                        block[y * DCTSIZE + x] = chroma_plane[y * plane_width + x];
                    }
                }
            }
            Subsampling::S422 => {
                // 2:1 horizontal downsampling
                for y in 0..DCTSIZE {
                    for x in 0..DCTSIZE {
                        let x2 = x * 2;
                        let val = (chroma_plane[y * plane_width + x2] as i32
                            + chroma_plane[y * plane_width + x2 + 1] as i32)
                            / 2;
                        block[y * DCTSIZE + x] = val as i16;
                    }
                }
            }
            Subsampling::S420 => {
                // 2:1 horizontal and vertical downsampling
                for y in 0..DCTSIZE {
                    for x in 0..DCTSIZE {
                        let x2 = x * 2;
                        let y2 = y * 2;
                        let val = (chroma_plane[y2 * plane_width + x2] as i32
                            + chroma_plane[y2 * plane_width + x2 + 1] as i32
                            + chroma_plane[(y2 + 1) * plane_width + x2] as i32
                            + chroma_plane[(y2 + 1) * plane_width + x2 + 1] as i32)
                            / 4;
                        block[y * DCTSIZE + x] = val as i16;
                    }
                }
            }
            Subsampling::S440 => {
                // 2:1 vertical downsampling
                for y in 0..DCTSIZE {
                    for x in 0..DCTSIZE {
                        let y2 = y * 2;
                        let val = (chroma_plane[y2 * plane_width + x] as i32
                            + chroma_plane[(y2 + 1) * plane_width + x] as i32)
                            / 2;
                        block[y * DCTSIZE + x] = val as i16;
                    }
                }
            }
        }

        let mut dct_block = [0i16; DCTSIZE2];
        (self.simd.forward_dct)(&block, &mut dct_block);

        // Convert to i32 for quantization
        let mut dct_i32 = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            dct_i32[i] = dct_block[i] as i32;
        }

        let mut quantized = [0i16; DCTSIZE2];
        quantize_block(&dct_i32, &self.chroma_qtable.values, &mut quantized);

        let dc = quantized[0] as i32;
        let dc_diff = dc - self.prev_dc[comp_idx];
        self.prev_dc[comp_idx] = dc;

        // Clone tables to avoid borrow conflicts
        let dc_table = self.dc_chroma_table.clone();
        let ac_table = self.ac_chroma_table.clone();
        self.encode_dc(dc_diff, &dc_table)?;
        self.encode_ac(&quantized, &ac_table)?;

        Ok(())
    }

    /// Encode a DC coefficient (differential).
    fn encode_dc(&mut self, diff: i32, table: &DerivedTable) -> Result<()> {
        let (size, bits) = if diff == 0 {
            (0, 0)
        } else {
            let abs_diff = diff.unsigned_abs();
            let size = 32 - abs_diff.leading_zeros();
            let bits = if diff > 0 {
                diff as u32
            } else {
                (diff - 1) as u32 & ((1 << size) - 1)
            };
            (size, bits)
        };

        // Write Huffman code for size
        let (code, code_len) = table.get_code(size as u8);
        self.write_bits(code as u64, code_len)?;

        // Write magnitude bits
        if size > 0 {
            self.write_bits(bits as u64, size as u8)?;
        }

        Ok(())
    }

    /// Encode AC coefficients in zigzag order.
    fn encode_ac(&mut self, quantized: &[i16; DCTSIZE2], table: &DerivedTable) -> Result<()> {
        let mut run = 0;

        for i in 1..DCTSIZE2 {
            let val = quantized[JPEG_NATURAL_ORDER[i] as usize];

            if val == 0 {
                run += 1;
            } else {
                // Write ZRL (16 zeros) codes if needed
                while run >= 16 {
                    let (code, code_len) = table.get_code(0xF0); // ZRL
                    self.write_bits(code as u64, code_len)?;
                    run -= 16;
                }

                // Compute size and bits
                let abs_val = val.unsigned_abs() as u32;
                let size = 32 - abs_val.leading_zeros();
                let bits = if val > 0 {
                    val as u32
                } else {
                    (val - 1) as u32 & ((1 << size) - 1)
                };

                // Symbol is (run << 4) | size
                let symbol = ((run as u8) << 4) | (size as u8);
                let (code, code_len) = table.get_code(symbol);
                self.write_bits(code as u64, code_len)?;
                self.write_bits(bits as u64, size as u8)?;

                run = 0;
            }
        }

        // End of block
        if run > 0 {
            let (code, code_len) = table.get_code(0x00); // EOB
            self.write_bits(code as u64, code_len)?;
        }

        Ok(())
    }

    /// Write bits to the output with byte stuffing.
    fn write_bits(&mut self, bits: u64, count: u8) -> Result<()> {
        self.bit_buffer |= bits << (64 - self.bits_in_buffer - count);
        self.bits_in_buffer += count;

        while self.bits_in_buffer >= 8 {
            let byte = (self.bit_buffer >> 56) as u8;
            self.writer.get_mut().write_all(&[byte])?;

            // Byte stuffing: 0xFF must be followed by 0x00
            if byte == 0xFF {
                self.writer.get_mut().write_all(&[0x00])?;
            }

            self.bit_buffer <<= 8;
            self.bits_in_buffer -= 8;
        }

        Ok(())
    }

    /// Write a restart marker and reset DC predictors.
    fn write_restart_marker(&mut self) -> Result<()> {
        // Flush remaining bits with 1s padding
        if self.bits_in_buffer > 0 {
            let padding = 8 - self.bits_in_buffer;
            self.write_bits((1u64 << padding) - 1, padding)?;
        }

        // Write RST marker
        let marker = 0xD0 + self.next_restart_num;
        self.writer.get_mut().write_all(&[0xFF, marker])?;

        // Reset state
        self.prev_dc = [0; 4];
        self.mcus_since_restart = 0;
        self.next_restart_num = (self.next_restart_num + 1) & 7;

        Ok(())
    }

    /// Finish encoding and write the EOI marker.
    ///
    /// This must be called after all scanlines have been written.
    /// Consumes the stream and returns the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        // Encode any remaining lines in the buffer (partial MCU row)
        if self.lines_in_buffer > 0 {
            // Pad the buffer with the last line
            let bytes_per_line = self.width as usize * self.bytes_per_pixel as usize;
            let last_line_start = (self.lines_in_buffer as usize - 1) * bytes_per_line;
            let last_line =
                self.scanline_buffer[last_line_start..last_line_start + bytes_per_line].to_vec();

            while self.lines_in_buffer < self.mcu_height {
                let buffer_offset = self.lines_in_buffer as usize * bytes_per_line;
                self.scanline_buffer[buffer_offset..buffer_offset + bytes_per_line]
                    .copy_from_slice(&last_line);
                self.lines_in_buffer += 1;
            }

            self.encode_mcu_row()?;
        }

        // Flush remaining bits with 1s padding
        if self.bits_in_buffer > 0 {
            let padding = 8 - self.bits_in_buffer;
            self.write_bits((1u64 << padding) - 1, padding)?;
        }

        // Write EOI marker
        self.writer.write_eoi()?;

        Ok(self.writer.into_inner())
    }
}

// Add streaming() method to Encoder
impl Encoder {
    /// Create a streaming encoder.
    ///
    /// Returns a [`StreamingEncoder`] which supports scanline-by-scanline encoding.
    /// Note that streaming mode does NOT support trellis quantization, progressive
    /// mode, or Huffman optimization (these require buffering the entire image).
    ///
    /// For full-featured encoding with all mozjpeg optimizations, use [`Encoder::new()`]
    /// with [`encode_rgb()`](Encoder::encode_rgb) or [`encode_gray()`](Encoder::encode_gray).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use mozjpeg_oxide::Encoder;
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

#[cfg(test)]
mod tests {
    use super::*;
    use dssim::Dssim;

    #[test]
    fn test_encoder_defaults() {
        let enc = Encoder::new();
        assert_eq!(enc.quality, 75);
        assert!(!enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S420);
        // mozjpeg defaults: trellis and Huffman optimization enabled
        assert!(enc.trellis.enabled);
        assert!(enc.optimize_huffman);
    }

    /// Verify JPEG output can be decoded by an external decoder
    #[test]
    fn test_decode_with_jpeg_decoder() {
        // Create a 16x16 gradient image
        let width = 16u32;
        let height = 16u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 16 + y * 8) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val / 2;
                rgb_data[i * 3 + 2] = 255 - val;
            }
        }

        let encoder = Encoder::new().quality(90).subsampling(Subsampling::S444);
        let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Decode with jpeg-decoder
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
        let decoded = decoder.decode().expect("Failed to decode JPEG");

        // Verify dimensions
        let info = decoder.info().unwrap();
        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);

        // Decoded data should have same number of pixels
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_encoder_builder() {
        let enc = Encoder::new()
            .quality(90)
            .progressive(true)
            .subsampling(Subsampling::S444);

        assert_eq!(enc.quality, 90);
        assert!(enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S444);
    }

    #[test]
    fn test_quality_clamping() {
        let enc = Encoder::new().quality(0);
        assert_eq!(enc.quality, 1);

        let enc = Encoder::new().quality(150);
        assert_eq!(enc.quality, 100);
    }

    #[test]
    fn test_natural_to_zigzag() {
        let mut natural = [0u16; 64];
        for i in 0..64 {
            natural[i] = i as u16;
        }
        let zigzag = natural_to_zigzag(&natural);

        // Position 0 (DC) should be 0
        assert_eq!(zigzag[0], 0);
        // Position 1 should be 1 (first AC in zigzag)
        assert_eq!(zigzag[1], 1);
    }

    #[test]
    fn test_encode_small_image() {
        // Create a small 16x16 red image
        let width = 16u32;
        let height = 16u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        // Fill with red
        for i in 0..(width * height) as usize {
            rgb_data[i * 3] = 255; // R
            rgb_data[i * 3 + 1] = 0; // G
            rgb_data[i * 3 + 2] = 0; // B
        }

        let encoder = Encoder::new().quality(75);
        let result = encoder.encode_rgb(&rgb_data, width, height);

        assert!(result.is_ok());
        let jpeg_data = result.unwrap();

        // Check JPEG markers
        assert_eq!(jpeg_data[0], 0xFF);
        assert_eq!(jpeg_data[1], 0xD8); // SOI
        assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
        assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI
    }

    #[test]
    fn test_encode_gradient() {
        // Create an 8x8 gradient image
        let width = 8u32;
        let height = 8u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x + y) * 16) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val;
                rgb_data[i * 3 + 2] = val;
            }
        }

        let encoder = Encoder::new().quality(90).subsampling(Subsampling::S444);
        let result = encoder.encode_rgb(&rgb_data, width, height);

        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_grayscale() {
        // Create a 16x16 grayscale gradient
        let width = 16u32;
        let height = 16u32;
        let mut gray_data = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                gray_data[i] = ((x + y) * 8) as u8;
            }
        }

        let encoder = Encoder::new().quality(85);
        let result = encoder.encode_gray(&gray_data, width, height);

        assert!(result.is_ok());
        let jpeg_data = result.unwrap();

        // Check JPEG markers
        assert_eq!(jpeg_data[0], 0xFF);
        assert_eq!(jpeg_data[1], 0xD8); // SOI
        assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
        assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9); // EOI

        // Decode and verify it's grayscale
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
        let decoded = decoder.decode().expect("Failed to decode grayscale JPEG");
        let info = decoder.info().unwrap();

        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);
        // Grayscale should have 1 component
        assert_eq!(decoded.len(), (width * height) as usize);
    }

    #[test]
    fn test_encode_with_exif() {
        // Create a small test image
        let width = 16u32;
        let height = 16u32;
        let rgb_data = vec![128u8; (width * height * 3) as usize];

        // Simple EXIF-like data (minimal TIFF header)
        // Real EXIF would have proper TIFF structure, this is just for marker presence testing
        let exif_data = vec![
            0x4D, 0x4D, // Big-endian TIFF
            0x00, 0x2A, // TIFF magic
            0x00, 0x00, 0x00, 0x08, // Offset to IFD
        ];

        let encoder = Encoder::new().quality(75).exif_data(exif_data.clone());

        let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Check for APP1 marker (0xFFE1) in the output
        let mut found_app1 = false;
        for i in 0..jpeg_data.len() - 1 {
            if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xE1 {
                found_app1 = true;
                // Check for "Exif\0\0" identifier
                if i + 4 < jpeg_data.len() {
                    let identifier = &jpeg_data[i + 4..i + 10];
                    assert_eq!(identifier, b"Exif\0\0");
                }
                break;
            }
        }
        assert!(found_app1, "APP1 (EXIF) marker not found in output");

        // Should still be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
        let decoded = decoder.decode().expect("Failed to decode JPEG with EXIF");
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_encode_with_restart_markers() {
        // Create a larger image to ensure restart markers are emitted
        let width = 64u32;
        let height = 64u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        // Create a pattern
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                rgb_data[i * 3] = (x * 4) as u8;
                rgb_data[i * 3 + 1] = (y * 4) as u8;
                rgb_data[i * 3 + 2] = 128;
            }
        }

        // Set restart interval (every 4 MCUs)
        let encoder = Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S444) // 1x1, so 64 MCUs for 64x64 image
            .optimize_huffman(false) // Use streaming path
            .trellis(TrellisConfig::disabled())
            .restart_interval(4);

        let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Check for DRI marker (0xFFDD) in the output
        let mut found_dri = false;
        for i in 0..jpeg_data.len() - 1 {
            if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDD {
                found_dri = true;
                // DRI marker should be followed by length (4) and interval
                if i + 5 < jpeg_data.len() {
                    let len = ((jpeg_data[i + 2] as u16) << 8) | (jpeg_data[i + 3] as u16);
                    assert_eq!(len, 4, "DRI marker length should be 4");
                    let interval = ((jpeg_data[i + 4] as u16) << 8) | (jpeg_data[i + 5] as u16);
                    assert_eq!(interval, 4, "Restart interval should be 4");
                }
                break;
            }
        }
        assert!(found_dri, "DRI marker not found in output");

        // Check for RST markers (0xFFD0-0xFFD7) in the entropy data
        // With 64 MCUs and interval=4, we should have 15 RST markers (after MCUs 4,8,12,...60)
        let mut rst_count = 0;
        for i in 0..jpeg_data.len() - 1 {
            if jpeg_data[i] == 0xFF && jpeg_data[i + 1] >= 0xD0 && jpeg_data[i + 1] <= 0xD7 {
                rst_count += 1;
            }
        }
        // 64 MCUs / 4 = 16 groups, so 15 restart markers between them
        assert_eq!(
            rst_count, 15,
            "Expected 15 RST markers, found {}",
            rst_count
        );

        // Should be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
        let decoded = decoder
            .decode()
            .expect("Failed to decode JPEG with restart markers");
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_encode_invalid_size() {
        let rgb_data = vec![0u8; 100]; // Wrong size
        let encoder = Encoder::new();
        let result = encoder.encode_rgb(&rgb_data, 16, 16);

        assert!(result.is_err());
    }

    #[test]
    fn test_encode_zero_dimensions() {
        let encoder = Encoder::new();

        // Zero width
        let result = encoder.encode_rgb(&[], 0, 16);
        assert!(matches!(
            result,
            Err(Error::InvalidDimensions {
                width: 0,
                height: 16
            })
        ));

        // Zero height
        let result = encoder.encode_rgb(&[], 16, 0);
        assert!(matches!(
            result,
            Err(Error::InvalidDimensions {
                width: 16,
                height: 0
            })
        ));

        // Both zero
        let result = encoder.encode_rgb(&[], 0, 0);
        assert!(matches!(
            result,
            Err(Error::InvalidDimensions {
                width: 0,
                height: 0
            })
        ));
    }

    #[test]
    fn test_encode_overflow_dimensions() {
        let encoder = Encoder::new();

        // Very large dimensions that would overflow usize multiplication
        // On 64-bit, this won't overflow but buffer size check will fail
        // On 32-bit, the checked_mul should catch it
        let result = encoder.encode_rgb(&[], u32::MAX, u32::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_progressive_encode_decode() {
        // Create a 16x16 gradient image
        let width = 16u32;
        let height = 16u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 16 + y * 8) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val / 2;
                rgb_data[i * 3 + 2] = 255 - val;
            }
        }

        let encoder = Encoder::new()
            .quality(85)
            .progressive(true)
            .subsampling(Subsampling::S420);

        let jpeg_data = encoder.encode_rgb(&rgb_data, width, height).unwrap();

        // Check JPEG markers
        assert_eq!(jpeg_data[0], 0xFF);
        assert_eq!(jpeg_data[1], 0xD8); // SOI

        // Check for SOF2 (progressive DCT marker) instead of SOF0 (baseline)
        let mut has_sof2 = false;
        let mut i = 2;
        while i < jpeg_data.len() - 1 {
            if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC2 {
                has_sof2 = true;
                break;
            }
            i += 1;
        }
        assert!(has_sof2, "Progressive JPEG should have SOF2 marker");

        // Decode with jpeg-decoder
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_data));
        let decoded = decoder.decode().expect("Failed to decode progressive JPEG");

        // Verify dimensions
        let info = decoder.info().unwrap();
        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);

        // Decoded data should have same number of pixels
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_progressive_vs_baseline_size() {
        // Create a larger image that might benefit from progressive
        let width = 64u32;
        let height = 64u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        // Fill with complex pattern
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = (((x as f32 * 0.1).sin() * 127.0 + 128.0) as u8)
                    .wrapping_add((((y as f32) * 0.1).cos() * 50.0) as u8);
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val.wrapping_add(30);
                rgb_data[i * 3 + 2] = 255 - val;
            }
        }

        // Encode baseline
        let baseline = Encoder::new()
            .quality(75)
            .progressive(false)
            .subsampling(Subsampling::S420);
        let baseline_data = baseline.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode progressive
        let progressive = Encoder::new()
            .quality(75)
            .progressive(true)
            .subsampling(Subsampling::S420);
        let progressive_data = progressive.encode_rgb(&rgb_data, width, height).unwrap();

        // Both should produce valid JPEGs
        assert!(!baseline_data.is_empty());
        assert!(!progressive_data.is_empty());

        println!("Baseline size: {} bytes", baseline_data.len());
        println!("Progressive size: {} bytes", progressive_data.len());

        // Both should be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline_data));
        decoder.decode().expect("Failed to decode baseline JPEG");

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive_data));
        decoder.decode().expect("Failed to decode progressive JPEG");
    }

    #[test]
    fn test_trellis_quantization_enabled() {
        // Create a test image with high-frequency content
        let width = 32u32;
        let height = 32u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        // Create a pattern that will have AC coefficients
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = (((x as i32 - y as i32).abs() * 10) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = 255 - val;
                rgb_data[i * 3 + 2] = val / 2;
            }
        }

        // Encode without trellis
        let no_trellis = Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S420)
            .trellis(TrellisConfig::disabled());
        let no_trellis_data = no_trellis.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with trellis
        let with_trellis = Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S420)
            .trellis(TrellisConfig::default());
        let with_trellis_data = with_trellis.encode_rgb(&rgb_data, width, height).unwrap();

        // Both should produce valid JPEGs
        assert!(!no_trellis_data.is_empty());
        assert!(!with_trellis_data.is_empty());

        println!("Without trellis: {} bytes", no_trellis_data.len());
        println!("With trellis: {} bytes", with_trellis_data.len());

        // Both should be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_trellis_data));
        decoder.decode().expect("Failed to decode non-trellis JPEG");

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_trellis_data));
        decoder.decode().expect("Failed to decode trellis JPEG");

        // Trellis optimization typically produces slightly different output
        // (may be slightly smaller due to better rate-distortion tradeoff)
    }

    #[test]
    fn test_trellis_presets() {
        // Create a larger test image with varied content
        let width = 64u32;
        let height = 64u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                // Create a pattern with gradients and edges
                let val = (((x as i32 - y as i32).abs() * 8) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = 255 - val;
                rgb_data[i * 3 + 2] = (val / 2).wrapping_add(64);
            }
        }

        // Test at high quality where the gap is most visible
        let quality = 97;

        // Encode with default trellis
        let default = Encoder::new()
            .quality(quality)
            .subsampling(Subsampling::S420)
            .trellis(TrellisConfig::default());
        let default_data = default.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with favor_size preset
        let favor_size = Encoder::new()
            .quality(quality)
            .subsampling(Subsampling::S420)
            .trellis(TrellisConfig::favor_size());
        let favor_size_data = favor_size.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with favor_quality preset
        let favor_quality = Encoder::new()
            .quality(quality)
            .subsampling(Subsampling::S420)
            .trellis(TrellisConfig::favor_quality());
        let favor_quality_data = favor_quality.encode_rgb(&rgb_data, width, height).unwrap();

        println!("Default trellis: {} bytes", default_data.len());
        println!("Favor size: {} bytes", favor_size_data.len());
        println!("Favor quality: {} bytes", favor_quality_data.len());

        // All should be valid
        for (name, data) in [
            ("default", &default_data),
            ("favor_size", &favor_size_data),
            ("favor_quality", &favor_quality_data),
        ] {
            let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
            decoder
                .decode()
                .expect(&format!("Failed to decode {} JPEG", name));
        }

        // favor_size should generally produce smaller files than default
        // favor_quality should generally produce larger files than default
        // (This may not always hold for small test images, so we just verify they're different)
        assert!(
            favor_size_data.len() != favor_quality_data.len(),
            "Presets should produce different sizes"
        );
    }

    #[test]
    fn test_trellis_rd_factor() {
        let width = 32u32;
        let height = 32u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 8 + y * 4) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val;
                rgb_data[i * 3 + 2] = val;
            }
        }

        // rd_factor(1.0) should behave like default
        let factor_1 = Encoder::new()
            .quality(85)
            .trellis(TrellisConfig::default().rd_factor(1.0));
        let factor_1_data = factor_1.encode_rgb(&rgb_data, width, height).unwrap();

        // rd_factor(2.0) should produce smaller files (more aggressive)
        let factor_2 = Encoder::new()
            .quality(85)
            .trellis(TrellisConfig::default().rd_factor(2.0));
        let factor_2_data = factor_2.encode_rgb(&rgb_data, width, height).unwrap();

        // Both should be valid JPEGs
        jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_1_data))
            .decode()
            .expect("Failed to decode rd_factor(1.0) JPEG");
        jpeg_decoder::Decoder::new(std::io::Cursor::new(&factor_2_data))
            .decode()
            .expect("Failed to decode rd_factor(2.0) JPEG");

        println!("rd_factor(1.0): {} bytes", factor_1_data.len());
        println!("rd_factor(2.0): {} bytes", factor_2_data.len());
    }

    #[test]
    fn test_max_compression_uses_all_optimizations() {
        let encoder = Encoder::max_compression();
        assert!(encoder.trellis.enabled);
        assert!(encoder.progressive);
        assert!(encoder.optimize_huffman);
    }

    #[test]
    fn test_huffman_optimization() {
        // Create a test image
        let width = 32u32;
        let height = 32u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 8 + y * 4) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val;
                rgb_data[i * 3 + 2] = val;
            }
        }

        // Encode without Huffman optimization
        let no_opt = Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S420)
            .optimize_huffman(false);
        let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode with Huffman optimization
        let with_opt = Encoder::new()
            .quality(75)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true);
        let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

        // Both should produce valid JPEGs
        assert!(!no_opt_data.is_empty());
        assert!(!with_opt_data.is_empty());

        println!("Without Huffman opt: {} bytes", no_opt_data.len());
        println!("With Huffman opt: {} bytes", with_opt_data.len());

        // Both should be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
        decoder
            .decode()
            .expect("Failed to decode non-optimized JPEG");

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
        decoder.decode().expect("Failed to decode optimized JPEG");

        // Optimized version should typically be smaller or equal
        // (though this depends on the image content)
    }

    #[test]
    fn test_color_encoding_accuracy() {
        // Test that solid colors encode correctly
        let test_cases = [
            ("black", 0u8, 0u8, 0u8),
            ("red", 255, 0, 0),
            ("green", 0, 255, 0),
            ("blue", 0, 0, 255),
            ("white", 255, 255, 255),
            ("gray", 128, 128, 128),
        ];

        let width = 16u32;
        let height = 16u32;

        for (name, r, g, b) in &test_cases {
            let mut rgb_data = vec![0u8; (width * height * 3) as usize];
            for i in 0..(width * height) as usize {
                rgb_data[i * 3] = *r;
                rgb_data[i * 3 + 1] = *g;
                rgb_data[i * 3 + 2] = *b;
            }

            // Encode with 4:4:4 to avoid chroma subsampling issues
            let encoder = Encoder::new().quality(95).subsampling(Subsampling::S444);
            let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

            // Decode and check
            let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
            let decoded = decoder.decode().expect("decode failed");

            let dr = decoded[0];
            let dg = decoded[1];
            let db = decoded[2];

            // Solid colors at Q95 with 4:4:4 should have minimal loss
            let tolerance = 2i16;
            let r_diff = (dr as i16 - *r as i16).abs();
            let g_diff = (dg as i16 - *g as i16).abs();
            let b_diff = (db as i16 - *b as i16).abs();

            println!(
                "{}: input=({},{},{}), output=({},{},{}), diff=({},{},{})",
                name, r, g, b, dr, dg, db, r_diff, g_diff, b_diff
            );

            assert!(
                r_diff <= tolerance,
                "{}: R mismatch - expected {}, got {} (diff {})",
                name,
                r,
                dr,
                r_diff
            );
            assert!(
                g_diff <= tolerance,
                "{}: G mismatch - expected {}, got {} (diff {})",
                name,
                g,
                dg,
                g_diff
            );
            assert!(
                b_diff <= tolerance,
                "{}: B mismatch - expected {}, got {} (diff {})",
                name,
                b,
                db,
                b_diff
            );
        }
    }

    #[test]
    fn test_optimize_scans() {
        // Create a test image with varied content
        let width = 64u32;
        let height = 64u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 4 + y * 3) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = 255 - val;
                rgb_data[i * 3 + 2] = ((val as u16 + 128) % 256) as u8;
            }
        }

        // Encode progressive without scan optimization
        let no_opt = Encoder::new()
            .quality(75)
            .progressive(true)
            .optimize_scans(false)
            .subsampling(Subsampling::S420);
        let no_opt_data = no_opt.encode_rgb(&rgb_data, width, height).unwrap();

        // Encode progressive with scan optimization
        let with_opt = Encoder::new()
            .quality(75)
            .progressive(true)
            .optimize_scans(true)
            .subsampling(Subsampling::S420);
        let with_opt_data = with_opt.encode_rgb(&rgb_data, width, height).unwrap();

        println!("Progressive without scan opt: {} bytes", no_opt_data.len());
        println!("Progressive with scan opt: {} bytes", with_opt_data.len());

        // Both should be decodable
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&no_opt_data));
        decoder
            .decode()
            .expect("Failed to decode non-optimized JPEG");

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&with_opt_data));
        decoder
            .decode()
            .expect("Failed to decode scan-optimized JPEG");

        // Optimized version should be valid (size comparison depends on image)
        assert!(!with_opt_data.is_empty());
    }

    #[test]
    fn test_max_compression_enables_optimize_scans() {
        let encoder = Encoder::max_compression();
        assert!(encoder.optimize_scans);
        assert!(encoder.progressive);
    }

    /// Regression test: Progressive encoding with non-MCU-aligned dimensions.
    ///
    /// This tests the fix for a bug where progressive AC scans were encoding
    /// MCU-padded blocks instead of actual component blocks. For non-interleaved
    /// scans (AC scans), the block count should be ceil(width/8) × ceil(height/8),
    /// not the MCU-padded count.
    ///
    /// The bug manifested as corrupted images (PSNR ~29 instead of ~43) for sizes
    /// like 17x17 with 4:2:0 subsampling, where the edge block row had v_idx=0.
    #[test]
    fn test_progressive_non_mcu_aligned_regression() {
        // Test sizes that triggered the bug: where (size-1)/8 % 2 == 0
        // These have edge blocks with v_idx=0 in 4:2:0 mode
        let failing_sizes = [17, 24, 33, 40, 49];

        for &size in &failing_sizes {
            let s = size as usize;
            let mut rgb = vec![0u8; s * s * 3];
            for y in 0..s {
                for x in 0..s {
                    let idx = (y * s + x) * 3;
                    rgb[idx] = (x * 15).min(255) as u8;
                    rgb[idx + 1] = (y * 15).min(255) as u8;
                    rgb[idx + 2] = 128;
                }
            }

            // Encode baseline (reference)
            let baseline = Encoder::new()
                .quality(95)
                .subsampling(Subsampling::S420)
                .progressive(false)
                .optimize_huffman(true)
                .trellis(TrellisConfig::disabled())
                .encode_rgb(&rgb, size, size)
                .unwrap();

            // Encode progressive (this was buggy)
            let progressive = Encoder::new()
                .quality(95)
                .subsampling(Subsampling::S420)
                .progressive(true)
                .optimize_huffman(true)
                .trellis(TrellisConfig::disabled())
                .encode_rgb(&rgb, size, size)
                .unwrap();

            // Decode both
            let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
                .decode()
                .expect("baseline decode failed");
            let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
                .decode()
                .expect("progressive decode failed");

            // Calculate PSNR for both
            let base_psnr = calculate_psnr(&rgb, &base_dec);
            let prog_psnr = calculate_psnr(&rgb, &prog_dec);

            // Progressive PSNR should be within 3 dB of baseline (not 14 dB worse!)
            let diff = (prog_psnr - base_psnr).abs();
            assert!(
                diff < 3.0,
                "{}x{}: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
                size,
                size,
                prog_psnr,
                base_psnr,
                diff
            );

            // DSSIM perceptual quality check - both modes should be high quality
            // Note: non-MCU-aligned images with 4:2:0 have higher DSSIM due to edge effects
            let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
            let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
            assert!(
                base_dssim < 0.01,
                "{}x{}: Baseline DSSIM too high: {:.6}",
                size,
                size,
                base_dssim
            );
            assert!(
                prog_dssim < 0.01,
                "{}x{}: Progressive DSSIM too high: {:.6}",
                size,
                size,
                prog_dssim
            );
        }
    }

    /// Also test 4:2:2 subsampling which had the same bug
    #[test]
    fn test_progressive_422_non_mcu_aligned_regression() {
        let size = 17u32;
        let s = size as usize;
        let mut rgb = vec![0u8; s * s * 3];
        for y in 0..s {
            for x in 0..s {
                let idx = (y * s + x) * 3;
                rgb[idx] = (x * 15).min(255) as u8;
                rgb[idx + 1] = (y * 15).min(255) as u8;
                rgb[idx + 2] = 128;
            }
        }

        let baseline = Encoder::new()
            .quality(95)
            .subsampling(Subsampling::S422)
            .progressive(false)
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let progressive = Encoder::new()
            .quality(95)
            .subsampling(Subsampling::S422)
            .progressive(true)
            .encode_rgb(&rgb, size, size)
            .unwrap();

        let base_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&baseline))
            .decode()
            .unwrap();
        let prog_dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&progressive))
            .decode()
            .unwrap();

        let base_psnr = calculate_psnr(&rgb, &base_dec);
        let prog_psnr = calculate_psnr(&rgb, &prog_dec);
        let diff = (prog_psnr - base_psnr).abs();

        assert!(
            diff < 3.0,
            "4:2:2 17x17: Progressive PSNR ({:.1}) differs from baseline ({:.1}) by {:.1} dB",
            prog_psnr,
            base_psnr,
            diff
        );

        // DSSIM perceptual quality check - non-MCU-aligned has higher DSSIM
        let base_dssim = calculate_dssim(&rgb, &base_dec, size, size);
        let prog_dssim = calculate_dssim(&rgb, &prog_dec, size, size);
        assert!(
            base_dssim < 0.01,
            "4:2:2 17x17: Baseline DSSIM too high: {:.6}",
            base_dssim
        );
        assert!(
            prog_dssim < 0.01,
            "4:2:2 17x17: Progressive DSSIM too high: {:.6}",
            prog_dssim
        );
    }

    fn calculate_psnr(orig: &[u8], decoded: &[u8]) -> f64 {
        let mse: f64 = orig
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| {
                let diff = a as f64 - b as f64;
                diff * diff
            })
            .sum::<f64>()
            / orig.len() as f64;

        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }

    fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
        use rgb::RGB8;

        let attr = Dssim::new();

        let orig_rgb: Vec<RGB8> = original
            .chunks(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = attr
            .create_image_rgb(&orig_rgb, width as usize, height as usize)
            .expect("Failed to create original image");

        let dec_rgb: Vec<RGB8> = decoded
            .chunks(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let dec_img = attr
            .create_image_rgb(&dec_rgb, width as usize, height as usize)
            .expect("Failed to create decoded image");

        let (dssim_val, _) = attr.compare(&orig_img, dec_img);
        dssim_val.into()
    }

    #[test]
    fn test_streaming_encoder_rgb() {
        // Create a 16x16 gradient image
        let width = 16u32;
        let height = 16u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 16 + y * 8) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val / 2;
                rgb_data[i * 3 + 2] = 255 - val;
            }
        }

        // Encode using streaming encoder (batch mode via trait)
        let streaming = StreamingEncoder::new().quality(85);
        let streaming_data = streaming.encode_rgb(&rgb_data, width, height).unwrap();

        // Verify JPEG markers
        assert_eq!(streaming_data[0], 0xFF);
        assert_eq!(streaming_data[1], 0xD8); // SOI
        assert_eq!(streaming_data[streaming_data.len() - 2], 0xFF);
        assert_eq!(streaming_data[streaming_data.len() - 1], 0xD9); // EOI

        // Verify can be decoded
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
        let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

        let info = decoder.info().unwrap();
        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_streaming_encoder_scanlines() {
        // Create a 16x16 gradient image
        let width = 16u32;
        let height = 16u32;
        let mut rgb_data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let val = ((x * 16 + y * 8) % 256) as u8;
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val / 2;
                rgb_data[i * 3 + 2] = 255 - val;
            }
        }

        // Encode using streaming mode with explicit scanline writes
        let mut output = Vec::new();
        let mut stream = StreamingEncoder::new()
            .quality(85)
            .subsampling(Subsampling::S420)
            .start_rgb(width, height, &mut output)
            .unwrap();

        // Write scanlines in chunks
        let bytes_per_line = (width * 3) as usize;
        for chunk in rgb_data.chunks(bytes_per_line * 8) {
            stream.write_scanlines(chunk).unwrap();
        }

        stream.finish().unwrap();

        // Verify JPEG markers
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xD8); // SOI
        assert_eq!(output[output.len() - 2], 0xFF);
        assert_eq!(output[output.len() - 1], 0xD9); // EOI

        // Verify can be decoded
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&output));
        let decoded = decoder.decode().expect("Failed to decode streaming JPEG");

        let info = decoder.info().unwrap();
        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);
        assert_eq!(decoded.len(), (width * height * 3) as usize);
    }

    #[test]
    fn test_streaming_encoder_gray() {
        // Create a 16x16 grayscale image
        let width = 16u32;
        let height = 16u32;
        let mut gray_data = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                gray_data[i] = ((x * 16 + y * 16) % 256) as u8;
            }
        }

        // Encode using streaming encoder
        let streaming = StreamingEncoder::new().quality(85);
        let streaming_data = streaming.encode_gray(&gray_data, width, height).unwrap();

        // Verify JPEG markers
        assert_eq!(streaming_data[0], 0xFF);
        assert_eq!(streaming_data[1], 0xD8); // SOI

        // Verify can be decoded
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&streaming_data));
        let decoded = decoder
            .decode()
            .expect("Failed to decode grayscale streaming JPEG");

        let info = decoder.info().unwrap();
        assert_eq!(info.width, width as u16);
        assert_eq!(info.height, height as u16);
        // Grayscale decodes as grayscale
        assert_eq!(decoded.len(), (width * height) as usize);
    }
}
