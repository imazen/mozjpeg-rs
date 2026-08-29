//! Streaming JPEG encoder.
//!
//! This module provides [`StreamingEncoder`] and [`EncodingStream`] for
//! scanline-by-scanline encoding, which is memory-efficient for large images.

use std::io::Write;

use crate::consts::{DCTSIZE, DCTSIZE2, JPEG_NATURAL_ORDER, QuantTableIdx};
use crate::error::{Error, Result};
use crate::huffman::DerivedTable;
use crate::marker::MarkerWriter;
use crate::progressive::generate_baseline_scan;
use crate::quant::quantize_block;
use crate::simd::SimdOps;
use crate::types::{ComponentInfo, Limits, PixelDensity, QuantTable, Subsampling};

use super::{
    Encode, create_std_ac_chroma_table, create_std_ac_luma_table, create_std_dc_chroma_table,
    create_std_dc_luma_table, try_alloc_vec,
};

/// Streaming JPEG encoder configuration.
///
/// This encoder supports scanline-by-scanline encoding, which is memory-efficient
/// for large images. It does NOT support trellis quantization, progressive mode,
/// or Huffman optimization (these require buffering the entire image).
///
/// Use [`Encoder`](super::Encoder) for full-featured batch encoding with optimizations.
///
/// # Example
///
/// ```ignore
/// use mozjpeg_rs::Encoder;
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
    /// Optional independent chroma quality (1-100). `None` = reuse
    /// luma quality (historical behaviour).
    chroma_quality: Option<u8>,
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
    /// Resource limits (all-off by default)
    limits: Limits,
}

impl Default for StreamingEncoder {
    fn default() -> Self {
        Self::baseline_fastest()
    }
}

impl StreamingEncoder {
    /// Create a streaming encoder with fastest settings.
    ///
    /// This matches [`Preset::BaselineFastest`](crate::Preset::BaselineFastest) but for streaming.
    ///
    /// Streaming mode does NOT support any optimizations that require buffering
    /// the entire image:
    /// - No trellis quantization (requires global context)
    /// - No progressive mode (requires multiple passes)
    /// - No Huffman optimization (requires 2-pass)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use mozjpeg_rs::StreamingEncoder;
    ///
    /// let mut stream = StreamingEncoder::baseline_fastest()
    ///     .quality(85)
    ///     .start_rgb(1920, 1080, output_file)?;
    /// ```
    pub fn baseline_fastest() -> Self {
        Self {
            quality: 75,
            chroma_quality: None,
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
            limits: Limits::none(),
        }
    }

    /// Set quality level (1-100).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Set an independent chroma quality (1-100).
    ///
    /// `None` (default) = reuse [`Self::quality`] for the chroma
    /// quant table. `Some(cq)` = scale the chrominance base table
    /// by `cq`'s quality-scale factor. See [`crate::Encoder::chroma_quality`]
    /// for semantics.
    ///
    /// *Hidden from rustdoc while the surface is experimental.*
    #[doc(hidden)]
    pub fn chroma_quality(mut self, chroma_quality: Option<u8>) -> Self {
        self.chroma_quality = chroma_quality.map(|q| q.clamp(1, 100));
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

    /// Set resource limits for the stream.
    ///
    /// Same type and same semantics as
    /// [`Encoder::limits`](crate::Encoder::limits): every cap is off by default
    /// (zero = unlimited), and every enabled cap is checked in
    /// [`start_rgb`](Self::start_rgb) / [`start_gray`](Self::start_gray) —
    /// before the scanline buffer is allocated, before a single marker byte is
    /// written, and long before any color conversion or DCT work. The same
    /// typed errors come back as on the batch path, so a caller can handle
    /// both identically.
    ///
    /// # What each cap means here
    ///
    /// | Cap | On the streaming path |
    /// |---|---|
    /// | [`max_width`](Limits::max_width) / [`max_height`](Limits::max_height) | Identical to the batch path — [`Error::DimensionLimitExceeded`] |
    /// | [`max_pixel_count`](Limits::max_pixel_count) | Identical — [`Error::PixelCountExceeded`] |
    /// | [`max_icc_profile_bytes`](Limits::max_icc_profile_bytes) | Identical — [`Error::IccProfileTooLarge`] |
    /// | [`max_exif_bytes`](Limits::max_exif_bytes) | Identical — [`Error::ExifDataTooLarge`] |
    /// | [`max_marker_bytes`](Limits::max_marker_bytes) | Identical (sum of all custom APP payloads) — [`Error::MarkerDataTooLarge`] |
    /// | [`max_alloc_bytes`](Limits::max_alloc_bytes) | Applies, but against a **much smaller** estimate — see below |
    ///
    /// No cap is structurally inapplicable to streaming. `max_alloc_bytes` is
    /// the only one whose *magnitude* differs, because streaming is the mode
    /// that exists to avoid whole-image buffers: it holds one MCU row, not the
    /// image. The estimate checked here is
    ///
    /// ```text
    /// mcu_height * width * components   // the scanline buffer
    ///   + 3 * mcu_width * mcu_height * 2 // three i16 MCU planes (RGB only)
    /// ```
    ///
    /// where `mcu_width`/`mcu_height` are 8 or 16 depending on subsampling.
    /// It does NOT include the output, which streaming hands to your `W: Write`
    /// rather than accumulating. A limit sized for
    /// [`Encoder::estimate_resources`](crate::Encoder::estimate_resources) will
    /// therefore essentially never trip here — that is the point of the mode,
    /// not an omission. Size it against the formula above if you want it to
    /// bind.
    ///
    /// # Example
    ///
    /// ```
    /// use mozjpeg_rs::{Encoder, Limits};
    ///
    /// let limits = Limits::default()
    ///     .max_width(20_000)
    ///     .max_pixel_count(120_000_000)
    ///     .max_exif_bytes(64 * 1024);
    ///
    /// let mut out = Vec::new();
    /// let mut stream = Encoder::streaming()
    ///     .quality(85)
    ///     .limits(limits)
    ///     .start_rgb(64, 64, &mut out)?;
    /// stream.write_scanlines(&vec![128u8; 64 * 64 * 3])?;
    /// stream.finish()?;
    /// # Ok::<(), mozjpeg_rs::Error>(())
    /// ```
    ///
    /// Note that [`Encoder::streaming()`](crate::Encoder::streaming) is an
    /// associated function with no receiver, so it cannot carry limits over
    /// from an `Encoder` — set them on the returned `StreamingEncoder`.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Check all resource limits before any buffer is allocated or marker written.
    ///
    /// Mirrors [`Encoder::check_limits`](crate::Encoder) field for field; see
    /// [`limits`](Self::limits) for how `max_alloc_bytes` is estimated on this
    /// path.
    fn check_limits(&self, width: u32, height: u32, num_components: u8) -> Result<()> {
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

        // Check allocation limit against the streaming working set, which is one
        // MCU row plus the per-MCU i16 planes — not a whole-image buffer.
        if limits.max_alloc_bytes > 0 {
            let estimated = self.estimate_streaming_bytes(width, num_components);
            if estimated > limits.max_alloc_bytes {
                return Err(Error::AllocationLimitExceeded {
                    estimated,
                    limit: limits.max_alloc_bytes,
                });
            }
        }

        // Check ICC profile size limit
        if limits.max_icc_profile_bytes > 0
            && let Some(ref icc) = self.icc_profile
            && icc.len() > limits.max_icc_profile_bytes
        {
            return Err(Error::IccProfileTooLarge {
                size: icc.len(),
                limit: limits.max_icc_profile_bytes,
            });
        }

        // Check EXIF payload size limit
        if limits.max_exif_bytes > 0
            && let Some(ref exif) = self.exif_data
            && exif.len() > limits.max_exif_bytes
        {
            return Err(Error::ExifDataTooLarge {
                size: exif.len(),
                limit: limits.max_exif_bytes,
            });
        }

        // Check combined custom APP marker size limit
        if limits.max_marker_bytes > 0 {
            let total: usize = self.custom_markers.iter().map(|(_, data)| data.len()).sum();
            if total > limits.max_marker_bytes {
                return Err(Error::MarkerDataTooLarge {
                    size: total,
                    limit: limits.max_marker_bytes,
                });
            }
        }

        Ok(())
    }

    /// Peak heap the streaming path holds, in bytes, for a given width.
    ///
    /// Independent of image height by construction — that is what makes this
    /// mode streaming. See [`limits`](Self::limits) for the documented formula.
    fn estimate_streaming_bytes(&self, width: u32, num_components: u8) -> usize {
        let (mcu_width, mcu_height) = Self::mcu_dimensions(self.subsampling, num_components);

        // The scanline buffer: one full MCU row of input pixels.
        let scanline = (mcu_height as usize)
            .saturating_mul(width as usize)
            .saturating_mul(num_components as usize);

        // Color MCU rows additionally build three i16 planes per MCU.
        let planes = if num_components == 1 {
            0
        } else {
            3usize
                .saturating_mul(mcu_width as usize)
                .saturating_mul(mcu_height as usize)
                .saturating_mul(core::mem::size_of::<i16>())
        };

        scanline.saturating_add(planes)
    }

    /// MCU dimensions in pixels for a component count and subsampling mode.
    ///
    /// Single source of truth shared by [`Self::estimate_streaming_bytes`] and
    /// [`EncodingStream::new`], so the limit estimate can never drift from the
    /// buffer it is estimating.
    fn mcu_dimensions(subsampling: Subsampling, num_components: u8) -> (u32, u32) {
        if num_components == 1 {
            (DCTSIZE as u32, DCTSIZE as u32)
        } else {
            match subsampling {
                Subsampling::S444 | Subsampling::Gray => (DCTSIZE as u32, DCTSIZE as u32),
                Subsampling::S422 => (DCTSIZE as u32 * 2, DCTSIZE as u32),
                Subsampling::S420 => (DCTSIZE as u32 * 2, DCTSIZE as u32 * 2),
                Subsampling::S440 => (DCTSIZE as u32, DCTSIZE as u32 * 2),
            }
        }
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
    ///
    /// # Errors
    /// Any limit set via [`limits`](Self::limits) is checked here, before the
    /// scanline buffer is allocated and before any marker is written.
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
    ///
    /// # Errors
    /// Any limit set via [`limits`](Self::limits) is checked here, before the
    /// scanline buffer is allocated and before any marker is written.
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

        // Check all resource limits. Ordered exactly as on the batch path:
        // after the zero-dimension check, before any allocation, marker write,
        // or pixel work.
        config.check_limits(width, height, num_components)?;

        // Determine MCU dimensions based on subsampling
        let (mcu_width, mcu_height) =
            StreamingEncoder::mcu_dimensions(config.subsampling, num_components);

        let mcus_per_row = (width + mcu_width - 1) / mcu_width;

        // Create quantization tables. Optional chroma_quality lets
        // callers scale the chroma base table with a different quality
        // than luma; `None` is bit-identical to pre-PR behaviour.
        let (luma_qtable, chroma_qtable) = crate::quant::create_quant_tables_split(
            config.quality,
            config.chroma_quality,
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

// ============================================================================
// Tests — streaming resource limits (issue #5 follow-up)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Encoder;
    use crate::types::Preset;

    /// 64x64 RGB solid gray, the shared fixture for these tests.
    fn rgb_pixels() -> Vec<u8> {
        vec![128u8; 64 * 64 * 3]
    }

    fn gray_pixels() -> Vec<u8> {
        vec![128u8; 64 * 64]
    }

    /// Run a full streaming RGB encode at 64x64 and return the result.
    fn stream_rgb(enc: StreamingEncoder) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut stream = enc.start_rgb(64, 64, &mut out)?;
            stream.write_scanlines(&rgb_pixels())?;
            stream.finish()?;
        }
        Ok(out)
    }

    /// Run a full streaming grayscale encode at 64x64 and return the result.
    fn stream_gray(enc: StreamingEncoder) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut stream = enc.start_gray(64, 64, &mut out)?;
            stream.write_scanlines(&gray_pixels())?;
            stream.finish()?;
        }
        Ok(out)
    }

    fn base() -> StreamingEncoder {
        StreamingEncoder::baseline_fastest().quality(85)
    }

    // ------------------------------------------------------------------
    // Off by default: existing streaming callers see byte-identical output
    // ------------------------------------------------------------------

    #[test]
    fn test_streaming_limits_off_by_default() {
        assert_eq!(base().limits, Limits::none());
        assert!(!base().limits.has_limits());
    }

    #[test]
    fn test_streaming_limits_off_by_default_byte_identical() {
        let exif = vec![7u8; 5000];
        let marker = vec![9u8; 5000];
        let icc = vec![3u8; 5000];

        let build = |limits: Option<Limits>| {
            let enc = base()
                .exif_data(exif.clone())
                .icc_profile(icc.clone())
                .add_marker(4, marker.clone());
            let enc = match limits {
                Some(l) => enc.limits(l),
                // No `.limits()` call at all — the pre-change call shape.
                None => enc,
            };
            stream_rgb(enc).expect("encode should succeed")
        };

        let untouched = build(None);
        // An explicit all-zero Limits must behave exactly like never calling it.
        assert_eq!(untouched, build(Some(Limits::default())));
        assert_eq!(untouched, build(Some(Limits::none())));
        // Generous caps must not perturb a single output byte either.
        assert_eq!(
            untouched,
            build(Some(
                Limits::default()
                    .max_width(100_000)
                    .max_height(100_000)
                    .max_pixel_count(1_000_000_000)
                    .max_alloc_bytes(1_000_000_000)
                    .max_icc_profile_bytes(1_000_000)
                    .max_exif_bytes(1_000_000)
                    .max_marker_bytes(1_000_000)
            ))
        );
        // Non-vacuous: the metadata really is in the stream being compared.
        assert!(untouched.windows(16).any(|w| w == [7u8; 16]));
        assert!(untouched.windows(16).any(|w| w == [9u8; 16]));
    }

    // ------------------------------------------------------------------
    // Boundary: limit-1 / limit / limit+1 for every enforced cap
    // ------------------------------------------------------------------

    #[test]
    fn test_streaming_width_limit_boundary() {
        // Cap width at 64: 63 and 64 pass, 65 is rejected.
        for (w, should_pass) in [(63u32, true), (64, true), (65, false)] {
            let mut out = Vec::new();
            let result = base()
                .limits(Limits::default().max_width(64))
                .start_rgb(w, 8, &mut out);
            match (result, should_pass) {
                (Ok(_), true) => {}
                (
                    Err(Error::DimensionLimitExceeded {
                        width, max_width, ..
                    }),
                    false,
                ) => {
                    assert_eq!((width, max_width), (65, 64));
                }
                (r, _) => panic!("width {w} against a 64 cap: unexpected {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_height_limit_boundary() {
        for (h, should_pass) in [(63u32, true), (64, true), (65, false)] {
            let mut out = Vec::new();
            let result = base()
                .limits(Limits::default().max_height(64))
                .start_rgb(8, h, &mut out);
            match (result, should_pass) {
                (Ok(_), true) => {}
                (
                    Err(Error::DimensionLimitExceeded {
                        height, max_height, ..
                    }),
                    false,
                ) => {
                    assert_eq!((height, max_height), (65, 64));
                }
                (r, _) => panic!("height {h} against a 64 cap: unexpected {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_pixel_count_limit_boundary() {
        // 64x64 = 4096 pixels. Vary the cap around it.
        for (cap, should_pass) in [(4095u64, false), (4096, true), (4097, true)] {
            let result = stream_rgb(base().limits(Limits::default().max_pixel_count(cap)));
            match (result, should_pass) {
                (Ok(_), true) => {}
                (Err(Error::PixelCountExceeded { pixel_count, limit }), false) => {
                    assert_eq!((pixel_count, limit), (4096, 4095));
                }
                (r, _) => panic!("cap {cap} on 4096 pixels: unexpected {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_alloc_estimate_matches_documented_formula() {
        // The formula in StreamingEncoder::limits docs, pinned so the estimate
        // cannot silently drift from the buffer it estimates.
        //
        // RGB 4:2:0 at width 64 -> MCU 16x16:
        //   scanline = 16 * 64 * 3          = 3072
        //   planes   = 3 * 16 * 16 * 2      = 1536
        assert_eq!(base().estimate_streaming_bytes(64, 3), 3072 + 1536);
        // Grayscale at width 64 -> MCU 8x8, no i16 planes:
        //   scanline = 8 * 64 * 1           = 512
        assert_eq!(base().estimate_streaming_bytes(64, 1), 512);
        // 4:4:4 RGB at width 64 -> MCU 8x8:
        //   scanline = 8 * 64 * 3           = 1536
        //   planes   = 3 * 8 * 8 * 2        = 384
        assert_eq!(
            base()
                .subsampling(Subsampling::S444)
                .estimate_streaming_bytes(64, 3),
            1536 + 384
        );
        // Height-independent by construction — that is what makes it streaming.
        assert_eq!(
            base().estimate_streaming_bytes(64, 3),
            base().estimate_streaming_bytes(64, 3)
        );
    }

    #[test]
    fn test_streaming_alloc_limit_boundary() {
        let estimate = base().estimate_streaming_bytes(64, 3);
        assert_eq!(estimate, 4608);
        for (cap, should_pass) in [
            (estimate - 1, false),
            (estimate, true),
            (estimate + 1, true),
        ] {
            let result = stream_rgb(base().limits(Limits::default().max_alloc_bytes(cap)));
            match (result, should_pass) {
                (Ok(_), true) => {}
                (Err(Error::AllocationLimitExceeded { estimated, limit }), false) => {
                    assert_eq!((estimated, limit), (estimate, estimate - 1));
                }
                (r, _) => panic!("cap {cap} on a {estimate}-byte estimate: {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_icc_limit_boundary() {
        for (len, should_pass) in [(999usize, true), (1000, true), (1001, false)] {
            let result = stream_rgb(
                base()
                    .limits(Limits::default().max_icc_profile_bytes(1000))
                    .icc_profile(vec![0u8; len]),
            );
            match (result, should_pass) {
                (Ok(_), true) => {}
                (Err(Error::IccProfileTooLarge { size, limit }), false) => {
                    assert_eq!((size, limit), (1001, 1000));
                }
                (r, _) => panic!("ICC of {len} bytes against a 1000 cap: {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_exif_limit_boundary() {
        for (len, should_pass) in [(999usize, true), (1000, true), (1001, false)] {
            let result = stream_rgb(
                base()
                    .limits(Limits::default().max_exif_bytes(1000))
                    .exif_data(vec![0u8; len]),
            );
            match (result, should_pass) {
                (Ok(_), true) => {}
                (Err(Error::ExifDataTooLarge { size, limit }), false) => {
                    assert_eq!((size, limit), (1001, 1000));
                }
                (r, _) => panic!("EXIF of {len} bytes against a 1000 cap: {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_marker_limit_boundary() {
        for (len, should_pass) in [(999usize, true), (1000, true), (1001, false)] {
            let result = stream_rgb(
                base()
                    .limits(Limits::default().max_marker_bytes(1000))
                    .add_marker(4, vec![0u8; len]),
            );
            match (result, should_pass) {
                (Ok(_), true) => {}
                (Err(Error::MarkerDataTooLarge { size, limit }), false) => {
                    assert_eq!((size, limit), (1001, 1000));
                }
                (r, _) => panic!("marker of {len} bytes against a 1000 cap: {:?}", r.err()),
            }
        }
    }

    #[test]
    fn test_streaming_marker_limit_sums_across_markers() {
        let result = stream_rgb(
            base()
                .limits(Limits::default().max_marker_bytes(1000))
                .add_marker(4, vec![0u8; 400])
                .add_marker(5, vec![0u8; 400])
                .add_marker(6, vec![0u8; 400]),
        );
        assert!(
            matches!(
                result,
                Err(Error::MarkerDataTooLarge {
                    size: 1200,
                    limit: 1000
                })
            ),
            "combined 1200 bytes should trip a 1000 cap, got {:?}",
            result.err()
        );
    }

    // ------------------------------------------------------------------
    // Enforcement happens before allocation / markers / pixel work
    // ------------------------------------------------------------------

    #[test]
    fn test_streaming_limits_reject_before_any_output() {
        // start_rgb writes SOI + APP0 + DQT + SOF + DHT + SOS before returning.
        // If the limit check ran late, the writer would already hold bytes.
        for limits in [
            Limits::default().max_width(8),
            Limits::default().max_pixel_count(16),
            Limits::default().max_alloc_bytes(16),
            Limits::default().max_exif_bytes(1),
            Limits::default().max_marker_bytes(1),
            Limits::default().max_icc_profile_bytes(1),
        ] {
            let mut out = Vec::new();
            let result = base()
                .limits(limits)
                .exif_data(vec![0u8; 100])
                .icc_profile(vec![0u8; 100])
                .add_marker(4, vec![0u8; 100])
                .start_rgb(64, 64, &mut out);
            assert!(result.is_err(), "{limits:?} should have been rejected");
            drop(result);
            assert!(
                out.is_empty(),
                "{limits:?} was rejected but {} bytes had already been written",
                out.len()
            );
        }
    }

    #[test]
    fn test_streaming_limits_apply_to_grayscale() {
        assert!(matches!(
            stream_gray(
                base()
                    .limits(Limits::default().max_exif_bytes(50))
                    .exif_data(vec![0u8; 100])
            ),
            Err(Error::ExifDataTooLarge { .. })
        ));
        assert!(matches!(
            stream_gray(
                base()
                    .limits(Limits::default().max_marker_bytes(50))
                    .add_marker(4, vec![0u8; 100])
            ),
            Err(Error::MarkerDataTooLarge { .. })
        ));
        assert!(matches!(
            stream_gray(base().limits(Limits::default().max_pixel_count(16))),
            Err(Error::PixelCountExceeded { .. })
        ));
        // Grayscale allocates no i16 planes, so its estimate is the bare
        // scanline buffer.
        assert!(matches!(
            stream_gray(base().limits(Limits::default().max_alloc_bytes(511))),
            Err(Error::AllocationLimitExceeded {
                estimated: 512,
                limit: 511
            })
        ));
    }

    #[test]
    fn test_streaming_limits_apply_via_encode_trait() {
        // The `Encode` impl routes through start_rgb/start_gray, so limits
        // must bind there too.
        use crate::encode::Encode;
        let enc = base()
            .limits(Limits::default().max_exif_bytes(50))
            .exif_data(vec![0u8; 100]);
        assert!(matches!(
            enc.encode_rgb(&rgb_pixels(), 64, 64),
            Err(Error::ExifDataTooLarge { .. })
        ));
        assert!(matches!(
            enc.encode_gray(&gray_pixels(), 64, 64),
            Err(Error::MarkerDataTooLarge { .. } | Error::ExifDataTooLarge { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Parity with the batch Encoder
    // ------------------------------------------------------------------

    #[test]
    fn test_streaming_and_encoder_reject_identically() {
        // Same over-limit input, both paths, same error variant. Each case is
        // (limits, exif, icc, marker).
        let cases: [(Limits, usize, usize, usize); 5] = [
            (Limits::default().max_width(32), 0, 0, 0),
            (Limits::default().max_pixel_count(16), 0, 0, 0),
            (Limits::default().max_exif_bytes(10), 100, 0, 0),
            (Limits::default().max_icc_profile_bytes(10), 0, 100, 0),
            (Limits::default().max_marker_bytes(10), 0, 0, 100),
        ];

        for (limits, exif, icc, marker) in cases {
            let mut batch = Encoder::new(Preset::BaselineFastest).limits(limits);
            let mut streaming = base().limits(limits);
            if exif > 0 {
                batch = batch.exif_data(vec![0u8; exif]);
                streaming = streaming.exif_data(vec![0u8; exif]);
            }
            if icc > 0 {
                batch = batch.icc_profile(vec![0u8; icc]);
                streaming = streaming.icc_profile(vec![0u8; icc]);
            }
            if marker > 0 {
                batch = batch.add_marker(4, vec![0u8; marker]);
                streaming = streaming.add_marker(4, vec![0u8; marker]);
            }

            let batch_err = batch
                .encode_rgb(&rgb_pixels(), 64, 64)
                .expect_err("batch path should reject");
            let streaming_err = stream_rgb(streaming).expect_err("streaming path should reject");

            assert_eq!(
                std::mem::discriminant(&batch_err),
                std::mem::discriminant(&streaming_err),
                "{limits:?}: batch gave {batch_err:?}, streaming gave {streaming_err:?}"
            );
            // For every cap except max_alloc_bytes the payloads match too,
            // because both paths compare the same numbers.
            assert_eq!(batch_err, streaming_err, "{limits:?}: payloads differ");
        }
    }

    #[test]
    fn test_streaming_and_encoder_alloc_limit_agree_on_variant_not_magnitude() {
        // max_alloc_bytes is the one cap whose estimate legitimately differs:
        // the batch path sizes a whole-image working set, streaming sizes one
        // MCU row. A 1-byte cap trips both, and the variant matches.
        let limits = Limits::default().max_alloc_bytes(1);

        let batch_err = Encoder::new(Preset::BaselineFastest)
            .limits(limits)
            .encode_rgb(&rgb_pixels(), 64, 64)
            .expect_err("batch path should reject");
        let streaming_err = stream_rgb(base().limits(limits)).expect_err("streaming should reject");

        let (
            Error::AllocationLimitExceeded {
                estimated: batch_est,
                ..
            },
            Error::AllocationLimitExceeded {
                estimated: stream_est,
                ..
            },
        ) = (&batch_err, &streaming_err)
        else {
            panic!(
                "expected AllocationLimitExceeded from both, got {batch_err:?} / {streaming_err:?}"
            );
        };

        // The whole point of streaming: its working set is far smaller.
        assert!(
            *stream_est < *batch_est,
            "streaming estimate {stream_est} should be below the batch estimate {batch_est}"
        );
        assert_eq!(*stream_est, 4608);
    }
}
