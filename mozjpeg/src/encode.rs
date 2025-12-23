//! JPEG encoder pipeline.
//!
//! This module provides the main encoder interface for encoding images to JPEG.
//!
//! # Example
//!
//! ```ignore
//! use mozjpeg::{Encoder, ColorSpace};
//!
//! let encoder = Encoder::new()
//!     .quality(85)
//!     .progressive(true);
//!
//! let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
//! ```

use std::io::Write;

use crate::bitstream::BitWriter;
use crate::color;
use crate::consts::{
    QuantTableIdx, DCTSIZE, DCTSIZE2, JPEG_NATURAL_ORDER,
    DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES,
    AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES,
    AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES,
};
use crate::dct;
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::huffman::{DerivedTable, HuffTable};
use crate::marker::MarkerWriter;
use crate::progressive::generate_baseline_scan;
use crate::quant::{create_quant_tables, get_luminance_quant_table, get_chrominance_quant_table, quantize_block};
use crate::sample;
use crate::types::{
    ComponentInfo, Subsampling, TrellisConfig,
};

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
    /// Trellis quantization configuration
    trellis: TrellisConfig,
    /// Force baseline-compatible output
    force_baseline: bool,
    /// Optimize Huffman tables (requires 2-pass)
    optimize_huffman: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Create a new encoder with default settings.
    ///
    /// Default configuration:
    /// - Quality: 75
    /// - Progressive: false
    /// - Subsampling: 4:2:0
    /// - Quant tables: ImageMagick (mozjpeg default)
    /// - Trellis: disabled for baseline
    pub fn new() -> Self {
        Self {
            quality: 75,
            progressive: false,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            trellis: TrellisConfig::disabled(),
            force_baseline: true,
            optimize_huffman: false,
        }
    }

    /// Create encoder with max compression settings (mozjpeg defaults).
    pub fn max_compression() -> Self {
        Self {
            quality: 75,
            progressive: true,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            trellis: TrellisConfig::default(),
            force_baseline: false,
            optimize_huffman: true,
        }
    }

    /// Create encoder with fastest settings (libjpeg-turbo compatible).
    pub fn fastest() -> Self {
        Self {
            quality: 75,
            progressive: false,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::JpegAnnexK,
            trellis: TrellisConfig::disabled(),
            force_baseline: true,
            optimize_huffman: false,
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

    /// Encode RGB image data to JPEG.
    ///
    /// # Arguments
    /// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// JPEG-encoded data as a Vec<u8>
    pub fn encode_rgb(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let expected_len = width as usize * height as usize * 3;
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
        let num_pixels = width * height;
        let mut y_plane = vec![0u8; num_pixels];
        let mut cb_plane = vec![0u8; num_pixels];
        let mut cr_plane = vec![0u8; num_pixels];

        color::convert_rgb_to_ycbcr(
            rgb_data,
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            width,
            height,
        );

        // Step 2: Downsample chroma if needed
        let (luma_h, luma_v) = self.subsampling.luma_factors();
        let (chroma_width, chroma_height) = sample::subsampled_dimensions(
            width, height,
            luma_h as usize, luma_v as usize,
        );

        let mut cb_subsampled = vec![0u8; chroma_width * chroma_height];
        let mut cr_subsampled = vec![0u8; chroma_width * chroma_height];

        sample::downsample_plane(
            &cb_plane, width, height,
            luma_h as usize, luma_v as usize,
            &mut cb_subsampled,
        );
        sample::downsample_plane(
            &cr_plane, width, height,
            luma_h as usize, luma_v as usize,
            &mut cr_subsampled,
        );

        // Step 3: Expand planes to MCU-aligned dimensions
        let (mcu_width, mcu_height) = sample::mcu_aligned_dimensions(
            width, height,
            luma_h as usize, luma_v as usize,
        );
        let (mcu_chroma_w, mcu_chroma_h) = (mcu_width / luma_h as usize, mcu_height / luma_v as usize);

        let mut y_mcu = vec![0u8; mcu_width * mcu_height];
        let mut cb_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];
        let mut cr_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];

        sample::expand_to_mcu(&y_plane, width, height, &mut y_mcu, mcu_width, mcu_height);
        sample::expand_to_mcu(&cb_subsampled, chroma_width, chroma_height, &mut cb_mcu, mcu_chroma_w, mcu_chroma_h);
        sample::expand_to_mcu(&cr_subsampled, chroma_width, chroma_height, &mut cr_mcu, mcu_chroma_w, mcu_chroma_h);

        // Step 4: Create quantization tables
        let (luma_qtable, chroma_qtable) = create_quant_tables(
            self.quality,
            self.quant_table_idx,
            self.force_baseline,
        );

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

        // APP0 (JFIF)
        marker_writer.write_jfif_app0(1, 72, 72)?;

        // DQT (quantization tables in zigzag order)
        let luma_qtable_zz = natural_to_zigzag(&luma_qtable.values);
        let chroma_qtable_zz = natural_to_zigzag(&chroma_qtable.values);
        marker_writer.write_dqt(0, &luma_qtable_zz, false)?;
        marker_writer.write_dqt(1, &chroma_qtable_zz, false)?;

        // SOF
        marker_writer.write_sof(
            self.progressive,
            8,
            height as u16,
            width as u16,
            &components,
        )?;

        // DHT (Huffman tables)
        marker_writer.write_dht(0, false, &dc_luma_huff)?;
        marker_writer.write_dht(1, false, &dc_chroma_huff)?;
        marker_writer.write_dht(0, true, &ac_luma_huff)?;
        marker_writer.write_dht(1, true, &ac_chroma_huff)?;

        // SOS and entropy data
        let scans = generate_baseline_scan(3);
        let scan = &scans[0]; // Baseline has only one scan
        marker_writer.write_sos(scan, &components)?;

        // Encode MCU data
        let output = marker_writer.into_inner();
        let mut bit_writer = BitWriter::new(output);
        let mut entropy = EntropyEncoder::new(&mut bit_writer);

        self.encode_mcus(
            &y_mcu, mcu_width, mcu_height,
            &cb_mcu, &cr_mcu, mcu_chroma_w, mcu_chroma_h,
            &luma_qtable.values, &chroma_qtable.values,
            &dc_luma_derived, &dc_chroma_derived,
            &ac_luma_derived, &ac_chroma_derived,
            &mut entropy,
            luma_h, luma_v,
        )?;

        // Flush bits and get output back
        bit_writer.flush()?;
        let mut output = bit_writer.into_inner();

        // EOI
        output.write_all(&[0xFF, 0xD9])?;

        Ok(())
    }

    /// Encode all MCUs (Minimum Coded Units).
    #[allow(clippy::too_many_arguments)]
    fn encode_mcus<W: Write>(
        &self,
        y_plane: &[u8], y_width: usize, y_height: usize,
        cb_plane: &[u8], cr_plane: &[u8],
        chroma_width: usize, chroma_height: usize,
        luma_qtable: &[u16; DCTSIZE2],
        chroma_qtable: &[u16; DCTSIZE2],
        dc_luma: &DerivedTable,
        dc_chroma: &DerivedTable,
        ac_luma: &DerivedTable,
        ac_chroma: &DerivedTable,
        entropy: &mut EntropyEncoder<W>,
        h_samp: u8, v_samp: u8,
    ) -> Result<()> {
        let mcu_rows = y_height / (DCTSIZE * v_samp as usize);
        let mcu_cols = y_width / (DCTSIZE * h_samp as usize);

        let mut dct_block = [0i16; DCTSIZE2];
        let mut quant_block = [0i16; DCTSIZE2];

        for mcu_row in 0..mcu_rows {
            for mcu_col in 0..mcu_cols {
                // Encode Y blocks (may be multiple per MCU for subsampling)
                for v in 0..v_samp as usize {
                    for h in 0..h_samp as usize {
                        let block_row = mcu_row * v_samp as usize + v;
                        let block_col = mcu_col * h_samp as usize + h;

                        self.encode_block(
                            y_plane, y_width,
                            block_row, block_col,
                            luma_qtable,
                            dc_luma, ac_luma,
                            0, // Y component
                            entropy,
                            &mut dct_block,
                            &mut quant_block,
                        )?;
                    }
                }

                // Encode Cb block
                self.encode_block(
                    cb_plane, chroma_width,
                    mcu_row, mcu_col,
                    chroma_qtable,
                    dc_chroma, ac_chroma,
                    1, // Cb component
                    entropy,
                    &mut dct_block,
                    &mut quant_block,
                )?;

                // Encode Cr block
                self.encode_block(
                    cr_plane, chroma_width,
                    mcu_row, mcu_col,
                    chroma_qtable,
                    dc_chroma, ac_chroma,
                    2, // Cr component
                    entropy,
                    &mut dct_block,
                    &mut quant_block,
                )?;
            }
        }

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

        // Forward DCT (includes level shift)
        dct::forward_dct(&samples, dct_block);

        // Quantize (convert i16 to i32 for quantize_block)
        let mut dct_i32 = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            dct_i32[i] = dct_block[i] as i32;
        }
        quantize_block(&dct_i32, qtable, quant_block);

        // Entropy encode
        entropy.encode_block(quant_block, component, dc_table, ac_table)?;

        Ok(())
    }
}

// Helper functions

fn create_ycbcr_components(subsampling: Subsampling) -> Vec<ComponentInfo> {
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

fn natural_to_zigzag(natural: &[u16; DCTSIZE2]) -> [u16; DCTSIZE2] {
    let mut zigzag = [0u16; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        zigzag[i] = natural[JPEG_NATURAL_ORDER[i] as usize];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_defaults() {
        let enc = Encoder::new();
        assert_eq!(enc.quality, 75);
        assert!(!enc.progressive);
        assert_eq!(enc.subsampling, Subsampling::S420);
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
            rgb_data[i * 3] = 255;     // R
            rgb_data[i * 3 + 1] = 0;   // G
            rgb_data[i * 3 + 2] = 0;   // B
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
    fn test_encode_invalid_size() {
        let rgb_data = vec![0u8; 100]; // Wrong size
        let encoder = Encoder::new();
        let result = encoder.encode_rgb(&rgb_data, 16, 16);

        assert!(result.is_err());
    }
}
