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
use crate::entropy::{EntropyEncoder, ProgressiveEncoder, ProgressiveSymbolCounter, SymbolCounter};
use crate::huffman::FrequencyCounter;
use crate::error::{Error, Result};
use crate::huffman::{DerivedTable, HuffTable};
use crate::marker::MarkerWriter;
use crate::progressive::{generate_baseline_scan, generate_simple_progressive_scans};
use crate::quant::{create_quant_tables, quantize_block};
use crate::trellis::{trellis_quantize_block, dc_trellis_optimize_indexed};
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
    /// Create a new encoder with default settings (mozjpeg defaults).
    ///
    /// Default configuration:
    /// - Quality: 75
    /// - Progressive: false
    /// - Subsampling: 4:2:0
    /// - Quant tables: ImageMagick (mozjpeg default)
    /// - Trellis: enabled (core mozjpeg optimization)
    /// - Huffman optimization: enabled (2-pass for optimal tables)
    pub fn new() -> Self {
        Self {
            quality: 75,
            progressive: false,
            subsampling: Subsampling::S420,
            quant_table_idx: QuantTableIdx::ImageMagick,
            trellis: TrellisConfig::default(),
            force_baseline: false,
            optimize_huffman: true,
        }
    }

    /// Create encoder with max compression settings (mozjpeg defaults).
    ///
    /// Enables progressive mode, trellis quantization, and Huffman optimization.
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

        // DHT (Huffman tables) - written here for non-optimized modes,
        // or later after frequency counting for optimized modes
        if !self.optimize_huffman {
            marker_writer.write_dht(0, false, &dc_luma_huff)?;
            marker_writer.write_dht(1, false, &dc_chroma_huff)?;
            marker_writer.write_dht(0, true, &ac_luma_huff)?;
            marker_writer.write_dht(1, true, &ac_chroma_huff)?;
        }

        if self.progressive {
            // Progressive mode: Store all blocks, then encode multiple scans
            let mcu_rows = mcu_height / (DCTSIZE * luma_v as usize);
            let mcu_cols = mcu_width / (DCTSIZE * luma_h as usize);
            let num_y_blocks = mcu_rows * mcu_cols * luma_h as usize * luma_v as usize;
            let num_chroma_blocks = mcu_rows * mcu_cols;

            // Collect all quantized blocks
            let mut y_blocks = vec![[0i16; DCTSIZE2]; num_y_blocks];
            let mut cb_blocks = vec![[0i16; DCTSIZE2]; num_chroma_blocks];
            let mut cr_blocks = vec![[0i16; DCTSIZE2]; num_chroma_blocks];

            // Optionally collect raw DCT for DC trellis
            let dc_trellis_enabled = self.trellis.enabled && self.trellis.dc_enabled;
            let mut y_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_y_blocks])
            } else {
                None
            };
            let mut cb_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_chroma_blocks])
            } else {
                None
            };
            let mut cr_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_chroma_blocks])
            } else {
                None
            };

            self.collect_blocks(
                &y_mcu, mcu_width, mcu_height,
                &cb_mcu, &cr_mcu, mcu_chroma_w, mcu_chroma_h,
                &luma_qtable.values, &chroma_qtable.values,
                &ac_luma_derived, &ac_chroma_derived,
                &mut y_blocks, &mut cb_blocks, &mut cr_blocks,
                y_raw_dct.as_deref_mut(), cb_raw_dct.as_deref_mut(), cr_raw_dct.as_deref_mut(),
                luma_h, luma_v,
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
                        y_raw, &mut y_blocks,
                        luma_qtable.values[0], &dc_luma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        y_block_rows, y_block_cols, mcu_cols, h, v,
                    );
                }
                // Chroma has 1x1 per MCU, so MCU order = row order
                if let Some(ref cb_raw) = cb_raw_dct {
                    run_dc_trellis_by_row(
                        cb_raw, &mut cb_blocks,
                        chroma_qtable.values[0], &dc_chroma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        mcu_rows, mcu_cols, mcu_cols, 1, 1,
                    );
                }
                if let Some(ref cr_raw) = cr_raw_dct {
                    run_dc_trellis_by_row(
                        cr_raw, &mut cr_blocks,
                        chroma_qtable.values[0], &dc_chroma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        mcu_rows, mcu_cols, mcu_cols, 1, 1,
                    );
                }
            }

            // Generate progressive scan script
            let scans = generate_simple_progressive_scans(3);

            // Count symbol frequencies for optimized Huffman tables
            let (opt_dc_luma, opt_dc_chroma, opt_ac_luma, opt_ac_chroma) = if self.optimize_huffman {
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
                            &y_blocks, &cb_blocks, &cr_blocks,
                            mcu_rows, mcu_cols,
                            luma_h, luma_v,
                            &mut dc_luma_freq, &mut dc_chroma_freq,
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
                        let ac_freq = if comp_idx == 0 { &mut ac_luma_freq } else { &mut ac_chroma_freq };
                        self.count_ac_scan_symbols(
                            scan, blocks,
                            mcu_rows, mcu_cols,
                            luma_h, luma_v, comp_idx,
                            ac_freq,
                        );
                    }
                }

                // Generate optimized Huffman tables
                let opt_dc_luma_huff = dc_luma_freq.generate_table()?;
                let opt_dc_chroma_huff = dc_chroma_freq.generate_table()?;
                let opt_ac_luma_huff = ac_luma_freq.generate_table()?;
                let opt_ac_chroma_huff = ac_chroma_freq.generate_table()?;

                // Write DHT with optimized tables
                marker_writer.write_dht(0, false, &opt_dc_luma_huff)?;
                marker_writer.write_dht(1, false, &opt_dc_chroma_huff)?;
                marker_writer.write_dht(0, true, &opt_ac_luma_huff)?;
                marker_writer.write_dht(1, true, &opt_ac_chroma_huff)?;

                // Create derived tables for encoding
                (
                    DerivedTable::from_huff_table(&opt_dc_luma_huff, true)?,
                    DerivedTable::from_huff_table(&opt_dc_chroma_huff, true)?,
                    DerivedTable::from_huff_table(&opt_ac_luma_huff, false)?,
                    DerivedTable::from_huff_table(&opt_ac_chroma_huff, false)?,
                )
            } else {
                // Use standard tables (already written)
                (dc_luma_derived.clone(), dc_chroma_derived.clone(),
                 ac_luma_derived.clone(), ac_chroma_derived.clone())
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
                    &y_blocks, &cb_blocks, &cr_blocks,
                    mcu_rows, mcu_cols,
                    luma_h, luma_v,
                    &opt_dc_luma, &opt_dc_chroma,
                    &opt_ac_luma, &opt_ac_chroma,
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
            let num_y_blocks = mcu_rows * mcu_cols * luma_h as usize * luma_v as usize;
            let num_chroma_blocks = mcu_rows * mcu_cols;

            let mut y_blocks = vec![[0i16; DCTSIZE2]; num_y_blocks];
            let mut cb_blocks = vec![[0i16; DCTSIZE2]; num_chroma_blocks];
            let mut cr_blocks = vec![[0i16; DCTSIZE2]; num_chroma_blocks];

            // Optionally collect raw DCT for DC trellis
            let dc_trellis_enabled = self.trellis.enabled && self.trellis.dc_enabled;
            let mut y_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_y_blocks])
            } else {
                None
            };
            let mut cb_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_chroma_blocks])
            } else {
                None
            };
            let mut cr_raw_dct = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; num_chroma_blocks])
            } else {
                None
            };

            self.collect_blocks(
                &y_mcu, mcu_width, mcu_height,
                &cb_mcu, &cr_mcu, mcu_chroma_w, mcu_chroma_h,
                &luma_qtable.values, &chroma_qtable.values,
                &ac_luma_derived, &ac_chroma_derived,
                &mut y_blocks, &mut cb_blocks, &mut cr_blocks,
                y_raw_dct.as_deref_mut(), cb_raw_dct.as_deref_mut(), cr_raw_dct.as_deref_mut(),
                luma_h, luma_v,
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
                        y_raw, &mut y_blocks,
                        luma_qtable.values[0], &dc_luma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        y_block_rows, y_block_cols, mcu_cols, h, v,
                    );
                }
                // Chroma has 1x1 per MCU, so MCU order = row order
                if let Some(ref cb_raw) = cb_raw_dct {
                    run_dc_trellis_by_row(
                        cb_raw, &mut cb_blocks,
                        chroma_qtable.values[0], &dc_chroma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        mcu_rows, mcu_cols, mcu_cols, 1, 1,
                    );
                }
                if let Some(ref cr_raw) = cr_raw_dct {
                    run_dc_trellis_by_row(
                        cr_raw, &mut cr_blocks,
                        chroma_qtable.values[0], &dc_chroma_derived,
                        self.trellis.lambda_log_scale1, self.trellis.lambda_log_scale2,
                        mcu_rows, mcu_cols, mcu_cols, 1, 1,
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
                        counter.count_block(&y_blocks[y_idx], 0, &mut dc_luma_freq, &mut ac_luma_freq);
                        y_idx += 1;
                    }
                    // Cb block
                    counter.count_block(&cb_blocks[c_idx], 1, &mut dc_chroma_freq, &mut ac_chroma_freq);
                    // Cr block
                    counter.count_block(&cr_blocks[c_idx], 2, &mut dc_chroma_freq, &mut ac_chroma_freq);
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

            // Write DHT with optimized tables
            marker_writer.write_dht(0, false, &opt_dc_luma_huff)?;
            marker_writer.write_dht(1, false, &opt_dc_chroma_huff)?;
            marker_writer.write_dht(0, true, &opt_ac_luma_huff)?;
            marker_writer.write_dht(1, true, &opt_ac_chroma_huff)?;

            // Write SOS and encode
            let scans = generate_baseline_scan(3);
            let scan = &scans[0];
            marker_writer.write_sos(scan, &components)?;

            let output = marker_writer.into_inner();
            let mut bit_writer = BitWriter::new(output);
            let mut entropy = EntropyEncoder::new(&mut bit_writer);

            // Encode from stored blocks
            y_idx = 0;
            c_idx = 0;
            for _mcu_row in 0..mcu_rows {
                for _mcu_col in 0..mcu_cols {
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
        }

        Ok(())
    }

    /// Encode all MCUs (Minimum Coded Units).
    #[allow(clippy::too_many_arguments)]
    fn encode_mcus<W: Write>(
        &self,
        y_plane: &[u8], y_width: usize, y_height: usize,
        cb_plane: &[u8], cr_plane: &[u8],
        chroma_width: usize, _chroma_height: usize,
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
        // Note: DCT output is scaled by factor of 8 (sqrt(8) per dimension)
        dct::forward_dct(&samples, dct_block);

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
            for i in 0..DCTSIZE2 {
                dct_i32[i] = (dct_i32[i] + 4) >> 3;
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
        y_plane: &[u8], y_width: usize, y_height: usize,
        cb_plane: &[u8], cr_plane: &[u8],
        chroma_width: usize, _chroma_height: usize,
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
        h_samp: u8, v_samp: u8,
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
                            y_plane, y_width,
                            block_row, block_col,
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
                    cb_plane, chroma_width,
                    mcu_row, mcu_col,
                    chroma_qtable,
                    ac_chroma,
                    &mut cb_blocks[c_idx],
                    &mut dct_block,
                    raw_dct_out,
                )?;

                // Collect Cr block
                let raw_dct_out = cr_raw_dct.as_mut().map(|arr| &mut arr[c_idx][..]);
                self.process_block_to_storage_with_raw(
                    cr_plane, chroma_width,
                    mcu_row, mcu_col,
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

        // Forward DCT (includes level shift)
        // Note: DCT output is scaled by factor of 8 (sqrt(8) per dimension)
        dct::forward_dct(&samples, dct_block);

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
            for i in 0..DCTSIZE2 {
                dct_i32[i] = (dct_i32[i] + 4) >> 3;
            }
            quantize_block(&dct_i32, qtable, out_block);
        }

        Ok(())
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
        h_samp: u8, v_samp: u8,
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
                scan, y_blocks, cb_blocks, cr_blocks,
                mcu_rows, mcu_cols, h_samp, v_samp,
                dc_luma, dc_chroma,
                is_refinement,
                encoder,
            )?;
        } else {
            // AC scan - single component only
            let comp_idx = scan.component_index[0] as usize;
            let blocks = match comp_idx {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => return Err(Error::InvalidComponentIndex(comp_idx)),
            };
            let ac_table = if comp_idx == 0 { ac_luma } else { ac_chroma };

            self.encode_ac_scan(
                scan, blocks,
                mcu_rows, mcu_cols, h_samp, v_samp, comp_idx,
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
        h_samp: u8, v_samp: u8,
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
    #[allow(clippy::too_many_arguments)]
    fn encode_ac_scan<W: Write>(
        &self,
        scan: &crate::types::ScanInfo,
        blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8, v_samp: u8,
        comp_idx: usize,
        ac_table: &DerivedTable,
        is_refinement: bool,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        // For Y component, blocks are interleaved per MCU
        // For chroma, one block per MCU
        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        let mut block_idx = 0;

        for _mcu_row in 0..mcu_rows {
            for _mcu_col in 0..mcu_cols {
                for _ in 0..blocks_per_mcu {
                    if is_refinement {
                        encoder.encode_ac_refine(
                            &blocks[block_idx],
                            scan.ss, scan.se, scan.al,
                            ac_table,
                        )?;
                    } else {
                        encoder.encode_ac_first(
                            &blocks[block_idx],
                            scan.ss, scan.se, scan.al,
                            ac_table,
                        )?;
                    }
                    block_idx += 1;
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
        h_samp: u8, v_samp: u8,
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
    #[allow(clippy::too_many_arguments)]
    fn count_ac_scan_symbols(
        &self,
        scan: &crate::types::ScanInfo,
        blocks: &[[i16; DCTSIZE2]],
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8, v_samp: u8,
        comp_idx: usize,
        ac_freq: &mut FrequencyCounter,
    ) {
        let blocks_per_mcu = if comp_idx == 0 {
            (h_samp * v_samp) as usize
        } else {
            1
        };

        let mut block_idx = 0;
        let mut counter = ProgressiveSymbolCounter::new();

        for _mcu_row in 0..mcu_rows {
            for _mcu_col in 0..mcu_cols {
                for _ in 0..blocks_per_mcu {
                    counter.count_ac_first(&blocks[block_idx], scan.ss, scan.se, scan.al, ac_freq);
                    block_idx += 1;
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
    for i in 0..scan.comps_in_scan as usize {
        let comp_idx = scan.component_index[i] as usize;
        let comp = &components[comp_idx];
        output.write_all(&[
            comp.component_id,
            (comp.dc_tbl_no << 4) | comp.ac_tbl_no,
        ])?;
    }

    // Spectral selection start (Ss), end (Se), successive approximation (Ah, Al)
    output.write_all(&[
        scan.ss,
        scan.se,
        (scan.ah << 4) | scan.al,
    ])?;

    Ok(())
}

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
        decoder.decode().expect("Failed to decode non-optimized JPEG");

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
            let encoder = Encoder::new()
                .quality(95)
                .subsampling(Subsampling::S444);
            let jpeg = encoder.encode_rgb(&rgb_data, width, height).unwrap();

            // Decode and check
            let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
            let decoded = decoder.decode().expect("decode failed");

            let dr = decoded[0];
            let dg = decoded[1];
            let db = decoded[2];

            // Allow tolerance of 10 for JPEG lossy compression
            let tolerance = 10i16;
            let r_diff = (dr as i16 - *r as i16).abs();
            let g_diff = (dg as i16 - *g as i16).abs();
            let b_diff = (db as i16 - *b as i16).abs();

            println!("{}: input=({},{},{}), output=({},{},{}), diff=({},{},{})",
                     name, r, g, b, dr, dg, db, r_diff, g_diff, b_diff);

            assert!(r_diff <= tolerance,
                    "{}: R mismatch - expected {}, got {} (diff {})", name, r, dr, r_diff);
            assert!(g_diff <= tolerance,
                    "{}: G mismatch - expected {}, got {} (diff {})", name, g, dg, g_diff);
            assert!(b_diff <= tolerance,
                    "{}: B mismatch - expected {}, got {} (diff {})", name, b, db, b_diff);
        }
    }
}
