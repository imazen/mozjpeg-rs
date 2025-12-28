//! Sequential scan trial encoder for optimize_scans.
//!
//! This module implements sequential trial encoding that maintains state between
//! scans, which is required for accurate refinement scan size estimation.
//!
//! The C mozjpeg implementation encodes all 64 candidate scans in sequence,
//! storing each scan's bytes. Refinement scans (Ah > 0) only work correctly
//! when they have context from the scans that came before them.
//!
//! IMPORTANT: Each scan uses its OWN optimal Huffman table, built from the
//! symbol frequencies in that specific scan. This matches C mozjpeg behavior
//! with optimize_scans enabled.

use std::io::Write;

use crate::bitstream::BitWriter;
use crate::consts::DCTSIZE2;
use crate::entropy::{ProgressiveEncoder, ProgressiveSymbolCounter};
use crate::error::Result;
use crate::huffman::{DerivedTable, FrequencyCounter};
use crate::types::ScanInfo;

/// State for a coefficient across the encoding sequence.
/// Tracks whether this coefficient has been "first-scanned" and at what Al level.
#[derive(Clone, Copy, Default)]
struct CoeffState {
    /// The Al level at which this coefficient was first coded (0 means not yet coded for AC)
    first_al: u8,
    /// Whether this coefficient has been first-scanned (Ah=0 scan)
    coded: bool,
}

/// Block-level state for sequential trial encoding.
#[derive(Clone)]
struct BlockState {
    /// State for each AC coefficient (indices 1-63 in zigzag order)
    ac_state: [CoeffState; 63],
    /// DC coefficient first Al level
    dc_first_al: u8,
    /// Whether DC has been coded
    dc_coded: bool,
}

impl Default for BlockState {
    fn default() -> Self {
        Self {
            ac_state: [CoeffState::default(); 63],
            dc_first_al: 0,
            dc_coded: false,
        }
    }
}

/// Sequential scan trial encoder.
///
/// Encodes scans in sequence while maintaining state about which coefficients
/// have been coded. This allows accurate size estimation for refinement scans.
///
/// Each scan uses its own optimal Huffman table, matching C mozjpeg's behavior.
pub struct ScanTrialEncoder<'a> {
    /// Y component blocks
    y_blocks: &'a [[i16; DCTSIZE2]],
    /// Cb component blocks (empty for grayscale)
    cb_blocks: &'a [[i16; DCTSIZE2]],
    /// Cr component blocks (empty for grayscale)
    cr_blocks: &'a [[i16; DCTSIZE2]],

    /// Block state for Y component
    y_state: Vec<BlockState>,
    /// Block state for Cb component
    cb_state: Vec<BlockState>,
    /// Block state for Cr component
    cr_state: Vec<BlockState>,

    /// Stored Huffman tables (fallback for DC scans)
    dc_luma: &'a DerivedTable,
    dc_chroma: &'a DerivedTable,
    #[allow(dead_code)]
    ac_luma: &'a DerivedTable,
    #[allow(dead_code)]
    ac_chroma: &'a DerivedTable,

    /// MCU dimensions
    mcu_rows: usize,
    mcu_cols: usize,
    h_samp: u8,
    v_samp: u8,

    /// Actual image dimensions
    actual_width: usize,
    actual_height: usize,
    chroma_width: usize,
    chroma_height: usize,

    /// Stored scan data for each trial-encoded scan
    scan_buffers: Vec<Vec<u8>>,
}

impl<'a> ScanTrialEncoder<'a> {
    /// Create a new sequential scan trial encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        y_blocks: &'a [[i16; DCTSIZE2]],
        cb_blocks: &'a [[i16; DCTSIZE2]],
        cr_blocks: &'a [[i16; DCTSIZE2]],
        dc_luma: &'a DerivedTable,
        dc_chroma: &'a DerivedTable,
        ac_luma: &'a DerivedTable,
        ac_chroma: &'a DerivedTable,
        mcu_rows: usize,
        mcu_cols: usize,
        h_samp: u8,
        v_samp: u8,
        actual_width: usize,
        actual_height: usize,
        chroma_width: usize,
        chroma_height: usize,
    ) -> Self {
        Self {
            y_blocks,
            cb_blocks,
            cr_blocks,
            y_state: vec![BlockState::default(); y_blocks.len()],
            cb_state: vec![BlockState::default(); cb_blocks.len()],
            cr_state: vec![BlockState::default(); cr_blocks.len()],
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
            scan_buffers: Vec::new(),
        }
    }

    /// Encode all candidate scans sequentially and return their sizes.
    ///
    /// This maintains state between scans so refinement scans produce correct sizes.
    pub fn encode_all_scans(&mut self, scans: &[ScanInfo]) -> Result<Vec<usize>> {
        let mut sizes = Vec::with_capacity(scans.len());

        for scan in scans {
            let size = self.encode_scan(scan)?;
            sizes.push(size);
        }

        Ok(sizes)
    }

    /// Encode a single scan and return its size.
    ///
    /// This does two-pass encoding like C mozjpeg:
    /// 1. First pass: count symbol frequencies
    /// 2. Build optimal Huffman table from frequencies
    /// 3. Second pass: encode with optimal table
    fn encode_scan(&mut self, scan: &ScanInfo) -> Result<usize> {
        let is_dc_scan = scan.ss == 0 && scan.se == 0;
        let is_refinement = scan.ah != 0;

        if is_dc_scan {
            // DC scans - use stored tables (DC scans are small, optimization less impactful)
            let mut buffer = Vec::new();
            let mut bit_writer = BitWriter::new(&mut buffer);
            let mut encoder = ProgressiveEncoder::new(&mut bit_writer);

            self.encode_dc_scan_with_stored_tables(scan, is_refinement, &mut encoder)?;
            encoder.finish_scan(None)?;
            bit_writer.flush()?;

            let size = buffer.len();
            self.scan_buffers.push(buffer);
            Ok(size)
        } else {
            // AC scans - use per-scan optimal Huffman tables
            // Pass 1: Count symbol frequencies
            let mut ac_counter = FrequencyCounter::new();
            self.count_ac_scan_symbols(scan, is_refinement, &mut ac_counter)?;

            // Build optimal Huffman table from frequencies
            let ac_huff = ac_counter.generate_table()?;
            let ac_table = DerivedTable::from_huff_table(&ac_huff, false)?;

            // Pass 2: Encode with optimal table
            let mut buffer = Vec::new();
            let mut bit_writer = BitWriter::new(&mut buffer);
            let mut encoder = ProgressiveEncoder::new(&mut bit_writer);

            self.encode_ac_scan_with_table(scan, is_refinement, &ac_table, &mut encoder)?;
            encoder.finish_scan(Some(&ac_table))?;
            bit_writer.flush()?;

            let size = buffer.len();
            self.scan_buffers.push(buffer);
            Ok(size)
        }
    }

    /// Count AC scan symbols (pass 1).
    fn count_ac_scan_symbols(
        &self,
        scan: &ScanInfo,
        is_refinement: bool,
        ac_counter: &mut FrequencyCounter,
    ) -> Result<()> {
        let comp_idx = scan.component_index[0] as usize;
        let blocks = match comp_idx {
            0 => self.y_blocks,
            1 => self.cb_blocks,
            2 => self.cr_blocks,
            _ => return Ok(()),
        };

        let ss = scan.ss;
        let se = scan.se;
        let al = scan.al;

        let (num_block_rows, num_block_cols) = if comp_idx == 0 {
            ((self.actual_height + 7) / 8, (self.actual_width + 7) / 8)
        } else {
            ((self.chroma_height + 7) / 8, (self.chroma_width + 7) / 8)
        };

        // For Y component with subsampling, blocks are stored in MCU-interleaved order
        // but AC scans encode them in component raster order.
        let blocks_per_mcu = if comp_idx == 0 {
            self.h_samp as usize * self.v_samp as usize
        } else {
            1
        };

        let mut counter = ProgressiveSymbolCounter::new();

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            let total_blocks = num_block_rows * num_block_cols;
            for block in blocks.iter().take(total_blocks) {
                if is_refinement {
                    counter.count_ac_refine(block, ss, se, scan.ah, al, ac_counter);
                } else {
                    counter.count_ac_first(block, ss, se, al, ac_counter);
                }
            }
        } else {
            // Y component with subsampling - iterate in raster order but access MCU storage
            let h = self.h_samp as usize;
            let v = self.v_samp as usize;

            for block_row in 0..num_block_rows {
                for block_col in 0..num_block_cols {
                    // Convert raster position to MCU-interleaved storage index
                    let mcu_row = block_row / v;
                    let mcu_col = block_col / h;
                    let v_idx = block_row % v;
                    let h_idx = block_col % h;
                    let storage_idx = mcu_row * (self.mcu_cols * blocks_per_mcu)
                        + mcu_col * blocks_per_mcu
                        + v_idx * h
                        + h_idx;

                    if storage_idx < blocks.len() {
                        let block = &blocks[storage_idx];

                        if is_refinement {
                            counter.count_ac_refine(block, ss, se, scan.ah, al, ac_counter);
                        } else {
                            counter.count_ac_first(block, ss, se, al, ac_counter);
                        }
                    }
                }
            }
        }

        // Finish to flush any pending EOBRUN
        counter.finish_scan(Some(ac_counter));

        Ok(())
    }

    /// Encode DC scan with stored Huffman tables.
    fn encode_dc_scan_with_stored_tables<W: Write>(
        &mut self,
        scan: &ScanInfo,
        is_refinement: bool,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        let al = scan.al;

        for mcu_row in 0..self.mcu_rows {
            for mcu_col in 0..self.mcu_cols {
                for i in 0..scan.comps_in_scan as usize {
                    let comp_idx = scan.component_index[i] as usize;
                    let (blocks, state, dc_table) = match comp_idx {
                        0 => (self.y_blocks, &mut self.y_state, self.dc_luma),
                        1 => (self.cb_blocks, &mut self.cb_state, self.dc_chroma),
                        2 => (self.cr_blocks, &mut self.cr_state, self.dc_chroma),
                        _ => continue,
                    };

                    let (h_blocks, v_blocks) = if comp_idx == 0 {
                        (self.h_samp as usize, self.v_samp as usize)
                    } else {
                        (1, 1)
                    };

                    for v in 0..v_blocks {
                        for h in 0..h_blocks {
                            let block_row = mcu_row * v_blocks + v;
                            let block_col = mcu_col * h_blocks + h;

                            let blocks_per_row = if comp_idx == 0 {
                                self.mcu_cols * h_blocks
                            } else {
                                self.mcu_cols
                            };
                            let block_idx = block_row * blocks_per_row + block_col;

                            if block_idx < blocks.len() {
                                let block = &blocks[block_idx];
                                let block_state = &mut state[block_idx];

                                if is_refinement {
                                    if block_state.dc_coded {
                                        encoder.encode_dc_refine(block, al)?;
                                    }
                                } else {
                                    encoder.encode_dc_first(block, comp_idx, dc_table, al)?;
                                    block_state.dc_coded = true;
                                    block_state.dc_first_al = al;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Encode AC scan with a specific Huffman table (pass 2).
    fn encode_ac_scan_with_table<W: Write>(
        &mut self,
        scan: &ScanInfo,
        is_refinement: bool,
        ac_table: &DerivedTable,
        encoder: &mut ProgressiveEncoder<W>,
    ) -> Result<()> {
        let comp_idx = scan.component_index[0] as usize;
        let blocks = match comp_idx {
            0 => self.y_blocks,
            1 => self.cb_blocks,
            2 => self.cr_blocks,
            _ => return Ok(()),
        };

        let ss = scan.ss as usize;
        let se = scan.se as usize;
        let al = scan.al;

        let (num_block_rows, num_block_cols) = if comp_idx == 0 {
            ((self.actual_height + 7) / 8, (self.actual_width + 7) / 8)
        } else {
            ((self.chroma_height + 7) / 8, (self.chroma_width + 7) / 8)
        };

        // For Y component with subsampling, blocks are stored in MCU-interleaved order
        // but AC scans encode them in component raster order.
        let blocks_per_mcu = if comp_idx == 0 {
            self.h_samp as usize * self.v_samp as usize
        } else {
            1
        };

        if blocks_per_mcu == 1 {
            // Chroma or 4:4:4 Y: storage order = raster order
            let total_blocks = num_block_rows * num_block_cols;
            for block in blocks.iter().take(total_blocks) {
                if is_refinement {
                    encoder.encode_ac_refine(block, ss as u8, se as u8, scan.ah, al, ac_table)?;
                } else {
                    encoder.encode_ac_first(block, ss as u8, se as u8, al, ac_table)?;
                }
            }
        } else {
            // Y component with subsampling - iterate in raster order but access MCU storage
            let h = self.h_samp as usize;
            let v = self.v_samp as usize;

            for block_row in 0..num_block_rows {
                for block_col in 0..num_block_cols {
                    // Convert raster position to MCU-interleaved storage index
                    let mcu_row = block_row / v;
                    let mcu_col = block_col / h;
                    let v_idx = block_row % v;
                    let h_idx = block_col % h;
                    let storage_idx = mcu_row * (self.mcu_cols * blocks_per_mcu)
                        + mcu_col * blocks_per_mcu
                        + v_idx * h
                        + h_idx;

                    if storage_idx < blocks.len() {
                        let block = &blocks[storage_idx];

                        if is_refinement {
                            encoder.encode_ac_refine(
                                block, ss as u8, se as u8, scan.ah, al, ac_table,
                            )?;
                        } else {
                            encoder.encode_ac_first(block, ss as u8, se as u8, al, ac_table)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the stored scan buffers (for output assembly).
    pub fn get_scan_buffers(&self) -> &[Vec<u8>] {
        &self.scan_buffers
    }

    /// Reset the encoder state for a new trial sequence.
    pub fn reset(&mut self) {
        for state in &mut self.y_state {
            *state = BlockState::default();
        }
        for state in &mut self.cb_state {
            *state = BlockState::default();
        }
        for state in &mut self.cr_state {
            *state = BlockState::default();
        }
        self.scan_buffers.clear();
    }
}
