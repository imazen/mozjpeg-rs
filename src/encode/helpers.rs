//! Helper functions for the encoder pipeline.
//!
//! This module contains utility functions used by both the main Encoder
//! and the StreamingEncoder.

use std::io::Write;

use crate::consts::{
    AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DCTSIZE2,
    DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES, JPEG_DHT,
    JPEG_NATURAL_ORDER, JPEG_SOS,
};
use crate::error::Result;
use crate::huffman::{DerivedTable, HuffTable};
use crate::trellis::dc_trellis_optimize_indexed;
use crate::types::{ComponentInfo, ScanInfo, Subsampling};

// ============================================================================
// Allocation Helpers
// ============================================================================

/// Helper to allocate a Vec with fallible allocation.
/// Returns Error::AllocationFailed if allocation fails.
#[inline]
pub(crate) fn try_alloc_vec<T: Clone>(value: T, len: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    v.resize(len, value);
    Ok(v)
}

/// Helper to allocate a Vec of arrays with fallible allocation.
#[inline]
pub(crate) fn try_alloc_vec_array<T: Copy + Default, const N: usize>(
    len: usize,
) -> Result<Vec<[T; N]>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    v.resize(len, [T::default(); N]);
    Ok(v)
}

// ============================================================================
// MCU/Block Indexing Helpers
// ============================================================================

/// Get MCU-order index for a block at (block_row, block_col) in image coordinates.
///
/// For multi-block MCUs (e.g., 4:2:0 luma with 2x2 blocks per MCU), blocks are stored
/// in MCU order: all blocks for MCU (0,0), then all for MCU (0,1), etc.
/// Within each MCU, blocks are stored row-by-row: (0,0), (0,1), (1,0), (1,1) for 2x2.
pub(crate) fn block_to_mcu_index(
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
pub(crate) fn get_row_indices(
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

// ============================================================================
// DC Trellis Helper
// ============================================================================

/// Run DC trellis optimization row by row (matching C mozjpeg behavior).
///
/// C mozjpeg processes DC trellis one block row at a time, with the last DC
/// value from each row propagating to the next row. This differs from
/// processing all blocks as one giant chain in that each row's optimization
/// is independent, but the differential encoding cost uses the previous row's
/// final DC value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dc_trellis_by_row(
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
    // Start with last_dc = 0 for the first row (matching C mozjpeg)
    let mut last_dc = 0i16;

    // Process each block row, propagating last_dc between rows
    for block_row in 0..block_rows {
        let indices = get_row_indices(block_row, block_cols, mcu_cols, h_samp, v_samp);

        // Use last_dc from previous row (or 0 for first row)
        // The function returns the final DC value for the next row
        last_dc = dc_trellis_optimize_indexed(
            raw_blocks,
            quantized_blocks,
            &indices,
            dc_quantval,
            dc_table,
            last_dc,
            lambda_log_scale1,
            lambda_log_scale2,
        );
    }
}

// ============================================================================
// Marker Writing Helpers
// ============================================================================

/// Write an SOS (Start of Scan) marker.
pub(crate) fn write_sos_marker<W: Write>(
    output: &mut W,
    scan: &ScanInfo,
    components: &[ComponentInfo],
) -> std::io::Result<()> {
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

/// Write a DHT marker directly to a writer.
///
/// Used for per-scan Huffman tables when optimize_scans is enabled.
/// Each AC scan gets its own optimal table written immediately before the scan.
///
/// # Arguments
/// * `output` - The output writer
/// * `table_index` - Huffman table index (0 for luma, 1 for chroma)
/// * `is_ac` - True for AC table, false for DC table
/// * `table` - The Huffman table to write
pub(crate) fn write_dht_marker<W: Write>(
    output: &mut W,
    table_index: u8,
    is_ac: bool,
    table: &HuffTable,
) -> std::io::Result<()> {
    // Count symbols in the table
    let num_symbols: usize = table.bits[1..=16].iter().map(|&b| b as usize).sum();

    // Length: 2 (length field) + 1 (Tc/Th) + 16 (bits) + num_symbols
    let length = 2 + 1 + 16 + num_symbols;

    // DHT marker
    output.write_all(&[0xFF, JPEG_DHT])?;

    // Length
    output.write_all(&[(length >> 8) as u8, (length & 0xFF) as u8])?;

    // Tc (table class) in high nibble, Th (table index) in low nibble
    let tc_th = if is_ac {
        0x10 | (table_index & 0x0F)
    } else {
        table_index & 0x0F
    };
    output.write_all(&[tc_th])?;

    // Bits array (counts for each code length)
    for i in 1..=16 {
        output.write_all(&[table.bits[i]])?;
    }

    // Huffval array (symbols)
    for i in 0..num_symbols {
        output.write_all(&[table.huffval[i]])?;
    }

    Ok(())
}

// ============================================================================
// Component Creation Helpers
// ============================================================================

/// Create component info for the given subsampling mode.
/// Returns 1 component for grayscale, 3 for color modes.
pub(crate) fn create_components(subsampling: Subsampling) -> Vec<ComponentInfo> {
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

/// Alias for backwards compatibility.
#[inline]
pub(crate) fn create_ycbcr_components(subsampling: Subsampling) -> Vec<ComponentInfo> {
    create_components(subsampling)
}

// ============================================================================
// Ordering Helper
// ============================================================================

/// Convert a quantization table from natural order to zigzag order.
#[allow(clippy::needless_range_loop)]
pub(crate) fn natural_to_zigzag(natural: &[u16; DCTSIZE2]) -> [u16; DCTSIZE2] {
    let mut zigzag = [0u16; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        zigzag[i] = natural[JPEG_NATURAL_ORDER[i]];
    }
    zigzag
}

// ============================================================================
// Standard Huffman Table Creation
// ============================================================================

/// Create standard DC luminance Huffman table.
pub(crate) fn create_std_dc_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    htbl.huffval[..DC_LUMINANCE_VALUES.len()].copy_from_slice(&DC_LUMINANCE_VALUES);
    htbl
}

/// Create standard DC chrominance Huffman table.
pub(crate) fn create_std_dc_chroma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_CHROMINANCE_BITS);
    htbl.huffval[..DC_CHROMINANCE_VALUES.len()].copy_from_slice(&DC_CHROMINANCE_VALUES);
    htbl
}

/// Create standard AC luminance Huffman table.
pub(crate) fn create_std_ac_luma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    htbl.huffval[..AC_LUMINANCE_VALUES.len()].copy_from_slice(&AC_LUMINANCE_VALUES);
    htbl
}

/// Create standard AC chrominance Huffman table.
pub(crate) fn create_std_ac_chroma_table() -> HuffTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
    htbl.huffval[..AC_CHROMINANCE_VALUES.len()].copy_from_slice(&AC_CHROMINANCE_VALUES);
    htbl
}
