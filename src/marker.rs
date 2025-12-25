//! JPEG marker emission for encoding.
//!
//! This module handles writing JPEG file format markers:
//! - SOI (Start of Image)
//! - APP0 (JFIF header)
//! - DQT (Define Quantization Table)
//! - SOF (Start of Frame)
//! - DHT (Define Huffman Table)
//! - SOS (Start of Scan)
//! - EOI (End of Image)
//!
//! Reference: ITU-T T.81 Section B

use std::io::Write;

use crate::consts::{
    DCTSIZE2, JPEG_DHT, JPEG_DQT, JPEG_EOI, JPEG_SOF0, JPEG_SOF2, JPEG_SOI, JPEG_SOS,
    JPEG_APP0, JPEG_RST0,
};
use crate::huffman::HuffTable;
use crate::types::{ComponentInfo, ScanInfo};

/// JFIF version string
const JFIF_ID: [u8; 5] = *b"JFIF\0";

/// JFIF version 1.01
const JFIF_VERSION: [u8; 2] = [1, 1];

/// Marker writer for JPEG encoding.
pub struct MarkerWriter<W: Write> {
    output: W,
    bytes_written: usize,
}

impl<W: Write> MarkerWriter<W> {
    /// Create a new marker writer.
    pub fn new(output: W) -> Self {
        Self {
            output,
            bytes_written: 0,
        }
    }

    /// Write a single byte.
    fn emit_byte(&mut self, byte: u8) -> std::io::Result<()> {
        self.output.write_all(&[byte])?;
        self.bytes_written += 1;
        Ok(())
    }

    /// Write a 2-byte value in big-endian order.
    fn emit_2bytes(&mut self, value: u16) -> std::io::Result<()> {
        self.emit_byte((value >> 8) as u8)?;
        self.emit_byte(value as u8)?;
        Ok(())
    }

    /// Write a marker (0xFF followed by marker code).
    fn emit_marker(&mut self, marker: u8) -> std::io::Result<()> {
        self.emit_byte(0xFF)?;
        self.emit_byte(marker)?;
        Ok(())
    }

    /// Write Start of Image marker.
    pub fn write_soi(&mut self) -> std::io::Result<()> {
        self.emit_marker(JPEG_SOI)
    }

    /// Write End of Image marker.
    pub fn write_eoi(&mut self) -> std::io::Result<()> {
        self.emit_marker(JPEG_EOI)
    }

    /// Write APP0 (JFIF) marker.
    ///
    /// # Arguments
    /// * `density_unit` - 0=no units, 1=dots/inch, 2=dots/cm
    /// * `x_density` - Horizontal pixel density
    /// * `y_density` - Vertical pixel density
    pub fn write_jfif_app0(
        &mut self,
        density_unit: u8,
        x_density: u16,
        y_density: u16,
    ) -> std::io::Result<()> {
        self.emit_marker(JPEG_APP0)?;

        // Length: 2 (length) + 5 (identifier) + 2 (version) + 1 (units) +
        //         2 (x_density) + 2 (y_density) + 1 (thumbnail_width) +
        //         1 (thumbnail_height) = 16
        self.emit_2bytes(16)?;

        // JFIF identifier
        for &b in &JFIF_ID {
            self.emit_byte(b)?;
        }

        // Version
        self.emit_byte(JFIF_VERSION[0])?;
        self.emit_byte(JFIF_VERSION[1])?;

        // Units and density
        self.emit_byte(density_unit)?;
        self.emit_2bytes(x_density)?;
        self.emit_2bytes(y_density)?;

        // No thumbnail
        self.emit_byte(0)?; // thumbnail width
        self.emit_byte(0)?; // thumbnail height

        Ok(())
    }

    /// Write Define Quantization Table marker for a single table.
    ///
    /// For better compression, prefer `write_dqt_multiple` to combine tables.
    ///
    /// # Arguments
    /// * `table_index` - Table slot (0-3)
    /// * `table` - 64 quantization values in zigzag order
    /// * `force_16bit` - Force 16-bit precision (for values > 255)
    pub fn write_dqt(
        &mut self,
        table_index: u8,
        table: &[u16; DCTSIZE2],
        force_16bit: bool,
    ) -> std::io::Result<()> {
        self.write_dqt_multiple(&[(table_index, table, force_16bit)])
    }

    /// Write Define Quantization Table marker for multiple tables.
    ///
    /// Combines multiple tables into a single DQT marker for smaller file size.
    /// C mozjpeg does this optimization - one marker instead of N markers saves
    /// (N-1) * 2 bytes of marker overhead.
    ///
    /// # Arguments
    /// * `tables` - Slice of (table_index, table values, force_16bit)
    pub fn write_dqt_multiple(
        &mut self,
        tables: &[(u8, &[u16; DCTSIZE2], bool)],
    ) -> std::io::Result<()> {
        if tables.is_empty() {
            return Ok(());
        }

        // Calculate total length
        let mut total_len = 2u16; // length field itself
        for (_, table, force_16bit) in tables {
            let use_16bit = *force_16bit || table.iter().any(|&v| v > 255);
            total_len += 1 + if use_16bit { 128 } else { 64 }; // Pq/Tq + values
        }

        self.emit_marker(JPEG_DQT)?;
        self.emit_2bytes(total_len)?;

        for (table_index, table, force_16bit) in tables {
            let use_16bit = *force_16bit || table.iter().any(|&v| v > 255);

            // Pq (precision) in high nibble, Tq (table index) in low nibble
            let pq_tq = if use_16bit {
                0x10 | (*table_index & 0x0F)
            } else {
                *table_index & 0x0F
            };
            self.emit_byte(pq_tq)?;

            // Table values in zigzag order
            for &value in table.iter() {
                if use_16bit {
                    self.emit_2bytes(value)?;
                } else {
                    self.emit_byte(value as u8)?;
                }
            }
        }

        Ok(())
    }

    /// Write Start of Frame marker (baseline or progressive).
    ///
    /// # Arguments
    /// * `progressive` - True for progressive (SOF2), false for baseline (SOF0)
    /// * `precision` - Sample precision (8 or 12 bits)
    /// * `height` - Image height in pixels
    /// * `width` - Image width in pixels
    /// * `components` - Component information
    pub fn write_sof(
        &mut self,
        progressive: bool,
        precision: u8,
        height: u16,
        width: u16,
        components: &[ComponentInfo],
    ) -> std::io::Result<()> {
        let marker = if progressive { JPEG_SOF2 } else { JPEG_SOF0 };
        self.emit_marker(marker)?;

        // Length: 2 (length) + 1 (precision) + 2 (height) + 2 (width) +
        //         1 (num_components) + 3 * num_components
        let num_components = components.len() as u16;
        self.emit_2bytes(8 + 3 * num_components)?;

        self.emit_byte(precision)?;
        self.emit_2bytes(height)?;
        self.emit_2bytes(width)?;
        self.emit_byte(num_components as u8)?;

        // Component specifications
        for comp in components {
            self.emit_byte(comp.component_id)?;
            // Sampling factors: (H << 4) | V
            let samp = (comp.h_samp_factor << 4) | comp.v_samp_factor;
            self.emit_byte(samp)?;
            self.emit_byte(comp.quant_tbl_no)?;
        }

        Ok(())
    }

    /// Write Define Huffman Table marker for a single table.
    ///
    /// For better compression, prefer `write_dht_multiple` to combine tables.
    ///
    /// # Arguments
    /// * `table_index` - Table slot (0-3)
    /// * `is_ac` - True for AC table, false for DC table
    /// * `table` - Huffman table
    pub fn write_dht(
        &mut self,
        table_index: u8,
        is_ac: bool,
        table: &HuffTable,
    ) -> std::io::Result<()> {
        self.write_dht_multiple(&[(table_index, is_ac, table)])
    }

    /// Write Define Huffman Table marker for multiple tables.
    ///
    /// Combines multiple tables into a single DHT marker for smaller file size.
    /// C mozjpeg does this optimization - one marker instead of N markers saves
    /// (N-1) * 2 bytes of marker overhead.
    ///
    /// # Arguments
    /// * `tables` - Slice of (table_index, is_ac, table)
    pub fn write_dht_multiple(
        &mut self,
        tables: &[(u8, bool, &HuffTable)],
    ) -> std::io::Result<()> {
        if tables.is_empty() {
            return Ok(());
        }

        // Calculate total length
        let mut total_len = 2u16; // length field itself
        for (_, _, table) in tables {
            let num_symbols: usize = table.bits[1..=16].iter().map(|&b| b as usize).sum();
            total_len += 1 + 16 + num_symbols as u16; // Tc/Th + bits + symbols
        }

        self.emit_marker(JPEG_DHT)?;
        self.emit_2bytes(total_len)?;

        for (table_index, is_ac, table) in tables {
            let num_symbols: usize = table.bits[1..=16].iter().map(|&b| b as usize).sum();

            // Tc (table class) in high nibble, Th (table index) in low nibble
            let tc_th = if *is_ac {
                0x10 | (*table_index & 0x0F)
            } else {
                *table_index & 0x0F
            };
            self.emit_byte(tc_th)?;

            // Bits array (counts for each code length)
            for i in 1..=16 {
                self.emit_byte(table.bits[i])?;
            }

            // Huffval array (symbols)
            for i in 0..num_symbols {
                self.emit_byte(table.huffval[i])?;
            }
        }

        Ok(())
    }

    /// Write Start of Scan marker.
    ///
    /// # Arguments
    /// * `scan` - Scan parameters
    /// * `components` - Component info for components in this scan
    pub fn write_sos(
        &mut self,
        scan: &ScanInfo,
        components: &[ComponentInfo],
    ) -> std::io::Result<()> {
        self.emit_marker(JPEG_SOS)?;

        // Length: 2 (length) + 1 (Ns) + 2*Ns (component specs) + 3 (Ss, Se, Ah/Al)
        self.emit_2bytes(6 + 2 * scan.comps_in_scan as u16)?;

        // Number of components in scan
        self.emit_byte(scan.comps_in_scan)?;

        // Component specifications
        for i in 0..scan.comps_in_scan as usize {
            let comp_idx = scan.component_index[i] as usize;
            if comp_idx < components.len() {
                let comp = &components[comp_idx];
                self.emit_byte(comp.component_id)?;
                // DC table in high nibble, AC table in low nibble
                let tables = (comp.dc_tbl_no << 4) | comp.ac_tbl_no;
                self.emit_byte(tables)?;
            }
        }

        // Spectral selection and successive approximation
        self.emit_byte(scan.ss)?;
        self.emit_byte(scan.se)?;
        self.emit_byte((scan.ah << 4) | scan.al)?;

        Ok(())
    }

    /// Write Define Restart Interval marker.
    ///
    /// # Arguments
    /// * `interval` - Number of MCUs between restart markers (0 to disable)
    pub fn write_dri(&mut self, interval: u16) -> std::io::Result<()> {
        if interval == 0 {
            return Ok(());
        }

        self.emit_marker(0xDD)?; // DRI marker
        self.emit_2bytes(4)?; // Length
        self.emit_2bytes(interval)?;

        Ok(())
    }

    /// Write a restart marker (RST0-RST7).
    ///
    /// # Arguments
    /// * `restart_num` - Restart marker number (0-7)
    pub fn write_rst(&mut self, restart_num: u8) -> std::io::Result<()> {
        self.emit_marker(JPEG_RST0 + (restart_num & 0x07))
    }

    /// Write a comment marker.
    ///
    /// # Arguments
    /// * `comment` - Comment text (ASCII)
    pub fn write_com(&mut self, comment: &str) -> std::io::Result<()> {
        self.emit_marker(0xFE)?; // COM marker
        let len = comment.len().min(65533) as u16;
        self.emit_2bytes(2 + len)?;
        for &b in comment.as_bytes().iter().take(len as usize) {
            self.emit_byte(b)?;
        }
        Ok(())
    }

    /// Get total bytes written.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    /// Consume the writer and return the underlying output.
    pub fn into_inner(self) -> W {
        self.output
    }

    /// Get a reference to the underlying output.
    pub fn get_ref(&self) -> &W {
        &self.output
    }

    /// Get a mutable reference to the underlying output.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_components() -> Vec<ComponentInfo> {
        vec![
            ComponentInfo {
                component_id: 1,
                component_index: 0,
                h_samp_factor: 2,
                v_samp_factor: 2,
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
    }

    #[test]
    fn test_write_soi() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);
        writer.write_soi().unwrap();

        assert_eq!(output, vec![0xFF, 0xD8]);
    }

    #[test]
    fn test_write_eoi() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);
        writer.write_eoi().unwrap();

        assert_eq!(output, vec![0xFF, 0xD9]);
    }

    #[test]
    fn test_write_jfif_app0() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);
        writer.write_jfif_app0(1, 72, 72).unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xE0);
        // Check length
        assert_eq!(output[2], 0x00);
        assert_eq!(output[3], 16);
        // Check JFIF identifier
        assert_eq!(&output[4..9], b"JFIF\0");
    }

    #[test]
    fn test_write_dqt_8bit() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let table = [16u16; DCTSIZE2]; // All 16s (fits in 8 bits)
        writer.write_dqt(0, &table, false).unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xDB);
        // Check length (3 + 64 = 67)
        assert_eq!(output[2], 0x00);
        assert_eq!(output[3], 67);
        // Check Pq/Tq (8-bit precision, table 0)
        assert_eq!(output[4], 0x00);
    }

    #[test]
    fn test_write_dqt_16bit() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let mut table = [16u16; DCTSIZE2];
        table[0] = 300; // > 255, needs 16-bit
        writer.write_dqt(1, &table, false).unwrap();

        // Check length (3 + 128 = 131)
        assert_eq!(output[2], 0x00);
        assert_eq!(output[3], 131);
        // Check Pq/Tq (16-bit precision, table 1)
        assert_eq!(output[4], 0x11);
    }

    #[test]
    fn test_write_sof_baseline() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let components = create_test_components();
        writer.write_sof(false, 8, 480, 640, &components).unwrap();

        // Check marker (SOF0 for baseline)
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xC0);
        // Check precision
        assert_eq!(output[4], 8);
        // Check dimensions
        assert_eq!((output[5] as u16) << 8 | output[6] as u16, 480);
        assert_eq!((output[7] as u16) << 8 | output[8] as u16, 640);
    }

    #[test]
    fn test_write_sof_progressive() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let components = create_test_components();
        writer.write_sof(true, 8, 480, 640, &components).unwrap();

        // Check marker (SOF2 for progressive)
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xC2);
    }

    #[test]
    fn test_write_dht() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let mut table = HuffTable::default();
        // Simple table: 2 codes of length 1, 1 code of length 2
        table.bits[1] = 2;
        table.bits[2] = 1;
        table.huffval[0] = 0;
        table.huffval[1] = 1;
        table.huffval[2] = 2;

        writer.write_dht(0, false, &table).unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xC4);
        // Check Tc/Th (DC table, slot 0)
        assert_eq!(output[4], 0x00);
    }

    #[test]
    fn test_write_dht_ac() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let table = HuffTable::default();
        writer.write_dht(1, true, &table).unwrap();

        // Check Tc/Th (AC table, slot 1)
        assert_eq!(output[4], 0x11);
    }

    #[test]
    fn test_write_sos() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        let components = create_test_components();
        let scan = ScanInfo {
            comps_in_scan: 3,
            component_index: [0, 1, 2, 0],
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
        };
        writer.write_sos(&scan, &components).unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xDA);
        // Check number of components
        assert_eq!(output[4], 3);
        // Check spectral selection
        assert_eq!(output[11], 0);  // Ss
        assert_eq!(output[12], 63); // Se
    }

    #[test]
    fn test_write_dri() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        writer.write_dri(100).unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xDD);
        // Check interval
        assert_eq!((output[4] as u16) << 8 | output[5] as u16, 100);
    }

    #[test]
    fn test_write_dri_zero() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        writer.write_dri(0).unwrap();

        // Should write nothing when interval is 0
        assert!(output.is_empty());
    }

    #[test]
    fn test_write_rst() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        writer.write_rst(3).unwrap();

        assert_eq!(output, vec![0xFF, 0xD3]);
    }

    #[test]
    fn test_write_com() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        writer.write_com("Test comment").unwrap();

        // Check marker
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xFE);
        // Check content
        assert_eq!(&output[4..16], b"Test comment");
    }

    #[test]
    fn test_bytes_written() {
        let mut output = Vec::new();
        let mut writer = MarkerWriter::new(&mut output);

        writer.write_soi().unwrap();
        writer.write_eoi().unwrap();

        assert_eq!(writer.bytes_written(), 4);
    }
}
