//! Huffman entropy encoder for JPEG.
//!
//! This module implements baseline Huffman encoding for DCT coefficients:
//! - DC coefficient encoding with differential coding
//! - AC coefficient encoding with run-length coding
//! - EOB (End of Block) and ZRL (Zero Run Length) symbols
//!
//! Reference: ITU-T T.81 Section F.1.2

use std::io::Write;

use crate::bitstream::BitWriter;
use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::huffman::{DerivedTable, FrequencyCounter};

/// Maximum coefficient bit size for 8-bit JPEG (10 bits for AC, 11 for DC diff)
#[allow(dead_code)]
const MAX_COEF_BITS: u8 = 10;

/// EOB (End of Block) symbol - encodes as run=0, size=0
const EOB: u8 = 0x00;

/// ZRL (Zero Run Length 16) symbol - encodes 16 consecutive zeros
const ZRL: u8 = 0xF0;

/// Calculate the number of bits needed to represent a value.
///
/// This is the "category" in JPEG terminology:
/// - 0 → 0 bits (value must be 0)
/// - 1 → 1 bit (values -1, 1)
/// - 2 → 2 bits (values -3..-2, 2..3)
/// - etc.
///
/// Uses the efficient bit-scan approach from mozjpeg.
#[inline]
pub fn jpeg_nbits(value: i16) -> u8 {
    if value == 0 {
        return 0;
    }
    let abs_value = value.unsigned_abs();
    16 - abs_value.leading_zeros() as u8
}

/// Calculate nbits for a non-zero value (faster, no zero check).
#[inline]
pub fn jpeg_nbits_nonzero(value: u16) -> u8 {
    16 - value.leading_zeros() as u8
}

/// Entropy encoder state for a single scan.
pub struct EntropyEncoder<'a, W: Write> {
    /// Bitstream writer
    writer: &'a mut BitWriter<W>,
    /// Last DC value for each component (for differential coding)
    last_dc_val: [i16; 4],
}

impl<'a, W: Write> EntropyEncoder<'a, W> {
    /// Create a new entropy encoder.
    pub fn new(writer: &'a mut BitWriter<W>) -> Self {
        Self {
            writer,
            last_dc_val: [0; 4],
        }
    }

    /// Reset DC predictions (called at restart markers).
    pub fn reset_dc(&mut self) {
        self.last_dc_val = [0; 4];
    }

    /// Emit a restart marker and reset DC predictions.
    ///
    /// This flushes the bit buffer (padding with 1-bits to byte boundary),
    /// writes the RST marker (0xFFD0 + restart_num), and resets DC predictions.
    ///
    /// # Arguments
    /// * `restart_num` - Restart marker number (0-7, will be masked to 3 bits)
    pub fn emit_restart(&mut self, restart_num: u8) -> std::io::Result<()> {
        // Flush bit buffer (pads with 1-bits to byte boundary)
        self.writer.flush()?;

        // Write RST marker: 0xFF followed by 0xD0 + (restart_num % 8)
        let rst_marker = 0xD0 + (restart_num & 0x07);
        self.writer.write_bytes(&[0xFF, rst_marker])?;

        // Reset DC predictions
        self.reset_dc();

        Ok(())
    }

    /// Get the last DC value for a component.
    pub fn last_dc(&self, component: usize) -> i16 {
        self.last_dc_val[component]
    }

    /// Set the last DC value for a component.
    pub fn set_last_dc(&mut self, component: usize, value: i16) {
        self.last_dc_val[component] = value;
    }

    /// Encode a single 8x8 block of DCT coefficients.
    ///
    /// # Arguments
    /// * `block` - 64 quantized DCT coefficients in natural (row-major) order
    /// * `component` - Component index (for DC prediction tracking)
    /// * `dc_table` - Derived Huffman table for DC coefficients
    /// * `ac_table` - Derived Huffman table for AC coefficients
    pub fn encode_block(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
    ) -> std::io::Result<()> {
        // Encode DC coefficient (differential)
        self.encode_dc(block[0], component, dc_table)?;

        // Encode AC coefficients in zigzag order
        self.encode_ac(block, ac_table)?;

        Ok(())
    }

    /// Encode the DC coefficient using differential coding.
    ///
    /// The DC value is encoded as the difference from the previous DC value
    /// of the same component, followed by the actual bits.
    fn encode_dc(
        &mut self,
        dc: i16,
        component: usize,
        dc_table: &DerivedTable,
    ) -> std::io::Result<()> {
        // Calculate difference from last DC
        let diff = dc.wrapping_sub(self.last_dc_val[component]);
        self.last_dc_val[component] = dc;

        // Handle the value encoding (Section F.1.2.1)
        // For negative values, we encode the complement
        let (nbits, value) = if diff < 0 {
            let nbits = jpeg_nbits(diff);
            // For negative, encode (diff - 1) which gives all-zeros for the sign-extended bits
            let value = (diff as u16).wrapping_sub(1) & ((1u16 << nbits) - 1);
            (nbits, value)
        } else {
            let nbits = jpeg_nbits(diff);
            (nbits, diff as u16)
        };

        // Emit Huffman code for the category (number of bits)
        let (code, size) = dc_table.get_code(nbits);
        if size > 0 {
            self.writer.put_bits(code, size)?;
        }

        // Emit the actual value bits
        if nbits > 0 {
            self.writer.put_bits(value as u32, nbits)?;
        }

        Ok(())
    }

    /// Encode AC coefficients using run-length coding.
    ///
    /// AC coefficients are encoded in zigzag order as (run, size) pairs
    /// where run is the number of preceding zeros and size is the magnitude bits.
    fn encode_ac(
        &mut self,
        block: &[i16; DCTSIZE2],
        ac_table: &DerivedTable,
    ) -> std::io::Result<()> {
        let mut run = 0u8; // Run length of zeros

        // Process coefficients 1-63 in zigzag order
        for &zigzag_idx in JPEG_NATURAL_ORDER[1..].iter() {
            let coef = block[zigzag_idx];

            if coef == 0 {
                run += 1;
            } else {
                // Emit ZRL codes for runs of 16+ zeros
                while run >= 16 {
                    let (code, size) = ac_table.get_code(ZRL);
                    self.writer.put_bits(code, size)?;
                    run -= 16;
                }

                // Calculate bits needed and value to encode
                let (nbits, value) = if coef < 0 {
                    let nbits = jpeg_nbits(coef);
                    let value = (coef as u16).wrapping_sub(1) & ((1u16 << nbits) - 1);
                    (nbits, value)
                } else {
                    let nbits = jpeg_nbits(coef);
                    (nbits, coef as u16)
                };

                // Symbol = (run << 4) | nbits
                let symbol = (run << 4) | nbits;
                let (code, size) = ac_table.get_code(symbol);
                self.writer.put_bits(code, size)?;

                // Emit the value bits
                if nbits > 0 {
                    self.writer.put_bits(value as u32, nbits)?;
                }

                run = 0;
            }
        }

        // If there are trailing zeros, emit EOB
        if run > 0 {
            let (code, size) = ac_table.get_code(EOB);
            self.writer.put_bits(code, size)?;
        }

        Ok(())
    }

    /// Flush any remaining bits to the output.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Encode a single block without maintaining state (for testing).
pub fn encode_block_standalone<W: Write>(
    writer: &mut BitWriter<W>,
    block: &[i16; DCTSIZE2],
    last_dc: i16,
    dc_table: &DerivedTable,
    ac_table: &DerivedTable,
) -> std::io::Result<i16> {
    let mut encoder = EntropyEncoder::new(writer);
    encoder.set_last_dc(0, last_dc);
    encoder.encode_block(block, 0, dc_table, ac_table)?;
    Ok(encoder.last_dc(0))
}

// =============================================================================
// Symbol Frequency Counting (for Huffman optimization)
// =============================================================================

/// Count symbol frequencies from a block for Huffman table optimization.
///
/// This is used in the first pass of 2-pass encoding to gather statistics
/// that will be used to generate optimal Huffman tables.
pub struct SymbolCounter {
    /// Last DC value for each component (for differential coding)
    last_dc_val: [i16; 4],
}

impl Default for SymbolCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolCounter {
    /// Create a new symbol counter.
    pub fn new() -> Self {
        Self {
            last_dc_val: [0; 4],
        }
    }

    /// Reset DC predictions.
    pub fn reset(&mut self) {
        self.last_dc_val = [0; 4];
    }

    /// Count symbols in a block, updating frequency counters.
    ///
    /// # Arguments
    /// * `block` - Quantized DCT coefficients
    /// * `component` - Component index for DC prediction
    /// * `dc_counter` - Frequency counter for DC symbols
    /// * `ac_counter` - Frequency counter for AC symbols
    pub fn count_block(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_counter: &mut FrequencyCounter,
        ac_counter: &mut FrequencyCounter,
    ) {
        // Count DC symbol
        let dc = block[0];
        let diff = dc.wrapping_sub(self.last_dc_val[component]);
        self.last_dc_val[component] = dc;

        let nbits = jpeg_nbits(diff);
        dc_counter.count(nbits);

        // Count AC symbols
        let mut run = 0u8;

        for &zigzag_idx in JPEG_NATURAL_ORDER[1..].iter() {
            let coef = block[zigzag_idx];

            if coef == 0 {
                run += 1;
            } else {
                // Count ZRL codes for runs of 16+ zeros
                while run >= 16 {
                    ac_counter.count(0xF0); // ZRL
                    run -= 16;
                }

                let nbits = jpeg_nbits(coef);
                let symbol = (run << 4) | nbits;
                ac_counter.count(symbol);
                run = 0;
            }
        }

        // Count EOB if there are trailing zeros
        if run > 0 {
            ac_counter.count(0x00); // EOB
        }
    }
}

// =============================================================================
// Progressive Entropy Encoder
// =============================================================================

/// Progressive entropy encoder for multi-scan JPEG encoding.
///
/// Progressive JPEG uses:
/// - DC scans: encode DC coefficients with optional successive approximation
/// - AC scans: encode AC coefficients in spectral bands (Ss..Se) for one component
/// - Refinement scans: encode additional bits of previously-coded coefficients
pub struct ProgressiveEncoder<'a, W: Write> {
    /// Bitstream writer
    writer: &'a mut BitWriter<W>,
    /// Last DC value for each component (for differential coding)
    last_dc_val: [i16; 4],
    /// End-of-block run count (for EOBRUN encoding)
    eobrun: u16,
    /// Whether to allow extended EOBRUN (requires optimized Huffman tables)
    allow_eobrun: bool,
    /// Buffered correction bits for AC refinement scans.
    /// These accumulate across blocks and are emitted with EOBRUN.
    /// C mozjpeg calls this BE (Buffered Errors) with bit_buffer.
    correction_bits: Vec<u32>,
}

impl<'a, W: Write> ProgressiveEncoder<'a, W> {
    /// Create a new progressive encoder.
    ///
    /// By default, allows extended EOBRUN (requires optimized Huffman tables).
    pub fn new(writer: &'a mut BitWriter<W>) -> Self {
        Self {
            writer,
            last_dc_val: [0; 4],
            eobrun: 0,
            allow_eobrun: true,
            correction_bits: Vec::new(),
        }
    }

    /// Create a progressive encoder for use with standard Huffman tables.
    ///
    /// Standard tables only include EOB (0x00), not extended EOBRUN symbols
    /// (0x10, 0x20, etc.). This mode flushes EOB after each block.
    pub fn new_standard_tables(writer: &'a mut BitWriter<W>) -> Self {
        Self {
            writer,
            last_dc_val: [0; 4],
            eobrun: 0,
            allow_eobrun: false,
            correction_bits: Vec::new(),
        }
    }

    /// Reset DC predictions (called at start of each scan).
    pub fn reset(&mut self) {
        self.last_dc_val = [0; 4];
        self.eobrun = 0;
        self.correction_bits.clear();
    }

    /// Encode a DC scan (first pass, Ah=0).
    ///
    /// # Arguments
    /// * `block` - DCT coefficients in natural order
    /// * `component` - Component index for DC prediction
    /// * `dc_table` - Huffman table for DC
    /// * `al` - Point transform (successive approximation low bit)
    pub fn encode_dc_first(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        al: u8,
    ) -> std::io::Result<()> {
        // Get DC coefficient with point transform
        let dc = block[0] >> al;

        // Calculate difference from last DC
        let diff = dc.wrapping_sub(self.last_dc_val[component]);
        self.last_dc_val[component] = dc;

        // Encode difference
        let (nbits, value) = if diff < 0 {
            let nbits = jpeg_nbits(diff);
            let value = (diff as u16).wrapping_sub(1) & ((1u16 << nbits) - 1);
            (nbits, value)
        } else {
            let nbits = jpeg_nbits(diff);
            (nbits, diff as u16)
        };

        // Emit Huffman code for the category
        let (code, size) = dc_table.get_code(nbits);
        if size > 0 {
            self.writer.put_bits(code, size)?;
        }

        // Emit the value bits
        if nbits > 0 {
            self.writer.put_bits(value as u32, nbits)?;
        }

        Ok(())
    }

    /// Encode a DC refinement scan (Ah != 0).
    ///
    /// Just outputs a single bit for each block.
    pub fn encode_dc_refine(&mut self, block: &[i16; DCTSIZE2], al: u8) -> std::io::Result<()> {
        // Output the next bit of DC coefficient
        let bit = ((block[0] >> al) & 1) as u32;
        self.writer.put_bits(bit, 1)?;
        Ok(())
    }

    /// Encode an AC first scan (Ah=0).
    ///
    /// # Arguments
    /// * `block` - DCT coefficients in natural order
    /// * `ss` - Spectral selection start (1..63)
    /// * `se` - Spectral selection end (1..63)
    /// * `al` - Point transform (successive approximation low bit)
    /// * `ac_table` - Huffman table for AC
    pub fn encode_ac_first(
        &mut self,
        block: &[i16; DCTSIZE2],
        ss: u8,
        se: u8,
        al: u8,
        ac_table: &DerivedTable,
    ) -> std::io::Result<()> {
        // Find last non-zero coefficient in this band.
        // C mozjpeg: checks (abs(coef) >> Al) != 0
        let mut k = se;
        while k >= ss {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            let abs_shifted = coef.unsigned_abs() >> al;
            if abs_shifted != 0 {
                break;
            }
            k -= 1;
        }
        let kex = k;

        let mut run = 0u32;

        for k in ss..=se {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            // Apply point transform to ABSOLUTE value (C mozjpeg does this in prepare)
            let abs_shifted = coef.unsigned_abs() >> al;

            if abs_shifted == 0 {
                run += 1;
                continue;
            }

            // Flush any pending EOBRUN
            if self.eobrun > 0 {
                self.flush_eobrun(ac_table)?;
            }

            // Emit ZRL codes for runs of 16+ zeros
            while run >= 16 {
                let (code, size) = ac_table.get_code(0xF0);
                if size == 0 {
                    eprintln!("ERROR: ZRL symbol 0xF0 has no Huffman code!");
                }
                self.writer.put_bits(code, size)?;
                run -= 16;
            }

            // Calculate category (number of bits needed for shifted absolute value)
            let nbits = jpeg_nbits_nonzero(abs_shifted);

            // Symbol = (run << 4) | nbits
            let symbol = ((run as u8) << 4) | nbits;
            let (code, size) = ac_table.get_code(symbol);
            if size == 0 {
                // This should never happen - means symbol wasn't in Huffman table
                eprintln!("ERROR: AC symbol 0x{:02X} has no Huffman code!", symbol);
            }
            self.writer.put_bits(code, size)?;

            // Emit value bits.
            // For negative: emit (abs_shifted - 1)
            // For positive: emit abs_shifted
            if coef < 0 {
                self.writer.put_bits((abs_shifted - 1) as u32, nbits)?;
            } else {
                self.writer.put_bits(abs_shifted as u32, nbits)?;
            }

            run = 0;

            // Check if we've reached the last non-zero coefficient
            if k == kex {
                break;
            }
        }

        // Emit EOB if we didn't encode all coefficients in the band.
        // This happens when:
        // - kex < se: last non-zero is before the end of the band
        // - kex < ss: all coefficients in the band are zero
        // Both cases are covered by kex < se (since ss <= se).
        if kex < se {
            self.eobrun += 1;
            // Standard tables only have EOB (0x00), not extended EOBRUN symbols.
            // Flush after each block if EOBRUN is disabled.
            if !self.allow_eobrun || self.eobrun == 0x7FFF {
                self.flush_eobrun(ac_table)?;
            }
        }

        Ok(())
    }

    /// Encode an AC refinement scan (Ah != 0).
    ///
    /// This is more complex because we need to:
    /// 1. Output correction bits for previously non-zero coefficients
    /// 2. Output new non-zero coefficients with their correction bits
    ///
    /// # Arguments
    /// * `block` - DCT coefficients in natural order
    /// * `ss` - Spectral selection start (1..63)
    /// * `se` - Spectral selection end (1..63)
    /// * `ah` - Successive approximation high bit (previous Al)
    /// * `al` - Successive approximation low bit (current precision)
    /// * `ac_table` - Huffman table for AC
    pub fn encode_ac_refine(
        &mut self,
        block: &[i16; DCTSIZE2],
        ss: u8,
        se: u8,
        _ah: u8,
        al: u8,
        ac_table: &DerivedTable,
    ) -> std::io::Result<()> {
        let mut run = 0u32;
        // br_start = index where THIS block's bits start in correction_bits.
        // BE (before) bits are 0..br_start, BR (current block) bits are br_start..
        let mut br_start = self.correction_bits.len();

        for k in ss..=se {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            let abs_coef = coef.unsigned_abs();

            // Check if this is a previously-coded non-zero coefficient.
            // C mozjpeg: temp >>= Al, then if (temp > 1) it was previously coded.
            // This is equivalent to: (abs_coef >> al) > 1
            // Note: The first scan encoded coef if (abs >> (al+1)) != 0
            //       In refinement, (abs >> al) > 1 implies (abs >> (al+1)) >= 1 > 0
            if (abs_coef >> al) > 1 {
                // Already coded - just queue the refinement bit
                // Use absolute value since magnitude bits are what we're refining
                let correction_bit = ((abs_coef >> al) & 1) as u32;
                self.correction_bits.push(correction_bit);
            } else if (abs_coef >> al) == 1 {
                // New non-zero coefficient
                // First, flush any pending EOBRUN with BE (previous blocks') correction bits
                if self.eobrun > 0 {
                    self.flush_eobrun_be_bits(ac_table, br_start)?;
                    // After flush, BE bits are gone but BR bits remain
                    // Move BR bits to start of buffer
                    let br_bits: Vec<u32> = self.correction_bits[br_start..].to_vec();
                    self.correction_bits.clear();
                    self.correction_bits.extend(br_bits);
                    br_start = 0;
                }

                // Emit ZRL for runs of 16
                while run >= 16 {
                    // Emit ZRL
                    let (code, size) = ac_table.get_code(0xF0);
                    self.writer.put_bits(code, size)?;

                    // Output correction bits accumulated so far in this block
                    for &bit in &self.correction_bits[br_start..] {
                        self.writer.put_bits(bit, 1)?;
                    }
                    self.correction_bits.truncate(br_start);
                    run -= 16;
                }

                // Emit the coefficient
                let symbol = ((run as u8) << 4) | 1;
                let (code, size) = ac_table.get_code(symbol);
                self.writer.put_bits(code, size)?;

                // Sign bit (1 for positive, 0 for negative)
                let sign_bit = if coef < 0 { 0u32 } else { 1u32 };
                self.writer.put_bits(sign_bit, 1)?;

                // Output correction bits accumulated so far in this block
                for &bit in &self.correction_bits[br_start..] {
                    self.writer.put_bits(bit, 1)?;
                }
                self.correction_bits.truncate(br_start);
                run = 0;
            } else {
                // Zero coefficient - increment run
                run += 1;

                // If run reaches 16, emit ZRL immediately with correction bits.
                if run == 16 {
                    // Flush EOBRUN first if needed (with only BE bits)
                    if self.eobrun > 0 {
                        self.flush_eobrun_be_bits(ac_table, br_start)?;
                        let br_bits: Vec<u32> = self.correction_bits[br_start..].to_vec();
                        self.correction_bits.clear();
                        self.correction_bits.extend(br_bits);
                        br_start = 0;
                    }

                    let (code, size) = ac_table.get_code(0xF0);
                    self.writer.put_bits(code, size)?;

                    // Output correction bits accumulated so far in this block
                    for &bit in &self.correction_bits[br_start..] {
                        self.writer.put_bits(bit, 1)?;
                    }
                    self.correction_bits.truncate(br_start);
                    run = 0;
                }
            }
        }

        // Handle remaining run (EOB)
        // BR = bits accumulated in this block = correction_bits.len() - br_start
        let br = self.correction_bits.len() - br_start;
        if run > 0 || br > 0 {
            self.eobrun += 1;
            // Correction bits stay in self.correction_bits (they're already there)
            // We force out the EOB if we risk either:
            // 1. overflow of the EOB counter (0x7FFF max)
            // 2. overflow of the correction bit buffer (use 1000 as max like C mozjpeg)
            // 3. EOBRUN is disabled (standard tables)
            if !self.allow_eobrun
                || self.eobrun == 0x7FFF
                || self.correction_bits.len() > (1000 - DCTSIZE2 + 1)
            {
                self.flush_eobrun_with_bits(ac_table)?;
            }
        }

        Ok(())
    }

    /// Flush the EOB run with only BE (previous blocks') correction bits.
    /// BR (current block) bits at indices br_start.. are NOT output.
    fn flush_eobrun_be_bits(
        &mut self,
        ac_table: &DerivedTable,
        br_start: usize,
    ) -> std::io::Result<()> {
        if self.eobrun == 0 {
            return Ok(());
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        let (code, size) = ac_table.get_code(symbol);
        self.writer.put_bits(code, size)?;

        // Output additional bits for EOBRUN
        if nbits > 0 {
            let mask = (1u16 << nbits) - 1;
            let extra = (self.eobrun & mask) as u32;
            self.writer.put_bits(extra, nbits)?;
        }

        self.eobrun = 0;

        // Emit only BE bits (0..br_start), not BR bits
        for &bit in &self.correction_bits[..br_start] {
            self.writer.put_bits(bit, 1)?;
        }
        // Note: don't clear correction_bits here - caller will handle it

        Ok(())
    }

    /// Flush the EOB run with all accumulated correction bits.
    fn flush_eobrun_with_bits(&mut self, ac_table: &DerivedTable) -> std::io::Result<()> {
        if self.eobrun == 0 {
            return Ok(());
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        let (code, size) = ac_table.get_code(symbol);
        self.writer.put_bits(code, size)?;

        // Output additional bits for EOBRUN
        if nbits > 0 {
            let mask = (1u16 << nbits) - 1;
            let extra = (self.eobrun & mask) as u32;
            self.writer.put_bits(extra, nbits)?;
        }

        self.eobrun = 0;

        // Emit all buffered correction bits
        for &bit in &self.correction_bits {
            self.writer.put_bits(bit, 1)?;
        }
        self.correction_bits.clear();

        Ok(())
    }

    /// Flush the EOB run to the bitstream.
    fn flush_eobrun(&mut self, ac_table: &DerivedTable) -> std::io::Result<()> {
        if self.eobrun == 0 {
            return Ok(());
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        // EOB0: EOBRUN=1 (nbits=0)
        // EOB1: EOBRUN=2-3 (nbits=1, 1 extra bit)
        // EOB2: EOBRUN=4-7 (nbits=2, 2 extra bits)
        // etc.
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        let (code, size) = ac_table.get_code(symbol);
        self.writer.put_bits(code, size)?;

        // Output additional bits for EOBRUN
        // Extra bits encode: EOBRUN - 2^nbits = EOBRUN & (2^nbits - 1)
        if nbits > 0 {
            let mask = (1u16 << nbits) - 1;
            let extra = self.eobrun & mask;
            self.writer.put_bits(extra as u32, nbits)?;
        }

        self.eobrun = 0;
        Ok(())
    }

    /// Finish the current scan, flushing any pending EOBRUN and correction bits.
    pub fn finish_scan(&mut self, ac_table: Option<&DerivedTable>) -> std::io::Result<()> {
        if let Some(table) = ac_table {
            // Use flush_eobrun_with_bits to ensure correction bits are emitted
            self.flush_eobrun_with_bits(table)?;
        }
        self.writer.flush()?;
        Ok(())
    }
}

// =============================================================================
// Progressive Symbol Counter (for Huffman optimization)
// =============================================================================

/// Count symbol frequencies for progressive JPEG scans.
///
/// This handles the different symbol sets needed for progressive encoding:
/// - DC scans use standard differential DC symbols
/// - AC scans use EOBRUN symbols (0x00, 0x10, 0x20, etc.) in addition to regular AC symbols
pub struct ProgressiveSymbolCounter {
    /// Last DC value for each component
    last_dc_val: [i16; 4],
    /// Accumulated EOB run count
    eobrun: u16,
}

impl Default for ProgressiveSymbolCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressiveSymbolCounter {
    /// Create a new progressive symbol counter.
    pub fn new() -> Self {
        Self {
            last_dc_val: [0; 4],
            eobrun: 0,
        }
    }

    /// Reset state for a new scan.
    pub fn reset(&mut self) {
        self.last_dc_val = [0; 4];
        self.eobrun = 0;
    }

    /// Count DC symbols for a first scan (Ah=0).
    pub fn count_dc_first(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        al: u8,
        dc_counter: &mut FrequencyCounter,
    ) {
        let dc = block[0] >> al;
        let diff = dc.wrapping_sub(self.last_dc_val[component]);
        self.last_dc_val[component] = dc;

        let nbits = jpeg_nbits(diff);
        dc_counter.count(nbits);
    }

    /// Count AC symbols for a first scan (Ah=0).
    pub fn count_ac_first(
        &mut self,
        block: &[i16; DCTSIZE2],
        ss: u8,
        se: u8,
        al: u8,
        ac_counter: &mut FrequencyCounter,
    ) {
        // Find last non-zero coefficient in this band
        // C mozjpeg: checks (abs(coef) >> Al) != 0
        let mut k = se;
        while k >= ss {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            let abs_shifted = coef.unsigned_abs() >> al;
            if abs_shifted != 0 {
                break;
            }
            k -= 1;
        }
        let kex = k;

        let mut run = 0u32;

        for k in ss..=se {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            // Apply point transform to ABSOLUTE value
            let abs_shifted = coef.unsigned_abs() >> al;

            if abs_shifted == 0 {
                run += 1;
                continue;
            }

            // Flush any pending EOBRUN
            if self.eobrun > 0 {
                self.flush_eobrun_count(ac_counter);
            }

            // Count ZRL codes for runs of 16+ zeros
            while run >= 16 {
                ac_counter.count(0xF0); // ZRL
                run -= 16;
            }

            // Symbol = (run << 4) | nbits
            let nbits = jpeg_nbits_nonzero(abs_shifted);
            let symbol = ((run as u8) << 4) | nbits;
            ac_counter.count(symbol);

            run = 0;

            if k == kex {
                break;
            }
        }

        // Accumulate EOB if we didn't encode all coefficients in the band
        if kex < se {
            self.eobrun += 1;
            if self.eobrun == 0x7FFF {
                self.flush_eobrun_count(ac_counter);
            }
        }
    }

    /// Count AC symbols for a refinement scan (Ah != 0).
    ///
    /// In refinement scans:
    /// - Previously-coded coefficients just output correction bits (not Huffman-coded)
    /// - Newly non-zero coefficients use (run, 1) symbols (always category 1)
    /// - ZRL (0xF0) for runs of 16 zeros
    /// - EOBn symbols for trailing zeros after last new non-zero
    ///
    /// # Arguments
    /// * `block` - DCT coefficients in natural order
    /// * `ss` - Spectral selection start (1..63)
    /// * `se` - Spectral selection end (1..63)
    /// * `ah` - Successive approximation high bit (previous Al)
    /// * `al` - Successive approximation low bit (current precision)
    /// * `ac_counter` - Frequency counter for AC symbols
    pub fn count_ac_refine(
        &mut self,
        block: &[i16; DCTSIZE2],
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        ac_counter: &mut FrequencyCounter,
    ) {
        let mut run = 0u32;

        for k in ss..=se {
            let coef = block[JPEG_NATURAL_ORDER[k as usize]];
            let abs_coef = coef.unsigned_abs();

            // Check if previously coded (matches C mozjpeg's temp > 1 check)
            if (abs_coef >> al) > 1 {
                // Previously coded - no Huffman symbol needed (just correction bit)
                // Correction bits are sent raw, not Huffman coded
                // Don't increment run - previously coded positions don't count as zeros
            } else if (abs_coef >> al) == 1 {
                // Newly non-zero coefficient
                // Flush any pending EOBRUN
                if self.eobrun > 0 {
                    self.flush_eobrun_count(ac_counter);
                }

                // Count ZRL for runs of 16+ zeros
                while run >= 16 {
                    ac_counter.count(0xF0); // ZRL
                    run -= 16;
                }

                // Symbol = (run << 4) | 1 (always category 1 for refinement)
                let symbol = ((run as u8) << 4) | 1;
                ac_counter.count(symbol);

                run = 0;
            } else {
                // Zero coefficient - increment run
                run += 1;

                // Emit ZRL incrementally when run reaches 16
                if run == 16 {
                    ac_counter.count(0xF0); // ZRL
                    run = 0;
                }
            }
        }

        // Handle remaining run (EOB)
        if run > 0 {
            self.eobrun += 1;
            if self.eobrun == 0x7FFF {
                self.flush_eobrun_count(ac_counter);
            }
        }
    }

    /// Flush and count the EOBRUN symbol.
    fn flush_eobrun_count(&mut self, ac_counter: &mut FrequencyCounter) {
        if self.eobrun == 0 {
            return;
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        // EOB0: EOBRUN=1 (nbits=0)
        // EOB1: EOBRUN=2-3 (nbits=1)
        // EOB2: EOBRUN=4-7 (nbits=2)
        // etc.
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        ac_counter.count(symbol);

        self.eobrun = 0;
    }

    /// Finish counting for a scan (flush any pending EOBRUN).
    pub fn finish_scan(&mut self, ac_counter: Option<&mut FrequencyCounter>) {
        if let Some(counter) = ac_counter {
            self.flush_eobrun_count(counter);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::VecBitWriter;
    use crate::consts::{
        AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    };
    use crate::huffman::HuffTable;

    fn create_dc_luma_table() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
        for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, true).unwrap()
    }

    fn create_ac_luma_table() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, false).unwrap()
    }

    #[test]
    fn test_jpeg_nbits() {
        assert_eq!(jpeg_nbits(0), 0);
        assert_eq!(jpeg_nbits(1), 1);
        assert_eq!(jpeg_nbits(-1), 1);
        assert_eq!(jpeg_nbits(2), 2);
        assert_eq!(jpeg_nbits(-2), 2);
        assert_eq!(jpeg_nbits(3), 2);
        assert_eq!(jpeg_nbits(-3), 2);
        assert_eq!(jpeg_nbits(4), 3);
        assert_eq!(jpeg_nbits(7), 3);
        assert_eq!(jpeg_nbits(8), 4);
        assert_eq!(jpeg_nbits(255), 8);
        assert_eq!(jpeg_nbits(-255), 8);
        assert_eq!(jpeg_nbits(1023), 10);
    }

    #[test]
    fn test_encode_zero_block() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        let block = [0i16; DCTSIZE2];
        let new_dc = encode_block_standalone(&mut writer, &block, 0, &dc_table, &ac_table).unwrap();
        writer.flush().unwrap();

        assert_eq!(new_dc, 0);
        // Should just have DC=0 (category 0) and EOB
        let bytes = writer.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_dc_only() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        // Block with only DC = 100, all AC = 0
        let mut block = [0i16; DCTSIZE2];
        block[0] = 100;

        let new_dc = encode_block_standalone(&mut writer, &block, 0, &dc_table, &ac_table).unwrap();
        writer.flush().unwrap();

        assert_eq!(new_dc, 100);
        let bytes = writer.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_dc_differential() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        // First block: DC = 100
        let mut block = [0i16; DCTSIZE2];
        block[0] = 100;

        let mut encoder = EntropyEncoder::new(&mut writer);
        encoder
            .encode_block(&block, 0, &dc_table, &ac_table)
            .unwrap();

        // Second block: DC = 105 (diff = 5)
        block[0] = 105;
        encoder
            .encode_block(&block, 0, &dc_table, &ac_table)
            .unwrap();

        // Third block: DC = 95 (diff = -10)
        block[0] = 95;
        encoder
            .encode_block(&block, 0, &dc_table, &ac_table)
            .unwrap();

        encoder.flush().unwrap();

        assert_eq!(encoder.last_dc(0), 95);
    }

    #[test]
    fn test_encode_with_ac_coefficients() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        // Block with some AC coefficients
        let mut block = [0i16; DCTSIZE2];
        block[0] = 50; // DC
        block[1] = 10; // AC at position 1
        block[8] = -5; // AC at position 8 (zigzag position 2)
        block[16] = 3; // AC at position 16 (zigzag position 3)

        encode_block_standalone(&mut writer, &block, 0, &dc_table, &ac_table).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_progressive_ac_encoding_matches_baseline() {
        let ac_table = create_ac_luma_table();

        // Test block with known AC coefficients
        let mut block = [0i16; DCTSIZE2];
        block[1] = 10; // AC at natural position 1 (zigzag 1)
        block[8] = -5; // AC at natural position 8 (zigzag 2)
        block[16] = 3; // AC at natural position 16 (zigzag 3)
                       // block[0] is DC, not encoded in AC scan

        // Baseline-style AC encoding
        let mut base_writer = VecBitWriter::new_vec();
        {
            let mut run = 0u8;
            for &zigzag_idx in JPEG_NATURAL_ORDER[1..].iter() {
                let coef = block[zigzag_idx];
                if coef == 0 {
                    run += 1;
                } else {
                    while run >= 16 {
                        let (code, size) = ac_table.get_code(0xF0);
                        base_writer.put_bits(code, size).unwrap();
                        run -= 16;
                    }
                    let nbits = jpeg_nbits(coef);
                    let symbol = (run << 4) | nbits;
                    let (code, size) = ac_table.get_code(symbol);
                    base_writer.put_bits(code, size).unwrap();
                    if coef < 0 {
                        let value = (coef as u16).wrapping_sub(1) & ((1u16 << nbits) - 1);
                        base_writer.put_bits(value as u32, nbits).unwrap();
                    } else {
                        base_writer.put_bits(coef as u32, nbits).unwrap();
                    }
                    run = 0;
                }
            }
            if run > 0 {
                let (code, size) = ac_table.get_code(0x00); // EOB
                base_writer.put_bits(code, size).unwrap();
            }
        }
        base_writer.flush().unwrap();
        let base_bytes = base_writer.into_bytes();

        // Progressive-style AC encoding (via ProgressiveEncoder)
        let mut prog_writer = VecBitWriter::new_vec();
        {
            let mut encoder = ProgressiveEncoder::new_standard_tables(&mut prog_writer);
            encoder
                .encode_ac_first(&block, 1, 63, 0, &ac_table)
                .unwrap();
            encoder.finish_scan(Some(&ac_table)).unwrap();
        }
        let prog_bytes = prog_writer.into_bytes();

        // They should produce identical output
        println!("Baseline bytes: {:02X?}", base_bytes);
        println!("Progressive bytes: {:02X?}", prog_bytes);
        assert_eq!(base_bytes, prog_bytes, "AC encoding should be identical");
    }

    #[test]
    fn test_progressive_multiple_blocks_ac_encoding() {
        // Test that progressive AC encoder handles multiple blocks correctly.
        //
        // Progressive encoding uses a continuous bitstream (no flush between blocks),
        // while baseline flushes after each block. This is expected to produce DIFFERENT
        // byte output, but both should decode to the same coefficients.
        //
        // This test validates that multi-block progressive encoding produces valid output
        // (no panics, reasonable size) rather than byte-exact matching.

        let ac_table = create_ac_luma_table();

        // Create 8 blocks with different AC patterns (simulating 2 MCUs of 4 blocks each)
        let mut blocks = [[0i16; DCTSIZE2]; 8];
        blocks[0][1] = 10; // Block 0: some AC
        blocks[1][1] = 20; // Block 1: different AC
        blocks[2][1] = 5; // Block 2
        blocks[3][1] = -5; // Block 3
        blocks[4][1] = 15; // Block 4
        blocks[5][1] = -10; // Block 5
        blocks[6][1] = 8; // Block 6
        blocks[7][1] = -3; // Block 7

        // Encode all 8 blocks with progressive AC encoder
        let mut prog_writer = VecBitWriter::new_vec();
        {
            let mut encoder = ProgressiveEncoder::new_standard_tables(&mut prog_writer);
            for i in 0..8 {
                encoder
                    .encode_ac_first(&blocks[i], 1, 63, 0, &ac_table)
                    .unwrap();
            }
            encoder.finish_scan(Some(&ac_table)).unwrap();
        }
        let prog_bytes = prog_writer.into_bytes();

        // Progressive encoding should produce valid output
        assert!(
            !prog_bytes.is_empty(),
            "Progressive encoding should produce output"
        );

        // The output should be reasonably sized (not empty, not excessively large)
        // Each block with 1 non-zero AC coefficient + EOB should be ~2-3 bytes per block
        // 8 blocks should produce roughly 16-24 bytes (tighter packing than baseline)
        assert!(
            prog_bytes.len() >= 8,
            "Output should have at least 1 byte per block"
        );
        assert!(
            prog_bytes.len() <= 30,
            "Output should not be excessively large"
        );

        println!(
            "Progressive multi-block encoding: {} bytes",
            prog_bytes.len()
        );
    }

    #[test]
    fn test_encode_run_length() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        // Block with zeros followed by a coefficient
        let mut block = [0i16; DCTSIZE2];
        block[0] = 10; // DC
        block[63] = 1; // Last AC coefficient (requires 62 zeros before it)

        encode_block_standalone(&mut writer, &block, 0, &dc_table, &ac_table).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        // Should contain ZRL codes (0xF0) for runs of 16 zeros
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_negative_values() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        // Block with negative coefficients
        let mut block = [0i16; DCTSIZE2];
        block[0] = -50; // DC
        block[1] = -10; // AC
        block[8] = -1; // AC

        encode_block_standalone(&mut writer, &block, 0, &dc_table, &ac_table).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_reset_dc() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        let mut encoder = EntropyEncoder::new(&mut writer);

        // Set DC values
        let mut block = [0i16; DCTSIZE2];
        block[0] = 100;
        encoder
            .encode_block(&block, 0, &dc_table, &ac_table)
            .unwrap();
        assert_eq!(encoder.last_dc(0), 100);

        // Reset
        encoder.reset_dc();
        assert_eq!(encoder.last_dc(0), 0);
    }

    #[test]
    fn test_multiple_components() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();
        let mut writer = VecBitWriter::new_vec();

        let mut encoder = EntropyEncoder::new(&mut writer);

        // Encode Y component
        let mut block = [0i16; DCTSIZE2];
        block[0] = 100;
        encoder
            .encode_block(&block, 0, &dc_table, &ac_table)
            .unwrap();

        // Encode Cb component
        block[0] = 128;
        encoder
            .encode_block(&block, 1, &dc_table, &ac_table)
            .unwrap();

        // Encode Cr component
        block[0] = 130;
        encoder
            .encode_block(&block, 2, &dc_table, &ac_table)
            .unwrap();

        encoder.flush().unwrap();

        // Each component should have its own DC prediction
        assert_eq!(encoder.last_dc(0), 100);
        assert_eq!(encoder.last_dc(1), 128);
        assert_eq!(encoder.last_dc(2), 130);
    }
}
