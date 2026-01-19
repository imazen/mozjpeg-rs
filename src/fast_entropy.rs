//! Fast entropy encoding for JPEG.
//!
//! This module provides optimized Huffman entropy encoding based on jpegli-rs approach:
//! - Owned Vec<u8> output instead of Write trait (no virtual dispatch)
//! - 64-bit bit buffer with delayed flush at 32+ bits
//! - SWAR (SIMD Within A Register) 0xFF detection for fast byte stuffing
//! - Combined code+extra bit writes
//! - Cold annotation on flush path to keep hot path small
//!
//! The encoder uses SIMD to build a 64-bit mask of non-zero coefficients,
//! enabling fast iteration through only the non-zero AC coefficients.

use wide::{i16x8, CmpEq};

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::huffman::DerivedTable;

/// Check if a u64 contains a 0xFF byte using SWAR (SIMD Within A Register).
///
/// XOR with all 1s to find bytes that are 0xFF (they become 0x00),
/// then use the "has zero byte" trick to detect.
#[inline(always)]
const fn has_byte_0xff_u64(v: u64) -> bool {
    let x = v ^ 0xFFFF_FFFF_FFFF_FFFF;
    (((x.wrapping_sub(0x0101_0101_0101_0101)) & !x) & 0x8080_8080_8080_8080) != 0
}

/// Check if a u32 contains a 0xFF byte using SWAR.
#[inline(always)]
const fn has_byte_0xff_u32(v: u32) -> bool {
    let x = v ^ 0xFFFF_FFFF;
    (((x.wrapping_sub(0x0101_0101)) & !x) & 0x8080_8080) != 0
}

/// Fast bit writer optimized for entropy encoding.
///
/// Uses a 64-bit accumulator and only flushes when we have 32+ bits,
/// reducing the number of flush operations in the hot path.
#[derive(Debug)]
pub struct FastBitWriter {
    /// Output buffer (owned, no trait object overhead)
    buffer: Vec<u8>,
    /// Current bit accumulator (64-bit for reduced flush frequency)
    bit_buffer: u64,
    /// Number of bits in accumulator (0-56, we flush at 32+)
    bits_in_buffer: u8,
}

impl FastBitWriter {
    /// Creates a new bit writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Creates a new bit writer with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Writes bits to the stream.
    ///
    /// # Arguments
    /// * `bits` - The bits to write (right-aligned)
    /// * `count` - Number of bits to write (1-24)
    #[inline(always)]
    pub fn write_bits(&mut self, bits: u32, count: u8) {
        debug_assert!(count <= 24);
        debug_assert!(count == 0 || bits < (1u32 << count));

        self.bit_buffer = (self.bit_buffer << count) | (bits as u64);
        self.bits_in_buffer += count;

        // Only flush when we have 32+ bits
        if self.bits_in_buffer >= 32 {
            self.flush_bytes();
        }
    }

    /// Writes Huffman code and extra bits in a single operation.
    ///
    /// This combines two write_bits calls, reducing function call overhead
    /// in the entropy coding hot path.
    #[inline(always)]
    pub fn write_code_and_extra(&mut self, code: u32, code_len: u8, extra: u16, extra_len: u8) {
        debug_assert!(code_len <= 16);
        debug_assert!(extra_len <= 16);
        let total_len = code_len + extra_len;
        debug_assert!(total_len <= 32);

        // Combine: code in high bits, extra in low bits
        let combined = ((code as u64) << extra_len) | (extra as u64);

        self.bit_buffer = (self.bit_buffer << total_len) | combined;
        self.bits_in_buffer += total_len;

        if self.bits_in_buffer >= 32 {
            self.flush_bytes();
        }
    }

    /// Flushes complete bytes from the bit buffer.
    /// Marked cold to keep write_bits hot path small.
    #[inline(never)]
    #[cold]
    fn flush_bytes(&mut self) {
        // Process 64 bits at a time
        while self.bits_in_buffer >= 64 {
            self.bits_in_buffer -= 64;
            let word = self.bit_buffer;
            self.emit_8_bytes(word);
        }

        // Handle 32-56 bits
        while self.bits_in_buffer >= 32 {
            self.bits_in_buffer -= 32;
            let word = (self.bit_buffer >> self.bits_in_buffer) as u32;
            self.emit_4_bytes(word);
        }

        // Handle remaining bytes (0-3)
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.emit_byte(byte);
        }
    }

    /// Emits a single byte with 0xFF stuffing.
    #[inline(always)]
    fn emit_byte(&mut self, byte: u8) {
        self.buffer.push(byte);
        if byte == 0xFF {
            self.buffer.push(0x00);
        }
    }

    /// Emits 4 bytes with 0xFF stuffing using SWAR detection.
    #[inline(always)]
    fn emit_4_bytes(&mut self, word: u32) {
        if !has_byte_0xff_u32(word) {
            // Fast path: no 0xFF bytes, write directly as big-endian
            self.buffer.extend_from_slice(&word.to_be_bytes());
        } else {
            // Slow path: emit byte-by-byte
            self.emit_byte((word >> 24) as u8);
            self.emit_byte((word >> 16) as u8);
            self.emit_byte((word >> 8) as u8);
            self.emit_byte(word as u8);
        }
    }

    /// Emits 8 bytes with 0xFF stuffing using SWAR detection.
    #[inline(always)]
    fn emit_8_bytes(&mut self, word: u64) {
        if !has_byte_0xff_u64(word) {
            // Fast path: no 0xFF bytes
            self.buffer.extend_from_slice(&word.to_be_bytes());
        } else {
            // Slow path: emit byte-by-byte
            self.emit_byte((word >> 56) as u8);
            self.emit_byte((word >> 48) as u8);
            self.emit_byte((word >> 40) as u8);
            self.emit_byte((word >> 32) as u8);
            self.emit_byte((word >> 24) as u8);
            self.emit_byte((word >> 16) as u8);
            self.emit_byte((word >> 8) as u8);
            self.emit_byte(word as u8);
        }
    }

    /// Writes a single byte directly (no bit stuffing).
    #[inline]
    pub fn write_byte_raw(&mut self, byte: u8) {
        self.buffer.push(byte);
    }

    /// Writes bytes directly (no bit stuffing).
    pub fn write_bytes_raw(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Flushes any remaining bits, padding with 1s.
    pub fn flush(&mut self) {
        // First flush any complete bytes
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }

        // Pad remaining bits with 1s
        if self.bits_in_buffer > 0 {
            let padding = 8 - self.bits_in_buffer;
            let padded = (self.bit_buffer << padding) | ((1u64 << padding) - 1);
            let byte = padded as u8;
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Returns the accumulated bytes, consuming the writer.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Returns a reference to the current buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns the current byte position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for FastBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Lookup table for jpeg_nbits - maps absolute value to number of bits.
/// Covers values 0-255, which is the common case for quantized coefficients.
static NBITS_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 1u16;
    while i < 256 {
        table[i as usize] = 16 - i.leading_zeros() as u8;
        i += 1;
    }
    table
};

/// Calculate the number of bits needed to represent a value (category).
/// Uses a lookup table for values in [-255, 255], falls back to leading_zeros for larger.
#[inline(always)]
fn jpeg_nbits(value: i16) -> u8 {
    let abs_val = value.unsigned_abs();
    if abs_val < 256 {
        NBITS_TABLE[abs_val as usize]
    } else {
        16 - abs_val.leading_zeros() as u8
    }
}

/// Calculate additional bits for JPEG encoding.
/// For negative values, returns the one's complement representation.
#[inline(always)]
fn additional_bits(value: i16, nbits: u8) -> u16 {
    if value < 0 {
        (value as u16).wrapping_sub(1) & ((1u16 << nbits) - 1)
    } else {
        value as u16
    }
}

/// Build a 64-bit mask of non-zero coefficients IN ZIGZAG ORDER.
/// Each bit k is set if coeffs[JPEG_NATURAL_ORDER[k]] != 0.
/// This allows using tzcnt to find the next non-zero in zigzag order.
#[inline(always)]
fn build_nonzero_mask_zigzag(coeffs: &[i16; DCTSIZE2]) -> u64 {
    let zero = i16x8::ZERO;
    let mut nonzero_mask: u64 = 0;

    // Process 8 zigzag positions at a time
    for chunk in 0..8 {
        let base = chunk * 8;
        // Load coefficients in zigzag order
        let v = i16x8::new([
            coeffs[JPEG_NATURAL_ORDER[base]],
            coeffs[JPEG_NATURAL_ORDER[base + 1]],
            coeffs[JPEG_NATURAL_ORDER[base + 2]],
            coeffs[JPEG_NATURAL_ORDER[base + 3]],
            coeffs[JPEG_NATURAL_ORDER[base + 4]],
            coeffs[JPEG_NATURAL_ORDER[base + 5]],
            coeffs[JPEG_NATURAL_ORDER[base + 6]],
            coeffs[JPEG_NATURAL_ORDER[base + 7]],
        ]);
        let is_zero = v.simd_eq(zero);
        let zero_bits = is_zero.to_bitmask() as u8;
        let nonzero_bits = !zero_bits;
        nonzero_mask |= (nonzero_bits as u64) << base;
    }

    nonzero_mask
}

/// Fast entropy encoder for baseline JPEG.
///
/// Uses borrowed Huffman tables and an owned bit writer for maximum performance.
pub struct FastEntropyEncoder {
    /// Bit writer (owned for performance)
    writer: FastBitWriter,
    /// Previous DC values for each component
    prev_dc: [i16; 4],
}

impl FastEntropyEncoder {
    /// Creates a new entropy encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: FastBitWriter::new(),
            prev_dc: [0; 4],
        }
    }

    /// Creates a new entropy encoder with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            writer: FastBitWriter::with_capacity(capacity),
            prev_dc: [0; 4],
        }
    }

    /// Resets DC prediction (for restart markers).
    pub fn reset_dc(&mut self) {
        self.prev_dc = [0; 4];
    }

    /// Encodes a block of DCT coefficients.
    ///
    /// Uses SIMD to quickly find non-zero coefficients and skip runs of zeros.
    #[inline]
    pub fn encode_block(
        &mut self,
        coeffs: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
    ) {
        // Encode DC coefficient
        let dc = coeffs[0];
        let dc_diff = dc.wrapping_sub(self.prev_dc[component]);
        self.prev_dc[component] = dc;

        let dc_nbits = jpeg_nbits(dc_diff);
        let (dc_code, dc_len) = dc_table.get_code(dc_nbits);

        if dc_nbits > 0 {
            let dc_extra = additional_bits(dc_diff, dc_nbits);
            self.writer.write_code_and_extra(dc_code, dc_len, dc_extra, dc_nbits);
        } else {
            self.writer.write_bits(dc_code, dc_len);
        }

        // Simple linear iteration through AC coefficients
        // This avoids the overhead of building a zigzag mask for all blocks
        self.encode_ac_linear(coeffs, ac_table);
    }

    /// Encode AC coefficients using linear iteration.
    /// This is the simplest and most predictable path.
    #[inline(always)]
    fn encode_ac_linear(&mut self, coeffs: &[i16; DCTSIZE2], ac_table: &DerivedTable) {
        let mut run = 0u8;

        for &zigzag_idx in JPEG_NATURAL_ORDER[1..].iter() {
            let coef = coeffs[zigzag_idx];

            if coef == 0 {
                run += 1;
            } else {
                while run >= 16 {
                    let (zrl_code, zrl_len) = ac_table.get_code(0xF0);
                    self.writer.write_bits(zrl_code, zrl_len);
                    run -= 16;
                }

                let nbits = jpeg_nbits(coef);
                let symbol = (run << 4) | nbits;
                let (code, len) = ac_table.get_code(symbol);
                let extra = additional_bits(coef, nbits);
                self.writer.write_code_and_extra(code, len, extra, nbits);

                run = 0;
            }
        }

        if run > 0 {
            let (eob_code, eob_len) = ac_table.get_code(0x00);
            self.writer.write_bits(eob_code, eob_len);
        }
    }

    /// Encode AC coefficients using tzcnt for sparse blocks.
    #[inline(always)]
    fn encode_ac_sparse(
        &mut self,
        coeffs: &[i16; DCTSIZE2],
        mut ac_mask: u64,
        ac_table: &DerivedTable,
    ) {
        let mut prev_zigzag_pos = 0u32;

        while ac_mask != 0 {
            let zigzag_pos = ac_mask.trailing_zeros();
            let run = (zigzag_pos - prev_zigzag_pos - 1) as u8;

            // Emit ZRL codes for runs of 16+ zeros
            let mut remaining_run = run;
            while remaining_run >= 16 {
                let (zrl_code, zrl_len) = ac_table.get_code(0xF0);
                self.writer.write_bits(zrl_code, zrl_len);
                remaining_run -= 16;
            }

            let coef = coeffs[JPEG_NATURAL_ORDER[zigzag_pos as usize]];
            let nbits = jpeg_nbits(coef);
            let symbol = (remaining_run << 4) | nbits;
            let (code, len) = ac_table.get_code(symbol);
            let extra = additional_bits(coef, nbits);
            self.writer.write_code_and_extra(code, len, extra, nbits);

            prev_zigzag_pos = zigzag_pos;
            ac_mask &= ac_mask - 1;
        }

        if prev_zigzag_pos < 63 {
            let (eob_code, eob_len) = ac_table.get_code(0x00);
            self.writer.write_bits(eob_code, eob_len);
        }
    }

    /// Encode AC coefficients using linear iteration for dense blocks.
    #[inline(always)]
    fn encode_ac_dense(&mut self, coeffs: &[i16; DCTSIZE2], ac_table: &DerivedTable) {
        let mut run = 0u8;

        for &zigzag_idx in JPEG_NATURAL_ORDER[1..].iter() {
            let coef = coeffs[zigzag_idx];

            if coef == 0 {
                run += 1;
            } else {
                while run >= 16 {
                    let (zrl_code, zrl_len) = ac_table.get_code(0xF0);
                    self.writer.write_bits(zrl_code, zrl_len);
                    run -= 16;
                }

                let nbits = jpeg_nbits(coef);
                let symbol = (run << 4) | nbits;
                let (code, len) = ac_table.get_code(symbol);
                let extra = additional_bits(coef, nbits);
                self.writer.write_code_and_extra(code, len, extra, nbits);

                run = 0;
            }
        }

        if run > 0 {
            let (eob_code, eob_len) = ac_table.get_code(0x00);
            self.writer.write_bits(eob_code, eob_len);
        }
    }

    /// Emits a restart marker and resets DC prediction.
    pub fn emit_restart(&mut self, restart_num: u8) {
        self.writer.flush();
        self.writer.write_bytes_raw(&[0xFF, 0xD0 + (restart_num & 0x07)]);
        self.reset_dc();
    }

    /// Flushes any remaining bits to the output.
    pub fn flush(&mut self) {
        self.writer.flush();
    }

    /// Returns the encoded bytes, consuming the encoder.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.writer.into_bytes()
    }

    /// Returns a reference to the current buffer.
    pub fn as_bytes(&self) -> &[u8] {
        self.writer.as_bytes()
    }

    /// Returns the current byte position.
    pub fn position(&self) -> usize {
        self.writer.position()
    }

    /// Gets the last DC value for a component.
    pub fn last_dc(&self, component: usize) -> i16 {
        self.prev_dc[component]
    }

    /// Sets the last DC value for a component.
    pub fn set_last_dc(&mut self, component: usize, value: i16) {
        self.prev_dc[component] = value;
    }
}

impl Default for FastEntropyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_fast_bit_writer_basic() {
        let mut writer = FastBitWriter::new();
        writer.write_bits(0b10101010, 8);
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10101010]);
    }

    #[test]
    fn test_fast_bit_writer_byte_stuffing() {
        let mut writer = FastBitWriter::new();
        writer.write_bits(0xFF, 8);
        let bytes = writer.into_bytes();
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x00);
    }

    #[test]
    fn test_has_byte_0xff() {
        assert!(!has_byte_0xff_u64(0x0102030405060708));
        assert!(has_byte_0xff_u64(0x01020304FF060708));
        assert!(has_byte_0xff_u64(0xFF02030405060708));
        assert!(has_byte_0xff_u64(0x010203040506FF08));

        assert!(!has_byte_0xff_u32(0x01020304));
        assert!(has_byte_0xff_u32(0xFF020304));
        assert!(has_byte_0xff_u32(0x0102FF04));
    }

    #[test]
    fn test_fast_encoder_zero_block() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();

        let mut encoder = FastEntropyEncoder::new();
        let block = [0i16; DCTSIZE2];
        encoder.encode_block(&block, 0, &dc_table, &ac_table);

        let bytes = encoder.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_fast_encoder_dc_only() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();

        let mut encoder = FastEntropyEncoder::new();
        let mut block = [0i16; DCTSIZE2];
        block[0] = 100;
        encoder.encode_block(&block, 0, &dc_table, &ac_table);

        assert_eq!(encoder.last_dc(0), 100);
        let bytes = encoder.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_fast_encoder_matches_standard() {
        // Test that FastEntropyEncoder produces identical output to EntropyEncoder
        use crate::bitstream::VecBitWriter;
        use crate::entropy::EntropyEncoder;

        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();

        // Test block with various coefficients
        let mut block = [0i16; DCTSIZE2];
        block[0] = 50; // DC
        block[1] = 10; // AC
        block[8] = -5; // AC
        block[16] = 3; // AC
        block[63] = 1; // Last AC

        // Encode with standard encoder
        let mut std_writer = VecBitWriter::new_vec();
        {
            let mut std_encoder = EntropyEncoder::new(&mut std_writer);
            std_encoder.encode_block(&block, 0, &dc_table, &ac_table).unwrap();
            std_encoder.flush().unwrap();
        }
        let std_bytes = std_writer.into_bytes();

        // Encode with fast encoder
        let mut fast_encoder = FastEntropyEncoder::new();
        fast_encoder.encode_block(&block, 0, &dc_table, &ac_table);
        let fast_bytes = fast_encoder.into_bytes();

        assert_eq!(std_bytes, fast_bytes, "Fast encoder should produce identical output");
    }

    #[test]
    fn test_fast_encoder_multiple_blocks() {
        let dc_table = create_dc_luma_table();
        let ac_table = create_ac_luma_table();

        let mut encoder = FastEntropyEncoder::new();

        // Encode multiple blocks
        for i in 0..10 {
            let mut block = [0i16; DCTSIZE2];
            block[0] = (i * 10) as i16;
            block[1] = ((i % 5) as i16) - 2;
            encoder.encode_block(&block, 0, &dc_table, &ac_table);
        }

        let bytes = encoder.into_bytes();
        assert!(!bytes.is_empty());
    }
}
