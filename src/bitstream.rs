//! Bitstream writer for JPEG entropy coding.
//!
//! This module provides bit-level output with:
//! - Efficient bit buffering (64-bit buffer)
//! - Automatic 0xFF byte stuffing (0xFF -> 0xFF 0x00)
//! - Byte-aligned flushing for segment boundaries
//!
//! The writer accumulates bits in a buffer and flushes complete bytes
//! to the output, handling the JPEG requirement that 0xFF bytes be
//! followed by 0x00 to distinguish from markers.

use std::io::Write;

/// Size of the bit buffer in bits
const BIT_BUF_SIZE: u32 = 64;

/// Bitstream writer for JPEG encoding.
///
/// Accumulates bits and writes them to an output buffer with
/// proper byte stuffing for 0xFF bytes.
pub struct BitWriter<W: Write> {
    /// Output destination
    output: W,
    /// Bit accumulation buffer
    put_buffer: u64,
    /// Number of free bits remaining in the buffer
    free_bits: i32,
    /// Total bytes written (for statistics)
    bytes_written: usize,
}

impl<W: Write> BitWriter<W> {
    /// Create a new bitstream writer.
    pub fn new(output: W) -> Self {
        Self {
            output,
            put_buffer: 0,
            free_bits: BIT_BUF_SIZE as i32,
            bytes_written: 0,
        }
    }

    /// Write bits to the bitstream.
    ///
    /// # Arguments
    /// * `code` - The bits to write (right-aligned)
    /// * `size` - Number of bits to write (1-16)
    ///
    /// # Returns
    /// Number of bytes written to output, or error
    #[inline]
    pub fn put_bits(&mut self, code: u32, size: u8) -> std::io::Result<()> {
        debug_assert!(size <= 16, "Size must be <= 16 bits");
        debug_assert!(code < (1 << size), "Code exceeds size bits");

        let size = size as i32;
        self.free_bits -= size;

        if self.free_bits < 0 {
            // Buffer is full, need to flush
            // -free_bits = number of bits that overflow into next buffer
            let overflow_bits = (-self.free_bits) as u32;

            // Put upper bits into current buffer before flush
            self.put_buffer =
                (self.put_buffer << (size + self.free_bits)) | ((code as u64) >> overflow_bits);
            self.flush_buffer()?;

            // Reset buffer with only the overflow (lower) bits
            self.free_bits += BIT_BUF_SIZE as i32;
            // Mask to keep only the lower overflow_bits
            self.put_buffer = (code as u64) & ((1u64 << overflow_bits) - 1);
        } else {
            self.put_buffer = (self.put_buffer << size) | (code as u64);
        }

        Ok(())
    }

    /// Write Huffman code and extra value bits in a single operation.
    ///
    /// This optimizes the common entropy coding pattern where a Huffman code
    /// is immediately followed by additional value bits. Combining them reduces
    /// function call overhead and allows the compiler to optimize better.
    ///
    /// # Arguments
    /// * `code` - Huffman code bits (right-aligned)
    /// * `code_size` - Number of bits in the Huffman code (1-16)
    /// * `value` - Extra value bits (right-aligned)
    /// * `value_size` - Number of extra bits (0-16)
    #[inline]
    pub fn put_bits_combined(
        &mut self,
        code: u32,
        code_size: u8,
        value: u32,
        value_size: u8,
    ) -> std::io::Result<()> {
        debug_assert!(code_size <= 16, "Code size must be <= 16 bits");
        debug_assert!(value_size <= 16, "Value size must be <= 16 bits");

        // Combine code and value: code in high bits, value in low bits
        let total_size = (code_size + value_size) as i32;
        let combined = ((code as u64) << value_size) | (value as u64);

        self.free_bits -= total_size;

        if self.free_bits < 0 {
            // Buffer is full, need to flush
            let overflow_bits = (-self.free_bits) as u32;

            // Put upper bits into current buffer before flush
            self.put_buffer = (self.put_buffer << (total_size + self.free_bits))
                | (combined >> overflow_bits);
            self.flush_buffer()?;

            // Reset buffer with only the overflow (lower) bits
            self.free_bits += BIT_BUF_SIZE as i32;
            self.put_buffer = combined & ((1u64 << overflow_bits) - 1);
        } else {
            self.put_buffer = (self.put_buffer << total_size) | combined;
        }

        Ok(())
    }

    /// Flush the bit buffer to output.
    ///
    /// Writes all complete bytes from the buffer, handling 0xFF stuffing.
    fn flush_buffer(&mut self) -> std::io::Result<()> {
        let buffer = self.put_buffer;

        // Check if any byte might be 0xFF using the SWAR technique
        // This checks if any byte has its high bit set and adding 1 doesn't carry
        if buffer & 0x8080808080808080 & !(buffer.wrapping_add(0x0101010101010101)) != 0 {
            // At least one byte might be 0xFF, emit with stuffing check
            self.emit_byte_stuffed((buffer >> 56) as u8)?;
            self.emit_byte_stuffed((buffer >> 48) as u8)?;
            self.emit_byte_stuffed((buffer >> 40) as u8)?;
            self.emit_byte_stuffed((buffer >> 32) as u8)?;
            self.emit_byte_stuffed((buffer >> 24) as u8)?;
            self.emit_byte_stuffed((buffer >> 16) as u8)?;
            self.emit_byte_stuffed((buffer >> 8) as u8)?;
            self.emit_byte_stuffed(buffer as u8)?;
        } else {
            // No 0xFF bytes, write directly
            let bytes = buffer.to_be_bytes();
            self.output.write_all(&bytes)?;
            self.bytes_written += 8;
        }

        Ok(())
    }

    /// Emit a single byte with 0xFF stuffing.
    #[inline]
    fn emit_byte_stuffed(&mut self, byte: u8) -> std::io::Result<()> {
        self.output.write_all(&[byte])?;
        self.bytes_written += 1;

        if byte == 0xFF {
            // Stuff with 0x00 after 0xFF
            self.output.write_all(&[0x00])?;
            self.bytes_written += 1;
        }

        Ok(())
    }

    /// Flush remaining bits to output, padding with 1s to byte boundary.
    ///
    /// This is called at the end of entropy-coded segments. The padding
    /// uses 1-bits as required by JPEG (to avoid creating false markers).
    pub fn flush(&mut self) -> std::io::Result<()> {
        let bits_in_buffer = (BIT_BUF_SIZE as i32) - self.free_bits;

        if bits_in_buffer > 0 {
            // Pad with 1-bits to fill to byte boundary
            let padding_bits = (8 - (bits_in_buffer % 8)) % 8;
            let total_bits = bits_in_buffer + padding_bits;
            let bytes_to_write = total_bits / 8;

            // Shift the buffer so bits are at the top, then add padding
            let shift_amount = (BIT_BUF_SIZE as i32) - bits_in_buffer;
            let mut buffer = self.put_buffer << shift_amount;

            // Add padding 1-bits at the end
            if padding_bits > 0 {
                let padding_shift = (BIT_BUF_SIZE as i32) - total_bits;
                let padding: u64 = ((1u64 << padding_bits) - 1) << padding_shift;
                buffer |= padding;
            }

            // Write each byte from the top of the buffer
            for i in 0..bytes_to_write {
                let byte = (buffer >> (56 - i * 8)) as u8;
                self.emit_byte_stuffed(byte)?;
            }

            // Reset buffer state
            self.put_buffer = 0;
            self.free_bits = BIT_BUF_SIZE as i32;
        }

        Ok(())
    }

    /// Write a raw byte directly (not bit-stuffed).
    ///
    /// This is used for marker bytes where stuffing is not wanted.
    /// The bit buffer must be byte-aligned before calling this.
    pub fn write_byte(&mut self, byte: u8) -> std::io::Result<()> {
        debug_assert!(
            self.free_bits == BIT_BUF_SIZE as i32,
            "Buffer must be flushed before writing raw bytes"
        );
        self.output.write_all(&[byte])?;
        self.bytes_written += 1;
        Ok(())
    }

    /// Write raw bytes directly (not bit-stuffed).
    ///
    /// Used for marker segments and other non-entropy-coded data.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        debug_assert!(
            self.free_bits == BIT_BUF_SIZE as i32,
            "Buffer must be flushed before writing raw bytes"
        );
        self.output.write_all(bytes)?;
        self.bytes_written += bytes.len();
        Ok(())
    }

    /// Get the number of bytes written so far.
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

/// A simple in-memory bitstream for testing and small outputs.
pub type VecBitWriter = BitWriter<Vec<u8>>;

impl VecBitWriter {
    /// Create a new bitstream writer backed by a Vec.
    pub fn new_vec() -> Self {
        Self::new(Vec::new())
    }

    /// Create a new bitstream writer with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity))
    }

    /// Get the written bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bits() {
        let mut writer = VecBitWriter::new_vec();

        // Write 8 bits (one byte)
        writer.put_bits(0b10101010, 8).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10101010]);
    }

    #[test]
    fn test_multiple_small_writes() {
        let mut writer = VecBitWriter::new_vec();

        // Write several small values
        writer.put_bits(0b11, 2).unwrap(); // 11
        writer.put_bits(0b00, 2).unwrap(); // 00
        writer.put_bits(0b1111, 4).unwrap(); // 1111
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b11001111]);
    }

    #[test]
    fn test_cross_byte_boundary() {
        let mut writer = VecBitWriter::new_vec();

        // Write 12 bits (crosses byte boundary)
        writer.put_bits(0b111100001111, 12).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        // 11110000 1111xxxx (padded with 1s) = 0xF0 0xFF
        // 0xFF requires byte stuffing -> 0xFF 0x00
        assert_eq!(bytes, vec![0xF0, 0xFF, 0x00]);
    }

    #[test]
    fn test_byte_stuffing() {
        let mut writer = VecBitWriter::new_vec();

        // Write 0xFF - should be stuffed
        writer.put_bits(0xFF, 8).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0x00]);
    }

    #[test]
    fn test_no_stuffing_for_non_ff() {
        let mut writer = VecBitWriter::new_vec();

        // Write 0xFE - should NOT be stuffed
        writer.put_bits(0xFE, 8).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFE]);
    }

    #[test]
    fn test_padding_with_ones() {
        let mut writer = VecBitWriter::new_vec();

        // Write 5 bits - should be padded with 3 ones
        writer.put_bits(0b10101, 5).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        // 10101 + 111 padding = 10101111
        assert_eq!(bytes, vec![0b10101111]);
    }

    #[test]
    fn test_large_write() {
        let mut writer = VecBitWriter::new_vec();

        // Write 16 bits
        writer.put_bits(0xABCD, 16).unwrap();
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_many_writes() {
        let mut writer = VecBitWriter::new_vec();

        // Write many bytes to test buffer flushing
        for i in 0..100u32 {
            writer.put_bits(i & 0xFF, 8).unwrap();
        }
        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        // Should have at least 100 bytes (possibly more due to stuffing)
        assert!(bytes.len() >= 100);
    }

    #[test]
    fn test_huffman_like_codes() {
        let mut writer = VecBitWriter::new_vec();

        // Simulate typical Huffman encoding pattern
        // DC: category 3, value 5 -> code + bits
        writer.put_bits(0b100, 3).unwrap(); // 3-bit code
        writer.put_bits(0b101, 3).unwrap(); // 3-bit value

        // AC: EOB (0x00) -> 4-bit code
        writer.put_bits(0b1010, 4).unwrap();

        writer.flush().unwrap();

        let bytes = writer.into_bytes();
        // 100 101 1010 + 11 padding = 10010110 10111111
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_bytes_written_count() {
        let mut writer = VecBitWriter::new_vec();

        writer.put_bits(0xAB, 8).unwrap();
        writer.put_bits(0xFF, 8).unwrap(); // Will be stuffed
        writer.put_bits(0xCD, 8).unwrap();
        writer.flush().unwrap();

        // Should count: AB, FF, 00 (stuffing), CD = 4 bytes
        assert_eq!(writer.bytes_written(), 4);
    }

    #[test]
    fn test_write_raw_bytes() {
        let mut writer = VecBitWriter::new_vec();

        // Write marker (raw bytes, no stuffing)
        writer.write_bytes(&[0xFF, 0xD8]).unwrap(); // SOI marker
        writer.write_bytes(&[0xFF, 0xD9]).unwrap(); // EOI marker

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0xD8, 0xFF, 0xD9]);
    }
}
