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
            self.put_buffer = (self.put_buffer << (size + self.free_bits))
                | ((code as u64) >> overflow_bits);
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

// =============================================================================
// Optimized Vec-based BitWriter (no Write trait indirection)
// =============================================================================

/// Optimized bitstream writer that writes directly to a Vec<u8>.
///
/// This avoids the overhead of the Write trait by:
/// - Direct Vec<u8> access (no trait dispatch)
/// - Infallible operations (no Result returns in hot path)
/// - Batch byte stuffing with pre-reserved capacity
pub struct FastBitWriter {
    /// Output buffer
    output: Vec<u8>,
    /// Bit accumulation buffer
    put_buffer: u64,
    /// Number of free bits remaining in the buffer
    free_bits: i32,
}

impl FastBitWriter {
    /// Create a new fast bitstream writer with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            output: Vec::with_capacity(capacity),
            put_buffer: 0,
            free_bits: BIT_BUF_SIZE as i32,
        }
    }

    /// Write bits to the bitstream.
    ///
    /// This is the hot path - optimized for speed with no error handling.
    #[inline(always)]
    pub fn put_bits(&mut self, code: u32, size: u8) {
        debug_assert!(size <= 16, "Size must be <= 16 bits");
        debug_assert!(code < (1u32 << size), "Code exceeds size bits");

        let size = size as i32;
        self.free_bits -= size;

        if self.free_bits < 0 {
            // Buffer is full, need to flush
            let overflow_bits = (-self.free_bits) as u32;

            // Put upper bits into current buffer before flush
            self.put_buffer = (self.put_buffer << (size + self.free_bits))
                | ((code as u64) >> overflow_bits);
            self.flush_buffer();

            // Reset buffer with only the overflow bits
            self.free_bits += BIT_BUF_SIZE as i32;
            self.put_buffer = (code as u64) & ((1u64 << overflow_bits) - 1);
        } else {
            self.put_buffer = (self.put_buffer << size) | (code as u64);
        }
    }

    /// Flush the 64-bit buffer to output with byte stuffing.
    #[inline]
    fn flush_buffer(&mut self) {
        let buffer = self.put_buffer;

        // Fast path: check if any byte might be 0xFF using SWAR
        // A byte is 0xFF iff it has high bit set and adding 1 causes carry to next byte
        if buffer & 0x8080808080808080 & !(buffer.wrapping_add(0x0101010101010101)) != 0 {
            // Slow path: at least one byte might be 0xFF
            self.emit_bytes_with_stuffing(buffer);
        } else {
            // Fast path: no 0xFF bytes, write all 8 bytes directly
            self.output.extend_from_slice(&buffer.to_be_bytes());
        }
    }

    /// Emit 8 bytes with potential 0xFF stuffing (slow path).
    #[inline(never)]
    fn emit_bytes_with_stuffing(&mut self, buffer: u64) {
        // Reserve space for worst case (all 0xFF = 16 bytes)
        self.output.reserve(16);

        for i in (0..8).rev() {
            let byte = ((buffer >> (i * 8)) & 0xFF) as u8;
            self.output.push(byte);
            if byte == 0xFF {
                self.output.push(0x00);
            }
        }
    }

    /// Flush remaining bits to output, padding with 1s to byte boundary.
    #[inline]
    pub fn flush(&mut self) {
        let bits_in_buffer = (BIT_BUF_SIZE as i32) - self.free_bits;

        if bits_in_buffer > 0 {
            // Pad with 1-bits to fill to byte boundary
            let padding_bits = (8 - (bits_in_buffer % 8)) % 8;
            let total_bits = bits_in_buffer + padding_bits;
            let bytes_to_write = (total_bits / 8) as usize;

            // Shift the buffer so bits are at the top, then add padding
            let shift_amount = (BIT_BUF_SIZE as i32) - bits_in_buffer;
            let mut buffer = self.put_buffer << shift_amount;

            // Add padding 1-bits at the end
            if padding_bits > 0 {
                let padding_shift = (BIT_BUF_SIZE as i32) - total_bits;
                let padding: u64 = ((1u64 << padding_bits) - 1) << padding_shift;
                buffer |= padding;
            }

            // Reserve and write bytes
            self.output.reserve(bytes_to_write * 2); // Worst case for stuffing
            for i in 0..bytes_to_write {
                let byte = (buffer >> (56 - i * 8)) as u8;
                self.output.push(byte);
                if byte == 0xFF {
                    self.output.push(0x00);
                }
            }

            // Reset buffer state
            self.put_buffer = 0;
            self.free_bits = BIT_BUF_SIZE as i32;
        }
    }

    /// Write raw bytes directly (not bit-stuffed).
    /// Buffer must be flushed first.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(
            self.free_bits == BIT_BUF_SIZE as i32,
            "Buffer must be flushed before writing raw bytes"
        );
        self.output.extend_from_slice(bytes);
    }

    /// Get the number of bytes written so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.output.len()
    }

    /// Check if the writer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.output.is_empty()
    }

    /// Consume the writer and return the output bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.output
    }

    /// Get a reference to the output bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.output
    }

    /// Reset the writer for reuse (keeps capacity).
    #[inline]
    pub fn clear(&mut self) {
        self.output.clear();
        self.put_buffer = 0;
        self.free_bits = BIT_BUF_SIZE as i32;
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
        writer.put_bits(0b11, 2).unwrap();  // 11
        writer.put_bits(0b00, 2).unwrap();  // 00
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
        writer.put_bits(0b100, 3).unwrap();  // 3-bit code
        writer.put_bits(0b101, 3).unwrap();  // 3-bit value

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
        writer.put_bits(0xFF, 8).unwrap();  // Will be stuffed
        writer.put_bits(0xCD, 8).unwrap();
        writer.flush().unwrap();

        // Should count: AB, FF, 00 (stuffing), CD = 4 bytes
        assert_eq!(writer.bytes_written(), 4);
    }

    #[test]
    fn test_write_raw_bytes() {
        let mut writer = VecBitWriter::new_vec();

        // Write marker (raw bytes, no stuffing)
        writer.write_bytes(&[0xFF, 0xD8]).unwrap();  // SOI marker
        writer.write_bytes(&[0xFF, 0xD9]).unwrap();  // EOI marker

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0xD8, 0xFF, 0xD9]);
    }

    // =========================================================================
    // FastBitWriter tests
    // =========================================================================

    #[test]
    fn test_fast_basic_bits() {
        let mut writer = FastBitWriter::with_capacity(16);

        // Write 8 bits (one byte)
        writer.put_bits(0b10101010, 8);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10101010]);
    }

    #[test]
    fn test_fast_multiple_small_writes() {
        let mut writer = FastBitWriter::with_capacity(16);

        writer.put_bits(0b11, 2);
        writer.put_bits(0b00, 2);
        writer.put_bits(0b1111, 4);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b11001111]);
    }

    #[test]
    fn test_fast_byte_stuffing() {
        let mut writer = FastBitWriter::with_capacity(16);

        writer.put_bits(0xFF, 8);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0x00]);
    }

    #[test]
    fn test_fast_padding_with_ones() {
        let mut writer = FastBitWriter::with_capacity(16);

        writer.put_bits(0b10101, 5);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10101111]);
    }

    #[test]
    fn test_fast_matches_vec_writer() {
        // Test that FastBitWriter produces identical output to VecBitWriter
        let mut fast = FastBitWriter::with_capacity(1024);
        let mut vec = VecBitWriter::new_vec();

        // Write a variety of bit patterns
        for i in 0..100u32 {
            let size = ((i % 16) + 1) as u8;
            // Mask code to fit in size bits
            let code = (i * 17) & ((1u32 << size) - 1);
            fast.put_bits(code, size);
            vec.put_bits(code, size).unwrap();
        }

        fast.flush();
        vec.flush().unwrap();

        assert_eq!(fast.into_bytes(), vec.into_bytes());
    }

    #[test]
    fn test_fast_many_writes() {
        let mut writer = FastBitWriter::with_capacity(1024);

        for i in 0..100u32 {
            writer.put_bits(i & 0xFF, 8);
        }
        writer.flush();

        let bytes = writer.into_bytes();
        assert!(bytes.len() >= 100);
    }

    #[test]
    fn test_fast_cross_byte_boundary() {
        let mut writer = FastBitWriter::with_capacity(16);

        writer.put_bits(0b111100001111, 12);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xF0, 0xFF, 0x00]);
    }
}
