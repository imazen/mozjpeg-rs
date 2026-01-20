//! SSE2/AVX2 optimized Huffman entropy encoding.
//!
//! Port of libjpeg-turbo's jchuff-sse2.asm to Rust with intrinsics.
//!
//! Key optimizations:
//! 1. Fused zigzag reorder + sign handling in SIMD
//! 2. 64-bit non-zero mask built during zigzag reorder
//! 3. tzcnt-based iteration through non-zero coefficients
//! 4. 64KB lookup table for nbits (avoids leading_zeros at runtime)
//!
//! Reference: libjpeg-turbo/simd/x86_64/jchuff-sse2.asm

// Note: unsafe code is needed for #[target_feature] functions which must be
// declared unsafe in stable Rust. The actual SIMD operations use archmage's
// safe wrappers where possible.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use archmage::tokens::x86::Sse2Token;

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::huffman::DerivedTable;

/// 64KB lookup table for nbits.
/// Maps value (0..65536) to number of bits needed to encode it.
/// This avoids calling leading_zeros() at runtime.
static NBITS_TABLE: [u8; 65536] = {
    let mut table = [0u8; 65536];
    let mut i = 1u32;
    let mut nbits = 1u8;
    while nbits <= 16 {
        let end = 1u32 << nbits;
        while i < end && i < 65536 {
            table[i as usize] = nbits;
            i += 1;
        }
        nbits += 1;
    }
    table
};

/// Mask bits table for extracting lower N bits.
static MASK_BITS: [u32; 17] = [
    0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff, 0x01ff, 0x03ff, 0x07ff,
    0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff,
];

/// Get the number of bits needed to encode a value using lookup table.
#[inline(always)]
fn jpeg_nbits_fast(value: i16) -> u8 {
    NBITS_TABLE[value.unsigned_abs() as usize]
}

// JPEG_NATURAL_ORDER from consts.rs maps zigzag position -> natural position
// i.e., to get coefficient at zigzag position i, read block[JPEG_NATURAL_ORDER[i]]

/// SIMD entropy encoder state.
pub struct SimdEntropyEncoder {
    /// Output buffer
    buffer: Vec<u8>,
    /// Bit accumulation buffer (64-bit)
    put_buffer: u64,
    /// Number of free bits in the buffer (starts at 64)
    free_bits: i32,
    /// Last DC value for each component
    last_dc_val: [i16; 4],
}

impl SimdEntropyEncoder {
    /// Create a new SIMD entropy encoder.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            put_buffer: 0,
            free_bits: 64,
            last_dc_val: [0; 4],
        }
    }

    /// Reset DC predictions (called at restart markers).
    pub fn reset_dc(&mut self) {
        self.last_dc_val = [0; 4];
    }

    /// Emit a restart marker and reset DC predictions.
    ///
    /// Flushes the bit buffer (padding with 1-bits to byte boundary),
    /// writes the RST marker (0xFFD0 + restart_num), and resets DC predictions.
    pub fn emit_restart(&mut self, restart_num: u8) {
        // Flush bit buffer (pads with 1-bits to byte boundary)
        let bits_in_buffer = 64 - self.free_bits;
        if bits_in_buffer > 0 {
            let padding_bits = (8 - (bits_in_buffer % 8)) % 8;
            let total_bits = bits_in_buffer + padding_bits;
            let bytes_to_write = total_bits / 8;

            let mut buffer = self.put_buffer << (64 - bits_in_buffer);
            if padding_bits > 0 {
                let padding = ((1u64 << padding_bits) - 1) << (64 - total_bits);
                buffer |= padding;
            }

            for i in 0..bytes_to_write {
                let byte = (buffer >> (56 - i * 8)) as u8;
                self.buffer.push(byte);
                if byte == 0xFF {
                    self.buffer.push(0x00);
                }
            }
        }

        // Reset bit buffer
        self.put_buffer = 0;
        self.free_bits = 64;

        // Write RST marker: 0xFF followed by 0xD0 + (restart_num % 8)
        let rst_marker = 0xD0 + (restart_num & 0x07);
        self.buffer.push(0xFF);
        self.buffer.push(rst_marker);

        // Reset DC predictions
        self.reset_dc();
    }

    /// Get current output size (for progress tracking).
    pub fn output_len(&self) -> usize {
        self.buffer.len()
    }

    /// Encode a single 8x8 block of DCT coefficients using SIMD.
    ///
    /// This is the safe wrapper that checks for SSE2 and dispatches to the
    /// SIMD implementation. On x86_64, SSE2 is always available (baseline).
    #[inline]
    pub fn encode_block(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
    ) {
        use archmage::SimdToken;
        // SSE2 is baseline on x86_64, so this should always succeed
        if let Some(_token) = Sse2Token::try_new() {
            // SAFETY: Token proves SSE2 is available
            unsafe {
                self.encode_block_sse2(block, component, dc_table, ac_table);
            }
        } else {
            // Fallback for non-x86_64 or hypothetical no-SSE2 scenarios
            // (This branch is effectively dead code on x86_64)
            self.encode_block_scalar(block, component, dc_table, ac_table);
        }
    }

    /// Scalar fallback for block encoding.
    fn encode_block_scalar(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
    ) {
        // Scalar zigzag reorder + sign handling
        let mut temp = [0i16; DCTSIZE2];
        for (zigzag_pos, &natural_pos) in JPEG_NATURAL_ORDER.iter().enumerate() {
            let value = block[natural_pos];
            // Sign handling: for negative values, compute value - 1
            temp[zigzag_pos] = if value < 0 { value - 1 } else { value };
        }

        // Encode DC
        self.encode_dc_fast(temp[0], component, dc_table);

        // Build non-zero mask
        let mut nonzero_mask: u64 = 0;
        for (i, &v) in temp.iter().enumerate() {
            if v != 0 {
                nonzero_mask |= 1u64 << i;
            }
        }

        // Encode AC using tzcnt iteration
        self.encode_ac_tzcnt(&temp, nonzero_mask, ac_table);
    }

    /// Encode a single 8x8 block of DCT coefficients using SIMD.
    ///
    /// # Safety
    /// Requires SSE2 support.
    #[target_feature(enable = "sse2")]
    unsafe fn encode_block_sse2(
        &mut self,
        block: &[i16; DCTSIZE2],
        component: usize,
        dc_table: &DerivedTable,
        ac_table: &DerivedTable,
    ) {
        // Step 1: Reorder to zigzag and handle sign, build non-zero mask
        let mut temp = [0i16; DCTSIZE2];
        let nonzero_mask = self.zigzag_reorder_and_sign_sse2(block, &mut temp);

        // Step 2: Encode DC coefficient
        self.encode_dc_fast(temp[0], component, dc_table);

        // Step 3: Encode AC coefficients using tzcnt-based iteration
        self.encode_ac_tzcnt(&temp, nonzero_mask, ac_table);
    }

    /// Reorder coefficients to zigzag order with sign handling using SSE2.
    ///
    /// Returns a 64-bit mask where bit i is set if temp[i] != 0.
    ///
    /// Sign handling: for negative values, we compute `value - 1` which gives
    /// the JPEG-format encoding where negative values are represented as
    /// (value - 1) & mask.
    #[target_feature(enable = "sse2")]
    unsafe fn zigzag_reorder_and_sign_sse2(
        &self,
        block: &[i16; DCTSIZE2],
        temp: &mut [i16; DCTSIZE2],
    ) -> u64 {
        let zero = _mm_setzero_si128();
        let mut nonzero_mask: u64 = 0;

        // Process in chunks of 8 coefficients
        // For each zigzag position, load the coefficient from natural order,
        // handle sign, and build the non-zero mask.

        // The C version uses complex shuffles to rearrange in SIMD.
        // For this Rust port, we'll use a simpler approach that still benefits
        // from SIMD for the sign handling and mask building.

        for chunk in 0..8 {
            let base = chunk * 8;

            // Load 8 coefficients in zigzag order
            // JPEG_NATURAL_ORDER[i] gives the natural position for zigzag position i
            let idx0 = JPEG_NATURAL_ORDER[base];
            let idx1 = JPEG_NATURAL_ORDER[base + 1];
            let idx2 = JPEG_NATURAL_ORDER[base + 2];
            let idx3 = JPEG_NATURAL_ORDER[base + 3];
            let idx4 = JPEG_NATURAL_ORDER[base + 4];
            let idx5 = JPEG_NATURAL_ORDER[base + 5];
            let idx6 = JPEG_NATURAL_ORDER[base + 6];
            let idx7 = JPEG_NATURAL_ORDER[base + 7];

            // Create SIMD vector from gathered coefficients
            // _mm_set_epi16 takes args in reverse order (highest element first)
            let values = _mm_set_epi16(
                block[idx7],
                block[idx6],
                block[idx5],
                block[idx4],
                block[idx3],
                block[idx2],
                block[idx1],
                block[idx0],
            );

            // Sign handling: for negative values, compute value + (value < 0 ? -1 : 0)
            // which is equivalent to value - 1 for negatives, value for positives
            let sign_mask = _mm_cmpgt_epi16(zero, values); // -1 where value < 0
            let adjusted = _mm_add_epi16(values, sign_mask); // value += sign_mask

            // Store the adjusted values - use bytemuck for safe type conversion
            let adjusted_arr: [i16; 8] = bytemuck::cast(adjusted);
            temp[base..base + 8].copy_from_slice(&adjusted_arr);

            // Build non-zero mask: compare with zero, pack to bytes
            let is_zero = _mm_cmpeq_epi16(adjusted, zero); // -1 where value == 0
            let mask_bits = _mm_movemask_epi8(is_zero) as u32;

            // Each i16 becomes 2 bytes in movemask, we need 1 bit per i16
            // So we take bits 0, 2, 4, 6, 8, 10, 12, 14 and compress them
            let compressed = (mask_bits & 0x0001)
                | ((mask_bits & 0x0004) >> 1)
                | ((mask_bits & 0x0010) >> 2)
                | ((mask_bits & 0x0040) >> 3)
                | ((mask_bits & 0x0100) >> 4)
                | ((mask_bits & 0x0400) >> 5)
                | ((mask_bits & 0x1000) >> 6)
                | ((mask_bits & 0x4000) >> 7);

            // is_zero gives -1 for zeros, so invert to get non-zero mask
            let nonzero_bits = (!compressed) & 0xFF;
            nonzero_mask |= (nonzero_bits as u64) << base;
        }

        nonzero_mask
    }

    /// Encode DC coefficient using fast lookup.
    #[inline]
    fn encode_dc_fast(&mut self, dc: i16, component: usize, dc_table: &DerivedTable) {
        let diff = dc.wrapping_sub(self.last_dc_val[component]);
        self.last_dc_val[component] = dc;

        // Handle sign: for negative values, encode (diff - 1) masked
        let (nbits, value) = if diff < 0 {
            let nbits = jpeg_nbits_fast(diff);
            let value = (diff as u16).wrapping_sub(1) & MASK_BITS[nbits as usize] as u16;
            (nbits, value as u32)
        } else if diff > 0 {
            let nbits = jpeg_nbits_fast(diff);
            (nbits, diff as u32)
        } else {
            (0, 0)
        };

        // Emit Huffman code and value
        let (code, code_size) = dc_table.get_code(nbits);
        let total_bits = code_size + nbits;
        let combined = ((code as u64) << nbits) | (value as u64);

        self.put_bits_fast(combined, total_bits);
    }

    /// Encode AC coefficients using tzcnt-based iteration.
    ///
    /// This is the core optimization from jchuff-sse2: instead of iterating
    /// through all 63 AC positions, we use trailing_zeros to jump to the
    /// next non-zero coefficient.
    #[inline]
    fn encode_ac_tzcnt(
        &mut self,
        temp: &[i16; DCTSIZE2],
        nonzero_mask: u64,
        ac_table: &DerivedTable,
    ) {
        // Clear DC bit (bit 0), keep only AC bits (1-63)
        let mut index = nonzero_mask & !1u64;

        // Fast path: all AC coefficients are zero
        if index == 0 {
            let (code, size) = ac_table.get_code(0x00); // EOB
            self.put_bits_fast(code as u64, size);
            return;
        }

        let mut last_pos = 0i32; // Track position for run-length calculation

        while index != 0 {
            // Find the next non-zero coefficient using tzcnt
            let pos = index.trailing_zeros() as i32;

            // Calculate run length (number of zeros since last non-zero)
            let run = pos - last_pos - 1;

            // Emit ZRL codes for runs >= 16
            let mut remaining_run = run;
            while remaining_run >= 16 {
                let (code, size) = ac_table.get_code(0xF0); // ZRL
                self.put_bits_fast(code as u64, size);
                remaining_run -= 16;
            }

            // Get the coefficient value (already sign-adjusted in temp)
            let coef = temp[pos as usize];

            // For sign-adjusted values:
            // - positive: nbits = NBITS(coef), value = coef
            // - negative (was adjusted to coef-1): nbits = NBITS(|original|), value = coef (masked)
            // We need to recover the original absolute value for nbits lookup
            let abs_val = if coef >= 0 {
                coef as u16
            } else {
                // coef was (original - 1), so |original| = |coef + 1| = -coef - 1 + 1 = -coef
                // Actually for negative original, coef = original - 1
                // So -original = -coef - 1, meaning |original| = -coef - 1...
                // Hmm, let me reconsider.
                // For negative original x (x < 0):
                //   coef = x + (-1) = x - 1 (since sign_mask is -1)
                // We want |x|. Since x < 0, |x| = -x.
                // coef = x - 1, so x = coef + 1, and |x| = -(coef + 1) = -coef - 1
                // But coef is also negative (coef = x - 1 < x < 0).
                // Wait, that's not right either.
                //
                // Let's trace through: original = -5
                // sign_mask = -1 (since original < 0)
                // adjusted = original + sign_mask = -5 + (-1) = -6
                // So coef = -6.
                //
                // For encoding: we need nbits = NBITS(|-5|) = NBITS(5) = 3
                // And value = (-5 - 1) & 0b111 = -6 & 0b111 = 0b010 = 2
                //
                // From coef = -6, we can get |original| = |coef + 1| = |-5| = 5
                // But coef + 1 = -5 which is negative, so |coef + 1| = -(coef + 1) = -coef - 1 = 6 - 1 = 5. Yes!
                (-(coef as i32) - 1) as u16
            };

            let nbits = NBITS_TABLE[abs_val as usize];
            let value = (coef as u32) & MASK_BITS[nbits as usize];

            // Symbol = (run << 4) | nbits
            let symbol = ((remaining_run as u8) << 4) | nbits;
            let (code, code_size) = ac_table.get_code(symbol);

            // Emit code and value
            let total_bits = code_size + nbits;
            let combined = ((code as u64) << nbits) | (value as u64);
            self.put_bits_fast(combined, total_bits);

            // Update position and clear this bit
            last_pos = pos;
            index &= index - 1; // Clear lowest set bit
        }

        // If last non-zero wasn't at position 63, emit EOB
        if last_pos < 63 {
            let (code, size) = ac_table.get_code(0x00); // EOB
            self.put_bits_fast(code as u64, size);
        }
    }

    /// Put bits into the buffer with fast path for common case.
    #[inline(always)]
    fn put_bits_fast(&mut self, bits: u64, size: u8) {
        let size = size as i32;
        self.free_bits -= size;

        if self.free_bits >= 0 {
            // Fast path: bits fit in buffer
            self.put_buffer = (self.put_buffer << size) | bits;
        } else {
            // Slow path: need to flush
            let overflow = (-self.free_bits) as u32;
            self.put_buffer = (self.put_buffer << (size + self.free_bits)) | (bits >> overflow);
            self.flush_buffer();
            self.free_bits += 64;
            self.put_buffer = bits & ((1u64 << overflow) - 1);
        }
    }

    /// Flush the 64-bit buffer to output with byte stuffing.
    fn flush_buffer(&mut self) {
        let buffer = self.put_buffer.to_be_bytes();

        // Check for 0xFF bytes using SWAR
        let has_ff = self.put_buffer
            & 0x8080808080808080
            & !(self.put_buffer.wrapping_add(0x0101010101010101))
            != 0;

        if has_ff {
            // Slow path: check each byte for 0xFF
            for &byte in &buffer {
                self.buffer.push(byte);
                if byte == 0xFF {
                    self.buffer.push(0x00);
                }
            }
        } else {
            // Fast path: no 0xFF bytes
            self.buffer.extend_from_slice(&buffer);
        }
    }

    /// Flush remaining bits and return the encoded data.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush_to_buffer();
        self.buffer
    }

    /// Flush remaining bits to the internal buffer (without consuming self).
    pub fn flush(&mut self) {
        self.flush_to_buffer();
    }

    /// Internal: flush remaining bits to buffer.
    fn flush_to_buffer(&mut self) {
        let bits_in_buffer = 64 - self.free_bits;

        if bits_in_buffer > 0 {
            // Pad with 1-bits to byte boundary
            let padding_bits = (8 - (bits_in_buffer % 8)) % 8;
            let total_bits = bits_in_buffer + padding_bits;
            let bytes_to_write = total_bits / 8;

            // Shift buffer so bits are at the top
            let mut buffer = self.put_buffer << (64 - bits_in_buffer);

            // Add padding 1-bits
            if padding_bits > 0 {
                let padding = ((1u64 << padding_bits) - 1) << (64 - total_bits);
                buffer |= padding;
            }

            // Write each byte with stuffing
            for i in 0..bytes_to_write {
                let byte = (buffer >> (56 - i * 8)) as u8;
                self.buffer.push(byte);
                if byte == 0xFF {
                    self.buffer.push(0x00);
                }
            }

            // Clear the bit buffer
            self.put_buffer = 0;
            self.free_bits = 64;
        }
    }

    /// Get the current output buffer for inspection.
    pub fn get_buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Take ownership of the output buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Write the encoded data to the given output and clear the internal buffer.
    pub fn write_to<W: std::io::Write>(&mut self, output: &mut W) -> std::io::Result<()> {
        output.write_all(&self.buffer)?;
        self.buffer.clear();
        Ok(())
    }
}

impl Default for SimdEntropyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbits_table() {
        // Verify the lookup table matches the formula
        for i in 0u16..=65535 {
            let expected = if i == 0 {
                0
            } else {
                16 - i.leading_zeros() as u8
            };
            assert_eq!(
                NBITS_TABLE[i as usize], expected,
                "Mismatch at i={}: table={}, expected={}",
                i, NBITS_TABLE[i as usize], expected
            );
        }
    }

    #[test]
    fn test_zigzag_reorder() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Test that zigzag reorder produces correct output
        // Create a block where each natural position i has value i
        let mut block = [0i16; 64];
        for i in 0..64 {
            block[i] = i as i16;
        }

        let encoder = SimdEntropyEncoder::new();
        let mut temp = [0i16; 64];
        unsafe {
            let _mask = encoder.zigzag_reorder_and_sign_sse2(&block, &mut temp);
        }

        // After zigzag reorder, temp[zigzag_pos] should contain block[JPEG_NATURAL_ORDER[zigzag_pos]]
        // Position 0 in zigzag = natural position 0, so temp[0] = block[0] = 0
        assert_eq!(temp[0], 0);
        // Position 1 in zigzag = natural position 1, so temp[1] = block[1] = 1
        assert_eq!(temp[1], 1);
        // Position 2 in zigzag = natural position 8, so temp[2] = block[8] = 8
        assert_eq!(temp[2], JPEG_NATURAL_ORDER[2] as i16); // = 8
    }

    #[test]
    fn test_sign_handling() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Test sign handling: negative values should be adjusted
        let mut block = [0i16; 64];
        block[0] = 5; // natural pos 0 = zigzag pos 0
        block[1] = -5; // natural pos 1 = zigzag pos 1
        block[8] = -1; // natural pos 8 = zigzag pos 2

        let encoder = SimdEntropyEncoder::new();
        let mut temp = [0i16; 64];
        unsafe {
            let mask = encoder.zigzag_reorder_and_sign_sse2(&block, &mut temp);

            // Position 0: value 5, should be unchanged
            assert_eq!(temp[0], 5);
            // Position 1: value -5, should be -6 (x + (x < 0 ? -1 : 0))
            assert_eq!(temp[1], -6);
            // Position 2: value -1 (from natural pos 8), should be -2
            assert_eq!(temp[2], -2);

            // Check mask: positions 0, 1, 2 should be non-zero
            assert_eq!(mask & 0b111, 0b111);
        }
    }

    #[test]
    fn test_mask_bits_table() {
        for i in 0..=16 {
            assert_eq!(MASK_BITS[i], (1u32 << i) - 1);
        }
    }

    #[test]
    fn test_encode_produces_valid_output() {
        use crate::consts::{
            AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
        };
        use crate::huffman::HuffTable;

        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Create standard Huffman tables
        let mut dc_htbl = HuffTable::default();
        dc_htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
        for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
            dc_htbl.huffval[i] = v;
        }
        let dc_table = DerivedTable::from_huff_table(&dc_htbl, true).unwrap();

        let mut ac_htbl = HuffTable::default();
        ac_htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            ac_htbl.huffval[i] = v;
        }
        let ac_table = DerivedTable::from_huff_table(&ac_htbl, false).unwrap();

        // Test with a simple block
        let mut block = [0i16; 64];
        block[0] = 100; // DC coefficient
        block[1] = 10; // AC at natural pos 1 (zigzag pos 1)
        block[8] = -5; // AC at natural pos 8 (zigzag pos 2)

        let mut encoder = SimdEntropyEncoder::new();
        unsafe {
            encoder.encode_block_sse2(&block, 0, &dc_table, &ac_table);
        }
        let output = encoder.finish();

        // Verify output is non-empty and reasonable size
        assert!(!output.is_empty(), "Output should not be empty");
        assert!(output.len() < 100, "Output should be reasonably sized");
    }

    #[test]
    fn test_compare_with_standard_encoder() {
        use crate::bitstream::VecBitWriter;
        use crate::consts::{
            AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
        };
        use crate::entropy::EntropyEncoder;
        use crate::huffman::HuffTable;

        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Create standard Huffman tables
        let mut dc_htbl = HuffTable::default();
        dc_htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
        for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
            dc_htbl.huffval[i] = v;
        }
        let dc_table = DerivedTable::from_huff_table(&dc_htbl, true).unwrap();

        let mut ac_htbl = HuffTable::default();
        ac_htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            ac_htbl.huffval[i] = v;
        }
        let ac_table = DerivedTable::from_huff_table(&ac_htbl, false).unwrap();

        // Test with various blocks
        let test_blocks: Vec<[i16; 64]> = vec![
            // All zeros except DC
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b
            },
            // Simple sparse block
            {
                let mut b = [0i16; 64];
                b[0] = 50;
                b[1] = 10;
                b[8] = -5;
                b[9] = 3;
                b
            },
            // Block with longer run
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b[63] = 1; // Last coefficient
                b
            },
        ];

        for (i, block) in test_blocks.iter().enumerate() {
            // Encode with standard encoder
            let mut writer = VecBitWriter::new_vec();
            {
                let mut std_encoder = EntropyEncoder::new(&mut writer);
                std_encoder
                    .encode_block(block, 0, &dc_table, &ac_table)
                    .unwrap();
                std_encoder.flush().unwrap();
            }
            let std_output = writer.into_bytes();

            // Encode with SIMD encoder
            let mut simd_encoder = SimdEntropyEncoder::new();
            unsafe {
                simd_encoder.encode_block_sse2(block, 0, &dc_table, &ac_table);
            }
            let simd_output = simd_encoder.finish();

            // Compare outputs
            assert_eq!(
                std_output, simd_output,
                "Block {} mismatch:\nStandard: {:?}\nSIMD: {:?}",
                i, std_output, simd_output
            );
        }
    }
}
