//! Quantization table construction and scaling.
//!
//! This module provides functions for:
//! - Quality to scale factor conversion (matching mozjpeg's formula)
//! - Quantization table scaling
//! - Quantization table selection from the 9 variants
//!
//! Reference: mozjpeg/jcparam.c

use crate::consts::{DCTSIZE2, QuantTableIdx, STD_CHROMINANCE_QUANT_TBL, STD_LUMINANCE_QUANT_TBL};
use crate::types::QuantTable;

/// Convert a quality value (1-100) to a scaling factor.
///
/// This matches mozjpeg's `jpeg_quality_scaling` / `jpeg_float_quality_scaling`:
/// - Quality 50 → scale factor 100 (use table as-is)
/// - Quality 100 → scale factor 0 (all values become 1)
/// - Quality 1 → scale factor 5000
/// - Quality < 50 → scale = 5000 / quality
/// - Quality > 50 → scale = 200 - 2 * quality
///
/// # Arguments
/// * `quality` - Quality value from 1 to 100
///
/// # Returns
/// Scale factor as percentage (100 = use table as-is)
pub fn quality_to_scale_factor(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as f32;

    let scale = if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - q * 2.0
    };

    scale as u32
}

/// Convert a quality value to floating-point scale factor.
///
/// This is the float version matching `jpeg_float_quality_scaling`.
pub fn quality_to_scale_factor_f32(quality: f32) -> f32 {
    let q = quality.clamp(1.0, 100.0);

    if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - q * 2.0
    }
}

/// Get the base luminance quantization table for a given variant.
///
/// # Arguments
/// * `idx` - Quantization table variant index (0-8)
///
/// # Returns
/// Reference to the 64-element quantization table
pub fn get_luminance_quant_table(idx: QuantTableIdx) -> &'static [u16; DCTSIZE2] {
    &STD_LUMINANCE_QUANT_TBL[idx as usize]
}

/// Get the base chrominance quantization table for a given variant.
///
/// # Arguments
/// * `idx` - Quantization table variant index (0-8)
///
/// # Returns
/// Reference to the 64-element quantization table
pub fn get_chrominance_quant_table(idx: QuantTableIdx) -> &'static [u16; DCTSIZE2] {
    &STD_CHROMINANCE_QUANT_TBL[idx as usize]
}

/// Create a scaled quantization table from a base table and quality.
///
/// This combines quality_to_scale_factor and QuantTable::scaled.
///
/// # Arguments
/// * `base` - Base quantization table
/// * `quality` - Quality value (1-100)
/// * `force_baseline` - If true, clamp values to 255 for baseline JPEG
///
/// # Returns
/// Scaled quantization table
pub fn create_quant_table(base: &[u16; DCTSIZE2], quality: u8, force_baseline: bool) -> QuantTable {
    let scale = quality_to_scale_factor(quality);
    QuantTable::scaled(base, scale, force_baseline)
}

/// Create luminance and chrominance quantization tables for a given quality.
///
/// # Arguments
/// * `quality` - Quality value (1-100)
/// * `table_idx` - Quantization table variant (default: ImageMagick)
/// * `force_baseline` - If true, clamp values to 255
///
/// # Returns
/// Tuple of (luminance_table, chrominance_table)
pub fn create_quant_tables(
    quality: u8,
    table_idx: QuantTableIdx,
    force_baseline: bool,
) -> (QuantTable, QuantTable) {
    create_quant_tables_split(quality, None, table_idx, force_baseline)
}

/// Create luminance and chrominance quantization tables with an optional
/// independent chroma quality.
///
/// `chroma_quality = None` → chroma table scaled with `quality` (the
/// historical behaviour, bit-identical to [`create_quant_tables`]).
///
/// `chroma_quality = Some(cq)` → chroma table scaled with `cq` instead.
/// Lets callers apply asymmetric compression: e.g. `quality=85, cq=70`
/// preserves luma detail while compressing chroma more aggressively —
/// useful on images evalchroma (or similar analyzers) has flagged as
/// flat-chroma.
pub fn create_quant_tables_split(
    quality: u8,
    chroma_quality: Option<u8>,
    table_idx: QuantTableIdx,
    force_baseline: bool,
) -> (QuantTable, QuantTable) {
    let luma = create_quant_table(
        get_luminance_quant_table(table_idx),
        quality,
        force_baseline,
    );
    let chroma = create_quant_table(
        get_chrominance_quant_table(table_idx),
        chroma_quality.unwrap_or(quality),
        force_baseline,
    );
    (luma, chroma)
}

/// Quantize a single coefficient.
///
/// # Arguments
/// * `coef` - DCT coefficient (can be negative)
/// * `quant` - Quantization step size
///
/// # Returns
/// Quantized coefficient (rounded to nearest)
#[inline]
pub fn quantize_coef(coef: i32, quant: u16) -> i16 {
    let q = quant as i32;
    // Round to nearest: (coef + q/2) / q for positive, (coef - q/2) / q for negative
    if coef >= 0 {
        ((coef + q / 2) / q) as i16
    } else {
        ((coef - q / 2) / q) as i16
    }
}

/// Dequantize a single coefficient.
///
/// # Arguments
/// * `qcoef` - Quantized coefficient
/// * `quant` - Quantization step size
///
/// # Returns
/// Dequantized coefficient
#[inline]
pub fn dequantize_coef(qcoef: i16, quant: u16) -> i32 {
    (qcoef as i32) * (quant as i32)
}

/// Quantize a full 8x8 block of DCT coefficients.
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients (64 values)
/// * `quant_table` - Quantization table (64 values)
/// * `output` - Output quantized coefficients (64 values)
pub fn quantize_block(
    coeffs: &[i32; DCTSIZE2],
    quant_table: &[u16; DCTSIZE2],
    output: &mut [i16; DCTSIZE2],
) {
    for i in 0..DCTSIZE2 {
        output[i] = quantize_coef(coeffs[i], quant_table[i]);
    }
}

/// Fixed-point reciprocal divisors matching C mozjpeg's `quantize()`
/// (jcdctmgr.c). C does not divide: it precomputes a `(recip, corr, shift)`
/// triple per quant entry via `compute_reciprocal()` and evaluates
/// `(coef + corr) * recip >> (shift + 16)` per coefficient. The reciprocal
/// approximation differs from true `(coef + d/2) / d` division at boundary
/// cases, so byte-parity requires reproducing it exactly.
///
/// Divisors are the post-DCT scaled values `quant_table[i] << 3` truncated
/// to `UINT16`, exactly as the C call site passes them.
#[derive(Clone, Debug)]
pub struct RecipQuantTable {
    /// `fq` reciprocal, `DCTELEM`-truncated (may wrap; read back as u16).
    recip: [u16; DCTSIZE2],
    /// `c` correction + round factor.
    corr: [u16; DCTSIZE2],
    /// `r - 16` shift (product is shifted right by `shift + 16`).
    shift: [i32; DCTSIZE2],
}

impl RecipQuantTable {
    /// Build the reciprocal divisor table for one 8-bit quantization table,
    /// replicating C `compute_reciprocal()` (jcdctmgr.c) element by element.
    pub fn new(quant_table: &[u16; DCTSIZE2]) -> Self {
        let mut t = RecipQuantTable {
            recip: [0; DCTSIZE2],
            corr: [0; DCTSIZE2],
            shift: [0; DCTSIZE2],
        };
        for i in 0..DCTSIZE2 {
            // C passes `qtbl->quantval[i] << 3` into a UINT16 parameter:
            // the shift happens in 32 bits, then truncates to 16.
            let divisor = ((quant_table[i] as u32) << 3) as u16;
            let (fq, c, r) = compute_reciprocal(divisor);
            t.recip[i] = fq;
            t.corr[i] = c;
            t.shift[i] = r - 16;
        }
        t
    }
}

/// Port of C `compute_reciprocal()` (jcdctmgr.c): returns
/// `(recip fq, correction c, total right-shift r)` for `divisor`.
fn compute_reciprocal(divisor: u16) -> (u16, u16, i32) {
    if divisor == 1 {
        // Identity mapping: `(temp + 0) * 1 >> 0`.
        return (1, 0, 0);
    }
    // b = flss(divisor) - 1 = index of the second-highest set bit region;
    // equivalently floor(log2(divisor)) - 1.
    let b = 15 - divisor.leading_zeros() as i32 - 1;
    let mut r = 16 + b;

    let mut fq = (1u32 << r) / divisor as u32;
    let fr = (1u32 << r) % divisor as u32;

    let mut c = divisor / 2;
    if fr == 0 {
        // Power of two: fq is one bit too large for DCTELEM.
        fq >>= 1;
        r -= 1;
    } else if fr <= divisor as u32 / 2 {
        c += 1;
    } else {
        fq += 1;
    }

    // C stores fq into a DCTELEM (i16) slot; the u16 bit pattern is what the
    // multiply reads back, so truncate rather than saturate.
    (fq as u16, c, r)
}

/// Quantize a full 8x8 block of raw DCT coefficients using C's reciprocal
/// method — `(|coef| + corr) * recip >> (shift + 16)`, truncated to DCTELEM.
/// No clamping: C relies on DCTELEM truncation for large quotients.
///
/// # Arguments
/// * `coeffs` - Raw DCT coefficients scaled by 8 (64 values)
/// * `table` - Precomputed reciprocal divisors
/// * `output` - Output quantized coefficients (64 values)
pub fn quantize_block_recip(
    coeffs: &[i32; DCTSIZE2],
    table: &RecipQuantTable,
    output: &mut [i16; DCTSIZE2],
) {
    for i in 0..DCTSIZE2 {
        let coef = coeffs[i];
        let (abs_coef, sign) = if coef < 0 {
            (-coef, -1i16)
        } else {
            (coef, 1i16)
        };
        // (temp + corr) is computed in C `int` then multiplied by the u16
        // reciprocal into a 32-bit unsigned product with wraparound.
        let product = (abs_coef + table.corr[i] as i32) as u32 * table.recip[i] as u32;
        output[i] = ((product >> (table.shift[i] + 16)) as i16).wrapping_mul(sign);
    }
}

/// Quantize a full 8x8 block of raw DCT coefficients (scaled by 8).
///
/// Convenience wrapper that builds the reciprocal divisor table on the fly;
/// hot paths should precompute [`RecipQuantTable`] and call
/// [`quantize_block_recip`].
///
/// # Arguments
/// * `coeffs` - Raw DCT coefficients scaled by 8 (64 values)
/// * `quant_table` - Quantization table (64 values)
/// * `output` - Output quantized coefficients (64 values)
pub fn quantize_block_raw(
    coeffs: &[i32; DCTSIZE2],
    quant_table: &[u16; DCTSIZE2],
    output: &mut [i16; DCTSIZE2],
) {
    let table = RecipQuantTable::new(quant_table);
    quantize_block_recip(coeffs, &table, output);
}

/// Dequantize a full 8x8 block of coefficients.
///
/// # Arguments
/// * `qcoeffs` - Input quantized coefficients (64 values)
/// * `quant_table` - Quantization table (64 values)
/// * `output` - Output dequantized coefficients (64 values)
pub fn dequantize_block(
    qcoeffs: &[i16; DCTSIZE2],
    quant_table: &[u16; DCTSIZE2],
    output: &mut [i32; DCTSIZE2],
) {
    for i in 0..DCTSIZE2 {
        output[i] = dequantize_coef(qcoeffs[i], quant_table[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::NUM_QUANT_TABLE_VARIANTS;

    #[test]
    fn test_quality_scaling_matches_mozjpeg() {
        // These values match mozjpeg's jpeg_quality_scaling exactly
        assert_eq!(quality_to_scale_factor(50), 100); // Q50 = 100%
        assert_eq!(quality_to_scale_factor(75), 50); // Q75 = 50%
        assert_eq!(quality_to_scale_factor(100), 0); // Q100 = 0%
        assert_eq!(quality_to_scale_factor(25), 200); // Q25 = 200%
        assert_eq!(quality_to_scale_factor(1), 5000); // Q1 = 5000%
        assert_eq!(quality_to_scale_factor(10), 500); // Q10 = 500%
    }

    #[test]
    fn test_quality_scaling_float() {
        assert!((quality_to_scale_factor_f32(50.0) - 100.0).abs() < 0.01);
        assert!((quality_to_scale_factor_f32(75.0) - 50.0).abs() < 0.01);
        assert!((quality_to_scale_factor_f32(100.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_clamping() {
        // Quality 0 should be treated as 1
        assert_eq!(quality_to_scale_factor(0), quality_to_scale_factor(1));
        // Quality > 100 should be clamped to 100
        // (using internal logic since we take u8)
    }

    #[test]
    fn test_quant_table_scaling() {
        let base = get_luminance_quant_table(QuantTableIdx::JpegAnnexK);

        // 100% scale should give same values
        let scaled = QuantTable::scaled(base, 100, false);
        assert_eq!(scaled.values[0], base[0]);

        // 50% scale should halve (with rounding)
        let scaled = QuantTable::scaled(base, 50, false);
        assert_eq!(scaled.values[0], (base[0] as u32 * 50 + 50) as u16 / 100);

        // 200% scale should double
        let scaled = QuantTable::scaled(base, 200, false);
        assert_eq!(scaled.values[0], base[0] * 2);
    }

    #[test]
    fn test_force_baseline() {
        // Create a table that would exceed 255
        let base = [300u16; DCTSIZE2];
        let scaled = QuantTable::scaled(&base, 100, true);

        // All values should be clamped to 255
        for v in scaled.values.iter() {
            assert!(*v <= 255);
        }
    }

    #[test]
    fn test_quant_table_nonzero() {
        // Even at Q100 (scale=0), values should never be 0
        let base = get_luminance_quant_table(QuantTableIdx::JpegAnnexK);
        let scaled = QuantTable::scaled(base, 0, false);

        for v in scaled.values.iter() {
            assert!(*v >= 1, "Quant value should be at least 1");
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let coef = 100;
        let quant = 10;

        let qcoef = quantize_coef(coef, quant);
        assert_eq!(qcoef, 10);

        let dcoef = dequantize_coef(qcoef, quant);
        assert_eq!(dcoef, 100);
    }

    #[test]
    fn test_quantize_rounding() {
        // Positive rounding
        assert_eq!(quantize_coef(14, 10), 1); // 14/10 rounds to 1
        assert_eq!(quantize_coef(15, 10), 2); // 15/10 rounds to 2 (round half up)
        assert_eq!(quantize_coef(16, 10), 2); // 16/10 rounds to 2

        // Negative rounding
        assert_eq!(quantize_coef(-14, 10), -1);
        assert_eq!(quantize_coef(-15, 10), -2);
        assert_eq!(quantize_coef(-16, 10), -2);
    }

    #[test]
    fn test_quantize_block() {
        let mut coeffs = [0i32; DCTSIZE2];
        coeffs[0] = 1000; // DC
        coeffs[1] = 100; // AC
        coeffs[63] = -50;

        let quant = [10u16; DCTSIZE2];
        let mut output = [0i16; DCTSIZE2];

        quantize_block(&coeffs, &quant, &mut output);

        assert_eq!(output[0], 100);
        assert_eq!(output[1], 10);
        assert_eq!(output[63], -5);
    }

    #[test]
    fn test_all_quant_table_variants() {
        // Verify all 9 variants are accessible and valid
        for i in 0..NUM_QUANT_TABLE_VARIANTS {
            let idx = QuantTableIdx::from_u8(i as u8).unwrap();
            let luma = get_luminance_quant_table(idx);
            let chroma = get_chrominance_quant_table(idx);

            // All values should be positive
            for v in luma.iter() {
                assert!(*v > 0, "Luminance table {} has zero value", i);
            }
            for v in chroma.iter() {
                assert!(*v > 0, "Chrominance table {} has zero value", i);
            }
        }
    }

    #[test]
    fn test_create_quant_tables() {
        let (luma, chroma) = create_quant_tables(75, QuantTableIdx::ImageMagick, true);

        // At Q75, scale factor is 50
        // Base ImageMagick luma[0] is 16, so scaled should be 8
        assert_eq!(luma.values[0], 8);

        // Verify baseline constraint
        for v in luma.values.iter() {
            assert!(*v <= 255);
        }
        for v in chroma.values.iter() {
            assert!(*v <= 255);
        }
    }
}
