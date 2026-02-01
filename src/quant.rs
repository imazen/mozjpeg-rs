//! Quantization table construction and scaling.
//!
//! This module provides functions for:
//! - Quality to scale factor conversion (matching mozjpeg's formula)
//! - Quantization table scaling
//! - Quantization table selection from the 9 variants
//!
//! Reference: mozjpeg/jcparam.c

use crate::consts::{QuantTableIdx, DCTSIZE2, STD_CHROMINANCE_QUANT_TBL, STD_LUMINANCE_QUANT_TBL};
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
    let luma = create_quant_table(
        get_luminance_quant_table(table_idx),
        quality,
        force_baseline,
    );
    let chroma = create_quant_table(
        get_chrominance_quant_table(table_idx),
        quality,
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

/// Quantize a full 8x8 block of raw DCT coefficients (scaled by 8).
///
/// This function matches C mozjpeg's non-trellis quantization approach:
/// - Takes raw DCT output (scaled by 8, NOT descaled)
/// - Uses scaled quantization: `q_scaled = 8 * quant_table[i]`
/// - Single rounding step: (abs(coef) + q_scaled/2) / q_scaled
///
/// This avoids the rounding differences that occur when descaling and
/// quantizing are done as separate steps.
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
    // max_coef_bits = data_precision + 2 = 8 + 2 = 10 for 8-bit JPEG
    const MAX_COEF_VAL: i32 = (1 << 10) - 1; // 1023

    for i in 0..DCTSIZE2 {
        let coef = coeffs[i];
        // Scaled quantization value (includes DCT scale factor of 8)
        let q = 8 * quant_table[i] as i32;

        // Single-step quantization with rounding
        let (abs_coef, sign) = if coef < 0 {
            (-coef, -1i16)
        } else {
            (coef, 1i16)
        };

        // Round to nearest and clamp to valid range
        let qval = ((abs_coef + q / 2) / q).min(MAX_COEF_VAL);
        output[i] = (qval as i16) * sign;
    }
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
