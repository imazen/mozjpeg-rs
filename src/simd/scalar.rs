//! Scalar (non-SIMD) reference implementations.
//!
//! These implementations are:
//! - **Correct**: Produce identical output to the reference C mozjpeg
//! - **Portable**: Work on any platform without SIMD support
//! - **Testable**: Used to verify SIMD implementations produce matching output
//!
//! All SIMD implementations must produce bit-exact results compared to these.

use crate::consts::{DCTSIZE, DCTSIZE2};

// ============================================================================
// Forward DCT (from dct.rs)
// ============================================================================

// Fixed-point constants for 13-bit precision (CONST_BITS = 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Pre-calculated fixed-point constants: FIX(x) = (x * (1 << CONST_BITS) + 0.5)
const FIX_0_298631336: i32 = 2446;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// DESCALE: Right-shift with rounding
#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Scalar forward DCT on one 8x8 block.
///
/// This is the Loeffler-Ligtenberg-Moschytz algorithm matching mozjpeg's jfdctint.c.
/// Output is scaled up by factor of 8 (removed during quantization).
pub fn forward_dct_8x8(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32
    for i in 0..DCTSIZE2 {
        data[i] = samples[i] as i32;
    }

    // Pass 1: process rows
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;

        let tmp0 = data[base] + data[base + 7];
        let tmp7 = data[base] - data[base + 7];
        let tmp1 = data[base + 1] + data[base + 6];
        let tmp6 = data[base + 1] - data[base + 6];
        let tmp2 = data[base + 2] + data[base + 5];
        let tmp5 = data[base + 2] - data[base + 5];
        let tmp3 = data[base + 3] + data[base + 4];
        let tmp4 = data[base + 3] - data[base + 4];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[base] = (tmp10 + tmp11) << PASS1_BITS;
        data[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        data[base + 6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        // Odd part
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        data[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    for col in 0..DCTSIZE {
        let tmp0 = data[col] + data[DCTSIZE * 7 + col];
        let tmp7 = data[col] - data[DCTSIZE * 7 + col];
        let tmp1 = data[DCTSIZE + col] + data[DCTSIZE * 6 + col];
        let tmp6 = data[DCTSIZE + col] - data[DCTSIZE * 6 + col];
        let tmp2 = data[DCTSIZE * 2 + col] + data[DCTSIZE * 5 + col];
        let tmp5 = data[DCTSIZE * 2 + col] - data[DCTSIZE * 5 + col];
        let tmp3 = data[DCTSIZE * 3 + col] + data[DCTSIZE * 4 + col];
        let tmp4 = data[DCTSIZE * 3 + col] - data[DCTSIZE * 4 + col];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[col] = descale(tmp10 + tmp11, PASS1_BITS);
        data[DCTSIZE * 4 + col] = descale(tmp10 - tmp11, PASS1_BITS);

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[DCTSIZE * 2 + col] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 6 + col] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);

        // Odd part
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        data[DCTSIZE * 7 + col] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 5 + col] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 3 + col] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE + col] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }

    // Copy results
    for i in 0..DCTSIZE2 {
        coeffs[i] = data[i] as i16;
    }
}

// Missing constant
const FIX_0_390180644: i32 = 3196;

// ============================================================================
// Color Conversion (from color.rs)
// ============================================================================

/// Fixed-point precision bits
const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);
const CBCR_CENTER: i32 = 128;

const fn fix(x: f64) -> i32 {
    (x * ((1i64 << SCALEBITS) as f64) + 0.5) as i32
}

const FIX_0_29900: i32 = fix(0.29900);
const FIX_0_58700: i32 = fix(0.58700);
const FIX_0_11400: i32 = fix(0.11400);
const FIX_0_16874: i32 = fix(0.16874);
const FIX_0_33126: i32 = fix(0.33126);
const FIX_0_50000: i32 = fix(0.50000);
const FIX_0_41869: i32 = fix(0.41869);
const FIX_0_08131: i32 = fix(0.08131);

/// Convert a single RGB pixel to YCbCr.
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    let y = (FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALEBITS;
    let cb = ((-FIX_0_16874 * r - FIX_0_33126 * g + FIX_0_50000 * b + ONE_HALF) >> SCALEBITS)
        + CBCR_CENTER;
    let cr = ((FIX_0_50000 * r - FIX_0_41869 * g - FIX_0_08131 * b + ONE_HALF) >> SCALEBITS)
        + CBCR_CENTER;

    let y = y.clamp(0, 255);
    let cb = cb.clamp(0, 255);
    let cr = cr.clamp(0, 255);

    (y as u8, cb as u8, cr as u8)
}

/// Scalar RGB to YCbCr conversion for a buffer.
///
/// This processes pixels one at a time. Used as reference for SIMD implementations.
pub fn convert_rgb_to_ycbcr(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    for i in 0..num_pixels {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
        y_out[i] = y;
        cb_out[i] = cb;
        cr_out[i] = cr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_flat_block() {
        let samples = [100i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8(&samples, &mut coeffs);

        // DC = 64 * value for flat block
        assert_eq!(coeffs[0], 6400);

        // AC should be ~0
        for i in 1..DCTSIZE2 {
            assert!(coeffs[i].abs() <= 1);
        }
    }

    #[test]
    fn test_dct_zero_block() {
        let samples = [0i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8(&samples, &mut coeffs);

        for c in coeffs.iter() {
            assert_eq!(*c, 0);
        }
    }

    #[test]
    fn test_color_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_color_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_color_gray() {
        let (y, cb, cr) = rgb_to_ycbcr(128, 128, 128);
        assert_eq!(y, 128);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }
}
