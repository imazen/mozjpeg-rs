//! Scalar reference implementations with automatic SIMD via multiversion.
//!
//! These implementations use `#[multiversion]` to automatically compile
//! optimized versions for different CPU features (AVX2, SSE4.1, NEON)
//! and dispatch at runtime. No unsafe code required.
//!
//! The DCT implementation is re-exported from `crate::dct` (single source of truth).

use multiversion::multiversion;

// Re-export the canonical DCT implementation (now with multiversion)
pub use crate::dct::forward_dct_8x8;

// ============================================================================
// Color Conversion
// ============================================================================

// Fixed-point precision (16 bits)
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

/// Convert a single RGB pixel to YCbCr (BT.601).
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

    (y.clamp(0, 255) as u8, cb.clamp(0, 255) as u8, cr.clamp(0, 255) as u8)
}

/// RGB to YCbCr conversion for a buffer.
///
/// Uses `multiversion` for automatic SIMD optimization via autovectorization.
#[multiversion(targets(
    "x86_64+avx2",
    "x86_64+sse4.1",
    "x86+avx2",
    "x86+sse4.1",
    "aarch64+neon",
))]
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
    use crate::consts::DCTSIZE2;

    #[test]
    fn test_dct_reexport_works() {
        let samples = [100i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8(&samples, &mut coeffs);
        // DC coefficient for flat block of 100 = 64 * 100 = 6400
        assert_eq!(coeffs[0], 6400);
    }

    #[test]
    fn test_gray_pixel() {
        let (y, cb, cr) = rgb_to_ycbcr(128, 128, 128);
        assert_eq!(y, 128);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_white_pixel() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_black_pixel() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_buffer_conversion() {
        let rgb = [128u8; 24]; // 8 gray pixels
        let mut y = [0u8; 8];
        let mut cb = [0u8; 8];
        let mut cr = [0u8; 8];

        convert_rgb_to_ycbcr(&rgb, &mut y, &mut cb, &mut cr, 8);

        assert!(y.iter().all(|&v| v == 128));
        assert!(cb.iter().all(|&v| v == 128));
        assert!(cr.iter().all(|&v| v == 128));
    }
}
