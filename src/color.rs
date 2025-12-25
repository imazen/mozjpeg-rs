//! Color space conversion routines.
//!
//! This module implements RGB to YCbCr conversion following the CCIR 601-1
//! (BT.601) standard, as used by JPEG. The conversion uses fixed-point
//! arithmetic for efficiency.
//!
//! The conversion equations are:
//! ```text
//! Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
//! Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
//! Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + 128
//! ```

/// Fixed-point precision bits (16 bits gives ~4 decimal digits precision)
const SCALEBITS: i32 = 16;

/// Half unit for rounding during right shift
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

/// Center value for Cb/Cr (added after shift, not before, to match C mozjpeg)
const CBCR_CENTER: i32 = 128;

/// Macro to compute fixed-point constant: FIX(x) = (x * (1 << SCALEBITS) + 0.5)
const fn fix(x: f64) -> i32 {
    (x * ((1i64 << SCALEBITS) as f64) + 0.5) as i32
}

// Pre-computed fixed-point conversion constants
const FIX_0_29900: i32 = fix(0.29900);  // Y coefficient for R
const FIX_0_58700: i32 = fix(0.58700);  // Y coefficient for G
const FIX_0_11400: i32 = fix(0.11400);  // Y coefficient for B
const FIX_0_16874: i32 = fix(0.16874);  // Cb coefficient for R (negated)
const FIX_0_33126: i32 = fix(0.33126);  // Cb coefficient for G (negated)
const FIX_0_50000: i32 = fix(0.50000);  // Cb coefficient for B, Cr coefficient for R
const FIX_0_41869: i32 = fix(0.41869);  // Cr coefficient for G (negated)
const FIX_0_08131: i32 = fix(0.08131);  // Cr coefficient for B (negated)

/// Convert a single RGB pixel to YCbCr.
///
/// # Arguments
/// * `r` - Red component (0-255)
/// * `g` - Green component (0-255)
/// * `b` - Blue component (0-255)
///
/// # Returns
/// Tuple of (Y, Cb, Cr) values, each in range 0-255
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // Y = 0.29900 * R + 0.58700 * G + 0.11400 * B
    // The ONE_HALF includes rounding
    let y = (FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALEBITS;

    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
    // Formula matches C mozjpeg: shift first, add 128 after, then clamp
    let cb = ((-FIX_0_16874 * r - FIX_0_33126 * g + FIX_0_50000 * b + ONE_HALF) >> SCALEBITS)
        + CBCR_CENTER;

    // Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B + 128
    let cr = ((FIX_0_50000 * r - FIX_0_41869 * g - FIX_0_08131 * b + ONE_HALF) >> SCALEBITS)
        + CBCR_CENTER;

    // Clamp to valid range (matches C mozjpeg behavior for extreme values)
    let y = y.clamp(0, 255);
    let cb = cb.clamp(0, 255);
    let cr = cr.clamp(0, 255);

    (y as u8, cb as u8, cr as u8)
}

/// Convert a single RGB pixel to grayscale (Y component only).
///
/// # Arguments
/// * `r` - Red component (0-255)
/// * `g` - Green component (0-255)
/// * `b` - Blue component (0-255)
///
/// # Returns
/// Y (luminance) value in range 0-255
#[inline]
pub fn rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    let y = (FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALEBITS;
    y as u8
}

/// Convert an RGB image buffer to YCbCr in-place component buffers.
///
/// The input is interleaved RGB data, and the output is three separate
/// component planes (Y, Cb, Cr), matching libjpeg's internal format.
///
/// # Arguments
/// * `rgb` - Input RGB data (3 bytes per pixel: R, G, B)
/// * `y_out` - Output Y component buffer
/// * `cb_out` - Output Cb component buffer
/// * `cr_out` - Output Cr component buffer
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
pub fn convert_rgb_to_ycbcr(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(y_out.len(), width * height);
    debug_assert_eq!(cb_out.len(), width * height);
    debug_assert_eq!(cr_out.len(), width * height);

    for i in 0..(width * height) {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
        y_out[i] = y;
        cb_out[i] = cb;
        cr_out[i] = cr;
    }
}

/// Convert an RGB image buffer to grayscale.
///
/// # Arguments
/// * `rgb` - Input RGB data (3 bytes per pixel: R, G, B)
/// * `gray_out` - Output grayscale buffer
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
pub fn convert_rgb_to_gray(rgb: &[u8], gray_out: &mut [u8], width: usize, height: usize) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(gray_out.len(), width * height);

    for i in 0..(width * height) {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        gray_out[i] = rgb_to_gray(r, g, b);
    }
}

/// Convert CMYK to YCCK color space.
///
/// This uses the Adobe convention where CMYK values represent
/// the ink density (0=no ink, 255=full ink). We compute:
/// R = 255 - C, G = 255 - M, B = 255 - Y
/// Then convert RGB to YCbCr, and pass K through unchanged.
///
/// # Arguments
/// * `c` - Cyan component (0-255)
/// * `m` - Magenta component (0-255)
/// * `y` - Yellow component (0-255)
/// * `k` - Black component (0-255)
///
/// # Returns
/// Tuple of (Y, Cb, Cr, K) values
#[inline]
pub fn cmyk_to_ycck(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8, u8) {
    // Convert CMYK to RGB (Adobe convention: ink density)
    let r = 255 - c;
    let g = 255 - m;
    let b = 255 - y;

    let (y_out, cb, cr) = rgb_to_ycbcr(r, g, b);
    (y_out, cb, cr, k)
}

/// Convert an 8x8 block of RGB samples to YCbCr for DCT processing.
///
/// This is optimized for the typical JPEG workflow where we process
/// 8x8 blocks at a time.
///
/// # Arguments
/// * `rgb_block` - Input 8x8 RGB block (192 bytes = 64 pixels * 3 channels)
/// * `y_block` - Output Y block (64 samples)
/// * `cb_block` - Output Cb block (64 samples)
/// * `cr_block` - Output Cr block (64 samples)
pub fn convert_block_rgb_to_ycbcr(
    rgb_block: &[u8; 192],
    y_block: &mut [u8; 64],
    cb_block: &mut [u8; 64],
    cr_block: &mut [u8; 64],
) {
    for i in 0..64 {
        let r = rgb_block[i * 3];
        let g = rgb_block[i * 3 + 1];
        let b = rgb_block[i * 3 + 2];
        let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
        y_block[i] = y;
        cb_block[i] = cb;
        cr_block[i] = cr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_ycbcr_black() {
        // Black (0,0,0) should give Y=0, Cb=128, Cr=128
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        // White (255,255,255) should give Y=255, Cb=128, Cr=128
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_gray() {
        // Gray (128,128,128) should give Y=128, Cb=128, Cr=128
        let (y, cb, cr) = rgb_to_ycbcr(128, 128, 128);
        assert_eq!(y, 128);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_red() {
        // Pure red should have high Cr (red-blue axis)
        let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
        // Y = 0.299 * 255 ≈ 76
        assert!((y as i32 - 76).abs() <= 1);
        // Cb should be below 128
        assert!(cb < 128);
        // Cr should be max (255)
        assert_eq!(cr, 255);
    }

    #[test]
    fn test_rgb_to_ycbcr_green() {
        // Pure green has high Y contribution
        let (y, cb, cr) = rgb_to_ycbcr(0, 255, 0);
        // Y = 0.587 * 255 ≈ 150
        assert!((y as i32 - 150).abs() <= 1);
        // Both Cb and Cr should be below 128
        assert!(cb < 128);
        assert!(cr < 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_blue() {
        // Pure blue should have high Cb (yellow-blue axis)
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 255);
        // Y = 0.114 * 255 ≈ 29
        assert!((y as i32 - 29).abs() <= 1);
        // Cb should be max (255)
        assert_eq!(cb, 255);
        // Cr should be below 128
        assert!(cr < 128);
    }

    #[test]
    fn test_rgb_to_gray() {
        // Test known values
        assert_eq!(rgb_to_gray(0, 0, 0), 0);
        assert_eq!(rgb_to_gray(255, 255, 255), 255);
        assert_eq!(rgb_to_gray(128, 128, 128), 128);

        // Red has Y ≈ 76
        let y_red = rgb_to_gray(255, 0, 0);
        assert!((y_red as i32 - 76).abs() <= 1);
    }

    #[test]
    fn test_cmyk_to_ycck() {
        // Black CMYK (0,0,0,255) = RGB white with full black = should give YCbCr of white
        let (y, cb, cr, k) = cmyk_to_ycck(0, 0, 0, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
        assert_eq!(k, 255);

        // Full CMYK (255,255,255,0) = RGB black with no black
        let (y, cb, cr, k) = cmyk_to_ycck(255, 255, 255, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
        assert_eq!(k, 0);
    }

    #[test]
    fn test_convert_block() {
        let mut rgb_block = [0u8; 192];
        // Fill with a gradient
        for i in 0..64 {
            rgb_block[i * 3] = (i * 4) as u8;     // R
            rgb_block[i * 3 + 1] = (i * 2) as u8; // G
            rgb_block[i * 3 + 2] = (i * 3) as u8; // B
        }

        let mut y_block = [0u8; 64];
        let mut cb_block = [0u8; 64];
        let mut cr_block = [0u8; 64];

        convert_block_rgb_to_ycbcr(&rgb_block, &mut y_block, &mut cb_block, &mut cr_block);

        // Verify first and last pixel conversions
        let (y0, cb0, cr0) = rgb_to_ycbcr(rgb_block[0], rgb_block[1], rgb_block[2]);
        assert_eq!(y_block[0], y0);
        assert_eq!(cb_block[0], cb0);
        assert_eq!(cr_block[0], cr0);

        let (y63, cb63, cr63) =
            rgb_to_ycbcr(rgb_block[189], rgb_block[190], rgb_block[191]);
        assert_eq!(y_block[63], y63);
        assert_eq!(cb_block[63], cb63);
        assert_eq!(cr_block[63], cr63);
    }

    #[test]
    fn test_fixed_point_constants() {
        // Verify the fixed-point constants are correctly computed
        // The sum of Y coefficients should equal 1.0 (within rounding)
        let y_sum = FIX_0_29900 + FIX_0_58700 + FIX_0_11400;
        let one = 1 << SCALEBITS;
        // Should be very close to 1.0 (within 1 unit)
        assert!((y_sum - one).abs() <= 1);
    }
}
