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
//!
//! ## Performance
//!
//! By default (with `fast-yuv` feature), this module uses the `yuv` crate for
//! SIMD-optimized color conversion. The `yuv` crate provides ~60% better
//! performance than our hand-written AVX2 code, with support for AVX-512,
//! AVX2, SSE, NEON, and WASM SIMD.
//!
//! The precision difference is ±1 level, which is invisible after JPEG
//! quantization (which loses 2-4+ levels at Q85).
//!
//! Without `fast-yuv`, uses the `wide` crate for portable SIMD (i32x8).

use wide::i32x8;

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
const FIX_0_29900: i32 = fix(0.29900); // Y coefficient for R
const FIX_0_58700: i32 = fix(0.58700); // Y coefficient for G
const FIX_0_11400: i32 = fix(0.11400); // Y coefficient for B
const FIX_0_16874: i32 = fix(0.16874); // Cb coefficient for R (negated)
const FIX_0_33126: i32 = fix(0.33126); // Cb coefficient for G (negated)
const FIX_0_50000: i32 = fix(0.50000); // Cb coefficient for B, Cr coefficient for R
const FIX_0_41869: i32 = fix(0.41869); // Cr coefficient for G (negated)
const FIX_0_08131: i32 = fix(0.08131); // Cr coefficient for B (negated)

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
/// With the `fast-yuv` feature (default), this uses the `yuv` crate for
/// ~60% better performance with AVX-512/AVX2/SSE/NEON/WASM SIMD support.
/// Without `fast-yuv`, uses the `wide` crate for portable SIMD (i32x8).
///
/// # Arguments
/// * `rgb` - Input RGB data (3 bytes per pixel: R, G, B)
/// * `y_out` - Output Y component buffer
/// * `cb_out` - Output Cb component buffer
/// * `cr_out` - Output Cr component buffer
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
#[cfg(feature = "fast-yuv")]
pub fn convert_rgb_to_ycbcr(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
) {
    use yuv::{
        rgb_to_yuv444, BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange,
        YuvStandardMatrix,
    };

    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(y_out.len(), width * height);
    debug_assert_eq!(cb_out.len(), width * height);
    debug_assert_eq!(cr_out.len(), width * height);

    let w = width as u32;
    let h = height as u32;

    // Create a YUV planar image that directly borrows our output buffers (zero-copy)
    let mut yuv_image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(y_out),
        y_stride: w,
        u_plane: BufferStoreMut::Borrowed(cb_out),
        u_stride: w,
        v_plane: BufferStoreMut::Borrowed(cr_out),
        v_stride: w,
        width: w,
        height: h,
    };

    // Convert RGB to YUV using the yuv crate's SIMD-optimized implementation
    // Uses BT.601 matrix (same as JPEG) with full range (0-255)
    rgb_to_yuv444(
        &mut yuv_image,
        rgb,
        w * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::default(), // Balanced mode: good precision + speed
    )
    .expect("yuv conversion failed");
}

/// Convert an RGB image buffer to YCbCr in-place component buffers.
///
/// The input is interleaved RGB data, and the output is three separate
/// component planes (Y, Cb, Cr), matching libjpeg's internal format.
///
/// This function uses SIMD to process 8 pixels at a time for better
/// performance on supported architectures.
///
/// # Arguments
/// * `rgb` - Input RGB data (3 bytes per pixel: R, G, B)
/// * `y_out` - Output Y component buffer
/// * `cb_out` - Output Cb component buffer
/// * `cr_out` - Output Cr component buffer
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
#[cfg(not(feature = "fast-yuv"))]
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

    let num_pixels = width * height;

    // AVX2 path: Process 8 pixels at a time using i32x8
    // This uses native AVX2 instructions when available via the `wide` crate
    let fix_y_r_8 = i32x8::splat(FIX_0_29900);
    let fix_y_g_8 = i32x8::splat(FIX_0_58700);
    let fix_y_b_8 = i32x8::splat(FIX_0_11400);
    let fix_cb_r_8 = i32x8::splat(-FIX_0_16874);
    let fix_cb_g_8 = i32x8::splat(-FIX_0_33126);
    let fix_cb_b_8 = i32x8::splat(FIX_0_50000);
    let fix_cr_r_8 = i32x8::splat(FIX_0_50000);
    let fix_cr_g_8 = i32x8::splat(-FIX_0_41869);
    let fix_cr_b_8 = i32x8::splat(-FIX_0_08131);
    let half_8 = i32x8::splat(ONE_HALF);
    let center_8 = i32x8::splat(CBCR_CENTER);
    let zero_8 = i32x8::splat(0);
    let max_val_8 = i32x8::splat(255);

    // Process 8 pixels at a time
    let chunks_8 = num_pixels / 8;
    for chunk in 0..chunks_8 {
        let base = chunk * 8;
        let rgb_base = base * 3;

        // Gather RGB values from interleaved format (8 pixels = 24 bytes)
        let r = i32x8::new([
            rgb[rgb_base] as i32,
            rgb[rgb_base + 3] as i32,
            rgb[rgb_base + 6] as i32,
            rgb[rgb_base + 9] as i32,
            rgb[rgb_base + 12] as i32,
            rgb[rgb_base + 15] as i32,
            rgb[rgb_base + 18] as i32,
            rgb[rgb_base + 21] as i32,
        ]);
        let g = i32x8::new([
            rgb[rgb_base + 1] as i32,
            rgb[rgb_base + 4] as i32,
            rgb[rgb_base + 7] as i32,
            rgb[rgb_base + 10] as i32,
            rgb[rgb_base + 13] as i32,
            rgb[rgb_base + 16] as i32,
            rgb[rgb_base + 19] as i32,
            rgb[rgb_base + 22] as i32,
        ]);
        let b = i32x8::new([
            rgb[rgb_base + 2] as i32,
            rgb[rgb_base + 5] as i32,
            rgb[rgb_base + 8] as i32,
            rgb[rgb_base + 11] as i32,
            rgb[rgb_base + 14] as i32,
            rgb[rgb_base + 17] as i32,
            rgb[rgb_base + 20] as i32,
            rgb[rgb_base + 23] as i32,
        ]);

        // Y = 0.29900 * R + 0.58700 * G + 0.11400 * B
        let y = (fix_y_r_8 * r + fix_y_g_8 * g + fix_y_b_8 * b + half_8) >> SCALEBITS;
        let y = y.max(zero_8).min(max_val_8);

        // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
        let cb =
            ((fix_cb_r_8 * r + fix_cb_g_8 * g + fix_cb_b_8 * b + half_8) >> SCALEBITS) + center_8;
        let cb = cb.max(zero_8).min(max_val_8);

        // Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B + 128
        let cr =
            ((fix_cr_r_8 * r + fix_cr_g_8 * g + fix_cr_b_8 * b + half_8) >> SCALEBITS) + center_8;
        let cr = cr.max(zero_8).min(max_val_8);

        // Store results
        let y_arr = y.to_array();
        let cb_arr = cb.to_array();
        let cr_arr = cr.to_array();
        y_out[base] = y_arr[0] as u8;
        y_out[base + 1] = y_arr[1] as u8;
        y_out[base + 2] = y_arr[2] as u8;
        y_out[base + 3] = y_arr[3] as u8;
        y_out[base + 4] = y_arr[4] as u8;
        y_out[base + 5] = y_arr[5] as u8;
        y_out[base + 6] = y_arr[6] as u8;
        y_out[base + 7] = y_arr[7] as u8;
        cb_out[base] = cb_arr[0] as u8;
        cb_out[base + 1] = cb_arr[1] as u8;
        cb_out[base + 2] = cb_arr[2] as u8;
        cb_out[base + 3] = cb_arr[3] as u8;
        cb_out[base + 4] = cb_arr[4] as u8;
        cb_out[base + 5] = cb_arr[5] as u8;
        cb_out[base + 6] = cb_arr[6] as u8;
        cb_out[base + 7] = cb_arr[7] as u8;
        cr_out[base] = cr_arr[0] as u8;
        cr_out[base + 1] = cr_arr[1] as u8;
        cr_out[base + 2] = cr_arr[2] as u8;
        cr_out[base + 3] = cr_arr[3] as u8;
        cr_out[base + 4] = cr_arr[4] as u8;
        cr_out[base + 5] = cr_arr[5] as u8;
        cr_out[base + 6] = cr_arr[6] as u8;
        cr_out[base + 7] = cr_arr[7] as u8;
    }

    // Handle remaining pixels with scalar code
    for i in (chunks_8 * 8)..num_pixels {
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
/// This function uses SIMD to process 8 pixels at a time for better
/// performance using AVX2 when available.
///
/// # Arguments
/// * `rgb` - Input RGB data (3 bytes per pixel: R, G, B)
/// * `gray_out` - Output grayscale buffer
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
pub fn convert_rgb_to_gray(rgb: &[u8], gray_out: &mut [u8], width: usize, height: usize) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(gray_out.len(), width * height);

    let num_pixels = width * height;

    // AVX2 SIMD constants for Y calculation (8 pixels at a time)
    let fix_y_r = i32x8::splat(FIX_0_29900);
    let fix_y_g = i32x8::splat(FIX_0_58700);
    let fix_y_b = i32x8::splat(FIX_0_11400);
    let half = i32x8::splat(ONE_HALF);

    // Process 8 pixels at a time
    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let rgb_base = base * 3;

        // Gather RGB values from interleaved format
        let r = i32x8::new([
            rgb[rgb_base] as i32,
            rgb[rgb_base + 3] as i32,
            rgb[rgb_base + 6] as i32,
            rgb[rgb_base + 9] as i32,
            rgb[rgb_base + 12] as i32,
            rgb[rgb_base + 15] as i32,
            rgb[rgb_base + 18] as i32,
            rgb[rgb_base + 21] as i32,
        ]);
        let g = i32x8::new([
            rgb[rgb_base + 1] as i32,
            rgb[rgb_base + 4] as i32,
            rgb[rgb_base + 7] as i32,
            rgb[rgb_base + 10] as i32,
            rgb[rgb_base + 13] as i32,
            rgb[rgb_base + 16] as i32,
            rgb[rgb_base + 19] as i32,
            rgb[rgb_base + 22] as i32,
        ]);
        let b = i32x8::new([
            rgb[rgb_base + 2] as i32,
            rgb[rgb_base + 5] as i32,
            rgb[rgb_base + 8] as i32,
            rgb[rgb_base + 11] as i32,
            rgb[rgb_base + 14] as i32,
            rgb[rgb_base + 17] as i32,
            rgb[rgb_base + 20] as i32,
            rgb[rgb_base + 23] as i32,
        ]);

        // Y = 0.29900 * R + 0.58700 * G + 0.11400 * B
        let y = (fix_y_r * r + fix_y_g * g + fix_y_b * b + half) >> SCALEBITS;
        let y_arr = y.to_array();

        gray_out[base] = y_arr[0] as u8;
        gray_out[base + 1] = y_arr[1] as u8;
        gray_out[base + 2] = y_arr[2] as u8;
        gray_out[base + 3] = y_arr[3] as u8;
        gray_out[base + 4] = y_arr[4] as u8;
        gray_out[base + 5] = y_arr[5] as u8;
        gray_out[base + 6] = y_arr[6] as u8;
        gray_out[base + 7] = y_arr[7] as u8;
    }

    // Handle remaining pixels
    for i in (chunks * 8)..num_pixels {
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

// ============================================================================
// C mozjpeg-compatible color conversion
// ============================================================================

/// C mozjpeg-compatible RGB to YCbCr conversion.
///
/// Uses identical fixed-point arithmetic to match C mozjpeg output exactly.
/// The key difference from [`rgb_to_ycbcr`] is the `- 1` adjustment in the
/// Cb/Cr calculations, which matches C mozjpeg's `jccolor.c` implementation.
///
/// This produces bytewise-identical output to C mozjpeg, eliminating the
/// ±1 rounding differences that cause 3-5% larger baseline files.
///
/// Use [`Encoder::c_compat_color`] to enable this conversion.
#[inline]
pub fn rgb_to_ycbcr_c_compat(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    // C mozjpeg uses 16-bit fixed point with these constants:
    // #define SCALEBITS  16
    // #define CBCR_OFFSET  ((INT32) CENTERJSAMPLE << SCALEBITS)
    // #define ONE_HALF  ((INT32) 1 << (SCALEBITS-1))
    // #define FIX(x)  ((INT32) ((x) * (1L<<SCALEBITS) + 0.5))
    //
    // The critical detail is `+ ONE_HALF - 1` for Cb/Cr, not just `+ ONE_HALF`.

    const SCALE: i32 = 16;
    const ONE_HALF: i32 = 1 << (SCALE - 1);
    const CBCR_OFFSET: i32 = 128 << SCALE;

    // FIX(x) = (x * (1 << 16) + 0.5) as i32
    const FIX_0_29900: i32 = 19595; // FIX(0.29900)
    const FIX_0_58700: i32 = 38470; // FIX(0.58700)
    const FIX_0_11400: i32 = 7471; // FIX(0.11400)
    const FIX_0_16874: i32 = 11059; // FIX(0.16874)
    const FIX_0_33126: i32 = 21709; // FIX(0.33126)
    const FIX_0_50000: i32 = 32768; // FIX(0.50000)
    const FIX_0_41869: i32 = 27439; // FIX(0.41869)
    const FIX_0_08131: i32 = 5329; // FIX(0.08131)

    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // Y = 0.29900*R + 0.58700*G + 0.11400*B
    let y = ((FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALE) as u8;

    // Cb = -0.16874*R - 0.33126*G + 0.50000*B + 128
    // Note: C mozjpeg uses `+ CBCR_OFFSET + ONE_HALF - 1` (the -1 is critical!)
    let cb = (((-FIX_0_16874) * r + (-FIX_0_33126) * g + FIX_0_50000 * b + CBCR_OFFSET + ONE_HALF
        - 1)
        >> SCALE) as u8;

    // Cr = 0.50000*R - 0.41869*G - 0.08131*B + 128
    let cr = ((FIX_0_50000 * r + (-FIX_0_41869) * g + (-FIX_0_08131) * b + CBCR_OFFSET + ONE_HALF
        - 1)
        >> SCALE) as u8;

    (y, cb, cr)
}

/// Convert RGB image to YCbCr using C mozjpeg-compatible algorithm.
///
/// This produces bytewise-identical output to C mozjpeg, which is important
/// for baseline JPEG compression where ±1 coefficient differences can
/// accumulate to 3-5% file size differences.
///
/// # Arguments
/// * `rgb` - Interleaved RGB data (3 bytes per pixel)
/// * `y_out` - Output Y plane
/// * `cb_out` - Output Cb plane
/// * `cr_out` - Output Cr plane
/// * `num_pixels` - Number of pixels to convert
pub fn convert_rgb_to_ycbcr_c_compat(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    convert_rgb_to_ycbcr_c_compat_dispatch(rgb, y_out, cb_out, cr_out, num_pixels);
}

/// Scalar fallback for C-compat conversion.
fn convert_rgb_to_ycbcr_c_compat_scalar(
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
        let (y, cb, cr) = rgb_to_ycbcr_c_compat(r, g, b);
        y_out[i] = y;
        cb_out[i] = cb;
        cr_out[i] = cr;
    }
}

// ============================================================================
// AVX2-accelerated C-compat color conversion (archmage)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)] // Required for SIMD intrinsics
mod avx2_c_compat {
    use archmage::{arcane, X64V3Token};
    use std::arch::x86_64::*;

    const SCALEBITS: i32 = 16;

    // C mozjpeg coefficients (FIX(x) = (x * 65536 + 0.5) as i32)
    const FIX_0_29900: i32 = 19595;
    const FIX_0_58700: i32 = 38470;
    const FIX_0_11400: i32 = 7471;
    const FIX_0_16874: i32 = 11059;
    const FIX_0_33126: i32 = 21709;
    const FIX_0_50000: i32 = 32768;
    const FIX_0_41869: i32 = 27439;
    const FIX_0_08131: i32 = 5329;

    /// AVX2 C-compat RGB→YCbCr conversion.
    /// Processes 8 pixels per iteration using i32 arithmetic for exact C parity.
    #[arcane]
    pub fn convert_avx2(
        _token: X64V3Token,
        rgb: &[u8],
        y_out: &mut [u8],
        cb_out: &mut [u8],
        cr_out: &mut [u8],
        num_pixels: usize,
    ) {
        // Coefficient vectors (8 x i32)
        let coef_y_r = _mm256_set1_epi32(FIX_0_29900);
        let coef_y_g = _mm256_set1_epi32(FIX_0_58700);
        let coef_y_b = _mm256_set1_epi32(FIX_0_11400);

        let coef_cb_r = _mm256_set1_epi32(-FIX_0_16874);
        let coef_cb_g = _mm256_set1_epi32(-FIX_0_33126);
        let coef_cb_b = _mm256_set1_epi32(FIX_0_50000);

        let coef_cr_r = _mm256_set1_epi32(FIX_0_50000);
        let coef_cr_g = _mm256_set1_epi32(-FIX_0_41869);
        let coef_cr_b = _mm256_set1_epi32(-FIX_0_08131);

        // Rounding: Y uses ONE_HALF, Cb/Cr use CBCR_OFFSET + ONE_HALF - 1
        let y_round = _mm256_set1_epi32(1 << (SCALEBITS - 1));
        let cbcr_round = _mm256_set1_epi32((128 << SCALEBITS) + (1 << (SCALEBITS - 1)) - 1);

        // Process 8 pixels at a time
        let chunks = num_pixels / 8;
        let remainder = num_pixels % 8;

        for chunk in 0..chunks {
            let rgb_base = chunk * 24;
            let out_base = chunk * 8;

            // Load and de-interleave 8 pixels of RGB data
            // Using scalar gather is simpler and correct; LLVM can optimize
            let mut r_arr = [0i32; 8];
            let mut g_arr = [0i32; 8];
            let mut b_arr = [0i32; 8];

            for i in 0..8 {
                r_arr[i] = rgb[rgb_base + i * 3] as i32;
                g_arr[i] = rgb[rgb_base + i * 3 + 1] as i32;
                b_arr[i] = rgb[rgb_base + i * 3 + 2] as i32;
            }

            let r = _mm256_loadu_si256(r_arr.as_ptr() as *const __m256i);
            let g = _mm256_loadu_si256(g_arr.as_ptr() as *const __m256i);
            let b = _mm256_loadu_si256(b_arr.as_ptr() as *const __m256i);

            // Y = (0.29900*R + 0.58700*G + 0.11400*B + ONE_HALF) >> 16
            let y_r = _mm256_mullo_epi32(r, coef_y_r);
            let y_g = _mm256_mullo_epi32(g, coef_y_g);
            let y_b = _mm256_mullo_epi32(b, coef_y_b);
            let y_sum =
                _mm256_add_epi32(_mm256_add_epi32(y_r, y_g), _mm256_add_epi32(y_b, y_round));
            let y_32 = _mm256_srai_epi32(y_sum, SCALEBITS);

            // Cb = (-0.16874*R - 0.33126*G + 0.50000*B + CBCR_OFFSET + ONE_HALF - 1) >> 16
            let cb_r = _mm256_mullo_epi32(r, coef_cb_r);
            let cb_g = _mm256_mullo_epi32(g, coef_cb_g);
            let cb_b = _mm256_mullo_epi32(b, coef_cb_b);
            let cb_sum = _mm256_add_epi32(
                _mm256_add_epi32(cb_r, cb_g),
                _mm256_add_epi32(cb_b, cbcr_round),
            );
            let cb_32 = _mm256_srai_epi32(cb_sum, SCALEBITS);

            // Cr = (0.50000*R - 0.41869*G - 0.08131*B + CBCR_OFFSET + ONE_HALF - 1) >> 16
            let cr_r = _mm256_mullo_epi32(r, coef_cr_r);
            let cr_g = _mm256_mullo_epi32(g, coef_cr_g);
            let cr_b = _mm256_mullo_epi32(b, coef_cr_b);
            let cr_sum = _mm256_add_epi32(
                _mm256_add_epi32(cr_r, cr_g),
                _mm256_add_epi32(cr_b, cbcr_round),
            );
            let cr_32 = _mm256_srai_epi32(cr_sum, SCALEBITS);

            // Store results
            let mut y_arr = [0i32; 8];
            let mut cb_arr = [0i32; 8];
            let mut cr_arr = [0i32; 8];

            _mm256_storeu_si256(y_arr.as_mut_ptr() as *mut __m256i, y_32);
            _mm256_storeu_si256(cb_arr.as_mut_ptr() as *mut __m256i, cb_32);
            _mm256_storeu_si256(cr_arr.as_mut_ptr() as *mut __m256i, cr_32);

            for i in 0..8 {
                y_out[out_base + i] = y_arr[i] as u8;
                cb_out[out_base + i] = cb_arr[i] as u8;
                cr_out[out_base + i] = cr_arr[i] as u8;
            }
        }

        // Handle remaining pixels with scalar
        let scalar_start = chunks * 8;
        for i in 0..remainder {
            let idx = scalar_start + i;
            let r = rgb[idx * 3];
            let g = rgb[idx * 3 + 1];
            let b = rgb[idx * 3 + 2];
            let (y, cb, cr) = super::rgb_to_ycbcr_c_compat(r, g, b);
            y_out[idx] = y;
            cb_out[idx] = cb;
            cr_out[idx] = cr;
        }
    }
}

/// Dispatch C-compat color conversion to best available implementation.
#[cfg(target_arch = "x86_64")]
fn convert_rgb_to_ycbcr_c_compat_dispatch(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    use archmage::{SimdToken, X64V3Token};

    if let Some(token) = X64V3Token::try_new() {
        avx2_c_compat::convert_avx2(token, rgb, y_out, cb_out, cr_out, num_pixels);
    } else {
        convert_rgb_to_ycbcr_c_compat_scalar(rgb, y_out, cb_out, cr_out, num_pixels);
    }
}

/// Dispatch C-compat color conversion (non-x86_64 fallback).
#[cfg(not(target_arch = "x86_64"))]
fn convert_rgb_to_ycbcr_c_compat_dispatch(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    convert_rgb_to_ycbcr_c_compat_scalar(rgb, y_out, cb_out, cr_out, num_pixels);
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
            rgb_block[i * 3] = (i * 4) as u8; // R
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

        let (y63, cb63, cr63) = rgb_to_ycbcr(rgb_block[189], rgb_block[190], rgb_block[191]);
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

    #[test]
    fn test_convert_rgb_to_ycbcr_simd() {
        // Test the SIMD version of RGB to YCbCr conversion
        // Use a size that exercises both SIMD (8-pixel chunks) and scalar paths
        let width = 17; // Not a multiple of 8 to test scalar path
        let height = 3;
        let num_pixels = width * height;

        let mut rgb = vec![0u8; num_pixels * 3];
        let mut y_out = vec![0u8; num_pixels];
        let mut cb_out = vec![0u8; num_pixels];
        let mut cr_out = vec![0u8; num_pixels];

        // Fill with various colors
        for i in 0..num_pixels {
            rgb[i * 3] = ((i * 7) % 256) as u8; // R
            rgb[i * 3 + 1] = ((i * 11) % 256) as u8; // G
            rgb[i * 3 + 2] = ((i * 13) % 256) as u8; // B
        }

        convert_rgb_to_ycbcr(&rgb, &mut y_out, &mut cb_out, &mut cr_out, width, height);

        // Verify against scalar implementation
        // With fast-yuv feature, the yuv crate uses 15-bit precision vs our 16-bit,
        // resulting in ±1 level difference which is invisible after JPEG quantization
        #[cfg(feature = "fast-yuv")]
        let max_diff = 1i16;
        #[cfg(not(feature = "fast-yuv"))]
        let max_diff = 0i16;

        for i in 0..num_pixels {
            let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            let y_diff = (y_out[i] as i16 - y as i16).abs();
            let cb_diff = (cb_out[i] as i16 - cb as i16).abs();
            let cr_diff = (cr_out[i] as i16 - cr as i16).abs();
            assert!(
                y_diff <= max_diff,
                "Y mismatch at pixel {}: got {}, expected {} (diff {})",
                i,
                y_out[i],
                y,
                y_diff
            );
            assert!(
                cb_diff <= max_diff,
                "Cb mismatch at pixel {}: got {}, expected {} (diff {})",
                i,
                cb_out[i],
                cb,
                cb_diff
            );
            assert!(
                cr_diff <= max_diff,
                "Cr mismatch at pixel {}: got {}, expected {} (diff {})",
                i,
                cr_out[i],
                cr,
                cr_diff
            );
        }
    }

    #[test]
    fn test_convert_rgb_to_ycbcr_simd_exact_multiple() {
        // Test with exact multiple of 8 pixels (no scalar cleanup needed)
        let width = 16;
        let height = 2;
        let num_pixels = width * height;

        let mut rgb = vec![0u8; num_pixels * 3];
        let mut y_out = vec![0u8; num_pixels];
        let mut cb_out = vec![0u8; num_pixels];
        let mut cr_out = vec![0u8; num_pixels];

        // All white
        for i in 0..num_pixels * 3 {
            rgb[i] = 255;
        }

        convert_rgb_to_ycbcr(&rgb, &mut y_out, &mut cb_out, &mut cr_out, width, height);

        // White -> Y=255, Cb=128, Cr=128
        for i in 0..num_pixels {
            assert_eq!(y_out[i], 255);
            assert_eq!(cb_out[i], 128);
            assert_eq!(cr_out[i], 128);
        }
    }

    #[test]
    fn test_convert_rgb_to_gray_simd() {
        // Test the SIMD version of RGB to grayscale conversion
        let width = 19; // Not a multiple of 8
        let height = 2;
        let num_pixels = width * height;

        let mut rgb = vec![0u8; num_pixels * 3];
        let mut gray_out = vec![0u8; num_pixels];

        // Fill with gradient
        for i in 0..num_pixels {
            rgb[i * 3] = ((i * 5) % 256) as u8;
            rgb[i * 3 + 1] = ((i * 9) % 256) as u8;
            rgb[i * 3 + 2] = ((i * 3) % 256) as u8;
        }

        convert_rgb_to_gray(&rgb, &mut gray_out, width, height);

        // Verify against scalar implementation
        for i in 0..num_pixels {
            let y = rgb_to_gray(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            assert_eq!(gray_out[i], y, "Grayscale mismatch at pixel {}", i);
        }
    }

    #[test]
    fn test_convert_rgb_to_gray_simd_exact_multiple() {
        // Test with exact multiple of 8 pixels
        let width = 8;
        let height = 4;
        let num_pixels = width * height;

        let mut rgb = vec![0u8; num_pixels * 3];
        let mut gray_out = vec![0u8; num_pixels];

        // All black
        convert_rgb_to_gray(&rgb, &mut gray_out, width, height);

        for i in 0..num_pixels {
            assert_eq!(gray_out[i], 0);
        }

        // All white
        for i in 0..num_pixels * 3 {
            rgb[i] = 255;
        }
        convert_rgb_to_gray(&rgb, &mut gray_out, width, height);

        for i in 0..num_pixels {
            assert_eq!(gray_out[i], 255);
        }
    }

    #[test]
    fn test_fix_const_fn() {
        // Test that the fix() const fn produces expected values
        // FIX(0.5) should be approximately 32768 (1 << 15)
        assert_eq!(fix(0.5), 32768);

        // FIX(1.0) should be approximately 65536 (1 << 16)
        assert_eq!(fix(1.0), 65536);

        // FIX(0.0) should be 0
        assert_eq!(fix(0.0), 0);
    }

    #[test]
    fn test_rgb_to_ycbcr_c_compat_basic() {
        // Black
        let (y, cb, cr) = rgb_to_ycbcr_c_compat(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128); // Neutral chrominance (but with -1 offset internally)
        assert_eq!(cr, 128);

        // White
        let (y, cb, cr) = rgb_to_ycbcr_c_compat(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);

        // Gray 128
        let (y, cb, cr) = rgb_to_ycbcr_c_compat(128, 128, 128);
        assert_eq!(y, 128);
        // C compat has -1 in the rounding for Cb/Cr
        assert!((cb as i32 - 128).abs() <= 1);
        assert!((cr as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_c_compat_differs_from_default() {
        // The C-compat and default implementations should differ for some values
        // due to the `-1` adjustment in Cb/Cr. Pure gray doesn't show it clearly,
        // but colors with chroma should show ±1 differences.
        let mut c_compat_matches = 0;
        let mut c_compat_differs = 0;

        // Test a range of colors
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let (y1, cb1, cr1) = rgb_to_ycbcr(r, g, b);
                    let (y2, cb2, cr2) = rgb_to_ycbcr_c_compat(r, g, b);

                    // Y should always match (same formula)
                    assert_eq!(y1, y2, "Y differs for RGB({}, {}, {})", r, g, b);

                    // Cb/Cr may differ by up to 1 due to the -1 adjustment
                    if cb1 == cb2 && cr1 == cr2 {
                        c_compat_matches += 1;
                    } else {
                        c_compat_differs += 1;
                        assert!(
                            (cb1 as i32 - cb2 as i32).abs() <= 1,
                            "Cb differs by more than 1 for RGB({}, {}, {}): {} vs {}",
                            r,
                            g,
                            b,
                            cb1,
                            cb2
                        );
                        assert!(
                            (cr1 as i32 - cr2 as i32).abs() <= 1,
                            "Cr differs by more than 1 for RGB({}, {}, {}): {} vs {}",
                            r,
                            g,
                            b,
                            cr1,
                            cr2
                        );
                    }
                }
            }
        }

        // We expect SOME differences (not all values should match)
        assert!(
            c_compat_differs > 0,
            "C-compat should differ from default for some values"
        );
        eprintln!(
            "C-compat: {} matches, {} differs",
            c_compat_matches, c_compat_differs
        );
    }

    #[test]
    fn test_convert_rgb_to_ycbcr_c_compat() {
        let width = 16;
        let height = 4;
        let num_pixels = width * height;

        // Random-ish test data
        let mut rgb = vec![0u8; num_pixels * 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = ((i * 37 + 13) % 256) as u8;
        }

        let mut y_out = vec![0u8; num_pixels];
        let mut cb_out = vec![0u8; num_pixels];
        let mut cr_out = vec![0u8; num_pixels];

        convert_rgb_to_ycbcr_c_compat(&rgb, &mut y_out, &mut cb_out, &mut cr_out, num_pixels);

        // Verify against scalar reference
        for i in 0..num_pixels {
            let (y, cb, cr) = rgb_to_ycbcr_c_compat(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            assert_eq!(y_out[i], y);
            assert_eq!(cb_out[i], cb);
            assert_eq!(cr_out[i], cr);
        }
    }
}
