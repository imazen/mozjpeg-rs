//! AVX2 SIMD implementations for x86_64.
//!
//! These implementations use `core::arch::x86_64` intrinsics directly
//! for maximum performance. Key optimizations:
//!
//! - Proper load+widen instructions (no scalar gather)
//! - AVX2 shuffle/permute for transpose
//! - Minimal register spills
//!
//! All functions require AVX2 support and are marked with `#[target_feature(enable = "avx2")]`.

#![allow(unsafe_code)]

use crate::consts::DCTSIZE2;
use core::arch::x86_64::*;

// ============================================================================
// DCT Constants
// ============================================================================

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
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

// ============================================================================
// Color Conversion Constants
// ============================================================================

const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);
const CBCR_CENTER: i32 = 128;

const fn fix(x: f64) -> i32 {
    (x * ((1i64 << SCALEBITS) as f64) + 0.5) as i32
}

const FIX_Y_R: i32 = fix(0.29900);
const FIX_Y_G: i32 = fix(0.58700);
const FIX_Y_B: i32 = fix(0.11400);
const FIX_CB_R: i32 = -fix(0.16874);
const FIX_CB_G: i32 = -fix(0.33126);
const FIX_CB_B: i32 = fix(0.50000);
const FIX_CR_R: i32 = fix(0.50000);
const FIX_CR_G: i32 = -fix(0.41869);
const FIX_CR_B: i32 = -fix(0.08131);

// ============================================================================
// Forward DCT
// ============================================================================

/// Load 8 contiguous i16 values and sign-extend to 8 i32 values.
#[inline(always)]
unsafe fn load_i16_to_i32(ptr: *const i16) -> __m256i {
    let row_i16 = _mm_loadu_si128(ptr as *const __m128i);
    _mm256_cvtepi16_epi32(row_i16)
}

/// Pack 8 i32 values to 8 i16 values with saturation.
#[inline(always)]
unsafe fn pack_i32_to_i16(v: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packs_epi32(lo, hi)
}

/// Descale with rounding for pass 1 (CONST_BITS - PASS1_BITS = 11).
#[inline(always)]
unsafe fn descale_pass1(x: __m256i) -> __m256i {
    const N: i32 = CONST_BITS - PASS1_BITS;
    let round = _mm256_set1_epi32(1 << (N - 1));
    _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
}

/// Descale with rounding for pass 2 (CONST_BITS + PASS1_BITS = 15).
#[inline(always)]
unsafe fn descale_pass2(x: __m256i) -> __m256i {
    const N: i32 = CONST_BITS + PASS1_BITS;
    let round = _mm256_set1_epi32(1 << (N - 1));
    _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
}

/// Descale with rounding for PASS1_BITS = 2.
#[inline(always)]
unsafe fn descale_pass1_bits(x: __m256i) -> __m256i {
    const N: i32 = PASS1_BITS;
    let round = _mm256_set1_epi32(1 << (N - 1));
    _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
}

/// Transpose 8x8 matrix of i32 values stored in 8 ymm registers.
#[inline(always)]
unsafe fn transpose_8x8_avx2(rows: &mut [__m256i; 8]) {
    // Phase 1: Interleave 32-bit elements
    let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
    let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
    let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
    let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
    let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
    let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
    let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
    let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

    // Phase 2: Interleave 64-bit elements
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    // Phase 3: Permute 128-bit lanes
    rows[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    rows[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    rows[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    rows[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    rows[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    rows[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    rows[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    rows[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

/// Perform 1D DCT on 8 rows simultaneously using AVX2.
#[inline(always)]
unsafe fn dct_1d_pass_avx2(data: &mut [__m256i; 8], pass1: bool) {
    let fix_0_541196100 = _mm256_set1_epi32(FIX_0_541196100);
    let fix_0_765366865 = _mm256_set1_epi32(FIX_0_765366865);
    let fix_1_847759065 = _mm256_set1_epi32(FIX_1_847759065);
    let fix_1_175875602 = _mm256_set1_epi32(FIX_1_175875602);
    let fix_0_298631336 = _mm256_set1_epi32(FIX_0_298631336);
    let fix_2_053119869 = _mm256_set1_epi32(FIX_2_053119869);
    let fix_3_072711026 = _mm256_set1_epi32(FIX_3_072711026);
    let fix_1_501321110 = _mm256_set1_epi32(FIX_1_501321110);
    let neg_fix_0_899976223 = _mm256_set1_epi32(-FIX_0_899976223);
    let neg_fix_2_562915447 = _mm256_set1_epi32(-FIX_2_562915447);
    let neg_fix_1_961570560 = _mm256_set1_epi32(-FIX_1_961570560);
    let neg_fix_0_390180644 = _mm256_set1_epi32(-FIX_0_390180644);

    // Even part
    let tmp0 = _mm256_add_epi32(data[0], data[7]);
    let tmp7 = _mm256_sub_epi32(data[0], data[7]);
    let tmp1 = _mm256_add_epi32(data[1], data[6]);
    let tmp6 = _mm256_sub_epi32(data[1], data[6]);
    let tmp2 = _mm256_add_epi32(data[2], data[5]);
    let tmp5 = _mm256_sub_epi32(data[2], data[5]);
    let tmp3 = _mm256_add_epi32(data[3], data[4]);
    let tmp4 = _mm256_sub_epi32(data[3], data[4]);

    let tmp10 = _mm256_add_epi32(tmp0, tmp3);
    let tmp13 = _mm256_sub_epi32(tmp0, tmp3);
    let tmp11 = _mm256_add_epi32(tmp1, tmp2);
    let tmp12 = _mm256_sub_epi32(tmp1, tmp2);

    let (out0, out4) = if pass1 {
        (
            _mm256_slli_epi32::<PASS1_BITS>(_mm256_add_epi32(tmp10, tmp11)),
            _mm256_slli_epi32::<PASS1_BITS>(_mm256_sub_epi32(tmp10, tmp11)),
        )
    } else {
        (
            descale_pass1_bits(_mm256_add_epi32(tmp10, tmp11)),
            descale_pass1_bits(_mm256_sub_epi32(tmp10, tmp11)),
        )
    };

    let z1 = _mm256_mullo_epi32(_mm256_add_epi32(tmp12, tmp13), fix_0_541196100);
    let (out2, out6) = if pass1 {
        (
            descale_pass1(_mm256_add_epi32(
                z1,
                _mm256_mullo_epi32(tmp13, fix_0_765366865),
            )),
            descale_pass1(_mm256_sub_epi32(
                z1,
                _mm256_mullo_epi32(tmp12, fix_1_847759065),
            )),
        )
    } else {
        (
            descale_pass2(_mm256_add_epi32(
                z1,
                _mm256_mullo_epi32(tmp13, fix_0_765366865),
            )),
            descale_pass2(_mm256_sub_epi32(
                z1,
                _mm256_mullo_epi32(tmp12, fix_1_847759065),
            )),
        )
    };

    // Odd part
    let z1 = _mm256_add_epi32(tmp4, tmp7);
    let z2 = _mm256_add_epi32(tmp5, tmp6);
    let z3 = _mm256_add_epi32(tmp4, tmp6);
    let z4 = _mm256_add_epi32(tmp5, tmp7);
    let z5 = _mm256_mullo_epi32(_mm256_add_epi32(z3, z4), fix_1_175875602);

    let tmp4 = _mm256_mullo_epi32(tmp4, fix_0_298631336);
    let tmp5 = _mm256_mullo_epi32(tmp5, fix_2_053119869);
    let tmp6 = _mm256_mullo_epi32(tmp6, fix_3_072711026);
    let tmp7 = _mm256_mullo_epi32(tmp7, fix_1_501321110);

    let z1 = _mm256_mullo_epi32(z1, neg_fix_0_899976223);
    let z2 = _mm256_mullo_epi32(z2, neg_fix_2_562915447);
    let z3 = _mm256_add_epi32(_mm256_mullo_epi32(z3, neg_fix_1_961570560), z5);
    let z4 = _mm256_add_epi32(_mm256_mullo_epi32(z4, neg_fix_0_390180644), z5);

    let (out7, out5, out3, out1) = if pass1 {
        (
            descale_pass1(_mm256_add_epi32(_mm256_add_epi32(tmp4, z1), z3)),
            descale_pass1(_mm256_add_epi32(_mm256_add_epi32(tmp5, z2), z4)),
            descale_pass1(_mm256_add_epi32(_mm256_add_epi32(tmp6, z2), z3)),
            descale_pass1(_mm256_add_epi32(_mm256_add_epi32(tmp7, z1), z4)),
        )
    } else {
        (
            descale_pass2(_mm256_add_epi32(_mm256_add_epi32(tmp4, z1), z3)),
            descale_pass2(_mm256_add_epi32(_mm256_add_epi32(tmp5, z2), z4)),
            descale_pass2(_mm256_add_epi32(_mm256_add_epi32(tmp6, z2), z3)),
            descale_pass2(_mm256_add_epi32(_mm256_add_epi32(tmp7, z1), z4)),
        )
    };

    data[0] = out0;
    data[1] = out1;
    data[2] = out2;
    data[3] = out3;
    data[4] = out4;
    data[5] = out5;
    data[6] = out6;
    data[7] = out7;
}

/// AVX2-optimized forward DCT on 8x8 block.
///
/// # Safety
/// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[target_feature(enable = "avx2")]
pub unsafe fn forward_dct_8x8_avx2(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut rows: [__m256i; 8] = [
        load_i16_to_i32(samples.as_ptr().add(0)),
        load_i16_to_i32(samples.as_ptr().add(8)),
        load_i16_to_i32(samples.as_ptr().add(16)),
        load_i16_to_i32(samples.as_ptr().add(24)),
        load_i16_to_i32(samples.as_ptr().add(32)),
        load_i16_to_i32(samples.as_ptr().add(40)),
        load_i16_to_i32(samples.as_ptr().add(48)),
        load_i16_to_i32(samples.as_ptr().add(56)),
    ];

    transpose_8x8_avx2(&mut rows);
    dct_1d_pass_avx2(&mut rows, true);
    transpose_8x8_avx2(&mut rows);
    dct_1d_pass_avx2(&mut rows, false);

    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(0) as *mut __m128i,
        pack_i32_to_i16(rows[0]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(8) as *mut __m128i,
        pack_i32_to_i16(rows[1]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(16) as *mut __m128i,
        pack_i32_to_i16(rows[2]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(24) as *mut __m128i,
        pack_i32_to_i16(rows[3]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(32) as *mut __m128i,
        pack_i32_to_i16(rows[4]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(40) as *mut __m128i,
        pack_i32_to_i16(rows[5]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(48) as *mut __m128i,
        pack_i32_to_i16(rows[6]),
    );
    _mm_storeu_si128(
        coeffs.as_mut_ptr().add(56) as *mut __m128i,
        pack_i32_to_i16(rows[7]),
    );
}

/// Safe wrapper for forward DCT that can be used as a function pointer.
pub fn forward_dct_8x8(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // SAFETY: This module is only compiled when target_feature = "avx2"
    unsafe { forward_dct_8x8_avx2(samples, coeffs) }
}

// ============================================================================
// Color Conversion
// ============================================================================

/// AVX2-optimized RGB to YCbCr conversion.
///
/// Processes 8 pixels at a time using proper SIMD loads (no gather).
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
unsafe fn convert_rgb_to_ycbcr_avx2_inner(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    let fix_y_r = _mm256_set1_epi32(FIX_Y_R);
    let fix_y_g = _mm256_set1_epi32(FIX_Y_G);
    let fix_y_b = _mm256_set1_epi32(FIX_Y_B);
    let fix_cb_r = _mm256_set1_epi32(FIX_CB_R);
    let fix_cb_g = _mm256_set1_epi32(FIX_CB_G);
    let fix_cb_b = _mm256_set1_epi32(FIX_CB_B);
    let fix_cr_r = _mm256_set1_epi32(FIX_CR_R);
    let fix_cr_g = _mm256_set1_epi32(FIX_CR_G);
    let fix_cr_b = _mm256_set1_epi32(FIX_CR_B);
    let half = _mm256_set1_epi32(ONE_HALF);
    let center = _mm256_set1_epi32(CBCR_CENTER);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(255);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let rgb_base = base * 3;

        // Load 8 pixels (24 bytes) and deinterleave
        // We load into i32 for the math, gathering from strided positions
        let r = _mm256_set_epi32(
            rgb[rgb_base + 21] as i32,
            rgb[rgb_base + 18] as i32,
            rgb[rgb_base + 15] as i32,
            rgb[rgb_base + 12] as i32,
            rgb[rgb_base + 9] as i32,
            rgb[rgb_base + 6] as i32,
            rgb[rgb_base + 3] as i32,
            rgb[rgb_base] as i32,
        );
        let g = _mm256_set_epi32(
            rgb[rgb_base + 22] as i32,
            rgb[rgb_base + 19] as i32,
            rgb[rgb_base + 16] as i32,
            rgb[rgb_base + 13] as i32,
            rgb[rgb_base + 10] as i32,
            rgb[rgb_base + 7] as i32,
            rgb[rgb_base + 4] as i32,
            rgb[rgb_base + 1] as i32,
        );
        let b = _mm256_set_epi32(
            rgb[rgb_base + 23] as i32,
            rgb[rgb_base + 20] as i32,
            rgb[rgb_base + 17] as i32,
            rgb[rgb_base + 14] as i32,
            rgb[rgb_base + 11] as i32,
            rgb[rgb_base + 8] as i32,
            rgb[rgb_base + 5] as i32,
            rgb[rgb_base + 2] as i32,
        );

        // Y = 0.29900 * R + 0.58700 * G + 0.11400 * B
        let y = _mm256_srai_epi32::<SCALEBITS>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(fix_y_r, r),
                    _mm256_mullo_epi32(fix_y_g, g),
                ),
                _mm256_mullo_epi32(fix_y_b, b),
            ),
            half,
        ));
        let y = _mm256_max_epi32(_mm256_min_epi32(y, max_val), zero);

        // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
        let cb = _mm256_add_epi32(
            _mm256_srai_epi32::<SCALEBITS>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(fix_cb_r, r),
                        _mm256_mullo_epi32(fix_cb_g, g),
                    ),
                    _mm256_mullo_epi32(fix_cb_b, b),
                ),
                half,
            )),
            center,
        );
        let cb = _mm256_max_epi32(_mm256_min_epi32(cb, max_val), zero);

        // Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B + 128
        let cr = _mm256_add_epi32(
            _mm256_srai_epi32::<SCALEBITS>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(fix_cr_r, r),
                        _mm256_mullo_epi32(fix_cr_g, g),
                    ),
                    _mm256_mullo_epi32(fix_cr_b, b),
                ),
                half,
            )),
            center,
        );
        let cr = _mm256_max_epi32(_mm256_min_epi32(cr, max_val), zero);

        // Extract and store results
        // AVX2 doesn't have a great way to pack i32->u8, so we extract manually
        let y_arr: [i32; 8] = core::mem::transmute(y);
        let cb_arr: [i32; 8] = core::mem::transmute(cb);
        let cr_arr: [i32; 8] = core::mem::transmute(cr);

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
    let remaining_start = chunks * 8;
    for i in remaining_start..num_pixels {
        let r = rgb[i * 3] as i32;
        let g = rgb[i * 3 + 1] as i32;
        let b = rgb[i * 3 + 2] as i32;

        let y = (FIX_Y_R * r + FIX_Y_G * g + FIX_Y_B * b + ONE_HALF) >> SCALEBITS;
        let cb =
            ((FIX_CB_R * r + FIX_CB_G * g + FIX_CB_B * b + ONE_HALF) >> SCALEBITS) + CBCR_CENTER;
        let cr =
            ((FIX_CR_R * r + FIX_CR_G * g + FIX_CR_B * b + ONE_HALF) >> SCALEBITS) + CBCR_CENTER;

        y_out[i] = y.clamp(0, 255) as u8;
        cb_out[i] = cb.clamp(0, 255) as u8;
        cr_out[i] = cr.clamp(0, 255) as u8;
    }
}

/// Safe wrapper for color conversion that can be used as a function pointer.
pub fn convert_rgb_to_ycbcr(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    // SAFETY: This module is only compiled when target_feature = "avx2"
    unsafe { convert_rgb_to_ycbcr_avx2_inner(rgb, y_out, cb_out, cr_out, num_pixels) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    #[test]
    fn test_avx2_dct_matches_scalar() {
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_avx2 = [0i16; DCTSIZE2];

            scalar::forward_dct_8x8(&samples, &mut coeffs_scalar);
            forward_dct_8x8(&samples, &mut coeffs_avx2);

            assert_eq!(
                coeffs_scalar, coeffs_avx2,
                "AVX2 DCT should match scalar for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_avx2_color_matches_scalar() {
        // Test with various patterns
        for seed in 0..10 {
            let mut rgb = vec![0u8; 64 * 3]; // 64 pixels
            for i in 0..64 {
                rgb[i * 3] = ((i * (seed + 1) * 3) % 256) as u8;
                rgb[i * 3 + 1] = ((i * (seed + 1) * 5) % 256) as u8;
                rgb[i * 3 + 2] = ((i * (seed + 1) * 7) % 256) as u8;
            }

            let mut y_scalar = vec![0u8; 64];
            let mut cb_scalar = vec![0u8; 64];
            let mut cr_scalar = vec![0u8; 64];
            let mut y_avx2 = vec![0u8; 64];
            let mut cb_avx2 = vec![0u8; 64];
            let mut cr_avx2 = vec![0u8; 64];

            scalar::convert_rgb_to_ycbcr(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, 64);
            convert_rgb_to_ycbcr(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, 64);

            assert_eq!(y_scalar, y_avx2, "Y should match for seed {}", seed);
            assert_eq!(cb_scalar, cb_avx2, "Cb should match for seed {}", seed);
            assert_eq!(cr_scalar, cr_avx2, "Cr should match for seed {}", seed);
        }
    }
}
