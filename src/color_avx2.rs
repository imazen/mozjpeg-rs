//! AVX2-optimized RGB to YCbCr color conversion.
//!
//! **SUPERSEDED**: This module has been superseded by the `yuv` crate, which is
//! used by default when the `fast-yuv` feature is enabled. The `yuv` crate
//! provides ~60% better performance with support for AVX-512, AVX2, SSE, NEON,
//! and WASM SIMD. The precision difference (±1 level) is invisible after JPEG
//! quantization.
//!
//! This module is kept for reference and can be enabled by disabling the
//! `fast-yuv` feature. It provides an AVX2-optimized implementation of RGB→YCbCr
//! color conversion based on libjpeg-turbo's jccolext-avx2.asm.
//!
//! Key optimizations:
//! 1. SIMD loads of contiguous RGB data (32 pixels = 96 bytes per iteration)
//! 2. Efficient de-interleaving using vpunpcklbw/hi, vpslldq, vperm2i128
//! 3. vpmaddwd for coefficient multiply-accumulate
//! 4. Direct storage to planar output buffers
//!
//! Note: Functions with `#[target_feature]` must be `unsafe` in Rust <1.92,
//! but the memory operations use archmage's safe wrappers internally.

// Allow unsafe for #[target_feature] functions - memory ops use safe archmage wrappers
#![allow(unsafe_code)]
#![allow(clippy::too_many_lines)]

#[cfg(target_arch = "x86_64")]
use archmage::mem::avx as mem_avx;
#[cfg(target_arch = "x86_64")]
use archmage::tokens::x86::Avx2Token;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fixed-point precision bits (matching libjpeg-turbo)
const SCALEBITS: i32 = 16;

// libjpeg-turbo uses different coefficient organization for vpmaddwd:
// Y  = 0.29900 * R + 0.58700 * G + 0.11400 * B
// They split G coefficient: 0.58700 = 0.33700 + 0.25000
// So Y = 0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G
//
// For vpmaddwd, we pair coefficients:
// - [R, G] with [F_0_299, F_0_337] for Y
// - [B, G] with [F_0_114, F_0_250] for Y
// etc.

const F_0_081: i32 = 5329; // FIX(0.08131)
const F_0_114: i32 = 7471; // FIX(0.11400)
const F_0_168: i32 = 11059; // FIX(0.16874)
const F_0_250: i32 = 16384; // FIX(0.25000)
const F_0_299: i32 = 19595; // FIX(0.29900)
const F_0_331: i32 = 21709; // FIX(0.33126)
const F_0_418: i32 = 27439; // FIX(0.41869)
const F_0_587: i32 = 38470; // FIX(0.58700)
const F_0_337: i32 = F_0_587 - F_0_250; // FIX(0.58700) - FIX(0.25000) = 22086

/// Helper to pack two i16 values into an i32 for vpmaddwd constant
#[inline]
const fn pack_i16_pair(lo: i32, hi: i32) -> i32 {
    ((hi as i16 as u16 as i32) << 16) | (lo as i16 as u16 as i32)
}

/// Convert RGB to YCbCr using AVX2 intrinsics.
///
/// Processes 32 pixels per iteration for optimal performance.
/// Uses archmage for safe SIMD memory operations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_rgb_to_ycbcr_avx2_impl(
    token: Avx2Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    debug_assert_eq!(rgb.len(), num_pixels * 3);
    debug_assert_eq!(y_out.len(), num_pixels);
    debug_assert_eq!(cb_out.len(), num_pixels);
    debug_assert_eq!(cr_out.len(), num_pixels);

    // Constants for color conversion
    // PW_F0299_F0337: [F_0_299, F_0_337] repeated 8 times for vpmaddwd
    let pw_f0299_f0337 = _mm256_set1_epi32(pack_i16_pair(F_0_299, F_0_337));
    // PW_F0114_F0250: [F_0_114, F_0_250] repeated 8 times
    let pw_f0114_f0250 = _mm256_set1_epi32(pack_i16_pair(F_0_114, F_0_250));
    // PW_MF016_MF033: [-F_0_168, -F_0_331] for Cb
    let pw_mf016_mf033 = _mm256_set1_epi32(pack_i16_pair(-F_0_168, -F_0_331));
    // PW_MF008_MF041: [-F_0_081, -F_0_418] for Cr
    let pw_mf008_mf041 = _mm256_set1_epi32(pack_i16_pair(-F_0_081, -F_0_418));

    // PD_ONEHALFM1_CJ: (1 << (SCALEBITS-1)) - 1 + (128 << SCALEBITS) for Cb/Cr rounding + center
    let pd_onehalfm1_cj = _mm256_set1_epi32((1 << (SCALEBITS - 1)) - 1 + (128 << SCALEBITS));
    // PD_ONEHALF: (1 << (SCALEBITS-1)) for Y rounding
    let pd_onehalf = _mm256_set1_epi32(1 << (SCALEBITS - 1));

    let chunks = num_pixels / 32;

    for chunk in 0..chunks {
        let rgb_base = chunk * 96;
        let out_base = chunk * 32;

        // Load 96 bytes (32 RGB pixels) using 3 YMM loads with archmage safe wrappers
        let ymm_a = mem_avx::_mm256_loadu_si256(
            token,
            <&[u8] as TryInto<&[u8; 32]>>::try_into(&rgb[rgb_base..rgb_base + 32]).unwrap(),
        );
        let ymm_f = mem_avx::_mm256_loadu_si256(
            token,
            <&[u8] as TryInto<&[u8; 32]>>::try_into(&rgb[rgb_base + 32..rgb_base + 64]).unwrap(),
        );
        let ymm_b = mem_avx::_mm256_loadu_si256(
            token,
            <&[u8] as TryInto<&[u8; 32]>>::try_into(&rgb[rgb_base + 64..rgb_base + 96]).unwrap(),
        );

        // De-interleave RGB using the libjpeg-turbo shuffle sequence.
        // This complex sequence separates interleaved RGB into separate R, G, B vectors.
        // See jccolext-avx2.asm lines 150-228 for the original algorithm.

        // Step 1: Reorganize across 128-bit lanes
        let ymm_c = ymm_a;
        let ymm_a = _mm256_inserti128_si256(ymm_f, _mm256_castsi256_si128(ymm_a), 0);
        let ymm_c = _mm256_inserti128_si256(ymm_c, _mm256_castsi256_si128(ymm_b), 0);
        let ymm_b = _mm256_inserti128_si256(ymm_b, _mm256_castsi256_si128(ymm_f), 0);
        let ymm_f = _mm256_permute2x128_si256(ymm_c, ymm_c, 0x01);

        // Step 2: First level of interleaving
        let ymm_g = ymm_a;
        let ymm_a = _mm256_slli_si256(ymm_a, 8);
        let ymm_g = _mm256_srli_si256(ymm_g, 8);

        let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_f);
        let ymm_f = _mm256_slli_si256(ymm_f, 8);

        let ymm_g = _mm256_unpacklo_epi8(ymm_g, ymm_b);
        let ymm_f = _mm256_unpackhi_epi8(ymm_f, ymm_b);

        // Step 3: Second level of interleaving
        let ymm_d = ymm_a;
        let ymm_a = _mm256_slli_si256(ymm_a, 8);
        let ymm_d = _mm256_srli_si256(ymm_d, 8);

        let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_g);
        let ymm_g = _mm256_slli_si256(ymm_g, 8);

        let ymm_d = _mm256_unpacklo_epi8(ymm_d, ymm_f);
        let ymm_g = _mm256_unpackhi_epi8(ymm_g, ymm_f);

        // Step 4: Third level of interleaving
        let ymm_e = ymm_a;
        let ymm_a = _mm256_slli_si256(ymm_a, 8);
        let ymm_e = _mm256_srli_si256(ymm_e, 8);

        let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_d);
        let ymm_d = _mm256_slli_si256(ymm_d, 8);

        let ymm_e = _mm256_unpacklo_epi8(ymm_e, ymm_g);
        let ymm_d = _mm256_unpackhi_epi8(ymm_d, ymm_g);

        // Step 5: Final unpacking - separate into R, G, B
        let ymm_h = _mm256_setzero_si256();

        // ymm_a contains interleaved data, unpack to get R even/odd
        let ymm_c = ymm_a;
        let ymm_a = _mm256_unpacklo_epi8(ymm_a, ymm_h); // R even (16 values as i16)
        let ymm_c = _mm256_unpackhi_epi8(ymm_c, ymm_h); // G even

        let ymm_b_tmp = ymm_e;
        let ymm_e = _mm256_unpacklo_epi8(ymm_e, ymm_h); // B even
        let ymm_b_new = _mm256_unpackhi_epi8(ymm_b_tmp, ymm_h); // R odd

        let ymm_f_tmp = ymm_d;
        let ymm_d = _mm256_unpacklo_epi8(ymm_d, ymm_h); // G odd
        let ymm_f = _mm256_unpackhi_epi8(ymm_f_tmp, ymm_h); // B odd

        // Now we have:
        // ymm_a = R even (16 i16)
        // ymm_c = G even (16 i16)
        // ymm_e = B even (16 i16)
        // ymm_b_new = R odd (16 i16)
        // ymm_d = G odd (16 i16)
        // ymm_f = B odd (16 i16)

        // Process ODD pixels first (for pipelining)
        // Interleave R_odd with G_odd for vpmaddwd
        let ymm_6 = ymm_b_new;
        let ymm_1 = _mm256_unpacklo_epi16(ymm_b_new, ymm_d);
        let ymm_6 = _mm256_unpackhi_epi16(ymm_6, ymm_d);
        let ymm_7 = ymm_1;
        let ymm_4 = ymm_6;

        // ROL*FIX(0.299) + GOL*FIX(0.337)
        let ymm_1 = _mm256_madd_epi16(ymm_1, pw_f0299_f0337);
        // ROH*FIX(0.299) + GOH*FIX(0.337)
        let ymm_6 = _mm256_madd_epi16(ymm_6, pw_f0299_f0337);
        // ROL*-FIX(0.168) + GOL*-FIX(0.331) for Cb
        let ymm_7 = _mm256_madd_epi16(ymm_7, pw_mf016_mf033);
        // ROH*-FIX(0.168) + GOH*-FIX(0.331) for Cb
        let ymm_4 = _mm256_madd_epi16(ymm_4, pw_mf016_mf033);

        // Save for later
        let wk4 = ymm_1; // ROL*FIX(0.299) + GOL*FIX(0.337)
        let wk5 = ymm_6; // ROH*FIX(0.299) + GOH*FIX(0.337)

        // B odd contribution to Cb: BOL*FIX(0.500)
        let ymm_1 = _mm256_setzero_si256();
        let ymm_6_z = _mm256_setzero_si256();
        let ymm_1 = _mm256_unpacklo_epi16(ymm_1, ymm_f);
        let ymm_6_z = _mm256_unpackhi_epi16(ymm_6_z, ymm_f);
        let ymm_1 = _mm256_srli_epi32(ymm_1, 1); // BOL * 0.5 (shift instead of multiply)
        let ymm_6_z = _mm256_srli_epi32(ymm_6_z, 1); // BOH * 0.5

        // Cb odd = ROL*-FIX(0.168) + GOL*-FIX(0.331) + BOL*FIX(0.500) + rounding + center
        let ymm_7 = _mm256_add_epi32(ymm_7, ymm_1);
        let ymm_4 = _mm256_add_epi32(ymm_4, ymm_6_z);
        let ymm_7 = _mm256_add_epi32(ymm_7, pd_onehalfm1_cj);
        let ymm_4 = _mm256_add_epi32(ymm_4, pd_onehalfm1_cj);
        let ymm_7 = _mm256_srai_epi32(ymm_7, SCALEBITS);
        let ymm_4 = _mm256_srai_epi32(ymm_4, SCALEBITS);
        let cb_odd = _mm256_packs_epi32(ymm_7, ymm_4);

        // Process EVEN pixels
        let ymm_6 = ymm_a;
        let ymm_0 = _mm256_unpacklo_epi16(ymm_a, ymm_c);
        let ymm_6 = _mm256_unpackhi_epi16(ymm_6, ymm_c);
        let ymm_5 = ymm_0;
        let ymm_4 = ymm_6;

        // REL*FIX(0.299) + GEL*FIX(0.337)
        let ymm_0 = _mm256_madd_epi16(ymm_0, pw_f0299_f0337);
        // REH*FIX(0.299) + GEH*FIX(0.337)
        let ymm_6 = _mm256_madd_epi16(ymm_6, pw_f0299_f0337);
        // REL*-FIX(0.168) + GEL*-FIX(0.331) for Cb
        let ymm_5 = _mm256_madd_epi16(ymm_5, pw_mf016_mf033);
        // REH*-FIX(0.168) + GEH*-FIX(0.331) for Cb
        let ymm_4 = _mm256_madd_epi16(ymm_4, pw_mf016_mf033);

        let wk6 = ymm_0; // REL*FIX(0.299) + GEL*FIX(0.337)
        let wk7 = ymm_6; // REH*FIX(0.299) + GEH*FIX(0.337)

        // B even contribution to Cb
        let ymm_0 = _mm256_setzero_si256();
        let ymm_6_z = _mm256_setzero_si256();
        let ymm_0 = _mm256_unpacklo_epi16(ymm_0, ymm_e);
        let ymm_6_z = _mm256_unpackhi_epi16(ymm_6_z, ymm_e);
        let ymm_0 = _mm256_srli_epi32(ymm_0, 1);
        let ymm_6_z = _mm256_srli_epi32(ymm_6_z, 1);

        let ymm_5 = _mm256_add_epi32(ymm_5, ymm_0);
        let ymm_4 = _mm256_add_epi32(ymm_4, ymm_6_z);
        let ymm_5 = _mm256_add_epi32(ymm_5, pd_onehalfm1_cj);
        let ymm_4 = _mm256_add_epi32(ymm_4, pd_onehalfm1_cj);
        let ymm_5 = _mm256_srai_epi32(ymm_5, SCALEBITS);
        let ymm_4 = _mm256_srai_epi32(ymm_4, SCALEBITS);
        let cb_even = _mm256_packs_epi32(ymm_5, ymm_4);

        // Interleave Cb even and odd, store
        let cb_7 = _mm256_slli_epi16(cb_odd, 8);
        let cb_5 = _mm256_or_si256(cb_even, cb_7);
        mem_avx::_mm256_storeu_si256(
            token,
            <&mut [u8] as TryInto<&mut [u8; 32]>>::try_into(&mut cb_out[out_base..out_base + 32])
                .unwrap(),
            cb_5,
        );

        // Now compute Y using saved R*0.299+G*0.337 values
        // Need B*0.114 + G*0.250

        // Y odd: B_odd with G_odd
        let ymm_4 = ymm_f;
        let ymm_0 = _mm256_unpacklo_epi16(ymm_f, ymm_d);
        let ymm_4 = _mm256_unpackhi_epi16(ymm_4, ymm_d);
        let ymm_7 = ymm_0;
        let ymm_5 = ymm_4;

        // BOL*FIX(0.114) + GOL*FIX(0.250)
        let ymm_0 = _mm256_madd_epi16(ymm_0, pw_f0114_f0250);
        // BOH*FIX(0.114) + GOH*FIX(0.250)
        let ymm_4 = _mm256_madd_epi16(ymm_4, pw_f0114_f0250);
        // BOL*-FIX(0.081) + GOL*-FIX(0.418)
        let ymm_7 = _mm256_madd_epi16(ymm_7, pw_mf008_mf041);
        // BOH*-FIX(0.081) + GOH*-FIX(0.418)
        let ymm_5 = _mm256_madd_epi16(ymm_5, pw_mf008_mf041);

        // Y odd = (R*0.299+G*0.337) + (B*0.114+G*0.250) + rounding
        let ymm_0 = _mm256_add_epi32(ymm_0, wk4);
        let ymm_4 = _mm256_add_epi32(ymm_4, wk5);
        let ymm_0 = _mm256_add_epi32(ymm_0, pd_onehalf);
        let ymm_4 = _mm256_add_epi32(ymm_4, pd_onehalf);
        let ymm_0 = _mm256_srai_epi32(ymm_0, SCALEBITS);
        let ymm_4 = _mm256_srai_epi32(ymm_4, SCALEBITS);
        let y_odd = _mm256_packs_epi32(ymm_0, ymm_4);

        // Cr odd: R_odd * 0.5 + B*-0.081 + G*-0.418
        let ymm_3 = _mm256_setzero_si256();
        let ymm_4 = _mm256_setzero_si256();
        let ymm_3 = _mm256_unpacklo_epi16(ymm_3, ymm_b_new);
        let ymm_4 = _mm256_unpackhi_epi16(ymm_4, ymm_b_new);
        let ymm_3 = _mm256_srli_epi32(ymm_3, 1);
        let ymm_4 = _mm256_srli_epi32(ymm_4, 1);

        let ymm_7 = _mm256_add_epi32(ymm_7, ymm_3);
        let ymm_5 = _mm256_add_epi32(ymm_5, ymm_4);
        let ymm_7 = _mm256_add_epi32(ymm_7, pd_onehalfm1_cj);
        let ymm_5 = _mm256_add_epi32(ymm_5, pd_onehalfm1_cj);
        let ymm_7 = _mm256_srai_epi32(ymm_7, SCALEBITS);
        let ymm_5 = _mm256_srai_epi32(ymm_5, SCALEBITS);
        let cr_odd = _mm256_packs_epi32(ymm_7, ymm_5);

        // Y even: B_even with G_even
        let ymm_4 = ymm_e;
        let ymm_6 = _mm256_unpacklo_epi16(ymm_e, ymm_c);
        let ymm_4 = _mm256_unpackhi_epi16(ymm_4, ymm_c);
        let ymm_1 = ymm_6;
        let ymm_5 = ymm_4;

        let ymm_6 = _mm256_madd_epi16(ymm_6, pw_f0114_f0250);
        let ymm_4 = _mm256_madd_epi16(ymm_4, pw_f0114_f0250);
        let ymm_1 = _mm256_madd_epi16(ymm_1, pw_mf008_mf041);
        let ymm_5 = _mm256_madd_epi16(ymm_5, pw_mf008_mf041);

        let ymm_6 = _mm256_add_epi32(ymm_6, wk6);
        let ymm_4 = _mm256_add_epi32(ymm_4, wk7);
        let ymm_6 = _mm256_add_epi32(ymm_6, pd_onehalf);
        let ymm_4 = _mm256_add_epi32(ymm_4, pd_onehalf);
        let ymm_6 = _mm256_srai_epi32(ymm_6, SCALEBITS);
        let ymm_4 = _mm256_srai_epi32(ymm_4, SCALEBITS);
        let y_even = _mm256_packs_epi32(ymm_6, ymm_4);

        // Interleave Y even and odd, store
        let y_0 = _mm256_slli_epi16(y_odd, 8);
        let y_6 = _mm256_or_si256(y_even, y_0);
        mem_avx::_mm256_storeu_si256(
            token,
            <&mut [u8] as TryInto<&mut [u8; 32]>>::try_into(&mut y_out[out_base..out_base + 32])
                .unwrap(),
            y_6,
        );

        // Cr even
        let ymm_2 = _mm256_setzero_si256();
        let ymm_4 = _mm256_setzero_si256();
        let ymm_2 = _mm256_unpacklo_epi16(ymm_2, ymm_a);
        let ymm_4 = _mm256_unpackhi_epi16(ymm_4, ymm_a);
        let ymm_2 = _mm256_srli_epi32(ymm_2, 1);
        let ymm_4 = _mm256_srli_epi32(ymm_4, 1);

        let ymm_1 = _mm256_add_epi32(ymm_1, ymm_2);
        let ymm_5 = _mm256_add_epi32(ymm_5, ymm_4);
        let ymm_1 = _mm256_add_epi32(ymm_1, pd_onehalfm1_cj);
        let ymm_5 = _mm256_add_epi32(ymm_5, pd_onehalfm1_cj);
        let ymm_1 = _mm256_srai_epi32(ymm_1, SCALEBITS);
        let ymm_5 = _mm256_srai_epi32(ymm_5, SCALEBITS);
        let cr_even = _mm256_packs_epi32(ymm_1, ymm_5);

        // Interleave Cr even and odd, store
        let cr_7 = _mm256_slli_epi16(cr_odd, 8);
        let cr_1 = _mm256_or_si256(cr_even, cr_7);
        mem_avx::_mm256_storeu_si256(
            token,
            <&mut [u8] as TryInto<&mut [u8; 32]>>::try_into(&mut cr_out[out_base..out_base + 32])
                .unwrap(),
            cr_1,
        );
    }

    // Handle remaining pixels with scalar code
    let remaining_start = chunks * 32;
    for i in remaining_start..num_pixels {
        let rgb_idx = i * 3;
        let r = rgb[rgb_idx] as i32;
        let g = rgb[rgb_idx + 1] as i32;
        let b = rgb[rgb_idx + 2] as i32;

        const FIX_0_29900: i32 = 19595;
        const FIX_0_58700: i32 = 38470;
        const FIX_0_11400: i32 = 7471;
        const FIX_0_16874: i32 = 11059;
        const FIX_0_33126: i32 = 21709;
        const FIX_0_50000: i32 = 32768;
        const FIX_0_41869: i32 = 27439;
        const FIX_0_08131: i32 = 5329;
        const ONE_HALF: i32 = 1 << 15;

        let y = (FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> 16;
        let cb = ((-FIX_0_16874 * r - FIX_0_33126 * g + FIX_0_50000 * b + ONE_HALF) >> 16) + 128;
        let cr = ((FIX_0_50000 * r - FIX_0_41869 * g - FIX_0_08131 * b + ONE_HALF) >> 16) + 128;

        y_out[i] = y.clamp(0, 255) as u8;
        cb_out[i] = cb.clamp(0, 255) as u8;
        cr_out[i] = cr.clamp(0, 255) as u8;
    }
}

/// Convert RGB to YCbCr using AVX2 intrinsics.
///
/// Processes 32 pixels per iteration for optimal performance.
///
/// # Safety
///
/// Caller must ensure AVX2 is supported.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn convert_rgb_to_ycbcr_avx2(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    use archmage::SimdToken;
    // SAFETY: Token is guaranteed valid since we're in a #[target_feature(enable = "avx2")] function
    let token = Avx2Token::forge_token_dangerously();
    convert_rgb_to_ycbcr_avx2_impl(token, rgb, y_out, cb_out, cr_out, num_pixels);
}

/// Runtime dispatch wrapper for AVX2 color conversion.
///
/// Falls back to the scalar implementation if AVX2 is not available.
#[cfg(target_arch = "x86_64")]
pub fn convert_rgb_to_ycbcr_dispatch(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    use archmage::SimdToken;
    if let Some(token) = Avx2Token::try_new() {
        // SAFETY: Token proves AVX2 is available
        unsafe {
            convert_rgb_to_ycbcr_avx2_impl(token, rgb, y_out, cb_out, cr_out, num_pixels);
        }
    } else {
        // Fallback to scalar
        for i in 0..num_pixels {
            let (y, cb, cr) =
                crate::color::rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            y_out[i] = y;
            cb_out[i] = cb;
            cr_out[i] = cr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_matches_scalar() {
        use archmage::SimdToken;
        if Avx2Token::try_new().is_none() {
            eprintln!("AVX2 not available, skipping test");
            return;
        }

        // Test with 64 pixels (2 iterations of 32)
        let num_pixels = 64;
        let rgb: Vec<u8> = (0..num_pixels * 3).map(|i| (i * 17) as u8).collect();

        let mut y_avx2 = vec![0u8; num_pixels];
        let mut cb_avx2 = vec![0u8; num_pixels];
        let mut cr_avx2 = vec![0u8; num_pixels];

        let mut y_scalar = vec![0u8; num_pixels];
        let mut cb_scalar = vec![0u8; num_pixels];
        let mut cr_scalar = vec![0u8; num_pixels];

        // Run AVX2 version
        unsafe {
            convert_rgb_to_ycbcr_avx2(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, num_pixels);
        }

        // Run scalar version
        for i in 0..num_pixels {
            let (y, cb, cr) =
                crate::color::rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            y_scalar[i] = y;
            cb_scalar[i] = cb;
            cr_scalar[i] = cr;
        }

        // Compare - allow ±1 for rounding differences
        for i in 0..num_pixels {
            assert!(
                (y_avx2[i] as i16 - y_scalar[i] as i16).abs() <= 1,
                "Y mismatch at {}: AVX2={}, scalar={}",
                i,
                y_avx2[i],
                y_scalar[i]
            );
            assert!(
                (cb_avx2[i] as i16 - cb_scalar[i] as i16).abs() <= 1,
                "Cb mismatch at {}: AVX2={}, scalar={}",
                i,
                cb_avx2[i],
                cb_scalar[i]
            );
            assert!(
                (cr_avx2[i] as i16 - cr_scalar[i] as i16).abs() <= 1,
                "Cr mismatch at {}: AVX2={}, scalar={}",
                i,
                cr_avx2[i],
                cr_scalar[i]
            );
        }
    }

    #[test]
    fn test_avx2_remainder_handling() {
        use archmage::SimdToken;
        if Avx2Token::try_new().is_none() {
            eprintln!("AVX2 not available, skipping test");
            return;
        }

        // Test with non-multiple of 32 pixels
        let num_pixels = 50;
        let rgb: Vec<u8> = (0..num_pixels * 3).map(|i| (i * 11) as u8).collect();

        let mut y_avx2 = vec![0u8; num_pixels];
        let mut cb_avx2 = vec![0u8; num_pixels];
        let mut cr_avx2 = vec![0u8; num_pixels];

        let mut y_scalar = vec![0u8; num_pixels];
        let mut cb_scalar = vec![0u8; num_pixels];
        let mut cr_scalar = vec![0u8; num_pixels];

        unsafe {
            convert_rgb_to_ycbcr_avx2(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, num_pixels);
        }

        for i in 0..num_pixels {
            let (y, cb, cr) =
                crate::color::rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            y_scalar[i] = y;
            cb_scalar[i] = cb;
            cr_scalar[i] = cr;
        }

        for i in 0..num_pixels {
            assert!(
                (y_avx2[i] as i16 - y_scalar[i] as i16).abs() <= 1,
                "Y mismatch at {}: AVX2={}, scalar={}",
                i,
                y_avx2[i],
                y_scalar[i]
            );
            assert!(
                (cb_avx2[i] as i16 - cb_scalar[i] as i16).abs() <= 1,
                "Cb mismatch at {}: AVX2={}, scalar={}",
                i,
                cb_avx2[i],
                cb_scalar[i]
            );
            assert!(
                (cr_avx2[i] as i16 - cr_scalar[i] as i16).abs() <= 1,
                "Cr mismatch at {}: AVX2={}, scalar={}",
                i,
                cr_avx2[i],
                cr_scalar[i]
            );
        }
    }
}
