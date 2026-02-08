//! NEON SIMD implementations for aarch64.
//!
//! Implements forward DCT using ARM NEON intrinsics with archmage for safe execution.
//! Based on the same algorithm as C mozjpeg's jfdctint-neon.c.

use crate::consts::DCTSIZE2;
use archmage::prelude::*;
use safe_unaligned_simd::aarch64::{vld1_s16, vld1q_s16, vst1q_s16};

// ============================================================================
// DCT Constants (same as x86_64 AVX2 version)
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

/// DCT constants stored in NEON-friendly layout.
#[repr(align(16))]
struct DctConstants {
    vals: [i16; 12],
}

impl DctConstants {
    const fn new() -> Self {
        Self {
            vals: [
                FIX_0_298631336 as i16,
                -FIX_0_390180644 as i16,
                FIX_0_541196100 as i16,
                FIX_0_765366865 as i16,
                -FIX_0_899976223 as i16,
                FIX_1_175875602 as i16,
                FIX_1_501321110 as i16,
                -FIX_1_847759065 as i16,
                -FIX_1_961570560 as i16,
                FIX_2_053119869 as i16,
                -FIX_2_562915447 as i16,
                FIX_3_072711026 as i16,
            ],
        }
    }
}

static DCT_CONSTS: DctConstants = DctConstants::new();

// ============================================================================
// Helper Functions
// ============================================================================

/// Descale and round for pass 1.
#[rite]
fn descale_p1(_token: NeonToken, v_l: int32x4_t, v_h: int32x4_t) -> int16x8_t {
    const SHIFT: i32 = CONST_BITS - PASS1_BITS;
    let low = vrshrn_n_s32::<SHIFT>(v_l);
    let high = vrshrn_n_s32::<SHIFT>(v_h);
    vcombine_s16(low, high)
}

/// Descale and round for pass 2.
#[rite]
fn descale_p2(_token: NeonToken, v_l: int32x4_t, v_h: int32x4_t) -> int16x8_t {
    const SHIFT: i32 = CONST_BITS + PASS1_BITS;
    let low = vrshrn_n_s32::<SHIFT>(v_l);
    let high = vrshrn_n_s32::<SHIFT>(v_h);
    vcombine_s16(low, high)
}

// ============================================================================
// Forward DCT
// ============================================================================

/// Forward DCT using NEON intrinsics.
///
/// Entry point - uses #[arcane] for safe target_feature dispatch.
/// Implements the same algorithm as C mozjpeg's jfdctint-neon.
#[arcane]
pub fn forward_dct_8x8_neon(
    token: NeonToken,
    samples: &[i16; DCTSIZE2],
    coeffs: &mut [i16; DCTSIZE2],
) {
    // Load DCT constants using vld1q_s16 from prelude (safe_unaligned_simd)
    let consts_arr: [i16; 8] = DCT_CONSTS.vals[0..8].try_into().unwrap();
    let consts = vld1q_s16(&consts_arr);
    let consts_lo = vget_low_s16(consts);
    let consts_hi = vget_high_s16(consts);

    let consts2_arr: [i16; 4] = DCT_CONSTS.vals[8..12].try_into().unwrap();
    let consts2 = vld1_s16(&consts2_arr);

    // Load 8x8 block
    let mut col0 = vld1q_s16(samples[0..8].try_into().unwrap());
    let mut col1 = vld1q_s16(samples[8..16].try_into().unwrap());
    let mut col2 = vld1q_s16(samples[16..24].try_into().unwrap());
    let mut col3 = vld1q_s16(samples[24..32].try_into().unwrap());
    let mut col4 = vld1q_s16(samples[32..40].try_into().unwrap());
    let mut col5 = vld1q_s16(samples[40..48].try_into().unwrap());
    let mut col6 = vld1q_s16(samples[48..56].try_into().unwrap());
    let mut col7 = vld1q_s16(samples[56..64].try_into().unwrap());

    // Transpose 8x8
    let cols_01 = vtrnq_s16(col0, col1);
    let cols_23 = vtrnq_s16(col2, col3);
    let cols_45 = vtrnq_s16(col4, col5);
    let cols_67 = vtrnq_s16(col6, col7);

    let cols_0145_l = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.0),
        vreinterpretq_s32_s16(cols_45.0),
    );
    let cols_0145_h = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.1),
        vreinterpretq_s32_s16(cols_45.1),
    );
    let cols_2367_l = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.0),
        vreinterpretq_s32_s16(cols_67.0),
    );
    let cols_2367_h = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.1),
        vreinterpretq_s32_s16(cols_67.1),
    );

    let rows_04 = vzipq_s32(cols_0145_l.0, cols_2367_l.0);
    let rows_15 = vzipq_s32(cols_0145_h.0, cols_2367_h.0);
    let rows_26 = vzipq_s32(cols_0145_l.1, cols_2367_l.1);
    let rows_37 = vzipq_s32(cols_0145_h.1, cols_2367_h.1);

    col0 = vreinterpretq_s16_s32(rows_04.0);
    col1 = vreinterpretq_s16_s32(rows_15.0);
    col2 = vreinterpretq_s16_s32(rows_26.0);
    col3 = vreinterpretq_s16_s32(rows_37.0);
    col4 = vreinterpretq_s16_s32(rows_04.1);
    col5 = vreinterpretq_s16_s32(rows_15.1);
    col6 = vreinterpretq_s16_s32(rows_26.1);
    col7 = vreinterpretq_s16_s32(rows_37.1);

    // Pass 1
    dct_pass(
        token, &mut col0, &mut col1, &mut col2, &mut col3, &mut col4, &mut col5, &mut col6,
        &mut col7, consts_lo, consts_hi, consts2, true,
    );

    // Transpose again
    let cols_01 = vtrnq_s16(col0, col1);
    let cols_23 = vtrnq_s16(col2, col3);
    let cols_45 = vtrnq_s16(col4, col5);
    let cols_67 = vtrnq_s16(col6, col7);

    let cols_0145_l = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.0),
        vreinterpretq_s32_s16(cols_45.0),
    );
    let cols_0145_h = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.1),
        vreinterpretq_s32_s16(cols_45.1),
    );
    let cols_2367_l = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.0),
        vreinterpretq_s32_s16(cols_67.0),
    );
    let cols_2367_h = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.1),
        vreinterpretq_s32_s16(cols_67.1),
    );

    let rows_04 = vzipq_s32(cols_0145_l.0, cols_2367_l.0);
    let rows_15 = vzipq_s32(cols_0145_h.0, cols_2367_h.0);
    let rows_26 = vzipq_s32(cols_0145_l.1, cols_2367_l.1);
    let rows_37 = vzipq_s32(cols_0145_h.1, cols_2367_h.1);

    col0 = vreinterpretq_s16_s32(rows_04.0);
    col1 = vreinterpretq_s16_s32(rows_15.0);
    col2 = vreinterpretq_s16_s32(rows_26.0);
    col3 = vreinterpretq_s16_s32(rows_37.0);
    col4 = vreinterpretq_s16_s32(rows_04.1);
    col5 = vreinterpretq_s16_s32(rows_15.1);
    col6 = vreinterpretq_s16_s32(rows_26.1);
    col7 = vreinterpretq_s16_s32(rows_37.1);

    // Pass 2
    dct_pass(
        token, &mut col0, &mut col1, &mut col2, &mut col3, &mut col4, &mut col5, &mut col6,
        &mut col7, consts_lo, consts_hi, consts2, false,
    );

    // Store results - use chunks to get mutable array references
    let cols = [col0, col1, col2, col3, col4, col5, col6, col7];
    for (i, chunk) in coeffs.chunks_exact_mut(8).enumerate() {
        vst1q_s16(chunk.try_into().unwrap(), cols[i]);
    }
}

/// Single DCT pass (row or column).
///
/// Inner helper - uses #[rite] for zero overhead.
#[allow(clippy::too_many_arguments)]
#[rite]
fn dct_pass(
    token: NeonToken,
    v0: &mut int16x8_t,
    v1: &mut int16x8_t,
    v2: &mut int16x8_t,
    v3: &mut int16x8_t,
    v4: &mut int16x8_t,
    v5: &mut int16x8_t,
    v6: &mut int16x8_t,
    v7: &mut int16x8_t,
    consts_lo: int16x4_t,
    consts_hi: int16x4_t,
    consts2: int16x4_t,
    pass1: bool,
) {
    // Butterfly stage
    let tmp0 = vaddq_s16(*v0, *v7);
    let tmp7 = vsubq_s16(*v0, *v7);
    let tmp1 = vaddq_s16(*v1, *v6);
    let tmp6 = vsubq_s16(*v1, *v6);
    let tmp2 = vaddq_s16(*v2, *v5);
    let tmp5 = vsubq_s16(*v2, *v5);
    let tmp3 = vaddq_s16(*v3, *v4);
    let tmp4 = vsubq_s16(*v3, *v4);

    // Even part
    let tmp10 = vaddq_s16(tmp0, tmp3);
    let tmp13 = vsubq_s16(tmp0, tmp3);
    let tmp11 = vaddq_s16(tmp1, tmp2);
    let tmp12 = vsubq_s16(tmp1, tmp2);

    if pass1 {
        *v0 = vshlq_n_s16::<PASS1_BITS>(vaddq_s16(tmp10, tmp11));
        *v4 = vshlq_n_s16::<PASS1_BITS>(vsubq_s16(tmp10, tmp11));
    } else {
        *v0 = vrshrq_n_s16::<PASS1_BITS>(vaddq_s16(tmp10, tmp11));
        *v4 = vrshrq_n_s16::<PASS1_BITS>(vsubq_s16(tmp10, tmp11));
    }

    let tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);

    // z1 = (tmp12 + tmp13) * FIX_0_541196100
    let z1_l = vmull_lane_s16::<2>(vget_low_s16(tmp12_add_tmp13), consts_lo);
    let z1_h = vmull_lane_s16::<2>(vget_high_s16(tmp12_add_tmp13), consts_lo);

    // v2 = z1 + tmp13 * FIX_0_765366865
    let v2_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp13), consts_lo);
    let v2_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp13), consts_lo);

    // v6 = z1 + tmp12 * (-FIX_1_847759065)
    let v6_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp12), consts_hi);
    let v6_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp12), consts_hi);

    if pass1 {
        *v2 = descale_p1(token, v2_l, v2_h);
        *v6 = descale_p1(token, v6_l, v6_h);
    } else {
        *v2 = descale_p2(token, v2_l, v2_h);
        *v6 = descale_p2(token, v6_l, v6_h);
    }

    // Odd part
    let z1 = vaddq_s16(tmp4, tmp7);
    let z2 = vaddq_s16(tmp5, tmp6);
    let z3 = vaddq_s16(tmp4, tmp6);
    let z4 = vaddq_s16(tmp5, tmp7);

    // z5 = (z3 + z4) * FIX_1_175875602
    let mut z5_l = vmull_lane_s16::<1>(vget_low_s16(z3), consts_hi);
    let mut z5_h = vmull_lane_s16::<1>(vget_high_s16(z3), consts_hi);
    z5_l = vmlal_lane_s16::<1>(z5_l, vget_low_s16(z4), consts_hi);
    z5_h = vmlal_lane_s16::<1>(z5_h, vget_high_s16(z4), consts_hi);

    // Compute rotations
    let mut tmp4_l = vmull_lane_s16::<0>(vget_low_s16(tmp4), consts_lo);
    let mut tmp4_h = vmull_lane_s16::<0>(vget_high_s16(tmp4), consts_lo);

    let mut tmp5_l = vmull_lane_s16::<1>(vget_low_s16(tmp5), consts2);
    let mut tmp5_h = vmull_lane_s16::<1>(vget_high_s16(tmp5), consts2);

    let mut tmp6_l = vmull_lane_s16::<3>(vget_low_s16(tmp6), consts2);
    let mut tmp6_h = vmull_lane_s16::<3>(vget_high_s16(tmp6), consts2);

    let mut tmp7_l = vmull_lane_s16::<2>(vget_low_s16(tmp7), consts_hi);
    let mut tmp7_h = vmull_lane_s16::<2>(vget_high_s16(tmp7), consts_hi);

    let z1_l = vmull_lane_s16::<0>(vget_low_s16(z1), consts_hi);
    let z1_h = vmull_lane_s16::<0>(vget_high_s16(z1), consts_hi);

    let z2_l = vmull_lane_s16::<2>(vget_low_s16(z2), consts2);
    let z2_h = vmull_lane_s16::<2>(vget_high_s16(z2), consts2);

    let z3_l = vmull_lane_s16::<0>(vget_low_s16(z3), consts2);
    let z3_h = vmull_lane_s16::<0>(vget_high_s16(z3), consts2);

    let z4_l = vmull_lane_s16::<1>(vget_low_s16(z4), consts_lo);
    let z4_h = vmull_lane_s16::<1>(vget_high_s16(z4), consts_lo);

    let z3_l = vaddq_s32(z3_l, z5_l);
    let z3_h = vaddq_s32(z3_h, z5_h);
    let z4_l = vaddq_s32(z4_l, z5_l);
    let z4_h = vaddq_s32(z4_h, z5_h);

    tmp4_l = vaddq_s32(tmp4_l, z1_l);
    tmp4_h = vaddq_s32(tmp4_h, z1_h);
    tmp4_l = vaddq_s32(tmp4_l, z3_l);
    tmp4_h = vaddq_s32(tmp4_h, z3_h);

    tmp5_l = vaddq_s32(tmp5_l, z2_l);
    tmp5_h = vaddq_s32(tmp5_h, z2_h);
    tmp5_l = vaddq_s32(tmp5_l, z4_l);
    tmp5_h = vaddq_s32(tmp5_h, z4_h);

    tmp6_l = vaddq_s32(tmp6_l, z2_l);
    tmp6_h = vaddq_s32(tmp6_h, z2_h);
    tmp6_l = vaddq_s32(tmp6_l, z3_l);
    tmp6_h = vaddq_s32(tmp6_h, z3_h);

    tmp7_l = vaddq_s32(tmp7_l, z1_l);
    tmp7_h = vaddq_s32(tmp7_h, z1_h);
    tmp7_l = vaddq_s32(tmp7_l, z4_l);
    tmp7_h = vaddq_s32(tmp7_h, z4_h);

    if pass1 {
        *v7 = descale_p1(token, tmp4_l, tmp4_h);
        *v5 = descale_p1(token, tmp5_l, tmp5_h);
        *v3 = descale_p1(token, tmp6_l, tmp6_h);
        *v1 = descale_p1(token, tmp7_l, tmp7_h);
    } else {
        *v7 = descale_p2(token, tmp4_l, tmp4_h);
        *v5 = descale_p2(token, tmp5_l, tmp5_h);
        *v3 = descale_p2(token, tmp6_l, tmp6_h);
        *v1 = descale_p2(token, tmp7_l, tmp7_h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_dct_flat_block() {
        let Some(token) = NeonToken::summon() else {
            eprintln!("NEON not available, skipping test");
            return;
        };

        let samples = [100i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];

        forward_dct_8x8_neon(token, &samples, &mut coeffs);

        // DC should be 64 * 100 = 6400
        assert_eq!(coeffs[0], 6400);

        // AC coefficients should be near zero for flat block
        for &coef in &coeffs[1..] {
            assert!(coef.abs() < 10, "AC coefficient too large: {}", coef);
        }
    }

    #[test]
    fn test_neon_vs_scalar() {
        let Some(token) = NeonToken::summon() else {
            eprintln!("NEON not available, skipping test");
            return;
        };

        // Test with various patterns
        for seed in 0..10 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut neon_coeffs = [0i16; DCTSIZE2];
            let mut scalar_coeffs = [0i16; DCTSIZE2];

            forward_dct_8x8_neon(token, &samples, &mut neon_coeffs);
            crate::dct::forward_dct_8x8_i32_multiversion(&samples, &mut scalar_coeffs);

            assert_eq!(
                neon_coeffs, scalar_coeffs,
                "NEON should match scalar for seed {}",
                seed
            );
        }
    }
}
