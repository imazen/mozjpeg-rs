//! Forward DCT (Discrete Cosine Transform) implementation.
//!
//! This implements the Loeffler-Ligtenberg-Moschytz algorithm for 8x8 DCT,
//! matching mozjpeg's jfdctint.c (integer slow DCT).
//!
//! The algorithm uses 12 multiplies and 32 adds per 1-D DCT.
//! A 2-D DCT is done by 1-D DCT on rows followed by 1-D DCT on columns.
//!
//! Note: The output is scaled up by a factor of 8 compared to a true DCT.
//! This scaling is removed during quantization (in the encoder pipeline).
//!
//! # SIMD Optimization
//!
//! The `forward_dct_8x8_i32_wide_gather` function uses the "row-parallel" approach:
//! process 4 rows simultaneously using `wide::i32x4`. Each lane of the
//! SIMD vector handles a different row.
//!
//! Reference: C. Loeffler, A. Ligtenberg and G. Moschytz,
//! "Practical Fast 1-D DCT Algorithms with 11 Multiplications",
//! Proc. ICASSP 1989, pp. 988-991.

use crate::consts::{DCTSIZE, DCTSIZE2};
use multiversion::multiversion;
use wide::{i32x4, i32x8};

// Fixed-point constants for 13-bit precision (CONST_BITS = 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Pre-calculated fixed-point constants: FIX(x) = (x * (1 << CONST_BITS) + 0.5)
const FIX_0_298631336: i32 = 2446; // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196; // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433; // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270; // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373; // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633; // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299; // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137; // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069; // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819; // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995; // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172; // FIX(3.072711026)

/// DESCALE: Right-shift with rounding (used to remove fixed-point scaling)
#[inline]
fn descale(x: i32, n: i32) -> i32 {
    // Round by adding 2^(n-1) before shifting
    (x + (1 << (n - 1))) >> n
}

/// Perform forward DCT on one 8x8 block of samples.
///
/// Input: 64 sample values in row-major order (0-255 for 8-bit JPEG)
/// Output: 64 DCT coefficients in row-major order
///
/// Note: The output is scaled up by a factor of 8. This is intentional
/// and matches libjpeg/mozjpeg behavior - the scaling is removed during
/// quantization.
///
/// Uses `multiversion` for automatic SIMD optimization via autovectorization.
///
/// # Arguments
/// * `samples` - Input 8x8 block of pixel samples (typically centered around 0)
/// * `coeffs` - Output 8x8 block of DCT coefficients
#[multiversion(targets(
    "x86_64+avx2",
    "x86_64+sse4.1",
    "x86+avx2",
    "x86+sse4.1",
    "aarch64+neon",
))]
pub fn forward_dct_8x8_i32_multiversion(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Work buffer (we modify in place across both passes)
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32 for processing
    for i in 0..DCTSIZE2 {
        data[i] = samples[i] as i32;
    }

    // Pass 1: process rows
    // Results are scaled up by sqrt(8) and by 2^PASS1_BITS
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

        // Even part (per Loeffler figure 1)
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[base] = (tmp10 + tmp11) << PASS1_BITS;
        data[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        data[base + 6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        // Odd part (per Loeffler figure 8)
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; // sqrt(2) * c3

        let tmp4 = tmp4 * FIX_0_298631336; // sqrt(2) * (-c1+c3+c5-c7)
        let tmp5 = tmp5 * FIX_2_053119869; // sqrt(2) * ( c1+c3-c5+c7)
        let tmp6 = tmp6 * FIX_3_072711026; // sqrt(2) * ( c1+c3+c5-c7)
        let tmp7 = tmp7 * FIX_1_501321110; // sqrt(2) * ( c1+c3-c5-c7)
        let z1 = z1 * (-FIX_0_899976223); // sqrt(2) * ( c7-c3)
        let z2 = z2 * (-FIX_2_562915447); // sqrt(2) * (-c1-c3)
        let z3 = z3 * (-FIX_1_961570560) + z5; // sqrt(2) * (-c3-c5)
        let z4 = z4 * (-FIX_0_390180644) + z5; // sqrt(2) * ( c5-c3)

        data[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    // We remove PASS1_BITS scaling but leave results scaled by factor of 8
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

    // Copy results to output
    for i in 0..DCTSIZE2 {
        coeffs[i] = data[i] as i16;
    }
}

/// SIMD descale operation for i32x4 vectors.
#[inline(always)]
fn descale_simd(x: i32x4, n: i32) -> i32x4 {
    let round = i32x4::splat(1 << (n - 1));
    (x + round) >> n
}

// Pre-computed SIMD constants for DCT - avoid recreating each call
const SIMD_FIX_0_298631336: i32x4 = i32x4::new([FIX_0_298631336; 4]);
const SIMD_FIX_0_541196100: i32x4 = i32x4::new([FIX_0_541196100; 4]);
const SIMD_FIX_0_765366865: i32x4 = i32x4::new([FIX_0_765366865; 4]);
const SIMD_FIX_1_175875602: i32x4 = i32x4::new([FIX_1_175875602; 4]);
const SIMD_FIX_1_501321110: i32x4 = i32x4::new([FIX_1_501321110; 4]);
const SIMD_FIX_1_847759065: i32x4 = i32x4::new([FIX_1_847759065; 4]);
const SIMD_FIX_2_053119869: i32x4 = i32x4::new([FIX_2_053119869; 4]);
const SIMD_FIX_3_072711026: i32x4 = i32x4::new([FIX_3_072711026; 4]);

// Negated constants to avoid runtime negation
const SIMD_NEG_FIX_0_390180644: i32x4 = i32x4::new([-FIX_0_390180644; 4]);
const SIMD_NEG_FIX_0_899976223: i32x4 = i32x4::new([-FIX_0_899976223; 4]);
const SIMD_NEG_FIX_1_961570560: i32x4 = i32x4::new([-FIX_1_961570560; 4]);
const SIMD_NEG_FIX_2_562915447: i32x4 = i32x4::new([-FIX_2_562915447; 4]);

/// Process one batch of 4 rows/columns with 1D DCT.
/// Fully inlined for performance - no function call overhead.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dct_1d_simd(
    d0: i32x4,
    d1: i32x4,
    d2: i32x4,
    d3: i32x4,
    d4: i32x4,
    d5: i32x4,
    d6: i32x4,
    d7: i32x4,
    shift_pass1: bool,
) -> [i32x4; 8] {
    // Even part
    let tmp0 = d0 + d7;
    let tmp7 = d0 - d7;
    let tmp1 = d1 + d6;
    let tmp6 = d1 - d6;
    let tmp2 = d2 + d5;
    let tmp5 = d2 - d5;
    let tmp3 = d3 + d4;
    let tmp4 = d3 - d4;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let (out0, out4) = if shift_pass1 {
        ((tmp10 + tmp11) << PASS1_BITS, (tmp10 - tmp11) << PASS1_BITS)
    } else {
        (
            descale_simd(tmp10 + tmp11, PASS1_BITS),
            descale_simd(tmp10 - tmp11, PASS1_BITS),
        )
    };

    let z1 = (tmp12 + tmp13) * SIMD_FIX_0_541196100;
    let (out2, out6) = if shift_pass1 {
        (
            descale_simd(z1 + tmp13 * SIMD_FIX_0_765366865, CONST_BITS - PASS1_BITS),
            descale_simd(z1 - tmp12 * SIMD_FIX_1_847759065, CONST_BITS - PASS1_BITS),
        )
    } else {
        (
            descale_simd(z1 + tmp13 * SIMD_FIX_0_765366865, CONST_BITS + PASS1_BITS),
            descale_simd(z1 - tmp12 * SIMD_FIX_1_847759065, CONST_BITS + PASS1_BITS),
        )
    };

    // Odd part
    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * SIMD_FIX_1_175875602;

    let tmp4 = tmp4 * SIMD_FIX_0_298631336;
    let tmp5 = tmp5 * SIMD_FIX_2_053119869;
    let tmp6 = tmp6 * SIMD_FIX_3_072711026;
    let tmp7 = tmp7 * SIMD_FIX_1_501321110;

    // Use pre-negated constants
    let neg_z1 = z1 * SIMD_NEG_FIX_0_899976223;
    let neg_z2 = z2 * SIMD_NEG_FIX_2_562915447;
    let z3 = z3 * SIMD_NEG_FIX_1_961570560 + z5;
    let z4 = z4 * SIMD_NEG_FIX_0_390180644 + z5;

    let scale = if shift_pass1 {
        CONST_BITS - PASS1_BITS
    } else {
        CONST_BITS + PASS1_BITS
    };
    let out7 = descale_simd(tmp4 + neg_z1 + z3, scale);
    let out5 = descale_simd(tmp5 + neg_z2 + z4, scale);
    let out3 = descale_simd(tmp6 + neg_z2 + z3, scale);
    let out1 = descale_simd(tmp7 + neg_z1 + z4, scale);

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// SIMD-optimized forward DCT on one 8x8 block.
///
/// Uses row-parallel approach: processes 4 rows simultaneously using i32x4.
/// Each lane of the SIMD vector handles a different row.
///
/// Optimizations applied:
/// - Pre-computed SIMD constants (no per-call allocation)
/// - Pre-negated constants (no runtime negation)
/// - Inlined 1D DCT helper (no function call overhead)
/// - Unrolled loops for row/column batches
#[deprecated(
    since = "0.5.0",
    note = "Not used by encoder. Use forward_dct_8x8 (with multiversion autovectorization) \
            or enable simd-intrinsics feature for hand-written AVX2."
)]
pub fn forward_dct_8x8_i32_wide_gather(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Work buffer - aligned to 64 bytes for potential future SIMD improvements
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32 - unrolled for better pipelining
    // Process 8 values at a time (one row)
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;
        data[base] = samples[base] as i32;
        data[base + 1] = samples[base + 1] as i32;
        data[base + 2] = samples[base + 2] as i32;
        data[base + 3] = samples[base + 3] as i32;
        data[base + 4] = samples[base + 4] as i32;
        data[base + 5] = samples[base + 5] as i32;
        data[base + 6] = samples[base + 6] as i32;
        data[base + 7] = samples[base + 7] as i32;
    }

    // Pass 1: Process rows 0-3
    {
        let col0 = i32x4::new([data[0], data[8], data[16], data[24]]);
        let col1 = i32x4::new([data[1], data[9], data[17], data[25]]);
        let col2 = i32x4::new([data[2], data[10], data[18], data[26]]);
        let col3 = i32x4::new([data[3], data[11], data[19], data[27]]);
        let col4 = i32x4::new([data[4], data[12], data[20], data[28]]);
        let col5 = i32x4::new([data[5], data[13], data[21], data[29]]);
        let col6 = i32x4::new([data[6], data[14], data[22], data[30]]);
        let col7 = i32x4::new([data[7], data[15], data[23], data[31]]);

        let out = dct_1d_simd(col0, col1, col2, col3, col4, col5, col6, col7, true);

        // Scatter results back
        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        data[0] = arr0[0];
        data[1] = arr1[0];
        data[2] = arr2[0];
        data[3] = arr3[0];
        data[4] = arr4[0];
        data[5] = arr5[0];
        data[6] = arr6[0];
        data[7] = arr7[0];
        data[8] = arr0[1];
        data[9] = arr1[1];
        data[10] = arr2[1];
        data[11] = arr3[1];
        data[12] = arr4[1];
        data[13] = arr5[1];
        data[14] = arr6[1];
        data[15] = arr7[1];
        data[16] = arr0[2];
        data[17] = arr1[2];
        data[18] = arr2[2];
        data[19] = arr3[2];
        data[20] = arr4[2];
        data[21] = arr5[2];
        data[22] = arr6[2];
        data[23] = arr7[2];
        data[24] = arr0[3];
        data[25] = arr1[3];
        data[26] = arr2[3];
        data[27] = arr3[3];
        data[28] = arr4[3];
        data[29] = arr5[3];
        data[30] = arr6[3];
        data[31] = arr7[3];
    }

    // Pass 1: Process rows 4-7
    {
        let col0 = i32x4::new([data[32], data[40], data[48], data[56]]);
        let col1 = i32x4::new([data[33], data[41], data[49], data[57]]);
        let col2 = i32x4::new([data[34], data[42], data[50], data[58]]);
        let col3 = i32x4::new([data[35], data[43], data[51], data[59]]);
        let col4 = i32x4::new([data[36], data[44], data[52], data[60]]);
        let col5 = i32x4::new([data[37], data[45], data[53], data[61]]);
        let col6 = i32x4::new([data[38], data[46], data[54], data[62]]);
        let col7 = i32x4::new([data[39], data[47], data[55], data[63]]);

        let out = dct_1d_simd(col0, col1, col2, col3, col4, col5, col6, col7, true);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        data[32] = arr0[0];
        data[33] = arr1[0];
        data[34] = arr2[0];
        data[35] = arr3[0];
        data[36] = arr4[0];
        data[37] = arr5[0];
        data[38] = arr6[0];
        data[39] = arr7[0];
        data[40] = arr0[1];
        data[41] = arr1[1];
        data[42] = arr2[1];
        data[43] = arr3[1];
        data[44] = arr4[1];
        data[45] = arr5[1];
        data[46] = arr6[1];
        data[47] = arr7[1];
        data[48] = arr0[2];
        data[49] = arr1[2];
        data[50] = arr2[2];
        data[51] = arr3[2];
        data[52] = arr4[2];
        data[53] = arr5[2];
        data[54] = arr6[2];
        data[55] = arr7[2];
        data[56] = arr0[3];
        data[57] = arr1[3];
        data[58] = arr2[3];
        data[59] = arr3[3];
        data[60] = arr4[3];
        data[61] = arr5[3];
        data[62] = arr6[3];
        data[63] = arr7[3];
    }

    // Pass 2: Process columns 0-3
    {
        let row0 = i32x4::new([data[0], data[1], data[2], data[3]]);
        let row1 = i32x4::new([data[8], data[9], data[10], data[11]]);
        let row2 = i32x4::new([data[16], data[17], data[18], data[19]]);
        let row3 = i32x4::new([data[24], data[25], data[26], data[27]]);
        let row4 = i32x4::new([data[32], data[33], data[34], data[35]]);
        let row5 = i32x4::new([data[40], data[41], data[42], data[43]]);
        let row6 = i32x4::new([data[48], data[49], data[50], data[51]]);
        let row7 = i32x4::new([data[56], data[57], data[58], data[59]]);

        let out = dct_1d_simd(row0, row1, row2, row3, row4, row5, row6, row7, false);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        data[0] = arr0[0];
        data[1] = arr0[1];
        data[2] = arr0[2];
        data[3] = arr0[3];
        data[8] = arr1[0];
        data[9] = arr1[1];
        data[10] = arr1[2];
        data[11] = arr1[3];
        data[16] = arr2[0];
        data[17] = arr2[1];
        data[18] = arr2[2];
        data[19] = arr2[3];
        data[24] = arr3[0];
        data[25] = arr3[1];
        data[26] = arr3[2];
        data[27] = arr3[3];
        data[32] = arr4[0];
        data[33] = arr4[1];
        data[34] = arr4[2];
        data[35] = arr4[3];
        data[40] = arr5[0];
        data[41] = arr5[1];
        data[42] = arr5[2];
        data[43] = arr5[3];
        data[48] = arr6[0];
        data[49] = arr6[1];
        data[50] = arr6[2];
        data[51] = arr6[3];
        data[56] = arr7[0];
        data[57] = arr7[1];
        data[58] = arr7[2];
        data[59] = arr7[3];
    }

    // Pass 2: Process columns 4-7
    {
        let row0 = i32x4::new([data[4], data[5], data[6], data[7]]);
        let row1 = i32x4::new([data[12], data[13], data[14], data[15]]);
        let row2 = i32x4::new([data[20], data[21], data[22], data[23]]);
        let row3 = i32x4::new([data[28], data[29], data[30], data[31]]);
        let row4 = i32x4::new([data[36], data[37], data[38], data[39]]);
        let row5 = i32x4::new([data[44], data[45], data[46], data[47]]);
        let row6 = i32x4::new([data[52], data[53], data[54], data[55]]);
        let row7 = i32x4::new([data[60], data[61], data[62], data[63]]);

        let out = dct_1d_simd(row0, row1, row2, row3, row4, row5, row6, row7, false);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        data[4] = arr0[0];
        data[5] = arr0[1];
        data[6] = arr0[2];
        data[7] = arr0[3];
        data[12] = arr1[0];
        data[13] = arr1[1];
        data[14] = arr1[2];
        data[15] = arr1[3];
        data[20] = arr2[0];
        data[21] = arr2[1];
        data[22] = arr2[2];
        data[23] = arr2[3];
        data[28] = arr3[0];
        data[29] = arr3[1];
        data[30] = arr3[2];
        data[31] = arr3[3];
        data[36] = arr4[0];
        data[37] = arr4[1];
        data[38] = arr4[2];
        data[39] = arr4[3];
        data[44] = arr5[0];
        data[45] = arr5[1];
        data[46] = arr5[2];
        data[47] = arr5[3];
        data[52] = arr6[0];
        data[53] = arr6[1];
        data[54] = arr6[2];
        data[55] = arr6[3];
        data[60] = arr7[0];
        data[61] = arr7[1];
        data[62] = arr7[2];
        data[63] = arr7[3];
    }

    // Copy results to output - unrolled
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;
        coeffs[base] = data[base] as i16;
        coeffs[base + 1] = data[base + 1] as i16;
        coeffs[base + 2] = data[base + 2] as i16;
        coeffs[base + 3] = data[base + 3] as i16;
        coeffs[base + 4] = data[base + 4] as i16;
        coeffs[base + 5] = data[base + 5] as i16;
        coeffs[base + 6] = data[base + 6] as i16;
        coeffs[base + 7] = data[base + 7] as i16;
    }
}

// ============================================================================
// Transpose-based SIMD DCT with contiguous memory access
// ============================================================================

// SIMD constants for i32x8 (8-wide operations)
const SIMD8_FIX_0_298631336: i32x8 = i32x8::new([FIX_0_298631336; 8]);
const SIMD8_FIX_0_541196100: i32x8 = i32x8::new([FIX_0_541196100; 8]);
const SIMD8_FIX_0_765366865: i32x8 = i32x8::new([FIX_0_765366865; 8]);
const SIMD8_FIX_1_175875602: i32x8 = i32x8::new([FIX_1_175875602; 8]);
const SIMD8_FIX_1_501321110: i32x8 = i32x8::new([FIX_1_501321110; 8]);
const SIMD8_FIX_1_847759065: i32x8 = i32x8::new([FIX_1_847759065; 8]);
const SIMD8_FIX_2_053119869: i32x8 = i32x8::new([FIX_2_053119869; 8]);
const SIMD8_FIX_3_072711026: i32x8 = i32x8::new([FIX_3_072711026; 8]);

// Negated constants for i32x8
const SIMD8_NEG_FIX_0_390180644: i32x8 = i32x8::new([-FIX_0_390180644; 8]);
const SIMD8_NEG_FIX_0_899976223: i32x8 = i32x8::new([-FIX_0_899976223; 8]);
const SIMD8_NEG_FIX_1_961570560: i32x8 = i32x8::new([-FIX_1_961570560; 8]);
const SIMD8_NEG_FIX_2_562915447: i32x8 = i32x8::new([-FIX_2_562915447; 8]);

/// SIMD descale for i32x8
#[inline(always)]
fn descale_simd8(x: i32x8, n: i32) -> i32x8 {
    let round = i32x8::splat(1 << (n - 1));
    (x + round) >> n
}

/// 1D DCT on 8 values simultaneously using i32x8.
/// Each element of the vectors corresponds to a different row/column.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dct_1d_8wide(
    d0: i32x8,
    d1: i32x8,
    d2: i32x8,
    d3: i32x8,
    d4: i32x8,
    d5: i32x8,
    d6: i32x8,
    d7: i32x8,
    shift_pass1: bool,
) -> [i32x8; 8] {
    // Even part
    let tmp0 = d0 + d7;
    let tmp7 = d0 - d7;
    let tmp1 = d1 + d6;
    let tmp6 = d1 - d6;
    let tmp2 = d2 + d5;
    let tmp5 = d2 - d5;
    let tmp3 = d3 + d4;
    let tmp4 = d3 - d4;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let (out0, out4) = if shift_pass1 {
        ((tmp10 + tmp11) << PASS1_BITS, (tmp10 - tmp11) << PASS1_BITS)
    } else {
        (
            descale_simd8(tmp10 + tmp11, PASS1_BITS),
            descale_simd8(tmp10 - tmp11, PASS1_BITS),
        )
    };

    let z1 = (tmp12 + tmp13) * SIMD8_FIX_0_541196100;
    let (out2, out6) = if shift_pass1 {
        (
            descale_simd8(z1 + tmp13 * SIMD8_FIX_0_765366865, CONST_BITS - PASS1_BITS),
            descale_simd8(z1 - tmp12 * SIMD8_FIX_1_847759065, CONST_BITS - PASS1_BITS),
        )
    } else {
        (
            descale_simd8(z1 + tmp13 * SIMD8_FIX_0_765366865, CONST_BITS + PASS1_BITS),
            descale_simd8(z1 - tmp12 * SIMD8_FIX_1_847759065, CONST_BITS + PASS1_BITS),
        )
    };

    // Odd part
    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * SIMD8_FIX_1_175875602;

    let tmp4 = tmp4 * SIMD8_FIX_0_298631336;
    let tmp5 = tmp5 * SIMD8_FIX_2_053119869;
    let tmp6 = tmp6 * SIMD8_FIX_3_072711026;
    let tmp7 = tmp7 * SIMD8_FIX_1_501321110;

    let neg_z1 = z1 * SIMD8_NEG_FIX_0_899976223;
    let neg_z2 = z2 * SIMD8_NEG_FIX_2_562915447;
    let z3 = z3 * SIMD8_NEG_FIX_1_961570560 + z5;
    let z4 = z4 * SIMD8_NEG_FIX_0_390180644 + z5;

    let scale = if shift_pass1 {
        CONST_BITS - PASS1_BITS
    } else {
        CONST_BITS + PASS1_BITS
    };
    let out7 = descale_simd8(tmp4 + neg_z1 + z3, scale);
    let out5 = descale_simd8(tmp5 + neg_z2 + z4, scale);
    let out3 = descale_simd8(tmp6 + neg_z2 + z3, scale);
    let out1 = descale_simd8(tmp7 + neg_z1 + z4, scale);

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// Transpose 8x8 matrix stored as 8 i32x8 vectors.
/// Uses standard SSE/AVX transpose algorithm in stages.
#[inline(always)]
fn transpose_8x8(rows: &mut [i32x8; 8]) {
    // Convert to arrays for element access - we need direct manipulation
    let mut data: [[i32; 8]; 8] = [
        rows[0].to_array(),
        rows[1].to_array(),
        rows[2].to_array(),
        rows[3].to_array(),
        rows[4].to_array(),
        rows[5].to_array(),
        rows[6].to_array(),
        rows[7].to_array(),
    ];

    // Transpose in-place using swap for upper triangle
    // We need both indices for 2D array element swap
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        for j in (i + 1)..8 {
            let tmp = data[i][j];
            data[i][j] = data[j][i];
            data[j][i] = tmp;
        }
    }

    // Write back
    rows[0] = i32x8::new(data[0]);
    rows[1] = i32x8::new(data[1]);
    rows[2] = i32x8::new(data[2]);
    rows[3] = i32x8::new(data[3]);
    rows[4] = i32x8::new(data[4]);
    rows[5] = i32x8::new(data[5]);
    rows[6] = i32x8::new(data[6]);
    rows[7] = i32x8::new(data[7]);
}

/// Transpose-based SIMD DCT with contiguous memory access.
///
/// This version:
/// 1. Loads rows contiguously (no gather operations)
/// 2. Transposes so that dct_1d_8wide operates within rows
/// 3. Does 1D DCT (row pass)
/// 4. Transposes again
/// 5. Does 1D DCT (column pass)
///
/// The key insight: when we call `dct_1d_8wide(rows[0], rows[1], ..., rows[7])`,
/// it combines `rows[0]+rows[7]`, `rows[1]+rows[6]`, etc. - that's column-wise processing.
/// To do row-wise processing, we transpose first so that rows become columns.
///
/// The contiguous loads should be faster than the gather-based approach.
pub fn forward_dct_8x8_i32_wide_transpose(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Load all 8 rows contiguously - this is the key optimization!
    // Each row is 8 contiguous i16 values converted to i32
    let mut data: [i32x8; 8] = [
        i32x8::new([
            samples[0] as i32,
            samples[1] as i32,
            samples[2] as i32,
            samples[3] as i32,
            samples[4] as i32,
            samples[5] as i32,
            samples[6] as i32,
            samples[7] as i32,
        ]),
        i32x8::new([
            samples[8] as i32,
            samples[9] as i32,
            samples[10] as i32,
            samples[11] as i32,
            samples[12] as i32,
            samples[13] as i32,
            samples[14] as i32,
            samples[15] as i32,
        ]),
        i32x8::new([
            samples[16] as i32,
            samples[17] as i32,
            samples[18] as i32,
            samples[19] as i32,
            samples[20] as i32,
            samples[21] as i32,
            samples[22] as i32,
            samples[23] as i32,
        ]),
        i32x8::new([
            samples[24] as i32,
            samples[25] as i32,
            samples[26] as i32,
            samples[27] as i32,
            samples[28] as i32,
            samples[29] as i32,
            samples[30] as i32,
            samples[31] as i32,
        ]),
        i32x8::new([
            samples[32] as i32,
            samples[33] as i32,
            samples[34] as i32,
            samples[35] as i32,
            samples[36] as i32,
            samples[37] as i32,
            samples[38] as i32,
            samples[39] as i32,
        ]),
        i32x8::new([
            samples[40] as i32,
            samples[41] as i32,
            samples[42] as i32,
            samples[43] as i32,
            samples[44] as i32,
            samples[45] as i32,
            samples[46] as i32,
            samples[47] as i32,
        ]),
        i32x8::new([
            samples[48] as i32,
            samples[49] as i32,
            samples[50] as i32,
            samples[51] as i32,
            samples[52] as i32,
            samples[53] as i32,
            samples[54] as i32,
            samples[55] as i32,
        ]),
        i32x8::new([
            samples[56] as i32,
            samples[57] as i32,
            samples[58] as i32,
            samples[59] as i32,
            samples[60] as i32,
            samples[61] as i32,
            samples[62] as i32,
            samples[63] as i32,
        ]),
    ];

    // Transpose first: now data[i] contains column i (all 8 rows' values for column i)
    // After this, dct_1d_8wide will process all 8 rows in parallel
    transpose_8x8(&mut data);

    // Pass 1: 1D DCT on rows (we transposed, so this processes row elements)
    // dct_1d_8wide(data[0], data[1], ...) now combines columns 0+7, 1+6, etc. within each row
    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], true,
    );
    data = result;

    // Transpose: data[i] now contains row i
    transpose_8x8(&mut data);

    // Pass 2: 1D DCT on columns (we transposed, so this processes column elements)
    // dct_1d_8wide(data[0], data[1], ...) now combines rows 0+7, 1+6, etc. within each column
    // Output: data[row][col] = final DCT coefficient - already row-major!
    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], false,
    );
    data = result;

    // Store results (data is already in row-major form)
    let arr0 = data[0].to_array();
    let arr1 = data[1].to_array();
    let arr2 = data[2].to_array();
    let arr3 = data[3].to_array();
    let arr4 = data[4].to_array();
    let arr5 = data[5].to_array();
    let arr6 = data[6].to_array();
    let arr7 = data[7].to_array();

    for i in 0..8 {
        coeffs[i] = arr0[i] as i16;
    }
    for i in 0..8 {
        coeffs[8 + i] = arr1[i] as i16;
    }
    for i in 0..8 {
        coeffs[16 + i] = arr2[i] as i16;
    }
    for i in 0..8 {
        coeffs[24 + i] = arr3[i] as i16;
    }
    for i in 0..8 {
        coeffs[32 + i] = arr4[i] as i16;
    }
    for i in 0..8 {
        coeffs[40 + i] = arr5[i] as i16;
    }
    for i in 0..8 {
        coeffs[48 + i] = arr6[i] as i16;
    }
    for i in 0..8 {
        coeffs[56 + i] = arr7[i] as i16;
    }
}

/// Prepare a sample block for DCT by level-shifting (centering around 0).
///
/// JPEG requires samples to be centered around 0 before DCT.
/// For 8-bit samples (0-255), subtract 128.
///
/// # Arguments
/// * `samples` - Input samples (0-255)
/// * `output` - Output level-shifted samples (-128 to 127)
pub fn level_shift(samples: &[u8; DCTSIZE2], output: &mut [i16; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        output[i] = (samples[i] as i16) - 128;
    }
}

/// Combined level-shift and forward DCT.
///
/// Uses SIMD-optimized DCT for better performance.
/// Uses transpose-based approach for contiguous memory access.
/// Uses AVX2 intrinsics when available for best performance.
///
/// # Arguments
/// * `samples` - Input 8x8 block of pixel samples (0-255)
/// * `coeffs` - Output 8x8 block of DCT coefficients
#[deprecated(
    since = "0.5.0",
    note = "Not used by encoder. The encoder calls level_shift and forward_dct_8x8 separately \
            via SimdOps dispatch. This function exists for standalone/testing use."
)]
#[allow(deprecated)]
pub fn forward_dct_u8_i32_multiversion(samples: &[u8; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut shifted = [0i16; DCTSIZE2];
    level_shift(samples, &mut shifted);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::tokens::x86::Avx2Token;
        use archmage::SimdToken;
        if let Some(token) = Avx2Token::try_new() {
            return avx2_archmage::forward_dct_8x8_i32(token, &shifted, coeffs);
        }
    }

    forward_dct_8x8_i32_wide_transpose(&shifted, coeffs);
}

/// Combined level-shift, overshoot deringing, and forward DCT.
///
/// Uses SIMD-optimized DCT for better performance.
/// Uses transpose-based approach for contiguous memory access.
/// Uses AVX2 intrinsics when available for best performance.
/// This variant applies mozjpeg's overshoot deringing preprocessing to reduce
/// visible ringing artifacts near hard edges on white backgrounds.
///
/// # Arguments
/// * `samples` - Input 8x8 block of pixel samples (0-255)
/// * `coeffs` - Output 8x8 block of DCT coefficients
/// * `dc_quant` - DC quantization value (used to limit overshoot amount)
///
/// # See Also
/// [`crate::deringing::preprocess_deringing`] for algorithm details.
#[deprecated(
    since = "0.5.0",
    note = "Not used by encoder. The encoder calls level_shift, preprocess_deringing, and \
            forward_dct_8x8 separately via SimdOps dispatch. This function exists for \
            standalone/testing use."
)]
#[allow(deprecated)]
pub fn forward_dct_u8_i32_multiversion_deringing(
    samples: &[u8; DCTSIZE2],
    coeffs: &mut [i16; DCTSIZE2],
    dc_quant: u16,
) {
    use crate::deringing::preprocess_deringing;

    let mut shifted = [0i16; DCTSIZE2];
    level_shift(samples, &mut shifted);
    preprocess_deringing(&mut shifted, dc_quant);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::tokens::x86::Avx2Token;
        use archmage::SimdToken;
        if let Some(token) = Avx2Token::try_new() {
            return avx2_archmage::forward_dct_8x8_i32(token, &shifted, coeffs);
        }
    }

    forward_dct_8x8_i32_wide_transpose(&shifted, coeffs);
}

// ============================================================================
// AVX2 Intrinsics-based DCT (x86_64 only) - DEPRECATED
// ============================================================================
//
// This module uses archmage for safe SIMD with capability tokens. However, it is
// NOT used by the encoder (which uses simd/x86_64/avx2.rs via SimdOps dispatch).
// This module exists for experimentation with archmage-based safe SIMD patterns.
//
// Key optimizations over the wide-based version:
// - Uses _mm256_cvtepi16_epi32 for proper load+sign-extend (no vpinsrw gather)
// - Uses proper shuffle instructions for transpose
// - Avoids to_array()/from_array() overhead

#[cfg(target_arch = "x86_64")]
#[deprecated(
    since = "0.5.0",
    note = "Not used by encoder. This archmage-based module exists for experimentation. \
            The encoder uses src/simd/x86_64/avx2.rs via SimdOps dispatch."
)]
pub mod avx2_archmage {
    //! AVX2 SIMD implementation of forward DCT using archmage (DEPRECATED).
    //!
    //! This module uses archmage for safe SIMD operations with capability tokens.
    //! The `#[arcane]` macro enables safe use of value-based intrinsics.
    //!
    //! **Note:** This module is not used by the encoder. It exists for experimentation
    //! with archmage-based safe SIMD patterns. The encoder uses `src/simd/x86_64/avx2.rs`.
    //! Runtime feature detection via `Avx2Token::try_new()` ensures safety.

    use super::*;
    use archmage::arcane;
    use archmage::mem::{avx, sse2};
    use archmage::tokens::x86::Avx2Token;
    use core::arch::x86_64::*;

    /// Load 8 contiguous i16 values and sign-extend to 8 i32 values in a ymm register.
    /// This is the key optimization - uses a single 128-bit load + vpmovsxwd.
    #[arcane]
    #[inline(always)]
    fn load_i16_to_i32(token: Avx2Token, data: &[i16; 8]) -> __m256i {
        let row_i16 = sse2::_mm_loadu_si128(token.sse2(), data);
        _mm256_cvtepi16_epi32(row_i16)
    }

    /// Pack 8 i32 values to 8 i16 values (with saturation).
    #[arcane]
    #[inline(always)]
    fn pack_i32_to_i16(_token: Avx2Token, v: __m256i) -> __m128i {
        // Extract low and high 128-bit lanes
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        // Pack with signed saturation
        _mm_packs_epi32(lo, hi)
    }

    /// Descale with rounding for pass 1 (CONST_BITS - PASS1_BITS = 11).
    #[arcane]
    #[inline(always)]
    fn descale_pass1(_token: Avx2Token, x: __m256i) -> __m256i {
        const N: i32 = CONST_BITS - PASS1_BITS; // 11
        let round = _mm256_set1_epi32(1 << (N - 1));
        _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
    }

    /// Descale with rounding for pass 2 (CONST_BITS + PASS1_BITS = 15).
    #[arcane]
    #[inline(always)]
    fn descale_pass2(_token: Avx2Token, x: __m256i) -> __m256i {
        const N: i32 = CONST_BITS + PASS1_BITS; // 15
        let round = _mm256_set1_epi32(1 << (N - 1));
        _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
    }

    /// Descale with rounding for PASS1_BITS = 2.
    #[arcane]
    #[inline(always)]
    fn descale_pass1_bits(_token: Avx2Token, x: __m256i) -> __m256i {
        const N: i32 = PASS1_BITS; // 2
        let round = _mm256_set1_epi32(1 << (N - 1));
        _mm256_srai_epi32::<N>(_mm256_add_epi32(x, round))
    }

    /// AVX2 intrinsics-based forward DCT.
    ///
    /// Uses proper load+widen instructions instead of gather pattern.
    /// This should generate much better assembly than the wide-based version.
    ///
    /// The token proves AVX2 is available. Memory operations use safe archmage wrappers.
    #[arcane]
    pub fn forward_dct_8x8_i32(
        token: Avx2Token,
        samples: &[i16; DCTSIZE2],
        coeffs: &mut [i16; DCTSIZE2],
    ) {
        // Helper to get a row reference
        #[inline(always)]
        fn row(samples: &[i16; DCTSIZE2], idx: usize) -> &[i16; 8] {
            samples[idx * 8..][..8].try_into().unwrap()
        }

        // Load all 8 rows with proper i16->i32 widening using safe memory operations
        let mut rows: [__m256i; 8] = [
            load_i16_to_i32(token, row(samples, 0)),
            load_i16_to_i32(token, row(samples, 1)),
            load_i16_to_i32(token, row(samples, 2)),
            load_i16_to_i32(token, row(samples, 3)),
            load_i16_to_i32(token, row(samples, 4)),
            load_i16_to_i32(token, row(samples, 5)),
            load_i16_to_i32(token, row(samples, 6)),
            load_i16_to_i32(token, row(samples, 7)),
        ];

        // Transpose 8x8 matrix using AVX2 shuffles
        transpose_8x8_avx2(token, &mut rows);

        // Pass 1: 1D DCT on rows (processing column elements after transpose)
        dct_1d_pass_avx2(token, &mut rows, true);

        // Transpose back
        transpose_8x8_avx2(token, &mut rows);

        // Pass 2: 1D DCT on columns
        // Output is already in row-major form (no final transpose needed)
        dct_1d_pass_avx2(token, &mut rows, false);

        // Store results - pack i32 back to i16 using safe memory operations
        let sse2_token = token.sse2();

        // Helper to get a mutable row reference
        #[inline(always)]
        fn row_mut(coeffs: &mut [i16; DCTSIZE2], idx: usize) -> &mut [i16; 8] {
            (&mut coeffs[idx * 8..][..8]).try_into().unwrap()
        }

        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 0),
            pack_i32_to_i16(token, rows[0]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 1),
            pack_i32_to_i16(token, rows[1]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 2),
            pack_i32_to_i16(token, rows[2]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 3),
            pack_i32_to_i16(token, rows[3]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 4),
            pack_i32_to_i16(token, rows[4]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 5),
            pack_i32_to_i16(token, rows[5]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 6),
            pack_i32_to_i16(token, rows[6]),
        );
        sse2::_mm_storeu_si128(
            sse2_token,
            row_mut(coeffs, 7),
            pack_i32_to_i16(token, rows[7]),
        );
    }

    /// Transpose 8x8 matrix of i32 values stored in 8 ymm registers.
    /// Uses AVX2 unpack and permute instructions.
    #[arcane]
    #[inline(always)]
    fn transpose_8x8_avx2(token: Avx2Token, rows: &mut [__m256i; 8]) {
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
        let _ = token; // silence unused warning
    }

    /// Perform 1D DCT on 8 rows simultaneously using AVX2.
    #[arcane]
    #[inline(always)]
    fn dct_1d_pass_avx2(token: Avx2Token, data: &mut [__m256i; 8], pass1: bool) {
        // Load constants
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

        // Output 0 and 4
        let (out0, out4) = if pass1 {
            (
                _mm256_slli_epi32::<PASS1_BITS>(_mm256_add_epi32(tmp10, tmp11)),
                _mm256_slli_epi32::<PASS1_BITS>(_mm256_sub_epi32(tmp10, tmp11)),
            )
        } else {
            (
                descale_pass1_bits(token, _mm256_add_epi32(tmp10, tmp11)),
                descale_pass1_bits(token, _mm256_sub_epi32(tmp10, tmp11)),
            )
        };

        // Output 2 and 6
        let z1 = _mm256_mullo_epi32(_mm256_add_epi32(tmp12, tmp13), fix_0_541196100);
        let (out2, out6) = if pass1 {
            (
                descale_pass1(
                    token,
                    _mm256_add_epi32(z1, _mm256_mullo_epi32(tmp13, fix_0_765366865)),
                ),
                descale_pass1(
                    token,
                    _mm256_sub_epi32(z1, _mm256_mullo_epi32(tmp12, fix_1_847759065)),
                ),
            )
        } else {
            (
                descale_pass2(
                    token,
                    _mm256_add_epi32(z1, _mm256_mullo_epi32(tmp13, fix_0_765366865)),
                ),
                descale_pass2(
                    token,
                    _mm256_sub_epi32(z1, _mm256_mullo_epi32(tmp12, fix_1_847759065)),
                ),
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
                descale_pass1(token, _mm256_add_epi32(_mm256_add_epi32(tmp4, z1), z3)),
                descale_pass1(token, _mm256_add_epi32(_mm256_add_epi32(tmp5, z2), z4)),
                descale_pass1(token, _mm256_add_epi32(_mm256_add_epi32(tmp6, z2), z3)),
                descale_pass1(token, _mm256_add_epi32(_mm256_add_epi32(tmp7, z1), z4)),
            )
        } else {
            (
                descale_pass2(token, _mm256_add_epi32(_mm256_add_epi32(tmp4, z1), z3)),
                descale_pass2(token, _mm256_add_epi32(_mm256_add_epi32(tmp5, z2), z4)),
                descale_pass2(token, _mm256_add_epi32(_mm256_add_epi32(tmp6, z2), z3)),
                descale_pass2(token, _mm256_add_epi32(_mm256_add_epi32(tmp7, z1), z4)),
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

    // ============================================================================
    // Mozjpeg-style 16-bit DCT using vpmaddwd
    // ============================================================================
    //
    // This implementation follows mozjpeg's jfdctint-avx2.asm approach:
    // - Data is packed as 16-bit (i16) values, two rows per YMM register
    // - Uses vpunpcklwd/hi for 16-bit transpose
    // - Uses vpmaddwd for combined multiply-add operations (key optimization!)
    //
    // vpmaddwd computes: result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
    // This allows computing expressions like (z1 + tmp13 * FIX_X) in one instruction.

    // Pre-computed constant vectors for vpmaddwd operations
    // Format: pairs of (multiplier_for_first, multiplier_for_second) repeated

    // For data2/data6: (FIX_0_541 + FIX_0_765, FIX_0_541) and (FIX_0_541 - FIX_1_847, FIX_0_541)
    const F_0_541_PLUS_0_765: i16 = (FIX_0_541196100 + FIX_0_765366865) as i16; // 10703
    const F_0_541_MINUS_1_847: i16 = (FIX_0_541196100 - FIX_1_847759065) as i16; // -10704
    const F_0_541: i16 = FIX_0_541196100 as i16;

    // For z3/z4: (FIX_1_175 - FIX_1_961, FIX_1_175) and (FIX_1_175 - FIX_0_390, FIX_1_175)
    const F_1_175_MINUS_1_961: i16 = (FIX_1_175875602 - FIX_1_961570560) as i16; // -6436
    const F_1_175_MINUS_0_390: i16 = (FIX_1_175875602 - FIX_0_390180644) as i16; // 6437
    const F_1_175: i16 = FIX_1_175875602 as i16;

    // For tmp4/tmp5: packed multipliers for odd part
    const F_0_298_MINUS_0_899: i16 = (FIX_0_298631336 - FIX_0_899976223) as i16; // -4927
    const F_NEG_0_899: i16 = (-FIX_0_899976223) as i16;
    const F_2_053_MINUS_2_562: i16 = (FIX_2_053119869 - FIX_2_562915447) as i16; // -4176
    const F_NEG_2_562: i16 = (-FIX_2_562915447) as i16;

    // For tmp6/tmp7: packed multipliers for odd part
    const F_3_072_MINUS_2_562: i16 = (FIX_3_072711026 - FIX_2_562915447) as i16; // 4177
    const F_1_501_MINUS_0_899: i16 = (FIX_1_501321110 - FIX_0_899976223) as i16; // 4926

    /// AVX2 forward DCT - direct translation of libjpeg-turbo's jfdctint-avx2.asm.
    ///
    /// Each assembly instruction is translated to the corresponding Rust intrinsic.
    /// This provides identical output to the reference C/asm implementation.
    ///
    /// The token proves AVX2 is available. Memory operations use safe archmage wrappers.
    #[arcane]
    pub fn forward_dct_8x8_i16(
        token: Avx2Token,
        samples: &[i16; DCTSIZE2],
        coeffs: &mut [i16; DCTSIZE2],
    ) {
        // Helper to get 16-element slice (256 bits) as array reference
        #[inline(always)]
        fn chunk(samples: &[i16; DCTSIZE2], idx: usize) -> &[i16; 16] {
            samples[idx * 16..][..16].try_into().unwrap()
        }

        #[inline(always)]
        fn chunk_mut(coeffs: &mut [i16; DCTSIZE2], idx: usize) -> &mut [i16; 16] {
            (&mut coeffs[idx * 16..][..16]).try_into().unwrap()
        }

        let avx_token = token.avx();

        // ---- Pass 1: process rows ----

        // Load: ymm4=(rows 0-1), ymm5=(rows 2-3), ymm6=(rows 4-5), ymm7=(rows 6-7)
        let ymm4 = avx::_mm256_loadu_si256(avx_token, chunk(samples, 0));
        let ymm5 = avx::_mm256_loadu_si256(avx_token, chunk(samples, 1));
        let ymm6 = avx::_mm256_loadu_si256(avx_token, chunk(samples, 2));
        let ymm7 = avx::_mm256_loadu_si256(avx_token, chunk(samples, 3));

        // Reorganize to (row0|row4), (row1|row5), (row2|row6), (row3|row7)
        let ymm0 = _mm256_permute2x128_si256(ymm4, ymm6, 0x20);
        let ymm1 = _mm256_permute2x128_si256(ymm4, ymm6, 0x31);
        let ymm2 = _mm256_permute2x128_si256(ymm5, ymm7, 0x20);
        let ymm3 = _mm256_permute2x128_si256(ymm5, ymm7, 0x31);

        // Transpose 8x8
        let (ymm0, ymm1, ymm2, ymm3) = dotranspose(token, ymm0, ymm1, ymm2, ymm3);

        // DCT pass 1
        let (ymm0, ymm1, ymm2, ymm3) = dodct(token, ymm0, ymm1, ymm2, ymm3, true);
        // After pass 1: ymm0=data0_4, ymm1=data3_1, ymm2=data2_6, ymm3=data7_5

        // ---- Pass 2: process columns ----

        // Reorganize for column pass
        let ymm4 = _mm256_permute2x128_si256(ymm1, ymm3, 0x20); // data3_7
        let ymm1 = _mm256_permute2x128_si256(ymm1, ymm3, 0x31); // data1_5

        // Transpose
        let (ymm0, ymm1, ymm2, ymm4) = dotranspose(token, ymm0, ymm1, ymm2, ymm4);

        // DCT pass 2
        let (ymm0, ymm1, ymm2, ymm4) = dodct(token, ymm0, ymm1, ymm2, ymm4, false);
        // After pass 2: ymm0=data0_4, ymm1=data3_1, ymm2=data2_6, ymm4=data7_5

        // Reorganize output to sequential rows
        let out01 = _mm256_permute2x128_si256(ymm0, ymm1, 0x30); // data0_1
        let out23 = _mm256_permute2x128_si256(ymm2, ymm1, 0x20); // data2_3
        let out45 = _mm256_permute2x128_si256(ymm0, ymm4, 0x31); // data4_5
        let out67 = _mm256_permute2x128_si256(ymm2, ymm4, 0x21); // data6_7

        // Store results using safe memory operations
        avx::_mm256_storeu_si256(avx_token, chunk_mut(coeffs, 0), out01);
        avx::_mm256_storeu_si256(avx_token, chunk_mut(coeffs, 1), out23);
        avx::_mm256_storeu_si256(avx_token, chunk_mut(coeffs, 2), out45);
        avx::_mm256_storeu_si256(avx_token, chunk_mut(coeffs, 3), out67);
    }

    /// 8x8x16-bit matrix transpose - exact translation of libjpeg-turbo's dotranspose macro
    #[arcane]
    #[inline(always)]
    fn dotranspose(
        token: Avx2Token,
        ymm0: __m256i,
        ymm1: __m256i,
        ymm2: __m256i,
        ymm3: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        // Phase 1: vpunpcklwd/vpunpckhwd
        let ymm4 = _mm256_unpacklo_epi16(ymm0, ymm1);
        let ymm5 = _mm256_unpackhi_epi16(ymm0, ymm1);
        let ymm6 = _mm256_unpacklo_epi16(ymm2, ymm3);
        let ymm7 = _mm256_unpackhi_epi16(ymm2, ymm3);

        // Phase 2: vpunpckldq/vpunpckhdq
        let ymm0 = _mm256_unpacklo_epi32(ymm4, ymm6);
        let ymm1 = _mm256_unpackhi_epi32(ymm4, ymm6);
        let ymm2 = _mm256_unpacklo_epi32(ymm5, ymm7);
        let ymm3 = _mm256_unpackhi_epi32(ymm5, ymm7);

        // Phase 3: vpermq
        let ymm0 = _mm256_permute4x64_epi64(ymm0, 0x8D);
        let ymm1 = _mm256_permute4x64_epi64(ymm1, 0x8D);
        let ymm2 = _mm256_permute4x64_epi64(ymm2, 0xD8);
        let ymm3 = _mm256_permute4x64_epi64(ymm3, 0xD8);

        let _ = token; // silence unused warning
        (ymm0, ymm1, ymm2, ymm3)
    }

    /// 1D DCT pass - exact translation of libjpeg-turbo's dodct macro
    #[arcane]
    #[inline(always)]
    fn dodct(
        token: Avx2Token,
        ymm0: __m256i,
        ymm1: __m256i,
        ymm2: __m256i,
        ymm3: __m256i,
        pass1: bool,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        // Step 1: Compute butterfly differences/sums
        let ymm4 = _mm256_sub_epi16(ymm0, ymm3); // tmp6_7
        let ymm5 = _mm256_add_epi16(ymm0, ymm3); // tmp1_0
        let ymm6 = _mm256_add_epi16(ymm1, ymm2); // tmp3_2
        let ymm7 = _mm256_sub_epi16(ymm1, ymm2); // tmp4_5

        // ---- Even part ----

        // Swap halves to get tmp0_1
        let ymm5 = _mm256_permute2x128_si256(ymm5, ymm5, 0x01);
        let ymm0 = _mm256_add_epi16(ymm5, ymm6); // tmp10_11
        let ymm5 = _mm256_sub_epi16(ymm5, ymm6); // tmp13_12

        let ymm6 = _mm256_permute2x128_si256(ymm0, ymm0, 0x01); // tmp11_10

        // PW_1_NEG1: high lane = 1, low lane = -1
        let pw_1_neg1 = _mm256_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        let ymm0 = _mm256_sign_epi16(ymm0, pw_1_neg1); // tmp10_neg11
        let ymm6 = _mm256_add_epi16(ymm6, ymm0); // (tmp10+tmp11)_(tmp10-tmp11)

        let out0_4 = if pass1 {
            _mm256_slli_epi16(ymm6, PASS1_BITS as i32)
        } else {
            let pw_descale = _mm256_set1_epi16(1 << (PASS1_BITS - 1));
            _mm256_srai_epi16(_mm256_add_epi16(ymm6, pw_descale), PASS1_BITS as i32)
        };

        // data2_6 calculation
        let ymm6 = _mm256_permute2x128_si256(ymm5, ymm5, 0x01); // tmp12_13
        let ymm1 = _mm256_unpacklo_epi16(ymm5, ymm6);
        let ymm5 = _mm256_unpackhi_epi16(ymm5, ymm6);

        // PW_F130_F054_MF130_F054 - libjpeg-turbo constant layout
        // NASM: times 4 dw A, B creates [A,B,A,B,...] - A at even indices
        // _mm256_set_epi16 is in reverse order, so we need [B,A,B,A,...] to get [A,B,A,B,...] in memory
        let pw_f130_f054 = _mm256_set_epi16(
            F_0_541,
            F_0_541_MINUS_1_847,
            F_0_541,
            F_0_541_MINUS_1_847,
            F_0_541,
            F_0_541_MINUS_1_847,
            F_0_541,
            F_0_541_MINUS_1_847,
            F_0_541,
            F_0_541_PLUS_0_765,
            F_0_541,
            F_0_541_PLUS_0_765,
            F_0_541,
            F_0_541_PLUS_0_765,
            F_0_541,
            F_0_541_PLUS_0_765,
        );

        let ymm1 = _mm256_madd_epi16(ymm1, pw_f130_f054);
        let ymm5 = _mm256_madd_epi16(ymm5, pw_f130_f054);

        let (ymm1, ymm5, pd_descale) = if pass1 {
            let pd = _mm256_set1_epi32(1 << (CONST_BITS - PASS1_BITS - 1));
            const N: i32 = CONST_BITS - PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm1, pd)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm5, pd)),
                pd,
            )
        } else {
            let pd = _mm256_set1_epi32(1 << (CONST_BITS + PASS1_BITS - 1));
            const N: i32 = CONST_BITS + PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm1, pd)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm5, pd)),
                pd,
            )
        };

        let out2_6 = _mm256_packs_epi32(ymm1, ymm5);

        // ---- Odd part ----

        let ymm6 = _mm256_add_epi16(ymm7, ymm4); // z3_4

        // z3/z4 calculation
        let ymm1 = _mm256_permute2x128_si256(ymm6, ymm6, 0x01); // z4_3
        let ymm5 = _mm256_unpacklo_epi16(ymm6, ymm1);
        let ymm6 = _mm256_unpackhi_epi16(ymm6, ymm1);

        // PW_MF078_F117_F078_F117 - pairs are [A,B] in memory
        let pw_mf078_f117 = _mm256_set_epi16(
            F_1_175,
            F_1_175_MINUS_0_390,
            F_1_175,
            F_1_175_MINUS_0_390,
            F_1_175,
            F_1_175_MINUS_0_390,
            F_1_175,
            F_1_175_MINUS_0_390,
            F_1_175,
            F_1_175_MINUS_1_961,
            F_1_175,
            F_1_175_MINUS_1_961,
            F_1_175,
            F_1_175_MINUS_1_961,
            F_1_175,
            F_1_175_MINUS_1_961,
        );

        let ymm5 = _mm256_madd_epi16(ymm5, pw_mf078_f117);
        let ymm6 = _mm256_madd_epi16(ymm6, pw_mf078_f117);

        // tmp4/tmp5 calculation
        let ymm3 = _mm256_permute2x128_si256(ymm4, ymm4, 0x01); // tmp7_6
        let ymm1 = _mm256_unpacklo_epi16(ymm7, ymm3);
        let ymm3 = _mm256_unpackhi_epi16(ymm7, ymm3);

        // PW_MF060_MF089_MF050_MF256 - pairs are [A,B] in memory
        let pw_mf060_mf089 = _mm256_set_epi16(
            F_NEG_2_562,
            F_2_053_MINUS_2_562,
            F_NEG_2_562,
            F_2_053_MINUS_2_562,
            F_NEG_2_562,
            F_2_053_MINUS_2_562,
            F_NEG_2_562,
            F_2_053_MINUS_2_562,
            F_NEG_0_899,
            F_0_298_MINUS_0_899,
            F_NEG_0_899,
            F_0_298_MINUS_0_899,
            F_NEG_0_899,
            F_0_298_MINUS_0_899,
            F_NEG_0_899,
            F_0_298_MINUS_0_899,
        );

        let ymm1 = _mm256_madd_epi16(ymm1, pw_mf060_mf089);
        let ymm3 = _mm256_madd_epi16(ymm3, pw_mf060_mf089);

        let ymm1 = _mm256_add_epi32(ymm1, ymm5); // data7_5L
        let ymm3 = _mm256_add_epi32(ymm3, ymm6); // data7_5H

        let (ymm1d, ymm3d) = if pass1 {
            const N: i32 = CONST_BITS - PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm1, pd_descale)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm3, pd_descale)),
            )
        } else {
            const N: i32 = CONST_BITS + PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm1, pd_descale)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm3, pd_descale)),
            )
        };

        let out7_5 = _mm256_packs_epi32(ymm1d, ymm3d);

        // tmp6/tmp7 calculation
        let ymm1 = _mm256_permute2x128_si256(ymm7, ymm7, 0x01); // tmp5_4
        let ymm7 = _mm256_unpacklo_epi16(ymm4, ymm1);
        let ymm4 = _mm256_unpackhi_epi16(ymm4, ymm1);

        // PW_F050_MF256_F060_MF089
        let pw_f050_mf256 = _mm256_set_epi16(
            F_NEG_0_899,
            F_1_501_MINUS_0_899,
            F_NEG_0_899,
            F_1_501_MINUS_0_899,
            F_NEG_0_899,
            F_1_501_MINUS_0_899,
            F_NEG_0_899,
            F_1_501_MINUS_0_899,
            F_NEG_2_562,
            F_3_072_MINUS_2_562,
            F_NEG_2_562,
            F_3_072_MINUS_2_562,
            F_NEG_2_562,
            F_3_072_MINUS_2_562,
            F_NEG_2_562,
            F_3_072_MINUS_2_562,
        );

        let ymm7 = _mm256_madd_epi16(ymm7, pw_f050_mf256);
        let ymm4 = _mm256_madd_epi16(ymm4, pw_f050_mf256);

        let ymm7 = _mm256_add_epi32(ymm7, ymm5); // data3_1L
        let ymm4 = _mm256_add_epi32(ymm4, ymm6); // data3_1H

        let (ymm7, ymm4) = if pass1 {
            const N: i32 = CONST_BITS - PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm7, pd_descale)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm4, pd_descale)),
            )
        } else {
            const N: i32 = CONST_BITS + PASS1_BITS;
            (
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm7, pd_descale)),
                _mm256_srai_epi32::<N>(_mm256_add_epi32(ymm4, pd_descale)),
            )
        };

        let out3_1 = _mm256_packs_epi32(ymm7, ymm4);

        // Output: data0_4, data3_1, data2_6, data7_5
        let _ = token; // silence unused warning
        (out0_4, out3_1, out2_6, out7_5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_shift() {
        let samples = [128u8; DCTSIZE2];
        let mut output = [0i16; DCTSIZE2];
        level_shift(&samples, &mut output);

        // 128 - 128 = 0
        for v in output.iter() {
            assert_eq!(*v, 0);
        }

        let mut samples2 = [0u8; DCTSIZE2];
        samples2[0] = 255;
        samples2[1] = 0;
        level_shift(&samples2, &mut output);
        assert_eq!(output[0], 127); // 255 - 128
        assert_eq!(output[1], -128); // 0 - 128
    }

    #[test]
    fn test_dc_coefficient() {
        // A flat block of all same values should have:
        // - DC coefficient = 8 * 8 * value (due to 2D DCT scaling, factor of 8 per dimension)
        // - All AC coefficients = 0
        let mut samples = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            samples[i] = 100; // Flat value after level shift
        }

        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs);

        // DC should be 8 * 8 * value = 64 * value = 6400
        // (Factor of 8 from row pass and factor of 8 from column pass)
        assert_eq!(
            coeffs[0], 6400,
            "DC coefficient should be 64 * input value for flat block"
        );

        // All AC coefficients should be 0 (or very close due to rounding)
        for i in 1..DCTSIZE2 {
            assert!(
                coeffs[i].abs() <= 1,
                "AC coefficient [{}] should be ~0 for flat block, got {}",
                i,
                coeffs[i]
            );
        }
    }

    #[test]
    fn test_zero_block() {
        let samples = [0i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs);

        // All coefficients should be 0
        for i in 0..DCTSIZE2 {
            assert_eq!(
                coeffs[i], 0,
                "Coefficient [{}] should be 0 for zero block",
                i
            );
        }
    }

    #[test]
    fn test_alternating_pattern() {
        // Horizontal stripes should produce non-zero vertical frequency components
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            let val = if row % 2 == 0 { 100 } else { -100 };
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = val;
            }
        }

        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs);

        // DC should be 0 (equal positive and negative)
        assert!(coeffs[0].abs() <= 1, "DC should be ~0 for balanced pattern");

        // For alternating rows (+100/-100), energy should be concentrated in
        // the vertical high frequency components. Check that there's significant
        // energy in AC coefficients (any row > 0, col 0).
        let mut max_vertical_ac = 0i16;
        for row in 1..DCTSIZE {
            max_vertical_ac = max_vertical_ac.max(coeffs[row * DCTSIZE].abs());
        }
        assert!(
            max_vertical_ac > 50,
            "Vertical AC frequencies should be present, got {}",
            max_vertical_ac
        );
    }

    #[test]
    fn test_gradient() {
        // Horizontal gradient should produce low-frequency horizontal component
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = (col as i16 - 4) * 20;
            }
        }

        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs);

        // The horizontal gradient should produce significant energy at position [0][1]
        // (first horizontal AC coefficient)
        assert!(
            coeffs[1].abs() > 100,
            "Horizontal low frequency should be present"
        );
    }

    #[test]
    fn test_descale_rounding() {
        // Test that descale rounds correctly (rounds toward negative infinity)
        assert_eq!(descale(7, 2), 2); // (7+2) >> 2 = 9 >> 2 = 2
        assert_eq!(descale(8, 2), 2); // (8+2) >> 2 = 10 >> 2 = 2
        assert_eq!(descale(9, 2), 2); // (9+2) >> 2 = 11 >> 2 = 2
        assert_eq!(descale(10, 2), 3); // (10+2) >> 2 = 12 >> 2 = 3

        // Negative values (arithmetic right shift rounds toward -infinity)
        assert_eq!(descale(-7, 2), -2); // (-7+2) >> 2 = -5 >> 2 = -2
        assert_eq!(descale(-8, 2), -2); // (-8+2) >> 2 = -6 >> 2 = -2
        assert_eq!(descale(-9, 2), -2); // (-9+2) >> 2 = -7 >> 2 = -2
        assert_eq!(descale(-10, 2), -2); // (-10+2) >> 2 = -8 >> 2 = -2
    }

    #[test]
    #[allow(deprecated)]
    fn test_simd_matches_scalar_flat() {
        // Test SIMD produces identical output to scalar for flat block
        let samples = [100i16; DCTSIZE2];
        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_gather(&samples, &mut coeffs_simd);

        assert_eq!(
            coeffs_scalar, coeffs_simd,
            "SIMD should match scalar for flat block"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_simd_matches_scalar_gradient() {
        // Test SIMD produces identical output to scalar for gradient pattern
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = (row as i16 - 4) * 20 + (col as i16 - 4) * 10;
            }
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_gather(&samples, &mut coeffs_simd);

        assert_eq!(
            coeffs_scalar, coeffs_simd,
            "SIMD should match scalar for gradient block"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_simd_matches_scalar_random() {
        // Test SIMD produces identical output to scalar for pseudo-random pattern
        let mut samples = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            // Deterministic pseudo-random values in range -128..127
            samples[i] = ((i as i32 * 73 + 17) % 256 - 128) as i16;
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_gather(&samples, &mut coeffs_simd);

        assert_eq!(
            coeffs_scalar, coeffs_simd,
            "SIMD should match scalar for random block"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_simd_matches_scalar_all_patterns() {
        // Exhaustive test with many patterns
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_simd = [0i16; DCTSIZE2];

            forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
            forward_dct_8x8_i32_wide_gather(&samples, &mut coeffs_simd);

            assert_eq!(
                coeffs_scalar, coeffs_simd,
                "SIMD should match scalar for pattern seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_transpose_matches_scalar_flat() {
        // Test transpose SIMD produces identical output to scalar for flat block
        let samples = [100i16; DCTSIZE2];
        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_transpose = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(
            coeffs_scalar, coeffs_transpose,
            "Transpose SIMD should match scalar for flat block"
        );
    }

    #[test]
    fn test_transpose_matches_scalar_gradient() {
        // Test transpose SIMD produces identical output to scalar for gradient pattern
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = (row as i16 - 4) * 20 + (col as i16 - 4) * 10;
            }
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_transpose = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(
            coeffs_scalar, coeffs_transpose,
            "Transpose SIMD should match scalar for gradient block"
        );
    }

    #[test]
    fn test_transpose_matches_scalar_random() {
        // Test transpose SIMD produces identical output to scalar for pseudo-random pattern
        let mut samples = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            // Deterministic pseudo-random values in range -128..127
            samples[i] = ((i as i32 * 73 + 17) % 256 - 128) as i16;
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_transpose = [0i16; DCTSIZE2];

        forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
        forward_dct_8x8_i32_wide_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(
            coeffs_scalar, coeffs_transpose,
            "Transpose SIMD should match scalar for random block"
        );
    }

    #[test]
    fn test_transpose_matches_scalar_all_patterns() {
        // Exhaustive test with many patterns
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_transpose = [0i16; DCTSIZE2];

            forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
            forward_dct_8x8_i32_wide_transpose(&samples, &mut coeffs_transpose);

            assert_eq!(
                coeffs_scalar, coeffs_transpose,
                "Transpose SIMD should match scalar for pattern seed {}",
                seed
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_matches_scalar_all_patterns() {
        use super::avx2::forward_dct_8x8_avx2;
        use archmage::tokens::x86::Avx2Token;
        use archmage::SimdToken;

        // SAFETY: Test runs only on x86_64 with AVX2 support
        let token = unsafe { Avx2Token::forge_token_dangerously() };

        // Exhaustive test with many patterns
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_avx2 = [0i16; DCTSIZE2];

            forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
            forward_dct_8x8_avx2(token, &samples, &mut coeffs_avx2);

            assert_eq!(
                coeffs_scalar, coeffs_avx2,
                "AVX2 SIMD should match scalar for pattern seed {}",
                seed
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_i16_matches_scalar_all_patterns() {
        use super::avx2::forward_dct_8x8_avx2_i16;
        use archmage::tokens::x86::Avx2Token;
        use archmage::SimdToken;

        // SAFETY: Test runs only on x86_64 with AVX2 support
        let token = unsafe { Avx2Token::forge_token_dangerously() };

        // Exhaustive test with many patterns
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_avx2_i16 = [0i16; DCTSIZE2];

            forward_dct_8x8_i32_multiversion(&samples, &mut coeffs_scalar);
            forward_dct_8x8_avx2_i16(token, &samples, &mut coeffs_avx2_i16);

            // Allow small differences due to 16-bit vs 32-bit intermediate precision
            for i in 0..DCTSIZE2 {
                let diff = (coeffs_scalar[i] as i32 - coeffs_avx2_i16[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "AVX2 i16 should match scalar within ±1 for pattern seed {}, coeff {}: scalar={}, i16={}",
                    seed,
                    i,
                    coeffs_scalar[i],
                    coeffs_avx2_i16[i]
                );
            }
        }
    }
}
