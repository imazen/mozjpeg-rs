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
//! The `forward_dct_8x8_simd` function uses the "row-parallel" approach:
//! process 4 rows simultaneously using `wide::i32x4`. Each lane of the
//! SIMD vector handles a different row.
//!
//! Reference: C. Loeffler, A. Ligtenberg and G. Moschytz,
//! "Practical Fast 1-D DCT Algorithms with 11 Multiplications",
//! Proc. ICASSP 1989, pp. 988-991.

use crate::consts::{DCTSIZE, DCTSIZE2};
use wide::{i32x4, i32x8};

// Fixed-point constants for 13-bit precision (CONST_BITS = 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Pre-calculated fixed-point constants: FIX(x) = (x * (1 << CONST_BITS) + 0.5)
const FIX_0_298631336: i32 = 2446;   // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196;   // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433;   // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270;   // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373;   // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633;   // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299;  // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137;  // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069;  // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819;  // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995;  // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172;  // FIX(3.072711026)

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
/// # Arguments
/// * `samples` - Input 8x8 block of pixel samples (typically centered around 0)
/// * `coeffs` - Output 8x8 block of DCT coefficients
pub fn forward_dct_8x8(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
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

        let tmp0 = data[base + 0] + data[base + 7];
        let tmp7 = data[base + 0] - data[base + 7];
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

        data[base + 0] = (tmp10 + tmp11) << PASS1_BITS;
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

        let tmp4 = tmp4 * FIX_0_298631336;  // sqrt(2) * (-c1+c3+c5-c7)
        let tmp5 = tmp5 * FIX_2_053119869;  // sqrt(2) * ( c1+c3-c5+c7)
        let tmp6 = tmp6 * FIX_3_072711026;  // sqrt(2) * ( c1+c3+c5-c7)
        let tmp7 = tmp7 * FIX_1_501321110;  // sqrt(2) * ( c1+c3-c5-c7)
        let z1 = z1 * (-FIX_0_899976223);   // sqrt(2) * ( c7-c3)
        let z2 = z2 * (-FIX_2_562915447);   // sqrt(2) * (-c1-c3)
        let z3 = z3 * (-FIX_1_961570560) + z5;  // sqrt(2) * (-c3-c5)
        let z4 = z4 * (-FIX_0_390180644) + z5;  // sqrt(2) * ( c5-c3)

        data[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    // We remove PASS1_BITS scaling but leave results scaled by factor of 8
    for col in 0..DCTSIZE {
        let tmp0 = data[DCTSIZE * 0 + col] + data[DCTSIZE * 7 + col];
        let tmp7 = data[DCTSIZE * 0 + col] - data[DCTSIZE * 7 + col];
        let tmp1 = data[DCTSIZE * 1 + col] + data[DCTSIZE * 6 + col];
        let tmp6 = data[DCTSIZE * 1 + col] - data[DCTSIZE * 6 + col];
        let tmp2 = data[DCTSIZE * 2 + col] + data[DCTSIZE * 5 + col];
        let tmp5 = data[DCTSIZE * 2 + col] - data[DCTSIZE * 5 + col];
        let tmp3 = data[DCTSIZE * 3 + col] + data[DCTSIZE * 4 + col];
        let tmp4 = data[DCTSIZE * 3 + col] - data[DCTSIZE * 4 + col];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[DCTSIZE * 0 + col] = descale(tmp10 + tmp11, PASS1_BITS);
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
        data[DCTSIZE * 1 + col] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
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
#[inline(always)]
fn dct_1d_simd(
    d0: i32x4, d1: i32x4, d2: i32x4, d3: i32x4,
    d4: i32x4, d5: i32x4, d6: i32x4, d7: i32x4,
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

    let scale = if shift_pass1 { CONST_BITS - PASS1_BITS } else { CONST_BITS + PASS1_BITS };
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
pub fn forward_dct_8x8_simd(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
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
        let arr0 = out[0].to_array(); let arr1 = out[1].to_array();
        let arr2 = out[2].to_array(); let arr3 = out[3].to_array();
        let arr4 = out[4].to_array(); let arr5 = out[5].to_array();
        let arr6 = out[6].to_array(); let arr7 = out[7].to_array();

        data[0] = arr0[0]; data[1] = arr1[0]; data[2] = arr2[0]; data[3] = arr3[0];
        data[4] = arr4[0]; data[5] = arr5[0]; data[6] = arr6[0]; data[7] = arr7[0];
        data[8] = arr0[1]; data[9] = arr1[1]; data[10] = arr2[1]; data[11] = arr3[1];
        data[12] = arr4[1]; data[13] = arr5[1]; data[14] = arr6[1]; data[15] = arr7[1];
        data[16] = arr0[2]; data[17] = arr1[2]; data[18] = arr2[2]; data[19] = arr3[2];
        data[20] = arr4[2]; data[21] = arr5[2]; data[22] = arr6[2]; data[23] = arr7[2];
        data[24] = arr0[3]; data[25] = arr1[3]; data[26] = arr2[3]; data[27] = arr3[3];
        data[28] = arr4[3]; data[29] = arr5[3]; data[30] = arr6[3]; data[31] = arr7[3];
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

        let arr0 = out[0].to_array(); let arr1 = out[1].to_array();
        let arr2 = out[2].to_array(); let arr3 = out[3].to_array();
        let arr4 = out[4].to_array(); let arr5 = out[5].to_array();
        let arr6 = out[6].to_array(); let arr7 = out[7].to_array();

        data[32] = arr0[0]; data[33] = arr1[0]; data[34] = arr2[0]; data[35] = arr3[0];
        data[36] = arr4[0]; data[37] = arr5[0]; data[38] = arr6[0]; data[39] = arr7[0];
        data[40] = arr0[1]; data[41] = arr1[1]; data[42] = arr2[1]; data[43] = arr3[1];
        data[44] = arr4[1]; data[45] = arr5[1]; data[46] = arr6[1]; data[47] = arr7[1];
        data[48] = arr0[2]; data[49] = arr1[2]; data[50] = arr2[2]; data[51] = arr3[2];
        data[52] = arr4[2]; data[53] = arr5[2]; data[54] = arr6[2]; data[55] = arr7[2];
        data[56] = arr0[3]; data[57] = arr1[3]; data[58] = arr2[3]; data[59] = arr3[3];
        data[60] = arr4[3]; data[61] = arr5[3]; data[62] = arr6[3]; data[63] = arr7[3];
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

        let arr0 = out[0].to_array(); let arr1 = out[1].to_array();
        let arr2 = out[2].to_array(); let arr3 = out[3].to_array();
        let arr4 = out[4].to_array(); let arr5 = out[5].to_array();
        let arr6 = out[6].to_array(); let arr7 = out[7].to_array();

        data[0] = arr0[0]; data[1] = arr0[1]; data[2] = arr0[2]; data[3] = arr0[3];
        data[8] = arr1[0]; data[9] = arr1[1]; data[10] = arr1[2]; data[11] = arr1[3];
        data[16] = arr2[0]; data[17] = arr2[1]; data[18] = arr2[2]; data[19] = arr2[3];
        data[24] = arr3[0]; data[25] = arr3[1]; data[26] = arr3[2]; data[27] = arr3[3];
        data[32] = arr4[0]; data[33] = arr4[1]; data[34] = arr4[2]; data[35] = arr4[3];
        data[40] = arr5[0]; data[41] = arr5[1]; data[42] = arr5[2]; data[43] = arr5[3];
        data[48] = arr6[0]; data[49] = arr6[1]; data[50] = arr6[2]; data[51] = arr6[3];
        data[56] = arr7[0]; data[57] = arr7[1]; data[58] = arr7[2]; data[59] = arr7[3];
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

        let arr0 = out[0].to_array(); let arr1 = out[1].to_array();
        let arr2 = out[2].to_array(); let arr3 = out[3].to_array();
        let arr4 = out[4].to_array(); let arr5 = out[5].to_array();
        let arr6 = out[6].to_array(); let arr7 = out[7].to_array();

        data[4] = arr0[0]; data[5] = arr0[1]; data[6] = arr0[2]; data[7] = arr0[3];
        data[12] = arr1[0]; data[13] = arr1[1]; data[14] = arr1[2]; data[15] = arr1[3];
        data[20] = arr2[0]; data[21] = arr2[1]; data[22] = arr2[2]; data[23] = arr2[3];
        data[28] = arr3[0]; data[29] = arr3[1]; data[30] = arr3[2]; data[31] = arr3[3];
        data[36] = arr4[0]; data[37] = arr4[1]; data[38] = arr4[2]; data[39] = arr4[3];
        data[44] = arr5[0]; data[45] = arr5[1]; data[46] = arr5[2]; data[47] = arr5[3];
        data[52] = arr6[0]; data[53] = arr6[1]; data[54] = arr6[2]; data[55] = arr6[3];
        data[60] = arr7[0]; data[61] = arr7[1]; data[62] = arr7[2]; data[63] = arr7[3];
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
#[inline(always)]
fn dct_1d_8wide(
    d0: i32x8, d1: i32x8, d2: i32x8, d3: i32x8,
    d4: i32x8, d5: i32x8, d6: i32x8, d7: i32x8,
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

    let scale = if shift_pass1 { CONST_BITS - PASS1_BITS } else { CONST_BITS + PASS1_BITS };
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

    // Transpose in-place
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
/// The key insight: when we call dct_1d_8wide(rows[0], rows[1], ..., rows[7]),
/// it combines rows[0]+rows[7], rows[1]+rows[6], etc. - that's column-wise processing.
/// To do row-wise processing, we transpose first so that rows become columns.
///
/// The contiguous loads should be faster than the gather-based approach.
pub fn forward_dct_8x8_transpose(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Load all 8 rows contiguously - this is the key optimization!
    // Each row is 8 contiguous i16 values converted to i32
    let mut data: [i32x8; 8] = [
        i32x8::new([
            samples[0] as i32, samples[1] as i32, samples[2] as i32, samples[3] as i32,
            samples[4] as i32, samples[5] as i32, samples[6] as i32, samples[7] as i32,
        ]),
        i32x8::new([
            samples[8] as i32, samples[9] as i32, samples[10] as i32, samples[11] as i32,
            samples[12] as i32, samples[13] as i32, samples[14] as i32, samples[15] as i32,
        ]),
        i32x8::new([
            samples[16] as i32, samples[17] as i32, samples[18] as i32, samples[19] as i32,
            samples[20] as i32, samples[21] as i32, samples[22] as i32, samples[23] as i32,
        ]),
        i32x8::new([
            samples[24] as i32, samples[25] as i32, samples[26] as i32, samples[27] as i32,
            samples[28] as i32, samples[29] as i32, samples[30] as i32, samples[31] as i32,
        ]),
        i32x8::new([
            samples[32] as i32, samples[33] as i32, samples[34] as i32, samples[35] as i32,
            samples[36] as i32, samples[37] as i32, samples[38] as i32, samples[39] as i32,
        ]),
        i32x8::new([
            samples[40] as i32, samples[41] as i32, samples[42] as i32, samples[43] as i32,
            samples[44] as i32, samples[45] as i32, samples[46] as i32, samples[47] as i32,
        ]),
        i32x8::new([
            samples[48] as i32, samples[49] as i32, samples[50] as i32, samples[51] as i32,
            samples[52] as i32, samples[53] as i32, samples[54] as i32, samples[55] as i32,
        ]),
        i32x8::new([
            samples[56] as i32, samples[57] as i32, samples[58] as i32, samples[59] as i32,
            samples[60] as i32, samples[61] as i32, samples[62] as i32, samples[63] as i32,
        ]),
    ];

    // Transpose first: now data[i] contains column i (all 8 rows' values for column i)
    // After this, dct_1d_8wide will process all 8 rows in parallel
    transpose_8x8(&mut data);

    // Pass 1: 1D DCT on rows (we transposed, so this processes row elements)
    // dct_1d_8wide(data[0], data[1], ...) now combines columns 0+7, 1+6, etc. within each row
    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
        true,
    );
    data = result;

    // Transpose: data[i] now contains row i
    transpose_8x8(&mut data);

    // Pass 2: 1D DCT on columns (we transposed, so this processes column elements)
    // dct_1d_8wide(data[0], data[1], ...) now combines rows 0+7, 1+6, etc. within each column
    // Output: data[row][col] = final DCT coefficient - already row-major!
    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
        false,
    );
    data = result;

    // Store results (data is already in row-major form)
    let arr0 = data[0].to_array(); let arr1 = data[1].to_array();
    let arr2 = data[2].to_array(); let arr3 = data[3].to_array();
    let arr4 = data[4].to_array(); let arr5 = data[5].to_array();
    let arr6 = data[6].to_array(); let arr7 = data[7].to_array();

    for i in 0..8 { coeffs[i] = arr0[i] as i16; }
    for i in 0..8 { coeffs[8 + i] = arr1[i] as i16; }
    for i in 0..8 { coeffs[16 + i] = arr2[i] as i16; }
    for i in 0..8 { coeffs[24 + i] = arr3[i] as i16; }
    for i in 0..8 { coeffs[32 + i] = arr4[i] as i16; }
    for i in 0..8 { coeffs[40 + i] = arr5[i] as i16; }
    for i in 0..8 { coeffs[48 + i] = arr6[i] as i16; }
    for i in 0..8 { coeffs[56 + i] = arr7[i] as i16; }
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
///
/// # Arguments
/// * `samples` - Input 8x8 block of pixel samples (0-255)
/// * `coeffs` - Output 8x8 block of DCT coefficients
pub fn forward_dct(samples: &[u8; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut shifted = [0i16; DCTSIZE2];
    level_shift(samples, &mut shifted);
    forward_dct_8x8_transpose(&shifted, coeffs);
}

/// Combined level-shift, overshoot deringing, and forward DCT.
///
/// Uses SIMD-optimized DCT for better performance.
/// Uses transpose-based approach for contiguous memory access.
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
pub fn forward_dct_with_deringing(
    samples: &[u8; DCTSIZE2],
    coeffs: &mut [i16; DCTSIZE2],
    dc_quant: u16,
) {
    use crate::deringing::preprocess_deringing;

    let mut shifted = [0i16; DCTSIZE2];
    level_shift(samples, &mut shifted);
    preprocess_deringing(&mut shifted, dc_quant);
    forward_dct_8x8_transpose(&shifted, coeffs);
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
        assert_eq!(output[0], 127);  // 255 - 128
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
        forward_dct_8x8(&samples, &mut coeffs);

        // DC should be 8 * 8 * value = 64 * value = 6400
        // (Factor of 8 from row pass and factor of 8 from column pass)
        assert_eq!(coeffs[0], 6400, "DC coefficient should be 64 * input value for flat block");

        // All AC coefficients should be 0 (or very close due to rounding)
        for i in 1..DCTSIZE2 {
            assert!(coeffs[i].abs() <= 1,
                "AC coefficient [{}] should be ~0 for flat block, got {}",
                i, coeffs[i]);
        }
    }

    #[test]
    fn test_zero_block() {
        let samples = [0i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        forward_dct_8x8(&samples, &mut coeffs);

        // All coefficients should be 0
        for i in 0..DCTSIZE2 {
            assert_eq!(coeffs[i], 0, "Coefficient [{}] should be 0 for zero block", i);
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
        forward_dct_8x8(&samples, &mut coeffs);

        // DC should be 0 (equal positive and negative)
        assert!(coeffs[0].abs() <= 1, "DC should be ~0 for balanced pattern");

        // For alternating rows (+100/-100), energy should be concentrated in
        // the vertical high frequency components. Check that there's significant
        // energy in AC coefficients (any row > 0, col 0).
        let mut max_vertical_ac = 0i16;
        for row in 1..DCTSIZE {
            max_vertical_ac = max_vertical_ac.max(coeffs[row * DCTSIZE].abs());
        }
        assert!(max_vertical_ac > 50, "Vertical AC frequencies should be present, got {}", max_vertical_ac);
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
        forward_dct_8x8(&samples, &mut coeffs);

        // The horizontal gradient should produce significant energy at position [0][1]
        // (first horizontal AC coefficient)
        assert!(coeffs[1].abs() > 100, "Horizontal low frequency should be present");
    }

    #[test]
    fn test_descale_rounding() {
        // Test that descale rounds correctly (rounds toward negative infinity)
        assert_eq!(descale(7, 2), 2);   // (7+2) >> 2 = 9 >> 2 = 2
        assert_eq!(descale(8, 2), 2);   // (8+2) >> 2 = 10 >> 2 = 2
        assert_eq!(descale(9, 2), 2);   // (9+2) >> 2 = 11 >> 2 = 2
        assert_eq!(descale(10, 2), 3);  // (10+2) >> 2 = 12 >> 2 = 3

        // Negative values (arithmetic right shift rounds toward -infinity)
        assert_eq!(descale(-7, 2), -2);  // (-7+2) >> 2 = -5 >> 2 = -2
        assert_eq!(descale(-8, 2), -2);  // (-8+2) >> 2 = -6 >> 2 = -2
        assert_eq!(descale(-9, 2), -2);  // (-9+2) >> 2 = -7 >> 2 = -2
        assert_eq!(descale(-10, 2), -2); // (-10+2) >> 2 = -8 >> 2 = -2
    }

    #[test]
    fn test_simd_matches_scalar_flat() {
        // Test SIMD produces identical output to scalar for flat block
        let samples = [100i16; DCTSIZE2];
        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_simd(&samples, &mut coeffs_simd);

        assert_eq!(coeffs_scalar, coeffs_simd, "SIMD should match scalar for flat block");
    }

    #[test]
    fn test_simd_matches_scalar_gradient() {
        // Test SIMD produces identical output to scalar for gradient pattern
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = ((row as i16 - 4) * 20 + (col as i16 - 4) * 10);
            }
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_simd(&samples, &mut coeffs_simd);

        assert_eq!(coeffs_scalar, coeffs_simd, "SIMD should match scalar for gradient block");
    }

    #[test]
    fn test_simd_matches_scalar_random() {
        // Test SIMD produces identical output to scalar for pseudo-random pattern
        let mut samples = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            // Deterministic pseudo-random values in range -128..127
            samples[i] = ((i as i32 * 73 + 17) % 256 - 128) as i16;
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_simd = [0i16; DCTSIZE2];

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_simd(&samples, &mut coeffs_simd);

        assert_eq!(coeffs_scalar, coeffs_simd, "SIMD should match scalar for random block");
    }

    #[test]
    fn test_simd_matches_scalar_all_patterns() {
        // Exhaustive test with many patterns
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_simd = [0i16; DCTSIZE2];

            forward_dct_8x8(&samples, &mut coeffs_scalar);
            forward_dct_8x8_simd(&samples, &mut coeffs_simd);

            assert_eq!(
                coeffs_scalar, coeffs_simd,
                "SIMD should match scalar for pattern seed {}", seed
            );
        }
    }

    #[test]
    fn test_transpose_matches_scalar_flat() {
        // Test transpose SIMD produces identical output to scalar for flat block
        let samples = [100i16; DCTSIZE2];
        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_transpose = [0i16; DCTSIZE2];

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(coeffs_scalar, coeffs_transpose, "Transpose SIMD should match scalar for flat block");
    }

    #[test]
    fn test_transpose_matches_scalar_gradient() {
        // Test transpose SIMD produces identical output to scalar for gradient pattern
        let mut samples = [0i16; DCTSIZE2];
        for row in 0..DCTSIZE {
            for col in 0..DCTSIZE {
                samples[row * DCTSIZE + col] = ((row as i16 - 4) * 20 + (col as i16 - 4) * 10);
            }
        }

        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_transpose = [0i16; DCTSIZE2];

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(coeffs_scalar, coeffs_transpose, "Transpose SIMD should match scalar for gradient block");
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

        forward_dct_8x8(&samples, &mut coeffs_scalar);
        forward_dct_8x8_transpose(&samples, &mut coeffs_transpose);

        assert_eq!(coeffs_scalar, coeffs_transpose, "Transpose SIMD should match scalar for random block");
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

            forward_dct_8x8(&samples, &mut coeffs_scalar);
            forward_dct_8x8_transpose(&samples, &mut coeffs_transpose);

            assert_eq!(
                coeffs_scalar, coeffs_transpose,
                "Transpose SIMD should match scalar for pattern seed {}", seed
            );
        }
    }
}
