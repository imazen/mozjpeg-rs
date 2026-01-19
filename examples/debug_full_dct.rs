//! Full DCT debug tracing both passes

use std::arch::x86_64::*;

const DCTSIZE: usize = 8;
const DCTSIZE2: usize = 64;
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Fixed-point constants
const F_0_541_PLUS_0_765: i16 = (4433 + 6270) as i16;
const F_0_541_MINUS_1_847: i16 = (4433 - 15137) as i16;
const F_0_541: i16 = 4433;

const F_1_175_MINUS_1_961: i16 = (9633 - 16069) as i16;
const F_1_175_MINUS_0_390: i16 = (9633 - 3196) as i16;
const F_1_175: i16 = 9633;

const F_0_298_MINUS_0_899: i16 = (2446 - 7373) as i16;
const F_NEG_0_899: i16 = -7373;
const F_2_053_MINUS_2_562: i16 = (16819 - 20995) as i16;
const F_NEG_2_562: i16 = -20995;

const F_3_072_MINUS_2_562: i16 = (25172 - 20995) as i16;
const F_1_501_MINUS_0_899: i16 = (12299 - 7373) as i16;

fn main() {
    // Random test pattern
    let mut samples = [0i16; DCTSIZE2];
    let mut seed = 12345u32;
    for i in 0..DCTSIZE2 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        samples[i] = ((seed >> 16) % 256) as i16 - 128;
    }

    println!("Input samples:");
    print_block(&samples);

    // Run scalar DCT for reference
    let mut scalar_coeffs = [0i16; DCTSIZE2];
    scalar_dct(&samples, &mut scalar_coeffs);
    println!("\nScalar DCT output:");
    print_block(&scalar_coeffs);

    // Run full AVX2 DCT with tracing
    unsafe {
        println!("\n=== AVX2 DCT with tracing ===");
        full_dct_traced(&samples);
    }
}

fn print_block(data: &[i16; DCTSIZE2]) {
    for row in 0..DCTSIZE {
        print!("  ");
        for col in 0..DCTSIZE {
            print!("{:5} ", data[row * DCTSIZE + col]);
        }
        println!();
    }
}

unsafe fn print_ymm_i16(name: &str, v: __m256i) {
    let mut arr = [0i16; 16];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v);
    println!(
        "{}: [{:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} | {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5}]",
        name,
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
        arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]
    );
}

#[target_feature(enable = "avx2")]
unsafe fn full_dct_traced(samples: &[i16; DCTSIZE2]) {
    // Load
    let ymm4 = _mm256_loadu_si256(samples.as_ptr().add(0) as *const __m256i);
    let ymm5 = _mm256_loadu_si256(samples.as_ptr().add(16) as *const __m256i);
    let ymm6 = _mm256_loadu_si256(samples.as_ptr().add(32) as *const __m256i);
    let ymm7 = _mm256_loadu_si256(samples.as_ptr().add(48) as *const __m256i);

    // Reorganize to (row0|row4), (row1|row5), (row2|row6), (row3|row7)
    let ymm0 = _mm256_permute2x128_si256(ymm4, ymm6, 0x20);
    let ymm1 = _mm256_permute2x128_si256(ymm4, ymm6, 0x31);
    let ymm2 = _mm256_permute2x128_si256(ymm5, ymm7, 0x20);
    let ymm3 = _mm256_permute2x128_si256(ymm5, ymm7, 0x31);

    println!("\n--- Pass 1: Input (row pairs) ---");
    print_ymm_i16("ymm0 (row0|row4)", ymm0);
    print_ymm_i16("ymm1 (row1|row5)", ymm1);
    print_ymm_i16("ymm2 (row2|row6)", ymm2);
    print_ymm_i16("ymm3 (row3|row7)", ymm3);

    // Transpose
    let (ymm0, ymm1, ymm2, ymm3) = dotranspose(ymm0, ymm1, ymm2, ymm3);
    println!("\n--- After transpose ---");
    print_ymm_i16("ymm0 (col1|col0)", ymm0);
    print_ymm_i16("ymm1 (col3|col2)", ymm1);
    print_ymm_i16("ymm2 (col4|col5)", ymm2);
    print_ymm_i16("ymm3 (col6|col7)", ymm3);

    // DCT pass 1
    let (ymm0, ymm1, ymm2, ymm3) = dodct(ymm0, ymm1, ymm2, ymm3, true);
    println!("\n--- After DCT pass 1 ---");
    print_ymm_i16("ymm0 (data0_4)", ymm0);
    print_ymm_i16("ymm1 (data3_1)", ymm1);
    print_ymm_i16("ymm2 (data2_6)", ymm2);
    print_ymm_i16("ymm3 (data7_5)", ymm3);

    // Reorganize for pass 2
    let ymm4 = _mm256_permute2x128_si256(ymm1, ymm3, 0x20); // data3_7
    let ymm1 = _mm256_permute2x128_si256(ymm1, ymm3, 0x31); // data1_5
    println!("\n--- Reorganized for pass 2 ---");
    print_ymm_i16("ymm0 (data0_4)", ymm0);
    print_ymm_i16("ymm1 (data1_5)", ymm1);
    print_ymm_i16("ymm2 (data2_6)", ymm2);
    print_ymm_i16("ymm4 (data3_7)", ymm4);

    // Transpose
    let (ymm0, ymm1, ymm2, ymm4) = dotranspose(ymm0, ymm1, ymm2, ymm4);
    println!("\n--- After transpose (for pass 2) ---");
    print_ymm_i16("ymm0", ymm0);
    print_ymm_i16("ymm1", ymm1);
    print_ymm_i16("ymm2", ymm2);
    print_ymm_i16("ymm4", ymm4);

    // DCT pass 2
    let (ymm0, ymm1, ymm2, ymm4) = dodct(ymm0, ymm1, ymm2, ymm4, false);
    println!("\n--- After DCT pass 2 ---");
    print_ymm_i16("ymm0 (data0_4)", ymm0);
    print_ymm_i16("ymm1 (data3_1)", ymm1);
    print_ymm_i16("ymm2 (data2_6)", ymm2);
    print_ymm_i16("ymm4 (data7_5)", ymm4);

    // Final output permutation
    let out01 = _mm256_permute2x128_si256(ymm0, ymm1, 0x30);
    let out23 = _mm256_permute2x128_si256(ymm2, ymm1, 0x20);
    let out45 = _mm256_permute2x128_si256(ymm0, ymm4, 0x31);
    let out67 = _mm256_permute2x128_si256(ymm2, ymm4, 0x21);

    println!("\n--- Final output ---");
    print_ymm_i16("out01 (row0|row1)", out01);
    print_ymm_i16("out23 (row2|row3)", out23);
    print_ymm_i16("out45 (row4|row5)", out45);
    print_ymm_i16("out67 (row6|row7)", out67);

    // Store and print as block
    let mut coeffs = [0i16; DCTSIZE2];
    _mm256_storeu_si256(coeffs.as_mut_ptr().add(0) as *mut __m256i, out01);
    _mm256_storeu_si256(coeffs.as_mut_ptr().add(16) as *mut __m256i, out23);
    _mm256_storeu_si256(coeffs.as_mut_ptr().add(32) as *mut __m256i, out45);
    _mm256_storeu_si256(coeffs.as_mut_ptr().add(48) as *mut __m256i, out67);
    println!("\nAVX2 output as block:");
    print_block(&coeffs);
}

#[target_feature(enable = "avx2")]
unsafe fn dotranspose(
    ymm0: __m256i,
    ymm1: __m256i,
    ymm2: __m256i,
    ymm3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let ymm4 = _mm256_unpacklo_epi16(ymm0, ymm1);
    let ymm5 = _mm256_unpackhi_epi16(ymm0, ymm1);
    let ymm6 = _mm256_unpacklo_epi16(ymm2, ymm3);
    let ymm7 = _mm256_unpackhi_epi16(ymm2, ymm3);

    let ymm0 = _mm256_unpacklo_epi32(ymm4, ymm6);
    let ymm1 = _mm256_unpackhi_epi32(ymm4, ymm6);
    let ymm2 = _mm256_unpacklo_epi32(ymm5, ymm7);
    let ymm3 = _mm256_unpackhi_epi32(ymm5, ymm7);

    let ymm0 = _mm256_permute4x64_epi64(ymm0, 0x8D);
    let ymm1 = _mm256_permute4x64_epi64(ymm1, 0x8D);
    let ymm2 = _mm256_permute4x64_epi64(ymm2, 0xD8);
    let ymm3 = _mm256_permute4x64_epi64(ymm3, 0xD8);

    (ymm0, ymm1, ymm2, ymm3)
}

#[target_feature(enable = "avx2")]
unsafe fn dodct(
    ymm0: __m256i,
    ymm1: __m256i,
    ymm2: __m256i,
    ymm3: __m256i,
    pass1: bool,
) -> (__m256i, __m256i, __m256i, __m256i) {
    // Step 1: butterflies
    let ymm4 = _mm256_sub_epi16(ymm0, ymm3); // tmp6_7
    let ymm5 = _mm256_add_epi16(ymm0, ymm3); // tmp1_0
    let ymm6 = _mm256_add_epi16(ymm1, ymm2); // tmp3_2
    let ymm7 = _mm256_sub_epi16(ymm1, ymm2); // tmp4_5

    // Even part
    let ymm5_perm = _mm256_permute2x128_si256(ymm5, ymm5, 0x01);
    let ymm0 = _mm256_add_epi16(ymm5_perm, ymm6); // tmp10_12 (NOT tmp10_11!)
    let ymm5 = _mm256_sub_epi16(ymm5_perm, ymm6); // tmp11_13 (NOT tmp13_12!)

    let ymm6 = _mm256_permute2x128_si256(ymm0, ymm0, 0x01); // tmp12_10

    let pw_1_neg1 = _mm256_set_epi16(
        -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
    );
    let ymm0_signed = _mm256_sign_epi16(ymm0, pw_1_neg1); // tmp10_neg12
    let ymm6 = _mm256_add_epi16(ymm6, ymm0_signed); // (tmp10+tmp12)_(tmp10-tmp12)

    let out0_4 = if pass1 {
        _mm256_slli_epi16(ymm6, PASS1_BITS as i32)
    } else {
        let pw_descale = _mm256_set1_epi16(1 << (PASS1_BITS - 1));
        _mm256_srai_epi16(_mm256_add_epi16(ymm6, pw_descale), PASS1_BITS as i32)
    };

    // data2_6 calculation
    let ymm6 = _mm256_permute2x128_si256(ymm5, ymm5, 0x01); // tmp13_11
    let ymm1 = _mm256_unpacklo_epi16(ymm5, ymm6);
    let ymm5 = _mm256_unpackhi_epi16(ymm5, ymm6);

    let pw_f130_f054 = _mm256_set_epi16(
        F_0_541, F_0_541_MINUS_1_847, F_0_541, F_0_541_MINUS_1_847,
        F_0_541, F_0_541_MINUS_1_847, F_0_541, F_0_541_MINUS_1_847,
        F_0_541, F_0_541_PLUS_0_765, F_0_541, F_0_541_PLUS_0_765,
        F_0_541, F_0_541_PLUS_0_765, F_0_541, F_0_541_PLUS_0_765,
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

    // Odd part
    let ymm6 = _mm256_add_epi16(ymm7, ymm4); // z3_4

    let ymm1 = _mm256_permute2x128_si256(ymm6, ymm6, 0x01); // z4_3
    let ymm5 = _mm256_unpacklo_epi16(ymm6, ymm1);
    let ymm6 = _mm256_unpackhi_epi16(ymm6, ymm1);

    let pw_mf078_f117 = _mm256_set_epi16(
        F_1_175, F_1_175_MINUS_0_390, F_1_175, F_1_175_MINUS_0_390,
        F_1_175, F_1_175_MINUS_0_390, F_1_175, F_1_175_MINUS_0_390,
        F_1_175, F_1_175_MINUS_1_961, F_1_175, F_1_175_MINUS_1_961,
        F_1_175, F_1_175_MINUS_1_961, F_1_175, F_1_175_MINUS_1_961,
    );

    let ymm5 = _mm256_madd_epi16(ymm5, pw_mf078_f117);
    let ymm6 = _mm256_madd_epi16(ymm6, pw_mf078_f117);

    // tmp4/tmp5 calculation
    let ymm3 = _mm256_permute2x128_si256(ymm4, ymm4, 0x01); // tmp7_6
    let ymm1 = _mm256_unpacklo_epi16(ymm7, ymm3);
    let ymm3 = _mm256_unpackhi_epi16(ymm7, ymm3);

    let pw_mf060_mf089 = _mm256_set_epi16(
        F_NEG_2_562, F_2_053_MINUS_2_562, F_NEG_2_562, F_2_053_MINUS_2_562,
        F_NEG_2_562, F_2_053_MINUS_2_562, F_NEG_2_562, F_2_053_MINUS_2_562,
        F_NEG_0_899, F_0_298_MINUS_0_899, F_NEG_0_899, F_0_298_MINUS_0_899,
        F_NEG_0_899, F_0_298_MINUS_0_899, F_NEG_0_899, F_0_298_MINUS_0_899,
    );

    let ymm1 = _mm256_madd_epi16(ymm1, pw_mf060_mf089);
    let ymm3 = _mm256_madd_epi16(ymm3, pw_mf060_mf089);

    let ymm1 = _mm256_add_epi32(ymm1, ymm5);
    let ymm3 = _mm256_add_epi32(ymm3, ymm6);

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

    let pw_f050_mf256 = _mm256_set_epi16(
        F_NEG_0_899, F_1_501_MINUS_0_899, F_NEG_0_899, F_1_501_MINUS_0_899,
        F_NEG_0_899, F_1_501_MINUS_0_899, F_NEG_0_899, F_1_501_MINUS_0_899,
        F_NEG_2_562, F_3_072_MINUS_2_562, F_NEG_2_562, F_3_072_MINUS_2_562,
        F_NEG_2_562, F_3_072_MINUS_2_562, F_NEG_2_562, F_3_072_MINUS_2_562,
    );

    let ymm7 = _mm256_madd_epi16(ymm7, pw_f050_mf256);
    let ymm4 = _mm256_madd_epi16(ymm4, pw_f050_mf256);

    let ymm7 = _mm256_add_epi32(ymm7, ymm5);
    let ymm4 = _mm256_add_epi32(ymm4, ymm6);

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

    (out0_4, out3_1, out2_6, out7_5)
}

fn scalar_dct(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
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

    fn descale(x: i32, n: i32) -> i32 {
        (x + (1 << (n - 1))) >> n
    }

    let mut workspace = [0i32; DCTSIZE2];

    // Pass 1: process rows
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;
        let d0 = samples[base] as i32;
        let d1 = samples[base + 1] as i32;
        let d2 = samples[base + 2] as i32;
        let d3 = samples[base + 3] as i32;
        let d4 = samples[base + 4] as i32;
        let d5 = samples[base + 5] as i32;
        let d6 = samples[base + 6] as i32;
        let d7 = samples[base + 7] as i32;

        let tmp0 = d0 + d7;
        let tmp1 = d1 + d6;
        let tmp2 = d2 + d5;
        let tmp3 = d3 + d4;
        let tmp4 = d3 - d4;
        let tmp5 = d2 - d5;
        let tmp6 = d1 - d6;
        let tmp7 = d0 - d7;

        let tmp10 = tmp0 + tmp3;
        let _tmp11 = tmp0 - tmp3;
        let tmp12 = tmp1 + tmp2;
        let tmp13 = tmp1 - tmp2;

        workspace[base] = (tmp10 + tmp12) << PASS1_BITS;
        workspace[base + 4] = (tmp10 - tmp12) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        workspace[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        workspace[base + 6] = descale(z1 - tmp12 * FIX_1_847759065, CONST_BITS - PASS1_BITS);

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = -z1 * FIX_0_899976223;
        let z2 = -z2 * FIX_2_562915447;
        let z3 = -z3 * FIX_1_961570560 + z5;
        let z4 = -z4 * FIX_0_390180644 + z5;

        workspace[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        workspace[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        workspace[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        workspace[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    for col in 0..DCTSIZE {
        let d0 = workspace[col];
        let d1 = workspace[col + DCTSIZE];
        let d2 = workspace[col + 2 * DCTSIZE];
        let d3 = workspace[col + 3 * DCTSIZE];
        let d4 = workspace[col + 4 * DCTSIZE];
        let d5 = workspace[col + 5 * DCTSIZE];
        let d6 = workspace[col + 6 * DCTSIZE];
        let d7 = workspace[col + 7 * DCTSIZE];

        let tmp0 = d0 + d7;
        let tmp1 = d1 + d6;
        let tmp2 = d2 + d5;
        let tmp3 = d3 + d4;
        let tmp4 = d3 - d4;
        let tmp5 = d2 - d5;
        let tmp6 = d1 - d6;
        let tmp7 = d0 - d7;

        let tmp10 = tmp0 + tmp3;
        let _tmp11 = tmp0 - tmp3;
        let tmp12 = tmp1 + tmp2;
        let tmp13 = tmp1 - tmp2;

        coeffs[col] = descale(tmp10 + tmp12, PASS1_BITS) as i16;
        coeffs[col + 4 * DCTSIZE] = descale(tmp10 - tmp12, PASS1_BITS) as i16;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        coeffs[col + 2 * DCTSIZE] =
            descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS) as i16;
        coeffs[col + 6 * DCTSIZE] =
            descale(z1 - tmp12 * FIX_1_847759065, CONST_BITS + PASS1_BITS) as i16;

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = -z1 * FIX_0_899976223;
        let z2 = -z2 * FIX_2_562915447;
        let z3 = -z3 * FIX_1_961570560 + z5;
        let z4 = -z4 * FIX_0_390180644 + z5;

        coeffs[col + 7 * DCTSIZE] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS) as i16;
        coeffs[col + 5 * DCTSIZE] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS) as i16;
        coeffs[col + 3 * DCTSIZE] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS) as i16;
        coeffs[col + 1 * DCTSIZE] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS) as i16;
    }
}
