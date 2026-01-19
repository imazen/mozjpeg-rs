//! Debug DCT by tracing intermediate values

use std::arch::x86_64::*;

const DCTSIZE: usize = 8;
const DCTSIZE2: usize = 64;

fn main() {
    // Random test pattern (pseudo-random to be reproducible)
    let mut samples = [0i16; DCTSIZE2];
    let mut seed = 12345u32;
    for i in 0..DCTSIZE2 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        samples[i] = ((seed >> 16) % 256) as i16 - 128;
    }

    println!("Input samples:");
    print_block(&samples);

    // Compute scalar DCT
    let mut scalar_coeffs = [0i16; DCTSIZE2];
    scalar_dct(&samples, &mut scalar_coeffs);
    println!("\nScalar DCT output:");
    print_block(&scalar_coeffs);

    // Compute AVX2 i16 DCT
    let mut avx2_coeffs = [0i16; DCTSIZE2];
    unsafe {
        avx2_i16_dct_debug(&samples, &mut avx2_coeffs);
    }
    println!("\nAVX2 i16 DCT output:");
    print_block(&avx2_coeffs);

    // Compare
    println!("\nDifference (scalar - avx2):");
    let mut max_diff = 0i16;
    for i in 0..DCTSIZE2 {
        let diff = scalar_coeffs[i] - avx2_coeffs[i];
        if diff.abs() > max_diff {
            max_diff = diff.abs();
        }
    }
    println!("Max difference: {}", max_diff);

    // Show first few differences
    println!("\nFirst row comparison:");
    for i in 0..8 {
        println!(
            "  coeff[{}]: scalar={:5}, avx2={:5}, diff={:5}",
            i,
            scalar_coeffs[i],
            avx2_coeffs[i],
            scalar_coeffs[i] - avx2_coeffs[i]
        );
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

// Scalar DCT for reference
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

        // NOTE: libjpeg uses non-standard naming here!
        // tmp13 = tmp0 - tmp3 (textbooks call this tmp11)
        // tmp11 = tmp1 + tmp2 (textbooks call this tmp12)
        // tmp12 = tmp1 - tmp2 (textbooks call this tmp13)
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;  // libjpeg naming
        let tmp11 = tmp1 + tmp2;  // libjpeg naming
        let tmp12 = tmp1 - tmp2;  // libjpeg naming

        workspace[base] = (tmp10 + tmp11) << PASS1_BITS;
        workspace[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

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

        // libjpeg naming (see pass 1 comment)
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        coeffs[col] = descale(tmp10 + tmp11, PASS1_BITS) as i16;
        coeffs[col + 4 * DCTSIZE] = descale(tmp10 - tmp11, PASS1_BITS) as i16;

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

#[target_feature(enable = "avx2")]
unsafe fn avx2_i16_dct_debug(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    mozjpeg_rs::dct::avx2::forward_dct_8x8_avx2_i16(samples, coeffs);
}
