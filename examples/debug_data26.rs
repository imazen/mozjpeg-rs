//! Debug data2_6 computation specifically

use std::arch::x86_64::*;

const DCTSIZE: usize = 8;
const DCTSIZE2: usize = 64;
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Fixed-point constants
const F_0_541_PLUS_0_765: i16 = (4433 + 6270) as i16; // 10703
const F_0_541_MINUS_1_847: i16 = (4433 - 15137) as i16; // -10704
const F_0_541: i16 = 4433;

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

    unsafe {
        debug_data26_computation(&samples);
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

unsafe fn print_ymm_i32(name: &str, v: __m256i) {
    let mut arr = [0i32; 8];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v);
    println!(
        "{}: [{:8} {:8} {:8} {:8} | {:8} {:8} {:8} {:8}]",
        name, arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]
    );
}

#[target_feature(enable = "avx2")]
unsafe fn debug_data26_computation(samples: &[i16; DCTSIZE2]) {
    println!("\n=== PASS 1: After loading and reorganizing ===");

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

    print_ymm_i16("ymm0 (row0|row4)", ymm0);
    print_ymm_i16("ymm1 (row1|row5)", ymm1);
    print_ymm_i16("ymm2 (row2|row6)", ymm2);
    print_ymm_i16("ymm3 (row3|row7)", ymm3);

    // Transpose
    let (ymm0, ymm1, ymm2, ymm3) = dotranspose(ymm0, ymm1, ymm2, ymm3);
    println!("\n=== After transpose ===");
    print_ymm_i16("ymm0 (col1|col0)", ymm0);
    print_ymm_i16("ymm1 (col3|col2)", ymm1);
    print_ymm_i16("ymm2 (col4|col5)", ymm2);
    print_ymm_i16("ymm3 (col6|col7)", ymm3);

    // DCT - just the even part up to data2_6
    println!("\n=== DCT Pass 1: Even part ===");

    // Step 1: butterflies
    let ymm4 = _mm256_sub_epi16(ymm0, ymm3); // tmp6_7
    let ymm5 = _mm256_add_epi16(ymm0, ymm3); // tmp1_0
    let ymm6 = _mm256_add_epi16(ymm1, ymm2); // tmp3_2
    let ymm7 = _mm256_sub_epi16(ymm1, ymm2); // tmp4_5

    print_ymm_i16("ymm4 (tmp6_7)", ymm4);
    print_ymm_i16("ymm5 (tmp1_0)", ymm5);
    print_ymm_i16("ymm6 (tmp3_2)", ymm6);
    print_ymm_i16("ymm7 (tmp4_5)", ymm7);

    // Even part computations
    let ymm5_perm = _mm256_permute2x128_si256(ymm5, ymm5, 0x01); // tmp0_1
    print_ymm_i16("ymm5_perm (tmp0_1)", ymm5_perm);

    let tmp10_11 = _mm256_add_epi16(ymm5_perm, ymm6);
    let tmp13_12 = _mm256_sub_epi16(ymm5_perm, ymm6);
    print_ymm_i16("tmp10_11", tmp10_11);
    print_ymm_i16("tmp13_12", tmp13_12);

    // data2_6 computation
    println!("\n=== data2_6 computation ===");
    let tmp12_13 = _mm256_permute2x128_si256(tmp13_12, tmp13_12, 0x01);
    print_ymm_i16("tmp12_13 (permuted)", tmp12_13);

    let interleaved_lo = _mm256_unpacklo_epi16(tmp13_12, tmp12_13);
    let interleaved_hi = _mm256_unpackhi_epi16(tmp13_12, tmp12_13);
    print_ymm_i16("interleaved_lo", interleaved_lo);
    print_ymm_i16("interleaved_hi", interleaved_hi);

    // Constant vector
    let pw_f130_f054 = _mm256_set_epi16(
        F_0_541, F_0_541_MINUS_1_847, F_0_541, F_0_541_MINUS_1_847,
        F_0_541, F_0_541_MINUS_1_847, F_0_541, F_0_541_MINUS_1_847,
        F_0_541, F_0_541_PLUS_0_765, F_0_541, F_0_541_PLUS_0_765,
        F_0_541, F_0_541_PLUS_0_765, F_0_541, F_0_541_PLUS_0_765,
    );
    print_ymm_i16("pw_f130_f054", pw_f130_f054);

    let madd_lo = _mm256_madd_epi16(interleaved_lo, pw_f130_f054);
    let madd_hi = _mm256_madd_epi16(interleaved_hi, pw_f130_f054);
    print_ymm_i32("madd_lo (data2_6L)", madd_lo);
    print_ymm_i32("madd_hi (data2_6H)", madd_hi);

    // Descale for pass 1
    let pd_descale = _mm256_set1_epi32(1 << (CONST_BITS - PASS1_BITS - 1));
    println!("pd_descale = {}", 1 << (CONST_BITS - PASS1_BITS - 1));

    let scaled_lo = _mm256_add_epi32(madd_lo, pd_descale);
    let scaled_hi = _mm256_add_epi32(madd_hi, pd_descale);
    print_ymm_i32("scaled_lo", scaled_lo);
    print_ymm_i32("scaled_hi", scaled_hi);

    const N: i32 = CONST_BITS - PASS1_BITS;
    let shifted_lo = _mm256_srai_epi32::<N>(scaled_lo);
    let shifted_hi = _mm256_srai_epi32::<N>(scaled_hi);
    print_ymm_i32("shifted_lo", shifted_lo);
    print_ymm_i32("shifted_hi", shifted_hi);

    let out2_6 = _mm256_packs_epi32(shifted_lo, shifted_hi);
    print_ymm_i16("out2_6 (packed)", out2_6);

    // Scalar verification for first column
    println!("\n=== Scalar verification (column 0) ===");
    let tmp13_col0 = tmp13_12.as_i16x16().as_array()[0] as i32;
    let tmp12_col0 = tmp13_12.as_i16x16().as_array()[8] as i32; // high lane
    println!("tmp13[0]={}, tmp12[0]={}", tmp13_col0, tmp12_col0);
    let z1 = (tmp12_col0 + tmp13_col0) * 4433; // FIX_0_541196100
    let data2 = (z1 + tmp13_col0 * 6270 + 1024) >> 11; // + FIX_0_765366865, descale
    let data6 = (z1 - tmp12_col0 * 15137 + 1024) >> 11; // - FIX_1_847759065, descale
    println!("Scalar: data2={}, data6={}", data2, data6);
    // This is getting complex - let me just verify the madd is working
    println!("\nManual verification of first madd element:");
    let a0 = interleaved_lo.as_i16x16().as_array()[0] as i32;
    let a1 = interleaved_lo.as_i16x16().as_array()[1] as i32;
    let b0 = pw_f130_f054.as_i16x16().as_array()[0] as i32;
    let b1 = pw_f130_f054.as_i16x16().as_array()[1] as i32;
    println!("  a0={}, a1={}, b0={}, b1={}", a0, a1, b0, b1);
    println!("  a0*b0 + a1*b1 = {} * {} + {} * {} = {}", a0, b0, a1, b1, a0 * b0 + a1 * b1);
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

// Helper trait to access array
trait AsArray<T, const N: usize> {
    fn as_array(&self) -> &[T; N];
}

impl AsArray<i16, 16> for std::arch::x86_64::__m256i {
    fn as_array(&self) -> &[i16; 16] {
        unsafe { &*(self as *const __m256i as *const [i16; 16]) }
    }
}

trait AsI16x16 {
    fn as_i16x16(&self) -> &__m256i;
}

impl AsI16x16 for __m256i {
    fn as_i16x16(&self) -> &__m256i {
        self
    }
}
