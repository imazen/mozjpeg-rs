//! Trace through the i16 transpose to verify correctness

use std::arch::x86_64::*;

fn main() {
    // Create a simple numbered pattern so we can trace the data flow
    // Each element is row*10 + col (e.g., 23 = row 2, col 3)
    let mut samples = [0i16; 64];
    for row in 0..8 {
        for col in 0..8 {
            samples[row * 8 + col] = (row * 10 + col) as i16;
        }
    }

    println!("Input (row*10 + col):");
    print_block(&samples);

    unsafe {
        test_transpose_detailed(&samples);
    }
}

fn print_block(data: &[i16; 64]) {
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            print!("{:3} ", data[row * 8 + col]);
        }
        println!();
    }
}

unsafe fn print_ymm(name: &str, v: __m256i) {
    let mut arr = [0i16; 16];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v);
    println!(
        "{}: [{:3} {:3} {:3} {:3} {:3} {:3} {:3} {:3} | {:3} {:3} {:3} {:3} {:3} {:3} {:3} {:3}]",
        name,
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
        arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]
    );
}

#[target_feature(enable = "avx2")]
unsafe fn test_transpose_detailed(samples: &[i16; 64]) {
    println!("\n=== Loading data ===");

    // Load data
    let ymm4 = _mm256_loadu_si256(samples.as_ptr().add(0) as *const __m256i); // rows 0-1
    let ymm5 = _mm256_loadu_si256(samples.as_ptr().add(16) as *const __m256i); // rows 2-3
    let ymm6 = _mm256_loadu_si256(samples.as_ptr().add(32) as *const __m256i); // rows 4-5
    let ymm7 = _mm256_loadu_si256(samples.as_ptr().add(48) as *const __m256i); // rows 6-7

    print_ymm("ymm4 (rows 0,1)", ymm4);
    print_ymm("ymm5 (rows 2,3)", ymm5);
    print_ymm("ymm6 (rows 4,5)", ymm6);
    print_ymm("ymm7 (rows 6,7)", ymm7);

    // Reorganize to (row0|row4), (row1|row5), (row2|row6), (row3|row7)
    // This matches mozjpeg's expected layout
    let ymm0 = _mm256_permute2x128_si256(ymm4, ymm6, 0x20); // row0|row4
    let ymm1 = _mm256_permute2x128_si256(ymm4, ymm6, 0x31); // row1|row5
    let ymm2 = _mm256_permute2x128_si256(ymm5, ymm7, 0x20); // row2|row6
    let ymm3 = _mm256_permute2x128_si256(ymm5, ymm7, 0x31); // row3|row7

    println!("\n=== After permute (row pairs) ===");
    print_ymm("ymm0 (row0|row4)", ymm0);
    print_ymm("ymm1 (row1|row5)", ymm1);
    print_ymm("ymm2 (row2|row6)", ymm2);
    print_ymm("ymm3 (row3|row7)", ymm3);

    // Now transpose - this should give us columns
    println!("\n=== Transpose phase 1: unpacklo/hi 16-bit ===");
    let t0 = _mm256_unpacklo_epi16(ymm0, ymm1);
    let t1 = _mm256_unpackhi_epi16(ymm0, ymm1);
    let t2 = _mm256_unpacklo_epi16(ymm2, ymm3);
    let t3 = _mm256_unpackhi_epi16(ymm2, ymm3);

    print_ymm("t0 = unpacklo(ymm0,ymm1)", t0);
    print_ymm("t1 = unpackhi(ymm0,ymm1)", t1);
    print_ymm("t2 = unpacklo(ymm2,ymm3)", t2);
    print_ymm("t3 = unpackhi(ymm2,ymm3)", t3);

    println!("\n=== Transpose phase 2: unpacklo/hi 32-bit ===");
    let u0 = _mm256_unpacklo_epi32(t0, t2);
    let u1 = _mm256_unpackhi_epi32(t0, t2);
    let u2 = _mm256_unpacklo_epi32(t1, t3);
    let u3 = _mm256_unpackhi_epi32(t1, t3);

    print_ymm("u0 = unpacklo(t0,t2)", u0);
    print_ymm("u1 = unpackhi(t0,t2)", u1);
    print_ymm("u2 = unpacklo(t1,t3)", u2);
    print_ymm("u3 = unpackhi(t1,t3)", u3);

    println!("\n=== Transpose phase 3: permute 64-bit ===");
    let out0 = _mm256_permute4x64_epi64(u0, 0x8D); // col1|col0
    let out1 = _mm256_permute4x64_epi64(u1, 0x8D); // col3|col2
    let out2 = _mm256_permute4x64_epi64(u2, 0xD8); // col4|col5
    let out3 = _mm256_permute4x64_epi64(u3, 0xD8); // col6|col7

    print_ymm("out0 (should be col1|col0)", out0);
    print_ymm("out1 (should be col3|col2)", out1);
    print_ymm("out2 (should be col4|col5)", out2);
    print_ymm("out3 (should be col6|col7)", out3);

    println!("\n=== Expected (from mozjpeg comments) ===");
    println!("out0 should be: [01 11 21 31 41 51 61 71 | 00 10 20 30 40 50 60 70]");
    println!("out1 should be: [03 13 23 33 43 53 63 73 | 02 12 22 32 42 52 62 72]");
    println!("out2 should be: [04 14 24 34 44 54 64 74 | 05 15 25 35 45 55 65 75]");
    println!("out3 should be: [06 16 26 36 46 56 66 76 | 07 17 27 37 47 57 67 77]");
}
