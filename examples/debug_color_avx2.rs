//! Debug tool for AVX2 color conversion
//!
//! Traces through the de-interleaving and color conversion step by step.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
fn print_m256i_u8(name: &str, v: __m256i) {
    let arr: [u8; 32] = unsafe { std::mem::transmute(v) };
    println!("{}: {:?}", name, arr);
}

#[cfg(target_arch = "x86_64")]
fn print_m256i_i16(name: &str, v: __m256i) {
    let arr: [i16; 16] = unsafe { std::mem::transmute(v) };
    println!("{}: {:?}", name, arr);
}

#[cfg(target_arch = "x86_64")]
fn print_m256i_i32(name: &str, v: __m256i) {
    let arr: [i32; 8] = unsafe { std::mem::transmute(v) };
    println!("{}: {:?}", name, arr);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn trace_deinterleave() {
    // Create test RGB data: 32 pixels = 96 bytes
    // Use simple pattern: pixel i has R=i*3, G=i*3+1, B=i*3+2
    let mut rgb = [0u8; 96];
    for i in 0..32 {
        rgb[i * 3] = (i * 3) as u8; // R
        rgb[i * 3 + 1] = (i * 3 + 1) as u8; // G
        rgb[i * 3 + 2] = (i * 3 + 2) as u8; // B
    }

    println!("Input RGB (first 12 values): {:?}", &rgb[0..12]);
    println!("Expected: R=[0,3,6,9,...], G=[1,4,7,10,...], B=[2,5,8,11,...]");
    println!();

    // Load 96 bytes (32 RGB pixels) using 3 YMM loads
    let rgb_ptr = rgb.as_ptr();
    let ymm_a = _mm256_loadu_si256(rgb_ptr.cast());
    let ymm_f = _mm256_loadu_si256(rgb_ptr.add(32).cast());
    let ymm_b = _mm256_loadu_si256(rgb_ptr.add(64).cast());

    println!("After loading:");
    print_m256i_u8("ymm_a (bytes 0-31)", ymm_a);
    print_m256i_u8("ymm_f (bytes 32-63)", ymm_f);
    print_m256i_u8("ymm_b (bytes 64-95)", ymm_b);
    println!();

    // Step 1: Reorganize across 128-bit lanes
    let ymm_c = ymm_a;
    let ymm_a = _mm256_inserti128_si256(ymm_f, _mm256_castsi256_si128(ymm_a), 0);
    let ymm_c = _mm256_inserti128_si256(ymm_c, _mm256_castsi256_si128(ymm_b), 0);
    let ymm_b = _mm256_inserti128_si256(ymm_b, _mm256_castsi256_si128(ymm_f), 0);
    let ymm_f = _mm256_permute2x128_si256(ymm_c, ymm_c, 0x01);

    println!("After Step 1 (lane reorganization):");
    print_m256i_u8("ymm_a", ymm_a);
    print_m256i_u8("ymm_b", ymm_b);
    print_m256i_u8("ymm_c", ymm_c);
    print_m256i_u8("ymm_f", ymm_f);
    println!();

    // Step 2: First level of interleaving
    let ymm_g = ymm_a;
    let ymm_a = _mm256_slli_si256(ymm_a, 8);
    let ymm_g = _mm256_srli_si256(ymm_g, 8);

    let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_f);
    let ymm_f = _mm256_slli_si256(ymm_f, 8);

    let ymm_g = _mm256_unpacklo_epi8(ymm_g, ymm_b);
    let ymm_f = _mm256_unpackhi_epi8(ymm_f, ymm_b);

    println!("After Step 2:");
    print_m256i_u8("ymm_a", ymm_a);
    print_m256i_u8("ymm_f", ymm_f);
    print_m256i_u8("ymm_g", ymm_g);
    println!();

    // Step 3: Second level of interleaving
    let ymm_d = ymm_a;
    let ymm_a = _mm256_slli_si256(ymm_a, 8);
    let ymm_d = _mm256_srli_si256(ymm_d, 8);

    let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_g);
    let ymm_g = _mm256_slli_si256(ymm_g, 8);

    let ymm_d = _mm256_unpacklo_epi8(ymm_d, ymm_f);
    let ymm_g = _mm256_unpackhi_epi8(ymm_g, ymm_f);

    println!("After Step 3:");
    print_m256i_u8("ymm_a", ymm_a);
    print_m256i_u8("ymm_d", ymm_d);
    print_m256i_u8("ymm_g", ymm_g);
    println!();

    // Step 4: Third level of interleaving
    let ymm_e = ymm_a;
    let ymm_a = _mm256_slli_si256(ymm_a, 8);
    let ymm_e = _mm256_srli_si256(ymm_e, 8);

    let ymm_a = _mm256_unpackhi_epi8(ymm_a, ymm_d);
    let ymm_d = _mm256_slli_si256(ymm_d, 8);

    let ymm_e = _mm256_unpacklo_epi8(ymm_e, ymm_g);
    let ymm_d = _mm256_unpackhi_epi8(ymm_d, ymm_g);

    println!("After Step 4:");
    print_m256i_u8("ymm_a", ymm_a);
    print_m256i_u8("ymm_d", ymm_d);
    print_m256i_u8("ymm_e", ymm_e);
    println!();

    // Step 5: Final unpacking - separate into R, G, B
    let ymm_h = _mm256_setzero_si256();

    let ymm_c = ymm_a;
    let ymm_a = _mm256_unpacklo_epi8(ymm_a, ymm_h);
    let ymm_c = _mm256_unpackhi_epi8(ymm_c, ymm_h);

    let ymm_b_tmp = ymm_e;
    let ymm_e = _mm256_unpacklo_epi8(ymm_e, ymm_h);
    let ymm_b_new = _mm256_unpackhi_epi8(ymm_b_tmp, ymm_h);

    let ymm_f_tmp = ymm_d;
    let ymm_d = _mm256_unpacklo_epi8(ymm_d, ymm_h);
    let ymm_f = _mm256_unpackhi_epi8(ymm_f_tmp, ymm_h);

    println!("After Step 5 (final unpack to i16):");
    println!("According to comments in color_avx2.rs:");
    println!("  ymm_a = R even, ymm_c = G even, ymm_e = B even");
    println!("  ymm_b_new = R odd, ymm_d = G odd, ymm_f = B odd");
    println!();

    print_m256i_i16("ymm_a (R even?)", ymm_a);
    print_m256i_i16("ymm_c (G even?)", ymm_c);
    print_m256i_i16("ymm_e (B even?)", ymm_e);
    print_m256i_i16("ymm_b_new (R odd?)", ymm_b_new);
    print_m256i_i16("ymm_d (G odd?)", ymm_d);
    print_m256i_i16("ymm_f (B odd?)", ymm_f);
    println!();

    // Verify: extract values and compare with expected
    let r_even: [i16; 16] = std::mem::transmute(ymm_a);
    let g_even: [i16; 16] = std::mem::transmute(ymm_c);
    let _b_even: [i16; 16] = std::mem::transmute(ymm_e);
    let r_odd: [i16; 16] = std::mem::transmute(ymm_b_new);
    let g_odd: [i16; 16] = std::mem::transmute(ymm_d);
    let b_odd: [i16; 16] = std::mem::transmute(ymm_f);

    println!("Verification:");
    println!("Expected R for even pixels: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]");
    println!("Got R even: {:?}", r_even);
    println!();
    println!("Expected G for even pixels: [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91]");
    println!("Got G even: {:?}", g_even);
    println!();
    println!(
        "Expected R for odd pixels: [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93]"
    );
    println!("Got R odd: {:?}", r_odd);
    println!();

    // Now trace the Cb calculation for odd pixels
    println!("=== Tracing Cb odd calculation ===");

    // Constants (from color_avx2.rs)
    const F_0_168: i32 = 11059;
    const F_0_331: i32 = 21709;
    const SCALEBITS: i32 = 16;

    // The code does:
    // let ymm_1 = _mm256_unpacklo_epi16(ymm_b_new, ymm_d);  // interleave R_odd with G_odd
    // let ymm_7 = _mm256_madd_epi16(ymm_7, pw_mf016_mf033); // ROL*-FIX(0.168) + GOL*-FIX(0.331)

    // pw_mf016_mf033 = pack_i16_pair(-F_0_168, -F_0_331)
    let pw_mf016_mf033 = _mm256_set1_epi32(pack_i16_pair(-F_0_168, -F_0_331));
    print_m256i_i16(
        "pw_mf016_mf033 (should be [-11059, -21709] repeated)",
        pw_mf016_mf033,
    );

    // Interleave R_odd with G_odd for low half
    let rg_interleaved = _mm256_unpacklo_epi16(ymm_b_new, ymm_d);
    print_m256i_i16("R_odd/G_odd interleaved (low)", rg_interleaved);

    // vpmaddwd: result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
    // So if rg_interleaved = [R0, G0, R1, G1, ...] and coefs = [-11059, -21709, ...]
    // Result = R0 * (-11059) + G0 * (-21709), R1 * (-11059) + G1 * (-21709), ...
    let madd_result = _mm256_madd_epi16(rg_interleaved, pw_mf016_mf033);
    print_m256i_i32("madd result (R*-0.168 + G*-0.331)", madd_result);

    // Expected for first odd pixel (pixel 1): R=3, G=4
    // R * (-11059) + G * (-21709) = 3 * (-11059) + 4 * (-21709) = -33177 + (-86836) = -120013
    println!(
        "Expected for pixel 1 (R=3, G=4): 3*(-11059) + 4*(-21709) = {}",
        3 * (-11059) + 4 * (-21709)
    );

    // Manual verification
    let r0 = r_odd[0] as i32;
    let g0 = g_odd[0] as i32;
    let b0 = b_odd[0] as i32;
    println!("Pixel 1 odd: R={}, G={}, B={}", r0, g0, b0);

    // Full Cb calculation
    const FIX_0_16874: i32 = 11059;
    const FIX_0_33126: i32 = 21709;
    const FIX_0_50000: i32 = 32768;
    const ONE_HALF: i32 = 1 << 15;

    let cb_scalar =
        ((-FIX_0_16874 * r0 - FIX_0_33126 * g0 + FIX_0_50000 * b0 + ONE_HALF) >> SCALEBITS) + 128;
    println!("Scalar Cb for pixel 1: {}", cb_scalar);

    // Now let's check the AVX2 path more carefully
    // The AVX2 code adds B*0.5 separately, so let's trace that

    // B contribution: unpack B with zero, shift right by 1 (divide by 2)
    let zero = _mm256_setzero_si256();
    let b_shifted_lo = _mm256_unpacklo_epi16(zero, ymm_f); // [0, B0, 0, B1, ...]
    let b_shifted_lo = _mm256_srli_epi32(b_shifted_lo, 1); // B * 0.5 in fixed point? No, this is wrong!
    print_m256i_i32("B shifted (unpack with zero, shift right 1)", b_shifted_lo);

    // Wait - the code unpacks [0, B] which puts B in high 16 bits of each i32
    // Then shifts right by 1, which is B * 32768 (since B is in high 16 bits = B << 16)
    // After shift: B << 15 = B * 32768 = B * FIX(0.5)
    println!("B0 * FIX(0.5) should be: {} * 32768 = {}", b0, b0 * 32768);

    // Full Cb AVX2 calculation:
    // (R * -11059 + G * -21709) + (B * 32768) + (1 << 15 - 1 + 128 << 16) >> 16
    let pd_onehalfm1_cj = (1 << (SCALEBITS - 1)) - 1 + (128 << SCALEBITS);
    println!("pd_onehalfm1_cj = {}", pd_onehalfm1_cj);

    let cb_avx2_manual = ((r0 * (-11059) + g0 * (-21709)) + (b0 << 15) + pd_onehalfm1_cj) >> 16;
    println!("AVX2 Cb manual calculation: {}", cb_avx2_manual);
    println!();

    // AH! The scalar code uses (shift >> 16) + 128, but AVX2 uses (value + 128<<16) >> 16
    // These should be equivalent, but let's verify
    println!("Scalar formula: ((-11059*R - 21709*G + 32768*B + 32768) >> 16) + 128");
    println!("AVX2 formula: ((-11059*R - 21709*G + B<<15) + pd_onehalfm1_cj) >> 16");
    println!(
        "pd_onehalfm1_cj = (1<<15) - 1 + (128<<16) = 32767 + 8388608 = {}",
        pd_onehalfm1_cj
    );

    // Scalar: (-33177 - 86836 + 163840 + 32768) >> 16 + 128
    //       = 76595 >> 16 + 128 = 1 + 128 = 129
    let scalar_inner = -FIX_0_16874 * r0 - FIX_0_33126 * g0 + FIX_0_50000 * b0 + ONE_HALF;
    println!("Scalar inner value: {}", scalar_inner);
    println!("Scalar >> 16: {}", scalar_inner >> 16);
    println!("Scalar final: {}", (scalar_inner >> 16) + 128);

    // AVX2: (-33177 - 86836 + 81920 + 8421375) >> 16
    //     = 8382282 >> 16 = 127
    let avx2_rg = r0 * (-11059) + g0 * (-21709);
    let avx2_b = b0 << 15;
    let avx2_inner = avx2_rg + avx2_b + pd_onehalfm1_cj;
    println!("AVX2 R*coef + G*coef: {}", avx2_rg);
    println!("AVX2 B<<15: {}", avx2_b);
    println!("AVX2 inner value: {}", avx2_inner);
    println!("AVX2 >> 16: {}", avx2_inner >> 16);
}

const fn pack_i16_pair(lo: i32, hi: i32) -> i32 {
    ((hi as i16 as u16 as i32) << 16) | (lo as i16 as u16 as i32)
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not available");
        return;
    }

    unsafe {
        trace_deinterleave();
    }
}
