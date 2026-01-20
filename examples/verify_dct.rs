//! Verify old vs new AVX2 DCT produce identical output

fn main() {
    const DCTSIZE2: usize = 64;
    
    // Generate test data
    let mut samples = [0i16; DCTSIZE2];
    let mut seed = 12345u32;
    for i in 0..DCTSIZE2 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        samples[i] = ((seed >> 16) % 256) as i16 - 128;
    }

    println!("Input samples (first 8): {:?}", &samples[..8]);

    // New implementation (dct.rs with archmage)
    let mut coeffs_new = [0i16; DCTSIZE2];
    
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        use archmage::tokens::x86::Avx2Token;
        
        if let Some(token) = Avx2Token::try_new() {
            mozjpeg_rs::dct::avx2::forward_dct_8x8_avx2(token, &samples, &mut coeffs_new);
            println!("\nNew (archmage) DCT output (first 8): {:?}", &coeffs_new[..8]);
        } else {
            println!("AVX2 not available");
            return;
        }
    }

    // Old implementation (simd/x86_64/avx2.rs) - requires simd-intrinsics feature
    #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
    {
        let mut coeffs_old = [0i16; DCTSIZE2];
        mozjpeg_rs::simd::x86_64::avx2::forward_dct_8x8(&samples, &mut coeffs_old);
        println!("Old (simd module) DCT output (first 8): {:?}", &coeffs_old[..8]);
        
        // Compare
        let mut max_diff = 0i16;
        let mut diff_count = 0;
        for i in 0..DCTSIZE2 {
            let diff = (coeffs_new[i] - coeffs_old[i]).abs();
            if diff > 0 {
                diff_count += 1;
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        
        if diff_count == 0 {
            println!("\n✓ Outputs are IDENTICAL");
        } else {
            println!("\n✗ {} differences, max diff: {}", diff_count, max_diff);
        }
    }
    
    #[cfg(not(feature = "simd-intrinsics"))]
    {
        println!("\nNote: simd-intrinsics feature not enabled, can't compare with old impl");
        println!("Run with: cargo run --example verify_dct --features simd-intrinsics");
    }
}
