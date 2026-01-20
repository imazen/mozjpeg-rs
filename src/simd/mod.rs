//! SIMD-optimized implementations for performance-critical operations.
//!
//! This module provides runtime dispatch between scalar and SIMD implementations
//! based on detected CPU capabilities. The design follows these principles:
//!
//! 1. **Scalar reference**: All operations have a scalar implementation for
//!    correctness testing and fallback on unsupported platforms.
//!
//! 2. **Platform-specific optimizations**: x86_64 (AVX2/SSE2), aarch64 (NEON)
//!    implementations live in separate submodules.
//!
//! 3. **Runtime dispatch**: CPU capabilities are detected once at initialization,
//!    and function pointers are set accordingly (no per-call overhead).
//!
//! 4. **Zero-cost when unused**: Platform-specific code is only compiled for
//!    the target architecture.
//!
//! # Architecture
//!
//! ```text
//! simd/
//! ├── mod.rs         # This file - public API and dispatch
//! ├── scalar.rs      # Reference implementations
//! └── x86_64/
//!     ├── mod.rs     # x86_64 dispatch
//!     └── avx2.rs    # AVX2 implementations
//! ```

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use crate::consts::DCTSIZE2;

#[cfg(target_arch = "x86_64")]
use archmage::{tokens::x86::Avx2Token, SimdToken};

// ============================================================================
// Fast YUV color conversion using the yuv crate (when feature enabled)
// ============================================================================

/// Convert RGB to YCbCr using the yuv crate's SIMD-optimized implementation.
/// This is ~10x faster than the scalar multiversion implementation.
#[cfg(feature = "fast-yuv")]
fn convert_rgb_to_ycbcr_yuv(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    num_pixels: usize,
) {
    use yuv::{
        rgb_to_yuv444, BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange,
        YuvStandardMatrix,
    };

    // Treat as 1D image (width=num_pixels, height=1)
    let w = num_pixels as u32;
    let h = 1u32;

    let mut yuv_image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(y_out),
        y_stride: w,
        u_plane: BufferStoreMut::Borrowed(cb_out),
        u_stride: w,
        v_plane: BufferStoreMut::Borrowed(cr_out),
        v_stride: w,
        width: w,
        height: h,
    };

    rgb_to_yuv444(
        &mut yuv_image,
        rgb,
        w * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::default(),
    )
    .expect("yuv conversion failed");
}

/// Function pointer type for forward DCT.
///
/// Signature: (samples, coeffs) where:
/// - samples: 64 level-shifted i16 values (-128 to 127)
/// - coeffs: output 64 DCT coefficients
pub type ForwardDctFn = fn(&[i16; DCTSIZE2], &mut [i16; DCTSIZE2]);

/// Function pointer type for RGB to YCbCr conversion.
///
/// Signature: (rgb, y_out, cb_out, cr_out, num_pixels) where:
/// - rgb: interleaved RGB data (3 bytes per pixel)
/// - y_out, cb_out, cr_out: output component planes
/// - num_pixels: total number of pixels to convert
pub type ColorConvertFn = fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize);

/// DCT implementation variant for dispatch.
#[derive(Clone, Copy, Debug)]
enum DctVariant {
    /// Scalar with multiversion autovectorization
    Multiversion,
    /// Hand-written AVX2 intrinsics (simd-intrinsics feature)
    #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
    Avx2Intrinsics,
    /// Archmage-based safe AVX2 with cached token
    #[cfg(target_arch = "x86_64")]
    Avx2Archmage,
}

/// SIMD operations dispatch table.
///
/// This struct holds function pointers to the best available implementations
/// for the current CPU. Create once at startup and reuse.
#[derive(Clone, Copy)]
pub struct SimdOps {
    /// Forward DCT function pointer (legacy, kept for compatibility)
    pub forward_dct: ForwardDctFn,
    /// RGB to YCbCr conversion function
    pub color_convert_rgb_to_ycbcr: ColorConvertFn,
    /// DCT implementation variant
    dct_variant: DctVariant,
    /// Cached AVX2 token for archmage-based DCT (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    avx2_token: Option<Avx2Token>,
}

impl SimdOps {
    /// Select the best available implementations for the current CPU.
    ///
    /// Priority order:
    /// 1. `fast-yuv` feature: Uses the `yuv` crate for color conversion (~10x faster)
    /// 2. Archmage AVX2: Safe SIMD with cached capability token (default on x86_64 with AVX2)
    /// 3. `simd-intrinsics` feature: Hand-written AVX2 intrinsics
    /// 4. Default: `multiversion` autovectorization (safe, ~87% of intrinsics perf)
    #[must_use]
    pub fn detect() -> Self {
        // Color conversion: prefer yuv crate (fastest), then intrinsics, then scalar
        #[cfg(feature = "fast-yuv")]
        let color_fn: ColorConvertFn = convert_rgb_to_ycbcr_yuv;

        #[cfg(all(
            not(feature = "fast-yuv"),
            target_arch = "x86_64",
            feature = "simd-intrinsics"
        ))]
        let color_fn: ColorConvertFn = if is_x86_feature_detected!("avx2") {
            x86_64::avx2::convert_rgb_to_ycbcr
        } else {
            scalar::convert_rgb_to_ycbcr
        };

        #[cfg(all(
            not(feature = "fast-yuv"),
            not(all(target_arch = "x86_64", feature = "simd-intrinsics"))
        ))]
        let color_fn: ColorConvertFn = scalar::convert_rgb_to_ycbcr;

        // DCT: Try archmage first (cached token), then intrinsics, then multiversion
        #[cfg(target_arch = "x86_64")]
        let (dct_fn, dct_variant, avx2_token) = if let Some(token) = Avx2Token::try_new() {
            // Archmage with cached token - use multiversion as the function pointer
            // but actual dispatch will use the token-based method
            (
                scalar::forward_dct_8x8 as ForwardDctFn,
                DctVariant::Avx2Archmage,
                Some(token),
            )
        } else {
            (
                scalar::forward_dct_8x8 as ForwardDctFn,
                DctVariant::Multiversion,
                None,
            )
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (dct_fn, dct_variant) = (
            scalar::forward_dct_8x8 as ForwardDctFn,
            DctVariant::Multiversion,
        );

        Self {
            forward_dct: dct_fn,
            color_convert_rgb_to_ycbcr: color_fn,
            dct_variant,
            #[cfg(target_arch = "x86_64")]
            avx2_token,
        }
    }

    /// Perform forward DCT using the best available implementation.
    ///
    /// This method uses the cached AVX2 token when available, avoiding
    /// per-call capability checks.
    #[inline]
    pub fn do_forward_dct(&self, samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
        match self.dct_variant {
            #[cfg(target_arch = "x86_64")]
            DctVariant::Avx2Archmage => {
                // Use archmage with cached token
                if let Some(token) = self.avx2_token {
                    #[allow(deprecated)]
                    crate::dct::avx2_archmage::forward_dct_8x8_i32(token, samples, coeffs);
                } else {
                    // Fallback (shouldn't happen if variant is Avx2Archmage)
                    crate::dct::forward_dct_8x8_i32_multiversion(samples, coeffs);
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
            DctVariant::Avx2Intrinsics => {
                x86_64::avx2::forward_dct_8x8_i32_avx2_intrinsics(samples, coeffs);
            }
            DctVariant::Multiversion => {
                crate::dct::forward_dct_8x8_i32_multiversion(samples, coeffs);
            }
        }
    }

    /// Get scalar implementations (with multiversion autovectorization).
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            forward_dct: scalar::forward_dct_8x8,
            color_convert_rgb_to_ycbcr: scalar::convert_rgb_to_ycbcr,
            dct_variant: DctVariant::Multiversion,
            #[cfg(target_arch = "x86_64")]
            avx2_token: None,
        }
    }

    /// Get explicit AVX2 intrinsics (requires x86_64 with AVX2).
    /// Returns None if not available.
    #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
    #[must_use]
    pub fn avx2_intrinsics() -> Option<Self> {
        if is_x86_feature_detected!("avx2") {
            Some(Self {
                forward_dct: x86_64::avx2::forward_dct_8x8_i32_avx2_intrinsics,
                color_convert_rgb_to_ycbcr: x86_64::avx2::convert_rgb_to_ycbcr,
                dct_variant: DctVariant::Avx2Intrinsics,
                avx2_token: Avx2Token::try_new(),
            })
        } else {
            None
        }
    }

    /// Get archmage-based AVX2 implementation with cached token.
    /// Returns None if AVX2 is not available.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn avx2_archmage() -> Option<Self> {
        Avx2Token::try_new().map(|token| Self {
            forward_dct: scalar::forward_dct_8x8,
            color_convert_rgb_to_ycbcr: scalar::convert_rgb_to_ycbcr,
            dct_variant: DctVariant::Avx2Archmage,
            avx2_token: Some(token),
        })
    }

    /// Check which DCT variant is active.
    pub fn dct_variant_name(&self) -> &'static str {
        match self.dct_variant {
            DctVariant::Multiversion => "multiversion",
            #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
            DctVariant::Avx2Intrinsics => "avx2_intrinsics",
            #[cfg(target_arch = "x86_64")]
            DctVariant::Avx2Archmage => "avx2_archmage",
        }
    }
}

impl Default for SimdOps {
    fn default() -> Self {
        Self::detect()
    }
}

impl std::fmt::Debug for SimdOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdOps")
            .field("dct_variant", &self.dct_variant_name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_valid_ops() {
        let ops = SimdOps::detect();

        // Test that DCT works via method
        let samples = [100i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        ops.do_forward_dct(&samples, &mut coeffs);

        // DC should be 64 * 100 = 6400 for flat block
        assert_eq!(coeffs[0], 6400);
    }

    #[test]
    fn test_scalar_matches_detect_for_flat_block() {
        let scalar_ops = SimdOps::scalar();
        let detect_ops = SimdOps::detect();

        let samples = [100i16; DCTSIZE2];
        let mut coeffs_scalar = [0i16; DCTSIZE2];
        let mut coeffs_detect = [0i16; DCTSIZE2];

        scalar_ops.do_forward_dct(&samples, &mut coeffs_scalar);
        detect_ops.do_forward_dct(&samples, &mut coeffs_detect);

        assert_eq!(coeffs_scalar, coeffs_detect);
    }

    #[test]
    fn test_color_convert_works() {
        let ops = SimdOps::detect();

        // Test with a small image
        let rgb = [128u8; 24]; // 8 pixels
        let mut y = [0u8; 8];
        let mut cb = [0u8; 8];
        let mut cr = [0u8; 8];

        (ops.color_convert_rgb_to_ycbcr)(&rgb, &mut y, &mut cb, &mut cr, 8);

        // Gray (128,128,128) should give Y=128, Cb=128, Cr=128
        assert_eq!(y[0], 128);
        assert_eq!(cb[0], 128);
        assert_eq!(cr[0], 128);
    }

    #[test]
    fn test_dct_variant_name() {
        let ops = SimdOps::detect();
        let name = ops.dct_variant_name();
        // Should be one of the known variants
        assert!(
            name == "multiversion" || name == "avx2_intrinsics" || name == "avx2_archmage",
            "Unknown variant: {}",
            name
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_archmage_matches_multiversion() {
        if let Some(archmage_ops) = SimdOps::avx2_archmage() {
            let scalar_ops = SimdOps::scalar();

            for seed in 0..20 {
                let mut samples = [0i16; DCTSIZE2];
                for i in 0..DCTSIZE2 {
                    samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
                }

                let mut coeffs_archmage = [0i16; DCTSIZE2];
                let mut coeffs_scalar = [0i16; DCTSIZE2];

                archmage_ops.do_forward_dct(&samples, &mut coeffs_archmage);
                scalar_ops.do_forward_dct(&samples, &mut coeffs_scalar);

                assert_eq!(
                    coeffs_archmage, coeffs_scalar,
                    "Archmage should match scalar for seed {}",
                    seed
                );
            }
        }
    }
}
