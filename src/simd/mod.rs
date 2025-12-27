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

/// SIMD operations dispatch table.
///
/// This struct holds function pointers to the best available implementations
/// for the current CPU. Create once at startup and reuse.
#[derive(Clone, Copy)]
pub struct SimdOps {
    // Note: Debug is implemented manually below since fn pointers print as addresses
    /// Forward DCT function
    pub forward_dct: ForwardDctFn,
    /// RGB to YCbCr conversion function
    pub color_convert_rgb_to_ycbcr: ColorConvertFn,
}

impl SimdOps {
    /// Select the best available implementations for the current CPU.
    ///
    /// By default, uses `multiversion` for automatic SIMD dispatch via
    /// autovectorization. This is safe and provides ~87% of intrinsics
    /// performance.
    ///
    /// With the `simd-intrinsics` feature, uses hand-written AVX2 intrinsics
    /// for maximum performance (~15% faster, but requires unsafe code).
    #[must_use]
    pub fn detect() -> Self {
        // With simd-intrinsics feature, use hand-written intrinsics for max perf
        #[cfg(all(target_arch = "x86_64", feature = "simd-intrinsics"))]
        {
            if is_x86_feature_detected!("avx2") {
                return Self {
                    forward_dct: x86_64::avx2::forward_dct_8x8,
                    color_convert_rgb_to_ycbcr: x86_64::avx2::convert_rgb_to_ycbcr,
                };
            }
        }

        // Default: use multiversion scalar (autovectorized, safe)
        Self {
            forward_dct: scalar::forward_dct_8x8,
            color_convert_rgb_to_ycbcr: scalar::convert_rgb_to_ycbcr,
        }
    }

    /// Get scalar implementations (with multiversion autovectorization).
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            forward_dct: scalar::forward_dct_8x8,
            color_convert_rgb_to_ycbcr: scalar::convert_rgb_to_ycbcr,
        }
    }

    /// Get explicit AVX2 intrinsics (requires x86_64 with AVX2).
    /// Returns None if not available.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub fn avx2_intrinsics() -> Option<Self> {
        if is_x86_feature_detected!("avx2") {
            Some(Self {
                forward_dct: x86_64::avx2::forward_dct_8x8,
                color_convert_rgb_to_ycbcr: x86_64::avx2::convert_rgb_to_ycbcr,
            })
        } else {
            None
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
        // Just indicate which implementation is in use, not the pointer addresses
        #[cfg(target_arch = "x86_64")]
        let variant = if is_x86_feature_detected!("avx2") {
            "AVX2"
        } else {
            "Scalar"
        };
        #[cfg(not(target_arch = "x86_64"))]
        let variant = "Scalar";

        f.debug_struct("SimdOps")
            .field("variant", &variant)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_valid_ops() {
        let ops = SimdOps::detect();

        // Test that DCT works
        let samples = [100i16; DCTSIZE2];
        let mut coeffs = [0i16; DCTSIZE2];
        (ops.forward_dct)(&samples, &mut coeffs);

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

        (scalar_ops.forward_dct)(&samples, &mut coeffs_scalar);
        (detect_ops.forward_dct)(&samples, &mut coeffs_detect);

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
}
