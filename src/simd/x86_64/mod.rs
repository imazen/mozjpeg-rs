//! x86_64-specific SIMD implementations.
//!
//! This module provides AVX2 and SSE2 optimized implementations for x86_64.
//! Runtime detection is handled by the parent `simd` module.
//!
//! Note: The avx2 module is always compiled on x86_64, but functions inside
//! use `#[target_feature(enable = "avx2")]` and are only called after
//! runtime feature detection confirms AVX2 support.

pub mod avx2;

// Future: SSE2 fallback
// pub mod sse2;
