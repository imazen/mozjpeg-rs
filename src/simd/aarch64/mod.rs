//! aarch64-specific SIMD implementations using NEON.
//!
//! This module provides NEON-optimized implementations for ARM64.
//! All functions use archmage's #[arcane] macro for safe intrinsics.

pub mod neon;
