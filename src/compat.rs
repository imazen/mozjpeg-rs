//! mozjpeg-sys configuration layer.
//!
//! This module provides methods to configure a C mozjpeg encoder (`jpeg_compress_struct`)
//! with settings matching our [`Encoder`](crate::Encoder) configuration.
//!
//! # Feature Flag
//!
//! This module requires the `mozjpeg-sys-config` feature:
//!
//! ```toml
//! [dependencies]
//! mozjpeg-rs = { version = "0.3", features = ["mozjpeg-sys-config"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use mozjpeg_rs::{Encoder, Preset};
//! use mozjpeg_sys::*;
//! use std::ptr;
//!
//! unsafe {
//!     let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
//!     let mut jerr: jpeg_error_mgr = std::mem::zeroed();
//!     cinfo.common.err = jpeg_std_error(&mut jerr);
//!     jpeg_CreateCompress(&mut cinfo, JPEG_LIB_VERSION as i32,
//!         std::mem::size_of::<jpeg_compress_struct>());
//!
//!     // Configure C encoder to match our Encoder settings
//!     let encoder = Encoder::new(Preset::ProgressiveBalanced).quality(85);
//!     encoder.configure_sys(&mut cinfo, 640, 480)
//!         .expect("Failed to configure C encoder");
//!
//!     // Now cinfo is configured identically to how our Encoder would encode
//!     // ... continue with jpeg_start_compress, etc.
//! }
//! ```
//!
//! # Limitations
//!
//! Some settings cannot be configured on `jpeg_compress_struct`:
//!
//! - **EXIF data**: Must be written as APP1 marker after `jpeg_start_compress`
//! - **ICC profile**: Must be written via `jpeg_write_icc_profile` after start
//! - **Custom markers**: Must be written via `jpeg_write_marker` after start
//!
//! These are returned as warnings, not errors. Use the returned [`ConfigWarnings`]
//! to check if any settings couldn't be applied.

use crate::consts::QuantTableIdx;
use crate::types::Subsampling;

/// Warnings from configuring a C mozjpeg encoder.
///
/// Some settings cannot be applied to `jpeg_compress_struct` directly
/// and must be handled separately after `jpeg_start_compress`.
#[derive(Debug, Clone, Default)]
pub struct ConfigWarnings {
    /// EXIF data was specified but must be written as APP1 marker after start
    pub has_exif: bool,
    /// ICC profile was specified but must be written after start
    pub has_icc_profile: bool,
    /// Custom markers were specified but must be written after start
    pub has_custom_markers: bool,
}

impl ConfigWarnings {
    /// Returns true if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        self.has_exif || self.has_icc_profile || self.has_custom_markers
    }
}

/// Error configuring a C mozjpeg encoder.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// The quant table index is not supported by C mozjpeg
    UnsupportedQuantTable(QuantTableIdx),
    /// Custom quant tables require manual configuration
    CustomQuantTablesNotSupported,
    /// The subsampling mode is not supported
    UnsupportedSubsampling(Subsampling),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::UnsupportedQuantTable(idx) => {
                write!(f, "Quant table {:?} not supported by C mozjpeg", idx)
            }
            ConfigError::CustomQuantTablesNotSupported => {
                write!(
                    f,
                    "Custom quant tables must be configured manually via jpeg_add_quant_table"
                )
            }
            ConfigError::UnsupportedSubsampling(s) => {
                write!(f, "Subsampling {:?} not supported", s)
            }
        }
    }
}

impl std::error::Error for ConfigError {}
