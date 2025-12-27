//! Error types for the mozjpeg encoder.

use std::fmt;

/// Result type for mozjpeg operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for mozjpeg operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Invalid image dimensions (zero width or height)
    InvalidDimensions {
        /// Image width
        width: u32,
        /// Image height
        height: u32,
    },
    /// Image buffer size doesn't match dimensions
    BufferSizeMismatch {
        /// Expected buffer size in bytes
        expected: usize,
        /// Actual buffer size in bytes
        actual: usize,
    },
    /// Invalid quality value (must be 1-100)
    InvalidQuality(u8),
    /// Invalid quantization table index
    InvalidQuantTableIndex(usize),
    /// Invalid component index
    InvalidComponentIndex(usize),
    /// Invalid Huffman table index
    InvalidHuffmanTableIndex(usize),
    /// Invalid sampling factor
    InvalidSamplingFactor {
        /// Horizontal sampling factor
        h: u8,
        /// Vertical sampling factor
        v: u8,
    },
    /// Invalid scan specification
    InvalidScanSpec {
        /// Reason for the invalid specification
        reason: &'static str,
    },
    /// Invalid Huffman table structure
    InvalidHuffmanTable,
    /// Huffman code length overflow (exceeds max allowed)
    HuffmanCodeLengthOverflow,
    /// Unsupported color space
    UnsupportedColorSpace,
    /// Unsupported feature
    UnsupportedFeature(&'static str),
    /// Internal encoder error
    InternalError(&'static str),
    /// I/O error
    IoError(String),
    /// Memory allocation failed
    AllocationFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height } => {
                write!(f, "Invalid image dimensions: {}x{}", width, height)
            }
            Error::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Error::InvalidQuality(q) => {
                write!(f, "Invalid quality value: {} (must be 1-100)", q)
            }
            Error::InvalidQuantTableIndex(idx) => {
                write!(f, "Invalid quantization table index: {}", idx)
            }
            Error::InvalidComponentIndex(idx) => {
                write!(f, "Invalid component index: {}", idx)
            }
            Error::InvalidHuffmanTableIndex(idx) => {
                write!(f, "Invalid Huffman table index: {}", idx)
            }
            Error::InvalidSamplingFactor { h, v } => {
                write!(f, "Invalid sampling factor: {}x{}", h, v)
            }
            Error::InvalidScanSpec { reason } => {
                write!(f, "Invalid scan specification: {}", reason)
            }
            Error::InvalidHuffmanTable => {
                write!(f, "Invalid Huffman table structure")
            }
            Error::HuffmanCodeLengthOverflow => {
                write!(f, "Huffman code length overflow (exceeds 16 bits)")
            }
            Error::UnsupportedColorSpace => {
                write!(f, "Unsupported color space")
            }
            Error::UnsupportedFeature(feature) => {
                write!(f, "Unsupported feature: {}", feature)
            }
            Error::InternalError(msg) => {
                write!(f, "Internal encoder error: {}", msg)
            }
            Error::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            Error::AllocationFailed => {
                write!(f, "Memory allocation failed")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e.to_string())
    }
}

impl From<std::collections::TryReserveError> for Error {
    fn from(_: std::collections::TryReserveError) -> Self {
        Error::AllocationFailed
    }
}
