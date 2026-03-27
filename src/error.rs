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
    /// Encoding was cancelled by the caller
    Cancelled,
    /// Encoding exceeded the timeout duration
    TimedOut,
    /// Image dimensions exceed the configured maximum
    DimensionLimitExceeded {
        /// Requested width
        width: u32,
        /// Requested height
        height: u32,
        /// Maximum allowed width
        max_width: u32,
        /// Maximum allowed height
        max_height: u32,
    },
    /// Estimated memory usage exceeds the configured limit
    AllocationLimitExceeded {
        /// Estimated memory in bytes
        estimated: usize,
        /// Maximum allowed memory in bytes
        limit: usize,
    },
    /// Pixel count (width × height) exceeds the configured limit
    PixelCountExceeded {
        /// Actual pixel count
        pixel_count: u64,
        /// Maximum allowed pixel count
        limit: u64,
    },
    /// ICC profile size exceeds the configured limit
    IccProfileTooLarge {
        /// Actual ICC profile size in bytes
        size: usize,
        /// Maximum allowed size in bytes
        limit: usize,
    },
    /// Row stride is too small for the image width
    InvalidStride {
        /// Provided stride in bytes
        stride: usize,
        /// Minimum required stride in bytes
        minimum: usize,
    },
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
            Error::Cancelled => {
                write!(f, "Encoding was cancelled")
            }
            Error::TimedOut => {
                write!(f, "Encoding timed out")
            }
            Error::DimensionLimitExceeded {
                width,
                height,
                max_width,
                max_height,
            } => {
                write!(
                    f,
                    "Image dimensions {}x{} exceed limit {}x{}",
                    width, height, max_width, max_height
                )
            }
            Error::AllocationLimitExceeded { estimated, limit } => {
                write!(
                    f,
                    "Estimated memory {} bytes exceeds limit {} bytes",
                    estimated, limit
                )
            }
            Error::PixelCountExceeded { pixel_count, limit } => {
                write!(f, "Pixel count {} exceeds limit {}", pixel_count, limit)
            }
            Error::IccProfileTooLarge { size, limit } => {
                write!(
                    f,
                    "ICC profile size {} bytes exceeds limit {} bytes",
                    size, limit
                )
            }
            Error::InvalidStride { stride, minimum } => {
                write!(
                    f,
                    "Invalid stride: {} bytes is less than minimum {} bytes",
                    stride, minimum
                )
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

impl From<enough::StopReason> for Error {
    fn from(reason: enough::StopReason) -> Self {
        match reason {
            enough::StopReason::TimedOut => Error::TimedOut,
            _ => Error::Cancelled,
        }
    }
}

#[cfg(feature = "zencodec")]
impl From<zencodec::LimitExceeded> for Error {
    fn from(limit: zencodec::LimitExceeded) -> Self {
        use zencodec::LimitExceeded;
        match limit {
            LimitExceeded::Width { actual, .. } | LimitExceeded::Height { actual, .. } => {
                Error::DimensionLimitExceeded {
                    width: actual,
                    height: 0,
                    max_width: 0,
                    max_height: 0,
                }
            }
            LimitExceeded::Pixels { actual, max } => Error::PixelCountExceeded {
                pixel_count: actual,
                limit: max,
            },
            LimitExceeded::Memory { actual, max } => Error::AllocationLimitExceeded {
                estimated: actual as usize,
                limit: max as usize,
            },
            _ => Error::InternalError("resource limit exceeded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        // Test that all error variants format correctly
        let errors = [
            (
                Error::InvalidDimensions {
                    width: 0,
                    height: 100,
                },
                "Invalid image dimensions: 0x100",
            ),
            (
                Error::BufferSizeMismatch {
                    expected: 1000,
                    actual: 500,
                },
                "Buffer size mismatch: expected 1000, got 500",
            ),
            (
                Error::InvalidQuality(0),
                "Invalid quality value: 0 (must be 1-100)",
            ),
            (
                Error::InvalidQuantTableIndex(5),
                "Invalid quantization table index: 5",
            ),
            (
                Error::InvalidComponentIndex(4),
                "Invalid component index: 4",
            ),
            (
                Error::InvalidHuffmanTableIndex(8),
                "Invalid Huffman table index: 8",
            ),
            (
                Error::InvalidSamplingFactor { h: 5, v: 3 },
                "Invalid sampling factor: 5x3",
            ),
            (
                Error::InvalidScanSpec {
                    reason: "test reason",
                },
                "Invalid scan specification: test reason",
            ),
            (
                Error::InvalidHuffmanTable,
                "Invalid Huffman table structure",
            ),
            (
                Error::HuffmanCodeLengthOverflow,
                "Huffman code length overflow (exceeds 16 bits)",
            ),
            (Error::UnsupportedColorSpace, "Unsupported color space"),
            (
                Error::UnsupportedFeature("arithmetic coding"),
                "Unsupported feature: arithmetic coding",
            ),
            (
                Error::InternalError("test error"),
                "Internal encoder error: test error",
            ),
            (Error::IoError("disk full".into()), "I/O error: disk full"),
            (Error::AllocationFailed, "Memory allocation failed"),
            (Error::Cancelled, "Encoding was cancelled"),
            (Error::TimedOut, "Encoding timed out"),
            (
                Error::DimensionLimitExceeded {
                    width: 5000,
                    height: 3000,
                    max_width: 4096,
                    max_height: 4096,
                },
                "Image dimensions 5000x3000 exceed limit 4096x4096",
            ),
            (
                Error::AllocationLimitExceeded {
                    estimated: 100_000_000,
                    limit: 50_000_000,
                },
                "Estimated memory 100000000 bytes exceeds limit 50000000 bytes",
            ),
            (
                Error::InvalidStride {
                    stride: 100,
                    minimum: 300,
                },
                "Invalid stride: 100 bytes is less than minimum 300 bytes",
            ),
        ];

        for (error, expected_msg) in errors {
            assert_eq!(error.to_string(), expected_msg);
        }
    }

    #[test]
    fn test_error_is_error_trait() {
        let error: &dyn std::error::Error = &Error::InvalidQuality(0);
        // Just verify it implements Error trait
        let _ = error.to_string();
    }

    #[test]
    fn test_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: Error = io_error.into();
        assert!(matches!(error, Error::IoError(_)));
        assert!(error.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_clone_and_eq() {
        let error1 = Error::InvalidQuality(50);
        let error2 = error1.clone();
        assert_eq!(error1, error2);

        let error3 = Error::InvalidQuality(60);
        assert_ne!(error1, error3);
    }
}
