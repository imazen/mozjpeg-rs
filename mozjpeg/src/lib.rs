//! mozjpeg-rs: Rust port of Mozilla's mozjpeg JPEG encoder.
//!
//! This library provides a high-quality JPEG encoder with features from mozjpeg:
//! - Trellis quantization for optimal rate-distortion
//! - Progressive JPEG with optimized scan order
//! - Huffman table optimization (2-pass encoding)
//! - Multiple perceptual quantization table variants
//!
//! # Example
//!
//! ```ignore
//! use mozjpeg::{Encoder, Subsampling};
//!
//! let encoder = Encoder::new()
//!     .quality(85)
//!     .progressive(true)
//!     .subsampling(Subsampling::S420);
//!
//! let jpeg_data = encoder.encode_rgb(&rgb_pixels, width, height)?;
//! ```

pub mod bitstream;
pub mod color;
pub mod consts;
pub mod dct;
pub mod encode;
pub mod entropy;
pub mod error;
pub mod huffman;
pub mod marker;
pub mod progressive;
pub mod quant;
pub mod sample;
pub mod trellis;
pub mod types;

/// Test encoder module for comparing Rust vs C implementations.
/// Hidden from public API but available for tests.
#[doc(hidden)]
pub mod test_encoder;

// Re-export commonly used items
pub use consts::{
    DCTSIZE, DCTSIZE2,
    JPEG_SOI, JPEG_EOI, JPEG_SOF0, JPEG_SOF2,
    JPEG_DHT, JPEG_DQT, JPEG_SOS,
    STD_LUMINANCE_QUANT_TBL, STD_CHROMINANCE_QUANT_TBL,
    QuantTableIdx,
};

pub use error::{Error, Result};

pub use types::{
    ColorSpace, CompressionProfile, DctMethod, Subsampling,
    ScanInfo, ComponentInfo, QuantTable, HuffmanTable,
    TrellisConfig, DctBlock, SampleBlock, FloatBlock,
};

pub use quant::{
    quality_to_scale_factor, quality_to_scale_factor_f32,
    get_luminance_quant_table, get_chrominance_quant_table,
    create_quant_table, create_quant_tables,
    quantize_coef, dequantize_coef,
    quantize_block, dequantize_block,
};

pub use dct::{forward_dct_8x8, forward_dct, level_shift};

pub use color::{
    rgb_to_ycbcr, rgb_to_gray, cmyk_to_ycck,
    convert_rgb_to_ycbcr, convert_rgb_to_gray,
    convert_block_rgb_to_ycbcr,
};

pub use huffman::{
    HuffTable, DerivedTable, FrequencyCounter,
    generate_optimal_table, MAX_CODE_LENGTH, NUM_HUFF_TBLS,
};

pub use bitstream::{BitWriter, VecBitWriter};

pub use entropy::{EntropyEncoder, ProgressiveEncoder, ProgressiveSymbolCounter, SymbolCounter, encode_block_standalone, jpeg_nbits};

pub use sample::{
    downsample_h2v1_row, downsample_h2v2_rows, downsample_plane,
    subsampled_dimensions, mcu_aligned_dimensions, expand_to_mcu,
};

pub use trellis::{
    trellis_quantize_block, trellis_quantize_block_with_eob_info,
    simple_quantize_block, optimize_eob_runs, BlockEobInfo, compute_block_eob_info,
};

pub use progressive::{
    generate_simple_progressive_scans, generate_standard_progressive_scans,
    generate_optimized_progressive_scans, generate_baseline_scan,
    validate_scan_script, is_progressive_script,
};

pub use marker::MarkerWriter;

pub use encode::Encoder;
