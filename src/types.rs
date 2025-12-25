//! Core type definitions for mozjpeg encoder.
//!
//! This module defines all the types needed for JPEG encoding,
//! matching the semantics of mozjpeg's C types but with idiomatic Rust design.

use crate::consts::{DCTSIZE2, MAX_COMPS_IN_SCAN};

// =============================================================================
// Color Spaces
// =============================================================================

/// Input color space for the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorSpace {
    /// Unknown/unspecified color space
    #[default]
    Unknown = 0,
    /// Grayscale (1 component)
    Grayscale = 1,
    /// RGB (3 components, standard order)
    Rgb = 2,
    /// YCbCr (3 components)
    YCbCr = 3,
    /// CMYK (4 components)
    Cmyk = 4,
    /// YCCK (4 components, Y/Cb/Cr/K)
    Ycck = 5,
    /// RGB with explicit order (red/green/blue)
    ExtRgb = 6,
    /// RGBX (RGB with padding byte)
    ExtRgbx = 7,
    /// BGR (blue/green/red)
    ExtBgr = 8,
    /// BGRX (BGR with padding byte)
    ExtBgrx = 9,
    /// XBGR (padding/blue/green/red)
    ExtXbgr = 10,
    /// XRGB (padding/red/green/blue)
    ExtXrgb = 11,
    /// RGBA (with alpha)
    ExtRgba = 12,
    /// BGRA (with alpha)
    ExtBgra = 13,
    /// ABGR (alpha first)
    ExtAbgr = 14,
    /// ARGB (alpha first)
    ExtArgb = 15,
}

impl ColorSpace {
    /// Returns the number of components for this color space.
    pub const fn num_components(self) -> usize {
        match self {
            ColorSpace::Unknown => 0,
            ColorSpace::Grayscale => 1,
            ColorSpace::Rgb | ColorSpace::YCbCr => 3,
            ColorSpace::ExtRgb | ColorSpace::ExtBgr => 3,
            ColorSpace::Cmyk | ColorSpace::Ycck => 4,
            ColorSpace::ExtRgbx
            | ColorSpace::ExtBgrx
            | ColorSpace::ExtXbgr
            | ColorSpace::ExtXrgb
            | ColorSpace::ExtRgba
            | ColorSpace::ExtBgra
            | ColorSpace::ExtAbgr
            | ColorSpace::ExtArgb => 4,
        }
    }

    /// Returns the bytes per pixel for this color space.
    pub const fn bytes_per_pixel(self) -> usize {
        self.num_components()
    }

    /// Returns true if this is an RGB variant.
    pub const fn is_rgb_variant(self) -> bool {
        matches!(
            self,
            ColorSpace::Rgb
                | ColorSpace::ExtRgb
                | ColorSpace::ExtRgbx
                | ColorSpace::ExtBgr
                | ColorSpace::ExtBgrx
                | ColorSpace::ExtXbgr
                | ColorSpace::ExtXrgb
                | ColorSpace::ExtRgba
                | ColorSpace::ExtBgra
                | ColorSpace::ExtAbgr
                | ColorSpace::ExtArgb
        )
    }
}

// =============================================================================
// Compression Profile
// =============================================================================

/// Compression profile controlling which mozjpeg features are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum CompressionProfile {
    /// Maximum compression - all mozjpeg features enabled.
    /// - Progressive mode
    /// - Trellis quantization
    /// - Optimized Huffman tables
    /// - ImageMagick quantization tables (index 3)
    #[default]
    MaxCompression = 0x5D083AAD,
    /// Fastest - libjpeg-turbo defaults, no mozjpeg extensions.
    /// - Baseline (non-progressive)
    /// - Standard quantization
    /// - Pre-computed Huffman tables
    Fastest = 0x2AEA5CB4,
}

// =============================================================================
// DCT Method
// =============================================================================

/// DCT algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DctMethod {
    /// Accurate integer method (default)
    #[default]
    IntSlow = 0,
    /// Less accurate but faster integer method
    IntFast = 1,
    /// Floating-point method
    Float = 2,
}

// =============================================================================
// Sampling Factor / Subsampling
// =============================================================================

/// Chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Subsampling {
    /// 4:4:4 - No subsampling (highest quality)
    #[default]
    S444,
    /// 4:2:2 - Horizontal subsampling
    S422,
    /// 4:2:0 - Horizontal and vertical subsampling (most common)
    S420,
    /// 4:4:0 - Vertical subsampling only
    S440,
    /// Grayscale (1 component)
    Gray,
}

impl Subsampling {
    /// Returns (h_samp_factor, v_samp_factor) for luminance component.
    pub const fn luma_factors(self) -> (u8, u8) {
        match self {
            Subsampling::S444 | Subsampling::Gray => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        }
    }

    /// Returns (h_samp_factor, v_samp_factor) for chroma components.
    pub const fn chroma_factors(self) -> (u8, u8) {
        (1, 1) // Chroma always 1x1 relative to max
    }
}

// =============================================================================
// Scan Info (for progressive JPEG)
// =============================================================================

/// Describes a single scan in a multi-scan (progressive) JPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanInfo {
    /// Number of components in this scan (1-4)
    pub comps_in_scan: u8,
    /// Component indices for this scan
    pub component_index: [u8; MAX_COMPS_IN_SCAN],
    /// Spectral selection start (0 for DC, 1-63 for AC)
    pub ss: u8,
    /// Spectral selection end (0 for DC-only, 63 for full AC)
    pub se: u8,
    /// Successive approximation high bit
    pub ah: u8,
    /// Successive approximation low bit (point transform)
    pub al: u8,
}

impl ScanInfo {
    /// Create a DC-only scan for all components.
    pub const fn dc_scan(num_components: u8) -> Self {
        Self {
            comps_in_scan: num_components,
            component_index: [0, 1, 2, 3],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }
    }

    /// Create an AC scan for a single component.
    pub const fn ac_scan(component: u8, ss: u8, se: u8, ah: u8, al: u8) -> Self {
        Self {
            comps_in_scan: 1,
            component_index: [component, 0, 0, 0],
            ss,
            se,
            ah,
            al,
        }
    }

    /// Returns true if this is a DC-only scan.
    pub const fn is_dc_scan(&self) -> bool {
        self.ss == 0 && self.se == 0
    }

    /// Returns true if this is a refinement scan (successive approximation).
    pub const fn is_refinement(&self) -> bool {
        self.ah != 0
    }

    /// Create a DC scan for a single component.
    pub const fn dc_scan_single(component: u8) -> Self {
        Self {
            comps_in_scan: 1,
            component_index: [component, 0, 0, 0],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }
    }

    /// Create a DC scan for two components (e.g., Cb and Cr).
    pub const fn dc_scan_pair(comp1: u8, comp2: u8) -> Self {
        Self {
            comps_in_scan: 2,
            component_index: [comp1, comp2, 0, 0],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }
    }
}

impl Default for ScanInfo {
    fn default() -> Self {
        Self::dc_scan(3)
    }
}

// =============================================================================
// Component Info
// =============================================================================

/// Information about a single image component (color channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentInfo {
    /// Component identifier (1=Y, 2=Cb, 3=Cr for YCbCr)
    pub component_id: u8,
    /// Index in component array
    pub component_index: u8,
    /// Horizontal sampling factor (1-4)
    pub h_samp_factor: u8,
    /// Vertical sampling factor (1-4)
    pub v_samp_factor: u8,
    /// Quantization table index (0-3)
    pub quant_tbl_no: u8,
    /// DC Huffman table index (0-3)
    pub dc_tbl_no: u8,
    /// AC Huffman table index (0-3)
    pub ac_tbl_no: u8,
}

impl Default for ComponentInfo {
    fn default() -> Self {
        Self {
            component_id: 1,
            component_index: 0,
            h_samp_factor: 1,
            v_samp_factor: 1,
            quant_tbl_no: 0,
            dc_tbl_no: 0,
            ac_tbl_no: 0,
        }
    }
}

// =============================================================================
// Quantization Table
// =============================================================================

/// A quantization table with 64 coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTable {
    /// Quantization values in natural (row-major) order
    pub values: [u16; DCTSIZE2],
    /// True if this table has been written to the output
    pub sent: bool,
}

impl QuantTable {
    /// Create a new quantization table from values.
    pub const fn new(values: [u16; DCTSIZE2]) -> Self {
        Self {
            values,
            sent: false,
        }
    }

    /// Create from a base table scaled by a quality factor.
    /// Scale factor is a percentage (100 = use table as-is).
    pub fn scaled(base: &[u16; DCTSIZE2], scale_factor: u32, force_baseline: bool) -> Self {
        let mut values = [0u16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            let mut temp = ((base[i] as u32) * scale_factor + 50) / 100;
            // Clamp to valid range
            if temp == 0 {
                temp = 1;
            }
            if temp > 32767 {
                temp = 32767;
            }
            if force_baseline && temp > 255 {
                temp = 255;
            }
            values[i] = temp as u16;
        }
        Self {
            values,
            sent: false,
        }
    }
}

impl Default for QuantTable {
    fn default() -> Self {
        Self {
            values: [16; DCTSIZE2], // Flat table
            sent: false,
        }
    }
}

// =============================================================================
// Huffman Table
// =============================================================================

/// A Huffman coding table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HuffmanTable {
    /// Number of codes of each length (`bits[k]` = # of symbols with k-bit codes).
    /// `bits[0]` is unused.
    pub bits: [u8; 17],
    /// Symbol values in order of increasing code length
    pub huffval: Vec<u8>,
    /// True if this table has been written to the output
    pub sent: bool,
}

impl HuffmanTable {
    /// Create a new Huffman table from bits and values.
    pub fn new(bits: [u8; 17], huffval: Vec<u8>) -> Self {
        Self {
            bits,
            huffval,
            sent: false,
        }
    }

    /// Returns the total number of symbols in this table.
    pub fn num_symbols(&self) -> usize {
        self.bits[1..].iter().map(|&b| b as usize).sum()
    }
}


// =============================================================================
// Trellis Configuration
// =============================================================================

/// Configuration for trellis quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrellisConfig {
    /// Enable trellis quantization for AC coefficients
    pub enabled: bool,
    /// Enable trellis quantization for DC coefficients
    pub dc_enabled: bool,
    /// Optimize for sequences of EOB
    pub eob_opt: bool,
    /// Use perceptual lambda weighting table
    pub use_lambda_weight_tbl: bool,
    /// Consider scan order in trellis optimization
    pub use_scans_in_trellis: bool,
    /// Optimize quantization table in trellis loop
    pub q_opt: bool,
    /// Lambda log scale parameter 1
    pub lambda_log_scale1: f32,
    /// Lambda log scale parameter 2
    pub lambda_log_scale2: f32,
    /// Frequency split point for spectral selection
    pub freq_split: i32,
    /// Number of trellis optimization loops
    pub num_loops: i32,
    /// DC delta weight for vertical gradient consideration
    pub delta_dc_weight: f32,
}

impl Default for TrellisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dc_enabled: true,
            eob_opt: true,
            use_lambda_weight_tbl: true,
            use_scans_in_trellis: false,
            q_opt: false,
            lambda_log_scale1: crate::consts::DEFAULT_LAMBDA_LOG_SCALE1,
            lambda_log_scale2: crate::consts::DEFAULT_LAMBDA_LOG_SCALE2,
            freq_split: crate::consts::DEFAULT_TRELLIS_FREQ_SPLIT,
            num_loops: crate::consts::DEFAULT_TRELLIS_NUM_LOOPS,
            delta_dc_weight: crate::consts::DEFAULT_TRELLIS_DELTA_DC_WEIGHT,
        }
    }
}

impl TrellisConfig {
    /// Configuration with trellis disabled (fastest mode).
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            dc_enabled: false,
            eob_opt: false,
            use_lambda_weight_tbl: false,
            use_scans_in_trellis: false,
            q_opt: false,
            lambda_log_scale1: 14.75,
            lambda_log_scale2: 16.5,
            freq_split: 8,
            num_loops: 1,
            delta_dc_weight: 0.0,
        }
    }
}

// =============================================================================
// DCT Block Types
// =============================================================================

/// A single 8x8 block of DCT coefficients.
pub type DctBlock = [i16; DCTSIZE2];

/// A single 8x8 block of pixel samples.
pub type SampleBlock = [u8; DCTSIZE2];

/// A single 8x8 block of floating-point values.
pub type FloatBlock = [f32; DCTSIZE2];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colorspace_components() {
        assert_eq!(ColorSpace::Grayscale.num_components(), 1);
        assert_eq!(ColorSpace::Rgb.num_components(), 3);
        assert_eq!(ColorSpace::YCbCr.num_components(), 3);
        assert_eq!(ColorSpace::Cmyk.num_components(), 4);
        assert_eq!(ColorSpace::ExtRgba.num_components(), 4);
    }

    #[test]
    fn test_subsampling_factors() {
        assert_eq!(Subsampling::S444.luma_factors(), (1, 1));
        assert_eq!(Subsampling::S422.luma_factors(), (2, 1));
        assert_eq!(Subsampling::S420.luma_factors(), (2, 2));
        assert_eq!(Subsampling::S440.luma_factors(), (1, 2));
    }

    #[test]
    fn test_scan_info() {
        let dc = ScanInfo::dc_scan(3);
        assert!(dc.is_dc_scan());
        assert!(!dc.is_refinement());

        let ac = ScanInfo::ac_scan(0, 1, 63, 0, 0);
        assert!(!ac.is_dc_scan());
        assert!(!ac.is_refinement());

        let refine = ScanInfo::ac_scan(0, 1, 63, 1, 0);
        assert!(refine.is_refinement());
    }

    #[test]
    fn test_quant_table_scaling() {
        let base = [16u16; DCTSIZE2];

        // 100% scale should give same values
        let scaled = QuantTable::scaled(&base, 100, false);
        assert_eq!(scaled.values, base);

        // 200% scale should double
        let scaled = QuantTable::scaled(&base, 200, false);
        assert_eq!(scaled.values[0], 32);

        // 50% scale should halve
        let scaled = QuantTable::scaled(&base, 50, false);
        assert_eq!(scaled.values[0], 8);

        // Force baseline should clamp to 255
        let high = [1000u16; DCTSIZE2];
        let scaled = QuantTable::scaled(&high, 100, true);
        assert_eq!(scaled.values[0], 255);
    }

    #[test]
    fn test_trellis_config_defaults() {
        let config = TrellisConfig::default();
        assert!(config.enabled);
        assert!(config.dc_enabled);
        assert_eq!(config.lambda_log_scale1, 14.75);
        assert_eq!(config.num_loops, 1);

        let disabled = TrellisConfig::disabled();
        assert!(!disabled.enabled);
    }
}
