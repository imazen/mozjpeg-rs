//! Unified test encoder interface for comparing Rust vs C implementations.
//!
//! This module provides identical interfaces for both implementations,
//! ensuring parameter parity for apples-to-apples comparison.

use crate::Subsampling;

/// Configuration for test encoding - shared between Rust and C.
#[derive(Clone, Debug)]
pub struct TestEncoderConfig {
    pub quality: u8,
    pub subsampling: Subsampling,
    pub progressive: bool,
    pub optimize_huffman: bool,
    pub trellis_quant: bool,
    pub trellis_dc: bool,
    pub overshoot_deringing: bool,
}

impl Default for TestEncoderConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            subsampling: Subsampling::S420,
            progressive: false,
            optimize_huffman: false,
            trellis_quant: false,
            trellis_dc: false,
            overshoot_deringing: false,
        }
    }
}

impl TestEncoderConfig {
    /// Baseline JPEG with no optimizations (for strict comparison)
    pub fn baseline() -> Self {
        Self::default()
    }

    /// Maximum compression settings
    pub fn max_compression() -> Self {
        Self {
            progressive: true,
            optimize_huffman: true,
            trellis_quant: true,
            trellis_dc: true,
            ..Self::default()
        }
    }
}

/// Encode using Rust implementation.
pub fn encode_rust(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
    use crate::{Encoder, TrellisConfig};

    let trellis = if config.trellis_quant || config.trellis_dc {
        TrellisConfig {
            enabled: config.trellis_quant,
            dc_enabled: config.trellis_dc,
            ..TrellisConfig::default()
        }
    } else {
        TrellisConfig::disabled()
    };

    Encoder::new()
        .quality(config.quality)
        .subsampling(config.subsampling)
        .progressive(config.progressive)
        .optimize_huffman(config.optimize_huffman)
        .trellis(trellis)
        .encode_rgb(rgb, width, height)
        .expect("Rust encoding failed")
}

/// Encode using C mozjpeg implementation via FFI.
#[cfg(feature = "ffi-test")]
pub fn encode_c(rgb: &[u8], width: u32, height: u32, config: &TestEncoderConfig) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::ptr;

    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_compress_struct>(),
        );

        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);

        // Set progressive mode
        if config.progressive {
            jpeg_simple_progression(&mut cinfo);
        } else {
            cinfo.num_scans = 0;
            cinfo.scan_info = ptr::null();
        }

        jpeg_set_quality(&mut cinfo, config.quality as i32, 1);

        // Set subsampling
        let (h_samp, v_samp) = match config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
        };
        (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
        (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        // Set optimization flags
        cinfo.optimize_coding = if config.optimize_huffman { 1 } else { 0 };

        // Set trellis options
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT,
            if config.trellis_quant { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_TRELLIS_QUANT_DC,
            if config.trellis_dc { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OVERSHOOT_DERINGING,
            if config.overshoot_deringing { 1 } else { 0 },
        );

        jpeg_start_compress(&mut cinfo, 1);

        let row_stride = width as usize * 3;
        while cinfo.next_scanline < cinfo.image_height {
            let row_idx = cinfo.next_scanline as usize;
            let row_ptr = rgb.as_ptr().add(row_idx * row_stride);
            jpeg_write_scanlines(&mut cinfo, &row_ptr as *const *const u8, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        jpeg_destroy_compress(&mut cinfo);

        let result = std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec();
        libc::free(outbuffer as *mut libc::c_void);
        result
    }
}

/// Compare two JPEGs by decoding and checking pixel differences.
#[cfg(test)]
pub fn compare_decoded(rust_jpeg: &[u8], c_jpeg: &[u8]) -> CompareResult {
    let rust_decoded = decode_jpeg(rust_jpeg);
    let c_decoded = decode_jpeg(c_jpeg);

    match (rust_decoded, c_decoded) {
        (Some((rust_pix, rw, rh)), Some((c_pix, cw, ch))) => {
            if (rw, rh) != (cw, ch) {
                return CompareResult {
                    dimensions_match: false,
                    ..Default::default()
                };
            }

            let mut max_diff = 0u8;
            let mut sum_diff = 0u64;
            let mut diff_count = 0usize;

            for (r, c) in rust_pix.iter().zip(c_pix.iter()) {
                let d = (*r as i16 - *c as i16).unsigned_abs() as u8;
                if d > 0 {
                    diff_count += 1;
                    sum_diff += d as u64;
                    max_diff = max_diff.max(d);
                }
            }

            let avg_diff = if diff_count > 0 {
                sum_diff as f64 / diff_count as f64
            } else {
                0.0
            };

            CompareResult {
                decode_failed: false,
                dimensions_match: true,
                total_components: rust_pix.len(),
                differing_components: diff_count,
                max_diff,
                avg_diff,
            }
        }
        _ => CompareResult {
            decode_failed: true,
            ..Default::default()
        },
    }
}

/// Decode JPEG using external decoder.
/// Note: jpeg_decoder is a dev-dependency, so this won't work in the main crate.
/// For test usage, import jpeg_decoder in your test and use this helper.
pub fn decode_with_decoder<D: JpegDecoder>(data: &[u8], decoder: &D) -> Option<(Vec<u8>, usize, usize)> {
    decoder.decode(data)
}

/// Trait for JPEG decoders (allows injecting different decoders in tests)
pub trait JpegDecoder {
    fn decode(&self, data: &[u8]) -> Option<(Vec<u8>, usize, usize)>;
}

/// Default implementation that calls jpeg_decoder (for tests)
#[cfg(test)]
fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as usize, info.height as usize))
}

#[derive(Debug, Default)]
pub struct CompareResult {
    pub decode_failed: bool,
    pub dimensions_match: bool,
    pub total_components: usize,
    pub differing_components: usize,
    pub max_diff: u8,
    pub avg_diff: f64,
}

impl CompareResult {
    /// Check if results are within acceptable tolerance for identical parameters.
    /// With truly identical parameters, we expect max_diff <= 1 (rounding only).
    pub fn is_acceptable(&self) -> bool {
        !self.decode_failed && self.dimensions_match && self.max_diff <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_rust_baseline() {
        let width = 16;
        let height = 16;
        let mut rgb = vec![0u8; width * height * 3];
        for i in 0..rgb.len() {
            rgb[i] = (i % 256) as u8;
        }

        let config = TestEncoderConfig::baseline();
        let jpeg = encode_rust(&rgb, width as u32, height as u32, &config);
        assert!(!jpeg.is_empty());
        assert_eq!(jpeg[0..2], [0xFF, 0xD8]); // JPEG SOI marker
    }
}
