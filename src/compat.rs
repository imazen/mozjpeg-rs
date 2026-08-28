//! C mozjpeg encoding layer.
//!
//! This module provides [`CMozjpeg`], which encodes images using the C mozjpeg
//! library with settings matching our [`Encoder`](crate::Encoder) configuration.
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
//!
//! let pixels: Vec<u8> = vec![128; 64 * 64 * 3];
//! let encoder = Encoder::new(Preset::ProgressiveBalanced).quality(85);
//!
//! // Encode with C mozjpeg using same settings as Rust encoder
//! let c_jpeg = encoder.to_c_mozjpeg().encode_rgb(&pixels, 64, 64)
//!     .expect("C encoding failed");
//! ```

#![allow(unsafe_code)]

use crate::consts::QuantTableIdx;
use crate::error::{Error, Result};
use crate::types::{Subsampling, TrellisConfig};

/// Warnings from configuring a C mozjpeg encoder.
///
/// Some settings cannot be applied to `jpeg_compress_struct` directly
/// and must be handled separately after `jpeg_start_compress`.
///
/// Non-exhaustive: this is an output type, produced by the crate and read by
/// callers. Build one for a test with `ConfigWarnings::default()` and assign
/// the fields you need.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
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
///
/// Non-exhaustive: new variants may be added in a patch release as C mozjpeg
/// gains settings this crate cannot map. Match with a `_` arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

/// C mozjpeg encoder with settings from a Rust [`Encoder`](crate::Encoder).
///
/// Created via [`Encoder::to_c_mozjpeg()`](crate::Encoder::to_c_mozjpeg).
/// Provides methods to encode images using the C mozjpeg library.
///
/// # Example
///
/// ```no_run
/// use mozjpeg_rs::{Encoder, Preset};
///
/// let pixels: Vec<u8> = vec![128; 64 * 64 * 3];
///
/// // Create encoder and convert to C mozjpeg
/// let c_encoder = Encoder::new(Preset::ProgressiveBalanced)
///     .quality(85)
///     .to_c_mozjpeg();
///
/// // Encode using C mozjpeg
/// let jpeg = c_encoder.encode_rgb(&pixels, 64, 64).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CMozjpeg {
    pub(crate) quality: u8,
    pub(crate) force_baseline: bool,
    pub(crate) subsampling: Subsampling,
    pub(crate) progressive: bool,
    pub(crate) optimize_huffman: bool,
    pub(crate) optimize_scans: bool,
    pub(crate) trellis: TrellisConfig,
    pub(crate) overshoot_deringing: bool,
    pub(crate) smoothing: u8,
    pub(crate) restart_interval: u16,
    pub(crate) quant_table_idx: QuantTableIdx,
    pub(crate) has_custom_qtables: bool,
    pub(crate) exif_data: Option<Vec<u8>>,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) custom_markers: Vec<(u8, Vec<u8>)>,
}

impl CMozjpeg {
    /// Configure a C mozjpeg `jpeg_compress_struct` with these settings.
    ///
    /// # Safety
    ///
    /// - `cinfo` must be a valid, initialized `jpeg_compress_struct`
    /// - `jpeg_CreateCompress` must have been called on `cinfo`
    /// - `jpeg_set_defaults` will be called by this method
    pub unsafe fn configure_cinfo(
        &self,
        cinfo: &mut mozjpeg_sys::jpeg_compress_struct,
        width: u32,
        height: u32,
    ) -> std::result::Result<ConfigWarnings, ConfigError> {
        use mozjpeg_sys::*;

        let mut warnings = ConfigWarnings::default();

        // Check for unsupported settings first
        if self.has_custom_qtables {
            return Err(ConfigError::CustomQuantTablesNotSupported);
        }

        // Set image dimensions and colorspace
        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        // Initialize defaults (this sets JCP_MAX_COMPRESSION profile)
        jpeg_set_defaults(cinfo);

        // Set quant table index BEFORE jpeg_set_quality
        let table_idx = self.quant_table_idx as i32;
        jpeg_c_set_int_param(cinfo, J_INT_PARAM::JINT_BASE_QUANT_TBL_IDX, table_idx);

        // Set quality (must come after quant table index)
        jpeg_set_quality(
            cinfo,
            self.quality as i32,
            if self.force_baseline { 1 } else { 0 },
        );

        // Set subsampling factors
        let (h_samp, v_samp) = match self.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => {
                // Grayscale: single component
                cinfo.input_components = 1;
                cinfo.in_color_space = J_COLOR_SPACE::JCS_GRAYSCALE;
                (1, 1)
            }
        };

        if self.subsampling != Subsampling::Gray {
            (*cinfo.comp_info.offset(0)).h_samp_factor = h_samp;
            (*cinfo.comp_info.offset(0)).v_samp_factor = v_samp;
            (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).v_samp_factor = 1;
        }

        // Huffman optimization
        cinfo.optimize_coding = if self.optimize_huffman { 1 } else { 0 };

        // Trellis quantization
        jpeg_c_set_bool_param(
            cinfo,
            J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT,
            if self.trellis.enabled { 1 } else { 0 },
        );
        jpeg_c_set_bool_param(
            cinfo,
            J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT_DC,
            if self.trellis.dc_enabled { 1 } else { 0 },
        );

        // Overshoot deringing
        jpeg_c_set_bool_param(
            cinfo,
            J_BOOLEAN_PARAM::JBOOLEAN_OVERSHOOT_DERINGING,
            if self.overshoot_deringing { 1 } else { 0 },
        );

        // Smoothing factor
        cinfo.smoothing_factor = self.smoothing as i32;

        // Restart interval
        cinfo.restart_interval = self.restart_interval as u32;

        // optimize_scans MUST be set BEFORE jpeg_simple_progression
        jpeg_c_set_bool_param(
            cinfo,
            J_BOOLEAN_PARAM::JBOOLEAN_OPTIMIZE_SCANS,
            if self.optimize_scans { 1 } else { 0 },
        );

        // Progressive mode (must come AFTER optimize_scans)
        if self.progressive {
            jpeg_simple_progression(cinfo);
        } else {
            // Ensure baseline mode
            cinfo.num_scans = 0;
            cinfo.scan_info = std::ptr::null();
        }

        // Check for settings that require post-start handling
        if self.exif_data.is_some() {
            warnings.has_exif = true;
        }
        if self.icc_profile.is_some() {
            warnings.has_icc_profile = true;
        }
        if !self.custom_markers.is_empty() {
            warnings.has_custom_markers = true;
        }

        Ok(warnings)
    }

    /// Encode RGB data using C mozjpeg with this configuration.
    ///
    /// # Arguments
    ///
    /// - `rgb`: RGB pixel data (3 bytes per pixel, row-major order)
    /// - `width`: Image width in pixels
    /// - `height`: Image height in pixels
    pub fn encode_rgb(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        use mozjpeg_sys::*;
        use std::ptr;

        let expected_size = (width as usize) * (height as usize) * 3;
        if rgb.len() != expected_size {
            return Err(Error::BufferSizeMismatch {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

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

            // Configure encoder
            self.configure_cinfo(&mut cinfo, width, height)
                .map_err(|e| {
                    Error::UnsupportedFeature(match e {
                        ConfigError::UnsupportedQuantTable(_) => {
                            "quant table not supported by C mozjpeg"
                        }
                        ConfigError::CustomQuantTablesNotSupported => {
                            "custom quant tables not supported"
                        }
                        ConfigError::UnsupportedSubsampling(_) => "subsampling mode not supported",
                    })
                })?;

            jpeg_start_compress(&mut cinfo, 1);

            // Write EXIF data if present (APP1 = 0xE1)
            if let Some(exif) = &self.exif_data {
                jpeg_write_marker(&mut cinfo, 0xE1, exif.as_ptr(), exif.len() as u32);
            }

            // Write ICC profile if present (APP2 markers with ICC_PROFILE prefix)
            if let Some(icc) = &self.icc_profile {
                self.write_icc_profile(&mut cinfo, icc);
            }

            // Write custom markers
            for (marker_type, data) in &self.custom_markers {
                jpeg_write_marker(
                    &mut cinfo,
                    *marker_type as i32,
                    data.as_ptr(),
                    data.len() as u32,
                );
            }

            // Write scanlines
            let row_stride = width as usize * 3;
            let mut row_pointer: [*const u8; 1] = [ptr::null()];

            while cinfo.next_scanline < cinfo.image_height {
                let row_offset = cinfo.next_scanline as usize * row_stride;
                row_pointer[0] = rgb.as_ptr().add(row_offset);
                jpeg_write_scanlines(&mut cinfo, row_pointer.as_ptr() as *mut *const u8, 1);
            }

            jpeg_finish_compress(&mut cinfo);

            let output_size = outsize as usize;
            let mut result = Vec::with_capacity(output_size);
            if !outbuffer.is_null() && output_size > 0 {
                result.extend_from_slice(std::slice::from_raw_parts(outbuffer, output_size));
            }

            jpeg_destroy_compress(&mut cinfo);

            if !outbuffer.is_null() {
                libc::free(outbuffer as *mut libc::c_void);
            }

            Ok(result)
        }
    }

    /// Encode grayscale data using C mozjpeg with this configuration.
    pub fn encode_grayscale(&self, gray: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        use mozjpeg_sys::*;
        use std::ptr;

        let expected_size = (width as usize) * (height as usize);
        if gray.len() != expected_size {
            return Err(Error::BufferSizeMismatch {
                expected: expected_size,
                actual: gray.len(),
            });
        }

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

            // Set grayscale mode
            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = 1;
            cinfo.in_color_space = J_COLOR_SPACE::JCS_GRAYSCALE;

            jpeg_set_defaults(&mut cinfo);

            jpeg_set_quality(
                &mut cinfo,
                self.quality as i32,
                if self.force_baseline { 1 } else { 0 },
            );

            cinfo.optimize_coding = if self.optimize_huffman { 1 } else { 0 };

            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT,
                if self.trellis.enabled { 1 } else { 0 },
            );
            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT_DC,
                if self.trellis.dc_enabled { 1 } else { 0 },
            );
            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_OVERSHOOT_DERINGING,
                if self.overshoot_deringing { 1 } else { 0 },
            );

            cinfo.smoothing_factor = self.smoothing as i32;

            // optimize_scans MUST be set BEFORE jpeg_simple_progression
            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_OPTIMIZE_SCANS,
                if self.optimize_scans { 1 } else { 0 },
            );

            // Progressive mode (must come AFTER optimize_scans)
            if self.progressive {
                jpeg_simple_progression(&mut cinfo);
            } else {
                // Ensure baseline mode - clear any scan script
                cinfo.num_scans = 0;
                cinfo.scan_info = std::ptr::null();
            }

            jpeg_start_compress(&mut cinfo, 1);

            let row_stride = width as usize;
            let mut row_pointer: [*const u8; 1] = [ptr::null()];

            while cinfo.next_scanline < cinfo.image_height {
                let row_offset = cinfo.next_scanline as usize * row_stride;
                row_pointer[0] = gray.as_ptr().add(row_offset);
                jpeg_write_scanlines(&mut cinfo, row_pointer.as_ptr() as *mut *const u8, 1);
            }

            jpeg_finish_compress(&mut cinfo);

            let output_size = outsize as usize;
            let mut result = Vec::with_capacity(output_size);
            if !outbuffer.is_null() && output_size > 0 {
                result.extend_from_slice(std::slice::from_raw_parts(outbuffer, output_size));
            }

            jpeg_destroy_compress(&mut cinfo);

            if !outbuffer.is_null() {
                libc::free(outbuffer as *mut libc::c_void);
            }

            Ok(result)
        }
    }

    /// Encode planar YCbCr data using C mozjpeg.
    ///
    /// Takes pre-separated Y, Cb, Cr planes where chroma planes are already
    /// subsampled according to the encoder's subsampling setting.
    ///
    /// # Arguments
    ///
    /// - `y`: Luma plane (width × height bytes)
    /// - `cb`: Cb chroma plane (subsampled according to encoder settings)
    /// - `cr`: Cr chroma plane (subsampled according to encoder settings)
    /// - `width`: Image width in pixels
    /// - `height`: Image height in pixels
    ///
    /// # Plane Sizes
    ///
    /// | Subsampling | Y size | Cb/Cr size |
    /// |-------------|--------|------------|
    /// | 4:4:4 | w × h | w × h |
    /// | 4:2:2 | w × h | (w/2) × h |
    /// | 4:2:0 | w × h | (w/2) × (h/2) |
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mozjpeg_rs::{Encoder, Preset};
    ///
    /// let width = 640;
    /// let height = 480;
    ///
    /// // Pre-subsampled 4:2:0 planes
    /// let y = vec![128u8; width * height];
    /// let cb = vec![128u8; (width / 2) * (height / 2)];
    /// let cr = vec![128u8; (width / 2) * (height / 2)];
    ///
    /// let jpeg = Encoder::new(Preset::BaselineBalanced)
    ///     .to_c_mozjpeg()
    ///     .encode_ycbcr_planar(&y, &cb, &cr, width as u32, height as u32)?;
    /// # Ok::<(), mozjpeg_rs::Error>(())
    /// ```
    pub fn encode_ycbcr_planar(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        use mozjpeg_sys::*;
        use std::ptr;

        // Calculate expected plane sizes based on subsampling
        let (h_factor, v_factor) = match self.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
            Subsampling::Gray => {
                return Err(Error::UnsupportedFeature(
                    "use encode_grayscale for grayscale images",
                ));
            }
        };

        let y_size = (width as usize) * (height as usize);
        let chroma_width = (width as usize + h_factor - 1) / h_factor;
        let chroma_height = (height as usize + v_factor - 1) / v_factor;
        let chroma_size = chroma_width * chroma_height;

        if y.len() != y_size {
            return Err(Error::BufferSizeMismatch {
                expected: y_size,
                actual: y.len(),
            });
        }
        if cb.len() != chroma_size {
            return Err(Error::BufferSizeMismatch {
                expected: chroma_size,
                actual: cb.len(),
            });
        }
        if cr.len() != chroma_size {
            return Err(Error::BufferSizeMismatch {
                expected: chroma_size,
                actual: cr.len(),
            });
        }

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

            // Set up for YCbCr input
            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = 3;
            cinfo.in_color_space = J_COLOR_SPACE::JCS_YCbCr;

            jpeg_set_defaults(&mut cinfo);

            // Set quant table
            let table_idx = self.quant_table_idx as i32;
            jpeg_c_set_int_param(&mut cinfo, J_INT_PARAM::JINT_BASE_QUANT_TBL_IDX, table_idx);

            jpeg_set_quality(
                &mut cinfo,
                self.quality as i32,
                if self.force_baseline { 1 } else { 0 },
            );

            // Set subsampling factors
            (*cinfo.comp_info.offset(0)).h_samp_factor = h_factor as i32;
            (*cinfo.comp_info.offset(0)).v_samp_factor = v_factor as i32;
            (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
            (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

            // Enable raw data input (planar YCbCr)
            cinfo.raw_data_in = 1;

            // Other settings
            cinfo.optimize_coding = if self.optimize_huffman { 1 } else { 0 };

            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT,
                if self.trellis.enabled { 1 } else { 0 },
            );
            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_TRELLIS_QUANT_DC,
                if self.trellis.dc_enabled { 1 } else { 0 },
            );
            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_OVERSHOOT_DERINGING,
                if self.overshoot_deringing { 1 } else { 0 },
            );

            cinfo.smoothing_factor = self.smoothing as i32;

            jpeg_c_set_bool_param(
                &mut cinfo,
                J_BOOLEAN_PARAM::JBOOLEAN_OPTIMIZE_SCANS,
                if self.optimize_scans { 1 } else { 0 },
            );

            if self.progressive {
                jpeg_simple_progression(&mut cinfo);
            } else {
                cinfo.num_scans = 0;
                cinfo.scan_info = std::ptr::null();
            }

            jpeg_start_compress(&mut cinfo, 1);

            // Write markers
            if let Some(exif) = &self.exif_data {
                jpeg_write_marker(&mut cinfo, 0xE1, exif.as_ptr(), exif.len() as u32);
            }
            if let Some(icc) = &self.icc_profile {
                self.write_icc_profile(&mut cinfo, icc);
            }
            for (marker_type, data) in &self.custom_markers {
                jpeg_write_marker(
                    &mut cinfo,
                    *marker_type as i32,
                    data.as_ptr(),
                    data.len() as u32,
                );
            }

            // Write raw data in MCU rows
            // For raw_data_in, we must provide data in units of max_v_samp_factor * DCTSIZE rows
            let mcu_rows = cinfo.max_v_samp_factor as usize * DCTSIZE as usize;

            // Allocate row pointer arrays
            let mut y_rows: Vec<*const u8> = vec![ptr::null(); mcu_rows];
            let mut cb_rows: Vec<*const u8> = vec![ptr::null(); mcu_rows];
            let mut cr_rows: Vec<*const u8> = vec![ptr::null(); mcu_rows];

            let mut row = 0usize;
            while row < height as usize {
                // Set up Y row pointers
                for i in 0..mcu_rows {
                    let src_row = (row + i).min(height as usize - 1);
                    y_rows[i] = y.as_ptr().add(src_row * width as usize);
                }

                // Set up Cb/Cr row pointers (subsampled)
                let chroma_row = row / v_factor;
                for i in 0..mcu_rows {
                    let src_row = (chroma_row + i / v_factor).min(chroma_height - 1);
                    cb_rows[i] = cb.as_ptr().add(src_row * chroma_width);
                    cr_rows[i] = cr.as_ptr().add(src_row * chroma_width);
                }

                // Build component pointer array for this batch
                let comp_ptrs: [*const *const u8; 3] =
                    [y_rows.as_ptr(), cb_rows.as_ptr(), cr_rows.as_ptr()];

                jpeg_write_raw_data(&mut cinfo, comp_ptrs.as_ptr(), mcu_rows as u32);

                row += mcu_rows;
            }

            jpeg_finish_compress(&mut cinfo);

            let output_size = outsize as usize;
            let mut result = Vec::with_capacity(output_size);
            if !outbuffer.is_null() && output_size > 0 {
                result.extend_from_slice(std::slice::from_raw_parts(outbuffer, output_size));
            }

            jpeg_destroy_compress(&mut cinfo);

            if !outbuffer.is_null() {
                libc::free(outbuffer as *mut libc::c_void);
            }

            Ok(result)
        }
    }

    /// Write ICC profile as APP2 markers with proper chunking.
    unsafe fn write_icc_profile(&self, cinfo: &mut mozjpeg_sys::jpeg_compress_struct, icc: &[u8]) {
        use mozjpeg_sys::jpeg_write_marker;

        const ICC_OVERHEAD: usize = 14; // "ICC_PROFILE\0" + seq_no + num_markers
        const MAX_DATA_PER_MARKER: usize = 65533 - ICC_OVERHEAD;

        let chunks: Vec<_> = icc.chunks(MAX_DATA_PER_MARKER).collect();
        let num_markers = chunks.len() as u8;

        for (i, chunk) in chunks.iter().enumerate() {
            let seq_no = (i + 1) as u8;
            let mut marker_data = Vec::with_capacity(ICC_OVERHEAD + chunk.len());
            marker_data.extend_from_slice(b"ICC_PROFILE\0");
            marker_data.push(seq_no);
            marker_data.push(num_markers);
            marker_data.extend_from_slice(chunk);

            // APP2 = 0xE2
            jpeg_write_marker(cinfo, 0xE2, marker_data.as_ptr(), marker_data.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Encoder, Preset};

    #[test]
    fn test_c_mozjpeg_encode_rgb() {
        let jpeg = Encoder::new(Preset::BaselineBalanced)
            .quality(75)
            .to_c_mozjpeg()
            .encode_rgb(&vec![128u8; 64 * 64 * 3], 64, 64)
            .expect("encoding failed");

        // Verify it's a valid JPEG
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // SOI marker
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]); // EOI marker
    }

    #[test]
    fn test_c_mozjpeg_encode_grayscale() {
        let jpeg = Encoder::new(Preset::BaselineBalanced)
            .quality(75)
            .to_c_mozjpeg()
            .encode_grayscale(&vec![128u8; 64 * 64], 64, 64)
            .expect("encoding failed");

        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn test_rust_vs_c_parity() {
        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let c_mozjpeg = encoder.to_c_mozjpeg();

        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

        let rust_jpeg = encoder
            .encode_rgb(&pixels, 64, 64)
            .expect("Rust encoding failed");
        let c_jpeg = c_mozjpeg
            .encode_rgb(&pixels, 64, 64)
            .expect("C encoding failed");

        // File sizes should be within 5% of each other
        let size_diff = (rust_jpeg.len() as f64 - c_jpeg.len() as f64).abs();
        let max_diff = c_jpeg.len() as f64 * 0.05;
        assert!(
            size_diff < max_diff,
            "File size difference too large: Rust={}, C={}, diff={}",
            rust_jpeg.len(),
            c_jpeg.len(),
            size_diff
        );
    }

    #[test]
    fn test_c_mozjpeg_encode_ycbcr_planar_420() {
        use crate::Subsampling;

        let width = 64usize;
        let height = 64usize;

        // Create planar 4:2:0 YCbCr data
        let y: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let cb: Vec<u8> = vec![128u8; (width / 2) * (height / 2)];
        let cr: Vec<u8> = vec![128u8; (width / 2) * (height / 2)];

        let jpeg = Encoder::new(Preset::BaselineBalanced)
            .quality(75)
            .subsampling(Subsampling::S420)
            .to_c_mozjpeg()
            .encode_ycbcr_planar(&y, &cb, &cr, width as u32, height as u32)
            .expect("encoding failed");

        // Verify it's a valid JPEG
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // SOI marker
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]); // EOI marker

        // Verify it decodes
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        let pixels = decoder.decode().expect("decode failed");
        assert_eq!(pixels.len(), width * height * 3);
    }

    #[test]
    fn test_c_mozjpeg_encode_ycbcr_planar_444() {
        use crate::Subsampling;

        let width = 64usize;
        let height = 64usize;

        // Create planar 4:4:4 YCbCr data (no subsampling)
        let y: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let cb: Vec<u8> = vec![128u8; width * height];
        let cr: Vec<u8> = vec![128u8; width * height];

        let jpeg = Encoder::new(Preset::BaselineBalanced)
            .quality(75)
            .subsampling(Subsampling::S444)
            .to_c_mozjpeg()
            .encode_ycbcr_planar(&y, &cb, &cr, width as u32, height as u32)
            .expect("encoding failed");

        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }
}
