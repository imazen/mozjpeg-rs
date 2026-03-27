//! zencodec trait implementations for mozjpeg-rs.
//!
//! Provides [`MozjpegEncoderConfig`] implementing the encode trait hierarchy
//! from zencodec, wrapping the native mozjpeg-rs [`Encoder`](crate::Encoder) API.
//!
//! This is a thin adapter layer — the native API remains untouched.
//!
//! # Trait mapping
//!
//! | zencodec | mozjpeg-rs adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`MozjpegEncoderConfig`] |
//! | `EncodeJob` | [`MozjpegEncodeJob`] |
//! | `Encoder` | [`MozjpegEncoder`] |
//! | `AnimationFrameEncoder` | `()` (JPEG has no animation) |

use zencodec::encode::{EncodeCapabilities, EncodeOutput, EncodePolicy};
use zencodec::{ImageFormat, Metadata, ResourceLimits, StopToken, UnsupportedOperation};
use zenpixels::{PixelDescriptor, PixelSlice};

use crate::error::Error;
use crate::types::{Preset, Subsampling};

// ============================================================================
// Capabilities
// ============================================================================

/// JPEG encode capabilities.
static MOZJPEG_ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_stop(true)
    .with_lossy(true)
    .with_push_rows(true)
    .with_native_gray(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true)
    .with_quality_range(1.0, 100.0)
    .with_effort_range(0, 3);

/// Supported encode pixel formats.
static ENCODE_DESCRIPTORS: &[PixelDescriptor] =
    &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB];

// ============================================================================
// EncoderConfig
// ============================================================================

/// mozjpeg-rs encoder configuration implementing [`zencodec::encode::EncoderConfig`].
///
/// Wraps [`crate::Encoder`] with the zencodec trait interface.
/// Defaults to `Preset::ProgressiveBalanced` at quality 85.
///
/// # Effort mapping
///
/// | Effort | Preset |
/// |--------|--------|
/// | 0 | `BaselineFastest` |
/// | 1 | `BaselineBalanced` |
/// | 2 | `ProgressiveBalanced` (default) |
/// | 3 | `ProgressiveSmallest` |
#[derive(Clone, Debug)]
pub struct MozjpegEncoderConfig {
    quality: u8,
    effort: i32,
    subsampling: Subsampling,
    /// Original generic quality value passed to `with_generic_quality()`.
    generic_quality_input: Option<f32>,
}

impl MozjpegEncoderConfig {
    /// Create a default config at quality 85, effort 2 (progressive balanced).
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality: 85,
            effort: 2,
            subsampling: Subsampling::S420,
            generic_quality_input: None,
        }
    }

    /// Set chroma subsampling mode.
    #[must_use]
    pub fn with_subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Build the native [`crate::Encoder`] from current settings.
    fn to_encoder(&self) -> crate::Encoder {
        let preset = match self.effort {
            0 => Preset::BaselineFastest,
            1 => Preset::BaselineBalanced,
            3 => Preset::ProgressiveSmallest,
            _ => Preset::ProgressiveBalanced,
        };
        crate::Encoder::new(preset)
            .quality(self.quality)
            .subsampling(self.subsampling)
    }
}

impl Default for MozjpegEncoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl zencodec::encode::EncoderConfig for MozjpegEncoderConfig {
    type Error = Error;
    type Job = MozjpegEncodeJob;

    fn format() -> ImageFormat {
        ImageFormat::Jpeg
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static EncodeCapabilities {
        &MOZJPEG_ENCODE_CAPS
    }

    fn with_generic_quality(mut self, quality: f32) -> Self {
        let clamped = quality.clamp(1.0, 100.0);
        self.generic_quality_input = Some(clamped);
        self.quality = clamped.round() as u8;
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        Some(self.generic_quality_input.unwrap_or(self.quality as f32))
    }

    fn with_generic_effort(mut self, effort: i32) -> Self {
        self.effort = effort.clamp(0, 3);
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        Some(self.effort)
    }

    fn job(self) -> Self::Job {
        MozjpegEncodeJob {
            config: self,
            stop: None,
            metadata: None,
            limits: ResourceLimits::none(),
            policy: None,
            image_size: None,
        }
    }
}

// ============================================================================
// EncodeJob
// ============================================================================

/// Per-operation mozjpeg encode job.
///
/// Created by [`MozjpegEncoderConfig::job()`]. Consumed by creating a
/// [`MozjpegEncoder`].
pub struct MozjpegEncodeJob {
    config: MozjpegEncoderConfig,
    stop: Option<StopToken>,
    metadata: Option<Metadata>,
    limits: ResourceLimits,
    policy: Option<EncodePolicy>,
    image_size: Option<(u32, u32)>,
}

impl zencodec::encode::EncodeJob for MozjpegEncodeJob {
    type Error = Error;
    type Enc = MozjpegEncoder;
    type AnimationFrameEnc = ();

    fn with_stop(mut self, stop: StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: Metadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_policy(mut self, policy: EncodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn with_canvas_size(mut self, width: u32, height: u32) -> Self {
        self.image_size = Some((width, height));
        self
    }

    fn encoder(self) -> Result<MozjpegEncoder, Self::Error> {
        Ok(MozjpegEncoder {
            inner: self.config.to_encoder(),
            stop: self.stop,
            metadata: self.metadata,
            limits: self.limits,
            policy: self.policy,
            _image_size: self.image_size,
            accumulator: None,
        })
    }

    fn animation_frame_encoder(self) -> Result<Self::AnimationFrameEnc, Self::Error> {
        Err(Error::UnsupportedFeature("JPEG does not support animation"))
    }
}

// ============================================================================
// Encoder
// ============================================================================

/// Single-image mozjpeg encoder implementing [`zencodec::encode::Encoder`].
///
/// Supports one-shot `encode()` and row-accumulation `push_rows()` + `finish()`.
pub struct MozjpegEncoder {
    inner: crate::Encoder,
    stop: Option<StopToken>,
    metadata: Option<Metadata>,
    limits: ResourceLimits,
    policy: Option<EncodePolicy>,
    _image_size: Option<(u32, u32)>,
    accumulator: Option<RowAccumulator>,
}

/// Internal buffer for accumulating pushed rows.
struct RowAccumulator {
    data: Vec<u8>,
    width: u32,
    total_rows: u32,
    is_gray: bool,
}

impl MozjpegEncoder {
    /// Apply metadata, limits, and policy to the inner encoder.
    fn prepare_encoder(&self) -> crate::Encoder {
        let mut enc = self.inner.clone();

        // Apply resource limits
        let mut native_limits = crate::Limits::none();
        if let Some(max_px) = self.limits.max_pixels {
            native_limits = native_limits.max_pixel_count(max_px);
        }
        if let Some(max_mem) = self.limits.max_memory_bytes {
            native_limits = native_limits.max_alloc_bytes(max_mem as usize);
        }
        if let Some(max_w) = self.limits.max_width {
            native_limits = native_limits.max_width(max_w);
        }
        if let Some(max_h) = self.limits.max_height {
            native_limits = native_limits.max_height(max_h);
        }
        if native_limits.has_limits() {
            enc = enc.limits(native_limits);
        }

        // Apply metadata per policy
        if let Some(ref meta) = self.metadata {
            let policy = self.policy.unwrap_or(EncodePolicy::none());
            if policy.resolve_icc(true)
                && let Some(ref icc) = meta.icc_profile
            {
                enc = enc.icc_profile(icc.to_vec());
            }
            if policy.resolve_exif(true)
                && let Some(ref exif) = meta.exif
            {
                enc = enc.exif_data(exif.to_vec());
            }
            if policy.resolve_xmp(true)
                && let Some(ref xmp) = meta.xmp
            {
                // XMP is stored in APP1 with "http://ns.adobe.com/xap/1.0/\0" prefix
                let namespace = b"http://ns.adobe.com/xap/1.0/\0";
                let mut marker_data = Vec::with_capacity(namespace.len() + xmp.len());
                marker_data.extend_from_slice(namespace);
                marker_data.extend_from_slice(xmp);
                enc = enc.add_marker(1, marker_data);
            }
        }

        enc
    }

    /// Get a reference to the stop token for passing to encode methods.
    fn stop_ref(&self) -> &dyn enough::Stop {
        match self.stop {
            Some(ref s) => s,
            None => &enough::Unstoppable,
        }
    }

    /// Encode RGB or grayscale data using the prepared encoder.
    fn encode_data(
        &self,
        enc: &crate::Encoder,
        data: &[u8],
        width: u32,
        height: u32,
        is_gray: bool,
    ) -> Result<Vec<u8>, Error> {
        let stop = self.stop_ref();
        if is_gray {
            enc.encode_gray_with_stop(data, width, height, stop)
        } else {
            enc.encode_rgb_with_stop(data, width, height, stop)
        }
    }
}

impl zencodec::encode::Encoder for MozjpegEncoder {
    type Error = Error;

    fn reject(op: UnsupportedOperation) -> Self::Error {
        Error::UnsupportedFeature(match op {
            UnsupportedOperation::AnimationEncode => "JPEG does not support animation",
            UnsupportedOperation::PullEncode => "pull-based encoding not supported",
            _ => "unsupported operation",
        })
    }

    fn preferred_strip_height(&self) -> u32 {
        16 // MCU height for 4:2:0
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
        let enc = self.prepare_encoder();
        let width = pixels.width();
        let height = pixels.rows();
        let data = pixels.contiguous_bytes();
        let is_gray = pixels.descriptor() == PixelDescriptor::GRAY8_SRGB;

        let jpeg_data = self.encode_data(&enc, &data, width, height, is_gray)?;
        Ok(EncodeOutput::new(jpeg_data, ImageFormat::Jpeg))
    }

    fn encode_srgba8(
        self,
        data: &mut [u8],
        _make_opaque: bool,
        width: u32,
        height: u32,
        stride_pixels: u32,
    ) -> Result<EncodeOutput, Error> {
        // Strip alpha: extract RGB from RGBA
        let stride_bytes = stride_pixels as usize * 4;
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height as usize {
            let row_start = y * stride_bytes;
            for x in 0..width as usize {
                let px = row_start + x * 4;
                rgb.push(data[px]);
                rgb.push(data[px + 1]);
                rgb.push(data[px + 2]);
            }
        }

        let enc = self.prepare_encoder();
        let jpeg_data = self.encode_data(&enc, &rgb, width, height, false)?;
        Ok(EncodeOutput::new(jpeg_data, ImageFormat::Jpeg))
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Error> {
        let width = rows.width();
        let is_gray = rows.descriptor() == PixelDescriptor::GRAY8_SRGB;
        let data = rows.contiguous_bytes();

        match &mut self.accumulator {
            None => {
                let mut buf = Vec::new();
                let estimated = data.len() * 4;
                buf.try_reserve(estimated)?;
                buf.extend_from_slice(&data);
                self.accumulator = Some(RowAccumulator {
                    data: buf,
                    width,
                    total_rows: rows.rows(),
                    is_gray,
                });
            }
            Some(acc) => {
                if acc.width != width || acc.is_gray != is_gray {
                    return Err(Error::UnsupportedFeature(
                        "push_rows: width or format changed between calls",
                    ));
                }
                acc.data.extend_from_slice(&data);
                acc.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<EncodeOutput, Error> {
        let enc = self.prepare_encoder();

        let acc = self.accumulator.ok_or(Error::UnsupportedFeature(
            "finish() called without any push_rows()",
        ))?;

        // Use stop token or Unstoppable — must resolve after moving accumulator
        // since stop_ref() borrows self.
        let stop: &dyn enough::Stop = match &self.stop {
            Some(s) => s,
            None => &enough::Unstoppable,
        };

        let jpeg_data = if acc.is_gray {
            enc.encode_gray_with_stop(&acc.data, acc.width, acc.total_rows, stop)?
        } else {
            enc.encode_rgb_with_stop(&acc.data, acc.width, acc.total_rows, stop)?
        };
        Ok(EncodeOutput::new(jpeg_data, ImageFormat::Jpeg))
    }
}

// ============================================================================
// Error interop
// ============================================================================

impl From<UnsupportedOperation> for Error {
    fn from(op: UnsupportedOperation) -> Self {
        Error::UnsupportedFeature(match op {
            UnsupportedOperation::AnimationEncode => "JPEG does not support animation",
            UnsupportedOperation::RowLevelEncode => "row-level encoding not supported",
            UnsupportedOperation::PullEncode => "pull-based encoding not supported",
            _ => "unsupported operation",
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};

    fn test_pixels_rgb(width: u32, height: u32) -> Vec<u8> {
        vec![128u8; (width * height * 3) as usize]
    }

    #[test]
    fn default_roundtrip() {
        let pixels = test_pixels_rgb(64, 64);
        let config = MozjpegEncoderConfig::new();
        let slice = PixelSlice::new(&pixels, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();

        let output = config.job().encoder().unwrap().encode(slice).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        assert_eq!(&output.data()[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn quality_and_effort() {
        let config = MozjpegEncoderConfig::new()
            .with_generic_quality(50.0)
            .with_generic_effort(0);

        assert_eq!(config.generic_quality(), Some(50.0));
        assert_eq!(config.generic_effort(), Some(0));
        assert_eq!(config.quality, 50);
        assert_eq!(config.effort, 0);
    }

    #[test]
    fn effort_clamping() {
        let config = MozjpegEncoderConfig::new().with_generic_effort(99);
        assert_eq!(config.effort, 3);

        let config = MozjpegEncoderConfig::new().with_generic_effort(-5);
        assert_eq!(config.effort, 0);
    }

    #[test]
    fn with_metadata() {
        let pixels = test_pixels_rgb(32, 32);
        let config = MozjpegEncoderConfig::new();
        let meta = Metadata::default().with_icc(vec![0u8; 100]);
        let slice = PixelSlice::new(&pixels, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();

        let output = config
            .job()
            .with_metadata(meta)
            .encoder()
            .unwrap()
            .encode(slice)
            .unwrap();
        assert!(!output.data().is_empty());
    }

    #[test]
    fn grayscale_encode() {
        let pixels = vec![128u8; 32 * 32];
        let config = MozjpegEncoderConfig::new();
        let slice = PixelSlice::new(&pixels, 32, 32, 32, PixelDescriptor::GRAY8_SRGB).unwrap();

        let output = config.job().encoder().unwrap().encode(slice).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(&output.data()[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn push_rows_encode() {
        let pixels = test_pixels_rgb(32, 32);
        let config = MozjpegEncoderConfig::new().with_generic_effort(0);

        let mut encoder = config.job().with_canvas_size(32, 32).encoder().unwrap();

        for chunk_start in (0..32u32).step_by(16) {
            let rows = 16u32.min(32 - chunk_start);
            let start = chunk_start as usize * 32 * 3;
            let end = start + rows as usize * 32 * 3;
            let slice = PixelSlice::new(
                &pixels[start..end],
                32,
                rows,
                32 * 3,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            encoder.push_rows(slice).unwrap();
        }

        let output = encoder.finish().unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(&output.data()[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn capabilities() {
        let caps = MozjpegEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.lossy());
        assert!(!caps.lossless());
        assert!(!caps.animation());
        assert!(caps.native_gray());
        assert!(caps.push_rows());
    }

    #[test]
    fn animation_returns_unsupported() {
        let config = MozjpegEncoderConfig::new();
        let result = config.job().animation_frame_encoder();
        assert!(result.is_err());
    }

    #[test]
    fn strip_metadata_with_policy() {
        let pixels = test_pixels_rgb(32, 32);
        let config = MozjpegEncoderConfig::new();
        let meta = Metadata::default()
            .with_icc(vec![0u8; 100])
            .with_exif(vec![0u8; 50]);

        let slice = PixelSlice::new(&pixels, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();

        let output_stripped = config
            .clone()
            .job()
            .with_metadata(meta.clone())
            .with_policy(EncodePolicy::strip_all())
            .encoder()
            .unwrap()
            .encode(slice)
            .unwrap();

        let slice2 = PixelSlice::new(&pixels, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();

        let output_with_meta = config
            .job()
            .with_metadata(meta)
            .encoder()
            .unwrap()
            .encode(slice2)
            .unwrap();

        // The version with metadata should be larger
        assert!(output_with_meta.data().len() > output_stripped.data().len());
    }
}
