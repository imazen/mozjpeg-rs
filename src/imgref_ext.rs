//! Integration with the `imgref` crate for type-safe pixel formats.
//!
//! This module provides [`Encoder`] methods that accept [`imgref::ImgRef`] directly,
//! giving you:
//! - Type-safe pixel formats (compile-time RGB vs RGBA vs Grayscale distinction)
//! - Automatic stride handling (subimages work without copying)
//! - No dimension mix-ups (width/height baked into the type)
//!
//! # Example
//!
//! ```ignore
//! use mozjpeg_rs::{Encoder, Preset};
//! use imgref::ImgVec;
//! use rgb::RGB8;
//!
//! // Create an image buffer
//! let pixels: Vec<RGB8> = vec![RGB8::new(128, 64, 32); 640 * 480];
//! let img = ImgVec::new(pixels, 640, 480);
//!
//! // Encode with automatic format detection
//! let encoder = Encoder::new(Preset::default()).quality(85);
//! let jpeg = encoder.encode_imgref(img.as_ref())?;
//! # Ok::<(), mozjpeg_rs::Error>(())
//! ```

use crate::encode::{try_alloc_vec, Encoder};
use crate::error::Result;
use imgref::ImgRef;
use rgb::{Gray, RGB, RGBA};

/// Trait for pixel types that can be encoded to JPEG.
///
/// This trait is implemented for common pixel types from the `rgb` crate.
/// The encoder will convert RGBA to RGB (discarding alpha) and encode
/// Gray as grayscale JPEG.
pub trait EncodeablePixel: Copy {
    /// Whether this is a grayscale format
    const IS_GRAYSCALE: bool;

    /// Extract grayscale value (for Gray types)
    fn to_gray(&self) -> u8;

    /// Extract RGB values (for color types)
    fn to_rgb(&self) -> (u8, u8, u8);
}

impl EncodeablePixel for RGB<u8> {
    const IS_GRAYSCALE: bool = false;

    #[inline]
    fn to_gray(&self) -> u8 {
        // Standard grayscale conversion (BT.601)
        ((self.r as u32 * 77 + self.g as u32 * 150 + self.b as u32 * 29) >> 8) as u8
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }
}

impl EncodeablePixel for RGBA<u8> {
    const IS_GRAYSCALE: bool = false;

    #[inline]
    fn to_gray(&self) -> u8 {
        ((self.r as u32 * 77 + self.g as u32 * 150 + self.b as u32 * 29) >> 8) as u8
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }
}

impl EncodeablePixel for Gray<u8> {
    const IS_GRAYSCALE: bool = true;

    #[inline]
    fn to_gray(&self) -> u8 {
        *self.as_ref()
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        let v = *self.as_ref();
        (v, v, v)
    }
}

impl EncodeablePixel for [u8; 3] {
    const IS_GRAYSCALE: bool = false;

    #[inline]
    fn to_gray(&self) -> u8 {
        ((self[0] as u32 * 77 + self[1] as u32 * 150 + self[2] as u32 * 29) >> 8) as u8
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        (self[0], self[1], self[2])
    }
}

impl EncodeablePixel for [u8; 4] {
    const IS_GRAYSCALE: bool = false;

    #[inline]
    fn to_gray(&self) -> u8 {
        ((self[0] as u32 * 77 + self[1] as u32 * 150 + self[2] as u32 * 29) >> 8) as u8
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        (self[0], self[1], self[2])
    }
}

impl EncodeablePixel for u8 {
    const IS_GRAYSCALE: bool = true;

    #[inline]
    fn to_gray(&self) -> u8 {
        *self
    }

    #[inline]
    fn to_rgb(&self) -> (u8, u8, u8) {
        (*self, *self, *self)
    }
}

impl Encoder {
    /// Encode an image from an [`ImgRef`] with automatic format detection.
    ///
    /// This method provides type-safe encoding with automatic stride handling.
    /// The pixel type determines the encoding format:
    /// - `RGB<u8>` or `[u8; 3]` → Color JPEG
    /// - `RGBA<u8>` or `[u8; 4]` → Color JPEG (alpha discarded)
    /// - `Gray<u8>` or `u8` → Grayscale JPEG
    ///
    /// # Arguments
    /// * `img` - Image reference with dimensions and optional stride
    ///
    /// # Returns
    /// JPEG-encoded data as a `Vec<u8>`.
    ///
    /// # Example
    /// ```ignore
    /// use mozjpeg_rs::{Encoder, Preset};
    /// use imgref::ImgVec;
    /// use rgb::RGB8;
    ///
    /// let pixels: Vec<RGB8> = vec![RGB8::new(128, 64, 32); 100 * 100];
    /// let img = ImgVec::new(pixels, 100, 100);
    ///
    /// let jpeg = Encoder::new(Preset::default())
    ///     .quality(85)
    ///     .encode_imgref(img.as_ref())?;
    /// # Ok::<(), mozjpeg_rs::Error>(())
    /// ```
    pub fn encode_imgref<P: EncodeablePixel>(&self, img: ImgRef<'_, P>) -> Result<Vec<u8>> {
        let width = img.width() as u32;
        let height = img.height() as u32;

        if P::IS_GRAYSCALE {
            // Grayscale path - extract gray values
            let mut gray_data = try_alloc_vec(0u8, (width * height) as usize)?;
            for (y, row) in img.rows().enumerate() {
                for (x, pixel) in row.iter().enumerate() {
                    gray_data[y * width as usize + x] = pixel.to_gray();
                }
            }
            self.encode_gray(&gray_data, width, height)
        } else {
            // Color path - extract RGB values
            let mut rgb_data = try_alloc_vec(0u8, (width * height * 3) as usize)?;
            for (y, row) in img.rows().enumerate() {
                for (x, pixel) in row.iter().enumerate() {
                    let (r, g, b) = pixel.to_rgb();
                    let i = (y * width as usize + x) * 3;
                    rgb_data[i] = r;
                    rgb_data[i + 1] = g;
                    rgb_data[i + 2] = b;
                }
            }
            self.encode_rgb(&rgb_data, width, height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Preset;
    use imgref::ImgVec;
    use rgb::{Gray, RGB8, RGBA8};

    #[test]
    fn test_encode_imgref_rgb8() {
        let pixels: Vec<RGB8> = (0..64 * 48)
            .map(|i| {
                RGB8::new(
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                )
            })
            .collect();
        let img = ImgVec::new(pixels, 64, 48);

        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg = encoder.encode_imgref(img.as_ref()).unwrap();

        // Verify we got valid JPEG
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // SOI marker

        // Verify it decodes
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        let decoded = decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 48);
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
        assert_eq!(decoded.len(), 64 * 48 * 3);
    }

    #[test]
    fn test_encode_imgref_rgba8() {
        let pixels: Vec<RGBA8> = (0..64 * 48)
            .map(|i| {
                RGBA8::new(
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                    255,
                )
            })
            .collect();
        let img = ImgVec::new(pixels, 64, 48);

        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg = encoder.encode_imgref(img.as_ref()).unwrap();

        // Verify we got valid JPEG
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);

        // Verify it decodes as RGB (not RGBA)
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
    }

    #[test]
    fn test_encode_imgref_gray() {
        let pixels: Vec<Gray<u8>> = (0..64 * 48).map(|i| Gray::new((i % 256) as u8)).collect();
        let img = ImgVec::new(pixels, 64, 48);

        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg = encoder.encode_imgref(img.as_ref()).unwrap();

        // Verify we got valid JPEG
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);

        // Verify it decodes as grayscale
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
    }

    #[test]
    fn test_encode_imgref_u8_slice() {
        let pixels: Vec<u8> = (0..64 * 48).map(|i| (i % 256) as u8).collect();
        let img = ImgVec::new(pixels, 64, 48);

        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg = encoder.encode_imgref(img.as_ref()).unwrap();

        // Verify we got valid grayscale JPEG
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
    }

    #[test]
    fn test_encode_imgref_with_stride() {
        // Create a larger buffer and take a subimage
        let full_width = 128;
        let full_height = 96;
        let mut full_pixels: Vec<RGB8> = vec![RGB8::new(0, 0, 0); full_width * full_height];

        // Fill a 64x48 region starting at (16, 8) with test pattern
        for y in 0..48 {
            for x in 0..64 {
                let i = (y + 8) * full_width + (x + 16);
                full_pixels[i] = RGB8::new((x * 4) as u8, (y * 5) as u8, 128);
            }
        }

        let full_img = ImgVec::new(full_pixels, full_width, full_height);
        let sub_img = full_img.sub_image(16, 8, 64, 48);

        // Encode the subimage (which has stride != width)
        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg = encoder.encode_imgref(sub_img).unwrap();

        // Verify dimensions
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder.decode().unwrap();
        let info = decoder.info().unwrap();
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 48);
    }

    #[test]
    fn test_encode_imgref_matches_encode_rgb() {
        let pixels: Vec<RGB8> = (0..64 * 48)
            .map(|i| {
                RGB8::new(
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                )
            })
            .collect();
        let img = ImgVec::new(pixels.clone(), 64, 48);

        // Convert to raw bytes for encode_rgb
        let rgb_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();

        let encoder = Encoder::new(Preset::BaselineBalanced).quality(85);
        let jpeg_imgref = encoder.encode_imgref(img.as_ref()).unwrap();
        let jpeg_rgb = encoder.encode_rgb(&rgb_bytes, 64, 48).unwrap();

        // Should be identical
        assert_eq!(jpeg_imgref, jpeg_rgb);
    }
}
