//! Chroma subsampling for JPEG encoding.
//!
//! This module implements downsampling routines for chroma components:
//! - 4:4:4 (no subsampling) - fullsize copy
//! - 4:2:2 (h2v1) - 2:1 horizontal, 1:1 vertical
//! - 4:2:0 (h2v2) - 2:1 horizontal, 2:1 vertical
//!
//! The downsampling uses a "box filter" (simple average) which is equivalent
//! to a triangle filter for 2:1 ratios. Rounding uses alternating bias to
//! avoid systematic rounding errors.
//!
//! Reference: mozjpeg jcsample.c

use crate::consts::DCTSIZE;

/// Downsample a row with 2:1 horizontal ratio (4:2:2).
///
/// Averages pairs of pixels horizontally. Uses alternating bias (0, 1, 0, 1...)
/// when rounding to avoid systematic bias toward larger values.
///
/// # Arguments
/// * `input` - Input row (width pixels)
/// * `output` - Output row (width/2 pixels)
///
/// # Panics
/// Panics if output length is not input.len() / 2 (rounded up).
pub fn downsample_h2v1_row(input: &[u8], output: &mut [u8]) {
    let output_len = (input.len() + 1) / 2;
    assert!(output.len() >= output_len, "output buffer too small");

    let mut bias = 0u16;
    for (i, out) in output.iter_mut().enumerate().take(output_len) {
        let idx = i * 2;
        let p0 = input[idx] as u16;
        let p1 = if idx + 1 < input.len() {
            input[idx + 1] as u16
        } else {
            p0 // edge replication
        };
        *out = ((p0 + p1 + bias) >> 1) as u8;
        bias ^= 1; // alternate 0, 1, 0, 1...
    }
}

/// Downsample a 2-row block with 2:1 horizontal, 2:1 vertical ratio (4:2:0).
///
/// Averages 2x2 blocks of pixels. Uses alternating bias (1, 2, 1, 2...)
/// when rounding to avoid systematic bias.
///
/// # Arguments
/// * `row0` - First input row
/// * `row1` - Second input row (or same as row0 for edge)
/// * `output` - Output row (width/2 pixels)
///
/// # Panics
/// Panics if rows have different lengths or output is too small.
pub fn downsample_h2v2_rows(row0: &[u8], row1: &[u8], output: &mut [u8]) {
    assert_eq!(row0.len(), row1.len(), "input rows must have same length");
    let output_len = (row0.len() + 1) / 2;
    assert!(output.len() >= output_len, "output buffer too small");

    let mut bias = 1u16;
    for (i, out) in output.iter_mut().enumerate().take(output_len) {
        let idx = i * 2;
        let p00 = row0[idx] as u16;
        let p10 = row1[idx] as u16;
        let (p01, p11) = if idx + 1 < row0.len() {
            (row0[idx + 1] as u16, row1[idx + 1] as u16)
        } else {
            (p00, p10) // edge replication
        };
        *out = ((p00 + p01 + p10 + p11 + bias) >> 2) as u8;
        bias ^= 3; // alternate 1, 2, 1, 2...
    }
}

/// Downsample a component plane with the specified ratios.
///
/// This is a higher-level function that handles full planes.
///
/// # Arguments
/// * `input` - Input plane (height x width)
/// * `input_width` - Width of input plane
/// * `input_height` - Height of input plane
/// * `h_ratio` - Horizontal downsampling ratio (1 or 2)
/// * `v_ratio` - Vertical downsampling ratio (1 or 2)
/// * `output` - Output plane buffer
///
/// # Returns
/// (output_width, output_height)
pub fn downsample_plane(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    h_ratio: usize,
    v_ratio: usize,
    output: &mut [u8],
) -> (usize, usize) {
    assert!(h_ratio == 1 || h_ratio == 2, "h_ratio must be 1 or 2");
    assert!(v_ratio == 1 || v_ratio == 2, "v_ratio must be 1 or 2");

    let output_width = (input_width + h_ratio - 1) / h_ratio;
    let output_height = (input_height + v_ratio - 1) / v_ratio;

    assert!(
        output.len() >= output_width * output_height,
        "output buffer too small"
    );

    match (h_ratio, v_ratio) {
        (1, 1) => {
            // Fullsize - just copy
            for y in 0..input_height {
                let src_start = y * input_width;
                let dst_start = y * output_width;
                output[dst_start..dst_start + input_width]
                    .copy_from_slice(&input[src_start..src_start + input_width]);
            }
        }
        (2, 1) => {
            // h2v1 - horizontal only
            for y in 0..input_height {
                let src_start = y * input_width;
                let dst_start = y * output_width;
                downsample_h2v1_row(
                    &input[src_start..src_start + input_width],
                    &mut output[dst_start..dst_start + output_width],
                );
            }
        }
        (1, 2) => {
            // h1v2 - vertical only (less common)
            for y in 0..output_height {
                let y0 = y * 2;
                let y1 = (y0 + 1).min(input_height - 1);
                let src0_start = y0 * input_width;
                let src1_start = y1 * input_width;
                let dst_start = y * output_width;

                // Average vertically, pixel by pixel
                for x in 0..input_width {
                    let p0 = input[src0_start + x] as u16;
                    let p1 = input[src1_start + x] as u16;
                    output[dst_start + x] = ((p0 + p1 + 1) >> 1) as u8;
                }
            }
        }
        (2, 2) => {
            // h2v2 - both directions (4:2:0)
            for y in 0..output_height {
                let y0 = y * 2;
                let y1 = (y0 + 1).min(input_height - 1);
                let src0_start = y0 * input_width;
                let src1_start = y1 * input_width;
                let dst_start = y * output_width;
                downsample_h2v2_rows(
                    &input[src0_start..src0_start + input_width],
                    &input[src1_start..src1_start + input_width],
                    &mut output[dst_start..dst_start + output_width],
                );
            }
        }
        _ => unreachable!(),
    }

    (output_width, output_height)
}

/// Calculate output dimensions for a given subsampling mode.
///
/// Rounds up to ensure all input pixels are covered.
pub fn subsampled_dimensions(
    width: usize,
    height: usize,
    h_ratio: usize,
    v_ratio: usize,
) -> (usize, usize) {
    (
        (width + h_ratio - 1) / h_ratio,
        (height + v_ratio - 1) / v_ratio,
    )
}

/// Calculate MCU-aligned dimensions for encoding.
///
/// JPEG requires dimensions to be multiples of 8 (DCTSIZE) times the
/// maximum sampling factor.
pub fn mcu_aligned_dimensions(
    width: usize,
    height: usize,
    max_h_samp: usize,
    max_v_samp: usize,
) -> (usize, usize) {
    let h_mcu = DCTSIZE * max_h_samp;
    let v_mcu = DCTSIZE * max_v_samp;
    (
        (width + h_mcu - 1) / h_mcu * h_mcu,
        (height + v_mcu - 1) / v_mcu * v_mcu,
    )
}

/// Expand a plane to MCU-aligned dimensions by replicating edge pixels.
///
/// This is needed before DCT processing to ensure all blocks are complete.
pub fn expand_to_mcu(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    output: &mut [u8],
    output_width: usize,
    output_height: usize,
) {
    assert!(output_width >= input_width);
    assert!(output_height >= input_height);
    assert!(output.len() >= output_width * output_height);

    for y in 0..output_height {
        let src_y = y.min(input_height - 1);
        let src_start = src_y * input_width;
        let dst_start = y * output_width;

        // Copy existing pixels
        let copy_width = input_width.min(output_width);
        output[dst_start..dst_start + copy_width]
            .copy_from_slice(&input[src_start..src_start + copy_width]);

        // Replicate right edge
        if output_width > input_width {
            let edge_pixel = input[src_start + input_width - 1];
            for x in input_width..output_width {
                output[dst_start + x] = edge_pixel;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_h2v1_basic() {
        let input = [10u8, 20, 30, 40, 50, 60];
        let mut output = [0u8; 3];
        downsample_h2v1_row(&input, &mut output);

        // (10+20+0)/2=15, (30+40+1)/2=35, (50+60+0)/2=55
        assert_eq!(output[0], 15);
        assert_eq!(output[1], 35); // bias=1
        assert_eq!(output[2], 55);
    }

    #[test]
    fn test_downsample_h2v1_odd_width() {
        let input = [10u8, 20, 30, 40, 50];
        let mut output = [0u8; 3];
        downsample_h2v1_row(&input, &mut output);

        // Last pixel replicates
        assert_eq!(output[0], 15);
        assert_eq!(output[1], 35);
        assert_eq!(output[2], 50); // (50+50+0)/2 = 50
    }

    #[test]
    fn test_downsample_h2v2_basic() {
        let row0 = [10u8, 20, 30, 40];
        let row1 = [12u8, 22, 32, 42];
        let mut output = [0u8; 2];
        downsample_h2v2_rows(&row0, &row1, &mut output);

        // (10+20+12+22+1)/4 = 65/4 = 16 (with bias=1)
        // (30+40+32+42+2)/4 = 146/4 = 36 (with bias=2)
        assert_eq!(output[0], 16);
        assert_eq!(output[1], 36);
    }

    #[test]
    fn test_downsample_h2v2_uniform() {
        // All same value should stay same
        let row0 = [128u8; 4];
        let row1 = [128u8; 4];
        let mut output = [0u8; 2];
        downsample_h2v2_rows(&row0, &row1, &mut output);

        assert_eq!(output[0], 128);
        assert_eq!(output[1], 128);
    }

    #[test]
    fn test_downsample_plane_fullsize() {
        let input = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut output = [0u8; 9];
        let (w, h) = downsample_plane(&input, 3, 3, 1, 1, &mut output);

        assert_eq!(w, 3);
        assert_eq!(h, 3);
        assert_eq!(&output, &input);
    }

    #[test]
    fn test_downsample_plane_h2v1() {
        // 4x2 -> 2x2
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 4];
        let (w, h) = downsample_plane(&input, 4, 2, 2, 1, &mut output);

        assert_eq!(w, 2);
        assert_eq!(h, 2);
        // Row 0: (10+20+0)/2=15, (30+40+1)/2=35
        // Row 1: (50+60+0)/2=55, (70+80+1)/2=75
        assert_eq!(output[0], 15);
        assert_eq!(output[1], 35);
        assert_eq!(output[2], 55);
        assert_eq!(output[3], 75);
    }

    #[test]
    fn test_downsample_plane_h2v2() {
        // 4x4 -> 2x2
        #[rustfmt::skip]
        let input = [
            10, 20, 30, 40,
            12, 22, 32, 42,
            14, 24, 34, 44,
            16, 26, 36, 46,
        ];
        let mut output = [0u8; 4];
        let (w, h) = downsample_plane(&input, 4, 4, 2, 2, &mut output);

        assert_eq!(w, 2);
        assert_eq!(h, 2);
    }

    #[test]
    fn test_subsampled_dimensions() {
        assert_eq!(subsampled_dimensions(640, 480, 1, 1), (640, 480));
        assert_eq!(subsampled_dimensions(640, 480, 2, 1), (320, 480));
        assert_eq!(subsampled_dimensions(640, 480, 2, 2), (320, 240));
        // Odd dimensions round up
        assert_eq!(subsampled_dimensions(641, 481, 2, 2), (321, 241));
    }

    #[test]
    fn test_mcu_aligned_dimensions() {
        // 4:4:4 (max samp = 1)
        assert_eq!(mcu_aligned_dimensions(640, 480, 1, 1), (640, 480));
        // 4:2:0 (max samp = 2)
        assert_eq!(mcu_aligned_dimensions(640, 480, 2, 2), (640, 480));
        // Needs padding
        assert_eq!(mcu_aligned_dimensions(641, 481, 2, 2), (656, 496));
        // Small image
        assert_eq!(mcu_aligned_dimensions(10, 10, 2, 2), (16, 16));
    }

    #[test]
    fn test_expand_to_mcu() {
        // 3x3 -> 8x8
        #[rustfmt::skip]
        let input = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ];
        let mut output = [0u8; 64];
        expand_to_mcu(&input, 3, 3, &mut output, 8, 8);

        // First row: 1, 2, 3, 3, 3, 3, 3, 3
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 3);
        assert_eq!(output[7], 3); // edge replicated

        // Last original row repeated
        assert_eq!(output[7 * 8], 7);
        assert_eq!(output[7 * 8 + 2], 9);
        assert_eq!(output[7 * 8 + 7], 9);
    }

    #[test]
    fn test_expand_to_mcu_exact() {
        // 8x8 -> 8x8 (no change needed)
        let input = [128u8; 64];
        let mut output = [0u8; 64];
        expand_to_mcu(&input, 8, 8, &mut output, 8, 8);
        assert_eq!(&output, &input);
    }

    #[test]
    fn test_alternating_bias_h2v1() {
        // Test that bias alternates to avoid systematic rounding
        // If always rounding up (bias=1), 127+128 -> 128
        // With alternating, some round down, some round up
        let input = [127u8, 128, 127, 128, 127, 128, 127, 128];
        let mut output = [0u8; 4];
        downsample_h2v1_row(&input, &mut output);

        // bias=0: (127+128+0)/2 = 127
        // bias=1: (127+128+1)/2 = 128
        // bias=0: (127+128+0)/2 = 127
        // bias=1: (127+128+1)/2 = 128
        assert_eq!(output[0], 127);
        assert_eq!(output[1], 128);
        assert_eq!(output[2], 127);
        assert_eq!(output[3], 128);
    }

    #[test]
    fn test_alternating_bias_h2v2() {
        // Test alternating bias for h2v2
        // 4 values of 127 each: (127*4 + bias) / 4
        let row = [127u8; 8];
        let mut output = [0u8; 4];
        downsample_h2v2_rows(&row, &row, &mut output);

        // bias=1: (508+1)/4 = 127
        // bias=2: (508+2)/4 = 127
        // All should be 127 since 508/4 = 127 exactly
        assert_eq!(output, [127, 127, 127, 127]);

        // Test with values that round differently
        let row0 = [126u8; 8];
        let row1 = [126u8; 8];
        let mut output2 = [0u8; 4];
        downsample_h2v2_rows(&row0, &row1, &mut output2);
        // (504+1)/4=126, (504+2)/4=126
        assert_eq!(output2, [126, 126, 126, 126]);
    }
}
