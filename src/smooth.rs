//! Input smoothing filter for noise reduction.
//!
//! This module implements mozjpeg's smoothing algorithm, which applies a
//! weighted average filter to reduce fine-scale noise in the input image.
//! It's particularly useful for converting dithered images (like GIFs) to JPEG.
//!
//! The algorithm replaces each pixel P with a weighted average:
//! - P's weight = 1 - 8*SF where SF = smoothing_factor/1024
//! - Each of 8 neighbors gets weight SF
//!
//! This is equivalent to a 3x3 kernel where the center has higher weight
//! and neighbors have equal lower weights.

/// Apply smoothing filter to RGB image data.
///
/// # Arguments
/// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `smoothing_factor` - Smoothing strength (0-100, 0 = no smoothing)
///
/// # Returns
/// Smoothed RGB data as a new Vec<u8>
pub fn smooth_rgb(rgb_data: &[u8], width: u32, height: u32, smoothing_factor: u8) -> Vec<u8> {
    if smoothing_factor == 0 || width < 3 || height < 3 {
        return rgb_data.to_vec();
    }

    let width = width as usize;
    let height = height as usize;
    let sf = smoothing_factor as i32;

    // Scale factors (matching mozjpeg's fixed-point arithmetic)
    // memberscale = 65536 - smoothing_factor * 512
    // neighscale = smoothing_factor * 64
    // At sf=100: memberscale=14336, neighscale=6400, total=14336+8*6400=65536 ✓
    let memberscale = 65536 - sf * 512;
    let neighscale = sf * 64;

    let mut output = vec![0u8; rgb_data.len()];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            for c in 0..3 {
                // Get center pixel
                let center = rgb_data[idx + c] as i32;

                // Get 8 neighbors (with edge clamping)
                let y_above = y.saturating_sub(1);
                let y_below = (y + 1).min(height - 1);
                let x_left = x.saturating_sub(1);
                let x_right = (x + 1).min(width - 1);

                let get_pixel =
                    |py: usize, px: usize| -> i32 { rgb_data[(py * width + px) * 3 + c] as i32 };

                // 8 neighbors
                let n_tl = get_pixel(y_above, x_left); // top-left
                let n_t = get_pixel(y_above, x); // top
                let n_tr = get_pixel(y_above, x_right); // top-right
                let n_l = get_pixel(y, x_left); // left
                let n_r = get_pixel(y, x_right); // right
                let n_bl = get_pixel(y_below, x_left); // bottom-left
                let n_b = get_pixel(y_below, x); // bottom
                let n_br = get_pixel(y_below, x_right); // bottom-right

                let neighbor_sum = n_tl + n_t + n_tr + n_l + n_r + n_bl + n_b + n_br;

                // Weighted average with fixed-point math
                let result = (center * memberscale + neighbor_sum * neighscale + 32768) >> 16;

                output[idx + c] = result.clamp(0, 255) as u8;
            }
        }
    }

    output
}

/// Apply smoothing filter to grayscale image data.
///
/// # Arguments
/// * `gray_data` - Grayscale pixel data (1 byte per pixel, row-major)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `smoothing_factor` - Smoothing strength (0-100, 0 = no smoothing)
///
/// # Returns
/// Smoothed grayscale data as a new Vec<u8>
pub fn smooth_grayscale(
    gray_data: &[u8],
    width: u32,
    height: u32,
    smoothing_factor: u8,
) -> Vec<u8> {
    if smoothing_factor == 0 || width < 3 || height < 3 {
        return gray_data.to_vec();
    }

    let width = width as usize;
    let height = height as usize;
    let sf = smoothing_factor as i32;

    let memberscale = 65536 - sf * 512;
    let neighscale = sf * 64;

    let mut output = vec![0u8; gray_data.len()];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            let center = gray_data[idx] as i32;

            let y_above = y.saturating_sub(1);
            let y_below = (y + 1).min(height - 1);
            let x_left = x.saturating_sub(1);
            let x_right = (x + 1).min(width - 1);

            let get_pixel = |py: usize, px: usize| -> i32 { gray_data[py * width + px] as i32 };

            let neighbor_sum = get_pixel(y_above, x_left)
                + get_pixel(y_above, x)
                + get_pixel(y_above, x_right)
                + get_pixel(y, x_left)
                + get_pixel(y, x_right)
                + get_pixel(y_below, x_left)
                + get_pixel(y_below, x)
                + get_pixel(y_below, x_right);

            let result = (center * memberscale + neighbor_sum * neighscale + 32768) >> 16;

            output[idx] = result.clamp(0, 255) as u8;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_rgb_no_change_at_zero() {
        let data = vec![100u8; 27]; // 3x3 RGB
        let result = smooth_rgb(&data, 3, 3, 0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_smooth_rgb_uniform_unchanged() {
        // Uniform image should remain unchanged regardless of smoothing
        let data = vec![128u8; 48]; // 4x4 RGB
        let result = smooth_rgb(&data, 4, 4, 50);
        // All pixels should be close to 128
        for pixel in result {
            assert!((pixel as i32 - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_smooth_grayscale_reduces_noise() {
        // Create a checkerboard pattern (dithered)
        let mut data = vec![0u8; 64]; // 8x8
        for y in 0..8 {
            for x in 0..8 {
                data[y * 8 + x] = if (x + y) % 2 == 0 { 255 } else { 0 };
            }
        }

        let result = smooth_grayscale(&data, 8, 8, 50);

        // After smoothing, values should be more uniform (closer to 128)
        let avg: i32 = result.iter().map(|&x| x as i32).sum::<i32>() / 64;
        assert!(
            (avg - 128).abs() < 20,
            "Average should be close to 128, got {}",
            avg
        );

        // Variance should be reduced
        let orig_variance: i32 = data.iter().map(|&x| (x as i32 - 128).pow(2)).sum();
        let smooth_variance: i32 = result.iter().map(|&x| (x as i32 - avg).pow(2)).sum();
        assert!(
            smooth_variance < orig_variance,
            "Smoothing should reduce variance"
        );
    }

    #[test]
    fn test_smooth_preserves_edges_somewhat() {
        // Create image with sharp edge
        let mut data = vec![0u8; 64]; // 8x8
        for y in 0..8 {
            for x in 4..8 {
                data[y * 8 + x] = 255;
            }
        }

        let result = smooth_grayscale(&data, 8, 8, 30);

        // Left side should still be dark
        assert!(result[0] < 50);
        // Right side should still be bright
        assert!(result[7] > 200);
    }

    #[test]
    fn test_smooth_factor_100_max_smoothing() {
        // At factor 100, memberscale=14336, neighscale=6400
        // Center gets ~22% weight, neighbors get ~78% total
        let mut data = vec![0u8; 9]; // 3x3
        data[4] = 255; // Center is white, all neighbors black

        let result = smooth_grayscale(&data, 3, 3, 100);

        // Center should be significantly reduced (but not to ~22% due to edge effects)
        assert!(
            result[4] < 100,
            "Center should be reduced, got {}",
            result[4]
        );
    }
}
