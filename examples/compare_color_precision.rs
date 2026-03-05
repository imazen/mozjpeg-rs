//! Compare precision between our AVX2 and yuv crate color conversion.

#![allow(unsafe_code)]

use mozjpeg_rs::color::rgb_to_ycbcr;
use yuv::{
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
    rgb_to_yuv444,
};

fn main() {
    // Test with various RGB values
    let test_cases: Vec<(u8, u8, u8)> = vec![
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (128, 128, 128), // Gray
        (255, 128, 64),  // Orange-ish
        (64, 128, 255),  // Blue-ish
    ];

    println!("=== Single pixel comparison ===");
    println!(
        "{:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>6} {:>6} {:>6}",
        "R", "G", "B", "Y_ours", "Cb_ours", "Cr_ours", "Y_diff", "Cb_diff", "Cr_diff"
    );
    println!("{}", "-".repeat(100));

    for (r, g, b) in &test_cases {
        let (y_ours, cb_ours, cr_ours) = rgb_to_ycbcr(*r, *g, *b);

        // yuv crate for single pixel
        let rgb = [*r, *g, *b];
        let mut yuv_image = YuvPlanarImageMut::alloc(1, 1, YuvChromaSubsampling::Yuv444);
        rgb_to_yuv444(
            &mut yuv_image,
            &rgb,
            3,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
            YuvConversionMode::default(),
        )
        .unwrap();

        let y_yuv = yuv_image.y_plane.borrow()[0];
        let cb_yuv = yuv_image.u_plane.borrow()[0];
        let cr_yuv = yuv_image.v_plane.borrow()[0];

        let y_diff = y_ours as i16 - y_yuv as i16;
        let cb_diff = cb_ours as i16 - cb_yuv as i16;
        let cr_diff = cr_ours as i16 - cr_yuv as i16;

        println!(
            "{:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>6} {:>6} {:>6}",
            r, g, b, y_ours, cb_ours, cr_ours, y_diff, cb_diff, cr_diff
        );
    }

    // Now test on a larger image to get statistics
    println!("\n=== Large image statistics (1920x1080) ===");

    let width = 1920u32;
    let height = 1080u32;
    let num_pixels = (width * height) as usize;

    // Generate test image with varied content
    let rgb: Vec<u8> = (0..num_pixels * 3)
        .map(|i| ((i * 17 + i / 1000) % 256) as u8)
        .collect();

    // Our implementation
    let mut y_ours = vec![0u8; num_pixels];
    let mut cb_ours = vec![0u8; num_pixels];
    let mut cr_ours = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        y_ours[i] = y;
        cb_ours[i] = cb;
        cr_ours[i] = cr;
    }

    // yuv crate
    let mut yuv_image = YuvPlanarImageMut::alloc(width, height, YuvChromaSubsampling::Yuv444);
    rgb_to_yuv444(
        &mut yuv_image,
        &rgb,
        width * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::default(),
    )
    .unwrap();

    let y_yuv = yuv_image.y_plane.borrow();
    let cb_yuv = yuv_image.u_plane.borrow();
    let cr_yuv = yuv_image.v_plane.borrow();

    // Calculate statistics
    let mut y_diffs = vec![0i16; num_pixels];
    let mut cb_diffs = vec![0i16; num_pixels];
    let mut cr_diffs = vec![0i16; num_pixels];

    for i in 0..num_pixels {
        y_diffs[i] = y_ours[i] as i16 - y_yuv[i] as i16;
        cb_diffs[i] = cb_ours[i] as i16 - cb_yuv[i] as i16;
        cr_diffs[i] = cr_ours[i] as i16 - cr_yuv[i] as i16;
    }

    fn stats(diffs: &[i16]) -> (i16, i16, f64, usize) {
        let min = *diffs.iter().min().unwrap();
        let max = *diffs.iter().max().unwrap();
        let mean: f64 = diffs.iter().map(|&x| x as f64).sum::<f64>() / diffs.len() as f64;
        let nonzero = diffs.iter().filter(|&&x| x != 0).count();
        (min, max, mean, nonzero)
    }

    let (y_min, y_max, y_mean, y_nonzero) = stats(&y_diffs);
    let (cb_min, cb_max, cb_mean, cb_nonzero) = stats(&cb_diffs);
    let (cr_min, cr_max, cr_mean, cr_nonzero) = stats(&cr_diffs);

    println!("Channel | Min | Max | Mean | Non-zero pixels | % different");
    println!("{}", "-".repeat(70));
    println!(
        "Y       | {:>3} | {:>3} | {:>5.2} | {:>15} | {:>5.2}%",
        y_min,
        y_max,
        y_mean,
        y_nonzero,
        100.0 * y_nonzero as f64 / num_pixels as f64
    );
    println!(
        "Cb      | {:>3} | {:>3} | {:>5.2} | {:>15} | {:>5.2}%",
        cb_min,
        cb_max,
        cb_mean,
        cb_nonzero,
        100.0 * cb_nonzero as f64 / num_pixels as f64
    );
    println!(
        "Cr      | {:>3} | {:>3} | {:>5.2} | {:>15} | {:>5.2}%",
        cr_min,
        cr_max,
        cr_mean,
        cr_nonzero,
        100.0 * cr_nonzero as f64 / num_pixels as f64
    );

    // Distribution of differences
    println!("\n=== Difference distribution ===");
    for channel_name in ["Y", "Cb", "Cr"] {
        let diffs = match channel_name {
            "Y" => &y_diffs,
            "Cb" => &cb_diffs,
            _ => &cr_diffs,
        };

        let mut hist = std::collections::HashMap::new();
        for &d in diffs {
            *hist.entry(d).or_insert(0usize) += 1;
        }

        let mut entries: Vec<_> = hist.into_iter().collect();
        entries.sort_by_key(|&(k, _)| k);

        print!("{}: ", channel_name);
        for (diff, count) in entries {
            if count > 0 {
                print!(
                    "[{}]={} ({:.1}%)  ",
                    diff,
                    count,
                    100.0 * count as f64 / num_pixels as f64
                );
            }
        }
        println!();
    }

    println!("\n=== Conclusion ===");
    let max_diff = y_max
        .abs()
        .max(y_min.abs())
        .max(cb_max.abs().max(cb_min.abs()))
        .max(cr_max.abs().max(cr_min.abs()));

    if max_diff <= 1 {
        println!("Maximum difference: {} level(s)", max_diff);
        println!("This is INVISIBLE after JPEG quantization (which loses 2-4+ levels at Q85).");
        println!("The yuv crate can safely be used as the default backend.");
    } else {
        println!("Maximum difference: {} levels", max_diff);
        println!("This MAY be visible in some edge cases. Consider testing with actual images.");
    }
}
