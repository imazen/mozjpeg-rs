//! Exhaustive parity testing at 20 quality levels using mozjpeg crate
use mozjpeg_rs::{Encoder, TrellisConfig, Subsampling};

fn create_gradient_image() -> (Vec<u8>, u32, u32) {
    let width = 512u32;
    let height = 512u32;
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            rgb[idx] = ((x * 255) / width) as u8;
            rgb[idx + 1] = ((y * 255) / height) as u8;
            rgb[idx + 2] = (((x + y) * 127) / (width + height)) as u8;
        }
    }
    println!("Using gradient image: {}x{}", width, height);
    (rgb, width, height)
}

fn main() {
    #[cfg(feature = "png")]
    let (rgb, width, height) = if let Some(path) = mozjpeg_rs::corpus::bundled_test_image("1.png") {
        if let Some((data, w, h)) = mozjpeg_rs::corpus::load_png_as_rgb(&path) {
            println!("Using real image: {}x{}", w, h);
            (data, w, h)
        } else {
            create_gradient_image()
        }
    } else {
        create_gradient_image()
    };

    #[cfg(not(feature = "png"))]
    let (rgb, width, height) = create_gradient_image();

    let qualities: Vec<u8> = (1..=20).map(|i| i * 5).collect();

    println!("\nExhaustive Parity Test: 20 Quality Levels (mozjpeg crate)");
    println!("All three modes: Baseline, Progressive, Max Compression");
    println!();
    println!("{:>3} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
             "Q", "Rust Base", "C Base", "Diff%", "Rust Prog", "C Prog", "Diff%", "Rust Max", "C Max", "Diff%");
    println!("{}", "-".repeat(106));

    let mut total_rust_baseline = 0usize;
    let mut total_c_baseline = 0usize;
    let mut total_rust_prog = 0usize;
    let mut total_c_prog = 0usize;
    let mut total_rust_max = 0usize;
    let mut total_c_max = 0usize;

    for q in &qualities {
        // Baseline: sequential DCT, no progressive
        let rust_baseline = encode_rust(&rgb, width, height, *q, false, false);
        let c_baseline = encode_c_mozjpeg(&rgb, width, height, *q, false, false);
        let diff_baseline = ((rust_baseline.len() as f64 / c_baseline.len() as f64) - 1.0) * 100.0;

        // Progressive: multi-scan, no optimize_scans
        let rust_prog = encode_rust(&rgb, width, height, *q, true, false);
        let c_prog = encode_c_mozjpeg(&rgb, width, height, *q, true, false);
        let diff_prog = ((rust_prog.len() as f64 / c_prog.len() as f64) - 1.0) * 100.0;

        // Max compression: progressive + optimize_scans
        let rust_max = encode_rust(&rgb, width, height, *q, true, true);
        let c_max = encode_c_mozjpeg(&rgb, width, height, *q, true, true);
        let diff_max = ((rust_max.len() as f64 / c_max.len() as f64) - 1.0) * 100.0;

        println!("{:3} {:>10} {:>10} {:>+7.2}% {:>10} {:>10} {:>+7.2}% {:>10} {:>10} {:>+7.2}%",
                 q,
                 rust_baseline.len(), c_baseline.len(), diff_baseline,
                 rust_prog.len(), c_prog.len(), diff_prog,
                 rust_max.len(), c_max.len(), diff_max);

        total_rust_baseline += rust_baseline.len();
        total_c_baseline += c_baseline.len();
        total_rust_prog += rust_prog.len();
        total_c_prog += c_prog.len();
        total_rust_max += rust_max.len();
        total_c_max += c_max.len();
    }

    println!("{}", "-".repeat(106));
    let avg_baseline = ((total_rust_baseline as f64 / total_c_baseline as f64) - 1.0) * 100.0;
    let avg_prog = ((total_rust_prog as f64 / total_c_prog as f64) - 1.0) * 100.0;
    let avg_max = ((total_rust_max as f64 / total_c_max as f64) - 1.0) * 100.0;
    println!("{:>3} {:>10} {:>10} {:>+7.2}% {:>10} {:>10} {:>+7.2}% {:>10} {:>10} {:>+7.2}%",
             "SUM", total_rust_baseline, total_c_baseline, avg_baseline,
             total_rust_prog, total_c_prog, avg_prog,
             total_rust_max, total_c_max, avg_max);
}

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, progressive: bool, optimize_scans: bool) -> Vec<u8> {
    Encoder::new(false)
        .quality(quality)
        .progressive(progressive)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true)
        .trellis(TrellisConfig::default())
        .overshoot_deringing(true)
        .optimize_scans(optimize_scans)
        .encode_rgb(rgb, width, height)
        .unwrap()
}

/// Encode using the mozjpeg crate (high-level safe wrapper).
/// C mozjpeg defaults: ImageMagick tables, trellis on, deringing on.
fn encode_c_mozjpeg(rgb: &[u8], width: u32, height: u32, quality: u8, progressive: bool, optimize_scans: bool) -> Vec<u8> {
    use mozjpeg::{Compress, ColorSpace};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);

    // 4:2:0 subsampling: luma 2x2, chroma 1x1
    comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2));

    comp.set_optimize_scans(optimize_scans);

    if progressive {
        comp.set_progressive_mode();
    }

    // Start compression to memory
    let mut comp = comp.start_compress(Vec::new()).unwrap();

    // Write all scanlines at once
    comp.write_scanlines(rgb).unwrap();

    comp.finish().unwrap()
}
