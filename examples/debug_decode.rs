//! Debug which encoder modes produce decodable JPEGs

use mozjpeg_rs::{Encoder, Subsampling, TrellisConfig};
use std::fs::File;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn try_decode(jpeg: &[u8]) -> Result<(), String> {
    jpeg_decoder::Decoder::new(std::io::Cursor::new(jpeg))
        .decode()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn main() {
    // Test all Kodak images
    let mut entries: Vec<_> = std::fs::read_dir("corpus/kodak")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    println!("Testing progressive mode (optimize_scans=false) on all images:\n");
    println!("{:<12} {:>8} {:>8}  Result", "Image", "W", "H");
    println!("{}", "-".repeat(50));

    let mut failed = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy();

        let (rgb, w, h) = match load_png(&path) {
            Some(d) => d,
            None => continue,
        };

        let prog = Encoder::baseline_optimized()
            .quality(85)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .optimize_scans(false)
            .subsampling(Subsampling::S420)
            .encode_rgb(&rgb, w, h)
            .unwrap();

        match try_decode(&prog) {
            Ok(()) => println!("{:<12} {:>8} {:>8}  OK", name, w, h),
            Err(e) => {
                println!("{:<12} {:>8} {:>8}  FAILED: {}", name, w, h, e);
                failed.push((name.to_string(), w, h));
            }
        }
    }

    println!("\n{} images failed:", failed.len());
    for (name, w, h) in &failed {
        println!("  {} ({}x{})", name, w, h);
    }

    // Test with different quality levels on a failing image
    if let Some((name, _, _)) = failed.first() {
        let path = Path::new("corpus/kodak").join(name);
        let (rgb, w, h) = load_png(&path).unwrap();

        println!("\nTesting {} at different quality levels:", name);
        for q in [50, 60, 70, 75, 80, 85, 90, 95] {
            let prog = Encoder::baseline_optimized()
                .quality(q)
                .progressive(true)
                .optimize_huffman(true)
                .trellis(TrellisConfig::default())
                .optimize_scans(false)
                .subsampling(Subsampling::S420)
                .encode_rgb(&rgb, w, h)
                .unwrap();

            match try_decode(&prog) {
                Ok(()) => println!("  Q{}: OK ({} bytes)", q, prog.len()),
                Err(e) => println!("  Q{}: FAILED - {}", q, e),
            }
        }

        // Test without trellis
        println!("\nTesting {} without trellis:", name);
        let prog = Encoder::baseline_optimized()
            .quality(85)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::disabled())
            .optimize_scans(false)
            .subsampling(Subsampling::S420)
            .encode_rgb(&rgb, w, h)
            .unwrap();

        match try_decode(&prog) {
            Ok(()) => println!("  Without trellis: OK ({} bytes)", prog.len()),
            Err(e) => println!("  Without trellis: FAILED - {}", e),
        }

        // Test with different subsampling
        println!("\nTesting {} with 4:4:4 subsampling:", name);
        let prog = Encoder::baseline_optimized()
            .quality(85)
            .progressive(true)
            .optimize_huffman(true)
            .trellis(TrellisConfig::default())
            .optimize_scans(false)
            .subsampling(Subsampling::S444)
            .encode_rgb(&rgb, w, h)
            .unwrap();

        match try_decode(&prog) {
            Ok(()) => println!("  4:4:4: OK ({} bytes)", prog.len()),
            Err(e) => println!("  4:4:4: FAILED - {}", e),
        }
    }
}
