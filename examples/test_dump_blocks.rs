//! Dump block values to compare baseline vs progressive encoding paths
//!
//! This test directly accesses the internal encoding functions to see
//! what values are being fed to each path.

fn main() {
    // Create minimal 16x16 gradient image
    let width = 16u32;
    let height = 16u32;
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            rgb_data[i * 3] = (x * 16) as u8; // R: 0-240 across
            rgb_data[i * 3 + 1] = (y * 16) as u8; // G: 0-240 down
            rgb_data[i * 3 + 2] = 128; // B: constant
        }
    }

    // Encode and save baseline
    let baseline = mozjpeg_rs::Encoder::new(false)
        .quality(85)
        .subsampling(mozjpeg_rs::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // Encode and save progressive
    let progressive = mozjpeg_rs::Encoder::max_compression()
        .quality(85)
        .subsampling(mozjpeg_rs::Subsampling::S444)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    std::fs::write("/tmp/dump_baseline.jpg", &baseline).unwrap();
    std::fs::write("/tmp/dump_progressive.jpg", &progressive).unwrap();

    println!(
        "Saved baseline ({} bytes) and progressive ({} bytes)",
        baseline.len(),
        progressive.len()
    );

    // Use djpeg to decode and compare
    println!("\n=== Decoding with djpeg (external tool) ===");

    // Decode baseline with djpeg
    let output = std::process::Command::new("djpeg")
        .args([
            "-pnm",
            "-outfile",
            "/tmp/dump_base.ppm",
            "/tmp/dump_baseline.jpg",
        ])
        .output();
    match output {
        Ok(o) => {
            if !o.status.success() {
                println!(
                    "djpeg baseline failed: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
            } else {
                println!("djpeg baseline: OK");
            }
        }
        Err(e) => println!("djpeg not available: {}", e),
    }

    // Decode progressive with djpeg
    let output = std::process::Command::new("djpeg")
        .args([
            "-pnm",
            "-outfile",
            "/tmp/dump_prog.ppm",
            "/tmp/dump_progressive.jpg",
        ])
        .output();
    match output {
        Ok(o) => {
            if !o.status.success() {
                println!(
                    "djpeg progressive failed: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
            } else {
                println!("djpeg progressive: OK");
            }
        }
        Err(e) => println!("djpeg not available: {}", e),
    }

    // Read and compare PPM files
    if let (Ok(base_ppm), Ok(prog_ppm)) = (
        std::fs::read("/tmp/dump_base.ppm"),
        std::fs::read("/tmp/dump_prog.ppm"),
    ) {
        // Skip PPM header (find the data after the header)
        fn parse_ppm(data: &[u8]) -> Option<&[u8]> {
            // Simple PPM parser - find the pixel data
            let s = std::str::from_utf8(data).ok()?;
            let mut lines = s.lines();
            let _magic = lines.next()?; // P6
            let _dims = lines.next()?; // width height
            let _max = lines.next()?; // 255
                                      // Find byte offset of pixel data
            let header_end = s.find("255\n")? + 4;
            Some(&data[header_end..])
        }

        if let (Some(base_pixels), Some(prog_pixels)) = (parse_ppm(&base_ppm), parse_ppm(&prog_ppm))
        {
            println!("\nPixel comparison (djpeg output):");
            println!("Original | Baseline | Progressive");
            for i in 0..8 {
                let orig = (rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
                let base = (
                    base_pixels.get(i * 3).copied().unwrap_or(0),
                    base_pixels.get(i * 3 + 1).copied().unwrap_or(0),
                    base_pixels.get(i * 3 + 2).copied().unwrap_or(0),
                );
                let prog = (
                    prog_pixels.get(i * 3).copied().unwrap_or(0),
                    prog_pixels.get(i * 3 + 1).copied().unwrap_or(0),
                    prog_pixels.get(i * 3 + 2).copied().unwrap_or(0),
                );
                println!(
                    "({:3},{:3},{:3}) | ({:3},{:3},{:3}) | ({:3},{:3},{:3})",
                    orig.0, orig.1, orig.2, base.0, base.1, base.2, prog.0, prog.1, prog.2
                );
            }
        }
    }

    // Also compare scan structures
    println!("\n=== Scan structure comparison ===");

    fn count_sos_markers(data: &[u8]) -> Vec<(usize, u8, u8)> {
        let mut markers = Vec::new();
        let mut i = 0;
        while i + 1 < data.len() {
            if data[i] == 0xFF && data[i + 1] == 0xDA {
                let length = if i + 3 < data.len() {
                    ((data[i + 2] as usize) << 8) | data[i + 3] as usize
                } else {
                    0
                };
                let ss = if i + length < data.len() {
                    data[i + length - 1]
                } else {
                    0
                };
                let se = if i + length + 1 < data.len() {
                    data[i + length]
                } else {
                    0
                };
                markers.push((i, ss, se));
                i += length + 2;
            } else {
                i += 1;
            }
        }
        markers
    }

    let base_scans = count_sos_markers(&baseline);
    let prog_scans = count_sos_markers(&progressive);

    println!("Baseline: {} scans", base_scans.len());
    println!("Progressive: {} scans", prog_scans.len());
}
