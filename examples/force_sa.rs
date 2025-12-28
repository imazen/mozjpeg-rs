//! Force SA script encoding to debug why it produces larger files
use mozjpeg_oxide::progressive::generate_mozjpeg_max_compression_scans;
use mozjpeg_oxide::types::ScanInfo;
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

fn main() {
    let (rgb, width, height) = load_png(Path::new("corpus/kodak/10.png")).unwrap();
    let quality = 85u8;

    // Get the SA scan script
    let sa_scans = generate_mozjpeg_max_compression_scans(3);

    println!("Testing individual scans for Q{}...", quality);
    println!();

    // For each scan, encode just that scan and measure size
    for (i, scan) in sa_scans.iter().enumerate() {
        let scan_type = if scan.ss == 0 && scan.se == 0 {
            "DC"
        } else if scan.ah > 0 {
            "AC refine"
        } else {
            "AC first"
        };

        println!(
            "Scan {:2}: {:10} | Ss={:2} Se={:2} Ah={} Al={} | comp={}",
            i, scan_type, scan.ss, scan.se, scan.ah, scan.al, scan.component_index[0]
        );
    }

    // Now let's compare what happens when we encode with the SA script
    println!();
    println!("Encoding with SA script requires modifying encode.rs - skipping for now");
    println!();
    println!("The issue is that when we use SA scans (Al>0), the coefficients");
    println!("are shifted right by Al bits. If the frequency counting or encoding");
    println!("doesn't properly account for this, we get wrong Huffman tables or");
    println!("wrong encoded values.");
    println!();
    println!("Key things to verify:");
    println!("1. encode_ac_first with Al>0 encodes (coef >> Al), not coef");
    println!("2. count_ac_first with Al>0 counts (coef >> Al) symbols");
    println!("3. Refinement scans only encode the bits being added");
    println!("4. EOBRUN handling works correctly for Al>0 scans");
}
