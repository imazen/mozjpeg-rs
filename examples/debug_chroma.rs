//! Debug chroma plane handling

use mozjpeg_rs::{color, sample};

fn main() {
    println!("=== Debugging chroma plane handling for 64x64 4:2:0 ===\n");

    let w = 64usize;
    let h = 64usize;

    // Create a simple test pattern
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Create a radial gradient pattern
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let max_dist = (cx * cx + cy * cy).sqrt();
            let r = (255.0 * (1.0 - dist / max_dist)).clamp(0.0, 255.0) as u8;
            let g = (200.0 * (x as f32 / w as f32)).clamp(0.0, 255.0) as u8;
            let b = (200.0 * (y as f32 / h as f32)).clamp(0.0, 255.0) as u8;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }

    // Step 1: Color conversion
    let mut y_plane = vec![0u8; w * h];
    let mut cb_plane = vec![0u8; w * h];
    let mut cr_plane = vec![0u8; w * h];

    color::convert_rgb_to_ycbcr(&rgb, &mut y_plane, &mut cb_plane, &mut cr_plane, w, h);

    println!("After color conversion:");
    println!(
        "  Y plane: {} bytes, range {:?}-{:?}",
        y_plane.len(),
        y_plane.iter().min().unwrap(),
        y_plane.iter().max().unwrap()
    );
    println!(
        "  Cb plane: {} bytes, range {:?}-{:?}",
        cb_plane.len(),
        cb_plane.iter().min().unwrap(),
        cb_plane.iter().max().unwrap()
    );
    println!(
        "  Cr plane: {} bytes, range {:?}-{:?}",
        cr_plane.len(),
        cr_plane.iter().min().unwrap(),
        cr_plane.iter().max().unwrap()
    );

    // Check specific positions
    for (name, py, px) in [
        ("top-left", 0, 0),
        ("top-right", 0, 63),
        ("center", 32, 32),
        ("bottom-left", 63, 0),
        ("bottom-right", 63, 63),
    ] {
        let idx = py * w + px;
        println!(
            "  {} ({},{}): Y={}, Cb={}, Cr={}",
            name, px, py, y_plane[idx], cb_plane[idx], cr_plane[idx]
        );
    }

    // Step 2: Downsample chroma
    let chroma_w = w / 2;
    let chroma_h = h / 2;
    let mut cb_sub = vec![0u8; chroma_w * chroma_h];
    let mut cr_sub = vec![0u8; chroma_w * chroma_h];

    sample::downsample_plane(&cb_plane, w, h, 2, 2, &mut cb_sub);
    sample::downsample_plane(&cr_plane, w, h, 2, 2, &mut cr_sub);

    println!("\nAfter downsampling ({}x{}):", chroma_w, chroma_h);
    println!(
        "  Cb_sub: {} bytes, range {:?}-{:?}",
        cb_sub.len(),
        cb_sub.iter().min().unwrap(),
        cb_sub.iter().max().unwrap()
    );
    println!(
        "  Cr_sub: {} bytes, range {:?}-{:?}",
        cr_sub.len(),
        cr_sub.iter().min().unwrap(),
        cr_sub.iter().max().unwrap()
    );

    // Check specific positions in downsampled chroma
    for (name, py, px) in [
        ("top-left", 0, 0),
        ("top-right", 0, 31),
        ("center", 16, 16),
        ("bottom-left", 31, 0),
        ("bottom-right", 31, 31),
    ] {
        let idx = py * chroma_w + px;
        println!(
            "  {} ({},{}): Cb={}, Cr={}",
            name, px, py, cb_sub[idx], cr_sub[idx]
        );
    }

    // Step 3: Expand to MCU (should be no-op for 32x32)
    let mcu_chroma_w = 32;
    let mcu_chroma_h = 32;
    let mut cb_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];
    let mut cr_mcu = vec![0u8; mcu_chroma_w * mcu_chroma_h];

    sample::expand_to_mcu(
        &cb_sub,
        chroma_w,
        chroma_h,
        &mut cb_mcu,
        mcu_chroma_w,
        mcu_chroma_h,
    );
    sample::expand_to_mcu(
        &cr_sub,
        chroma_w,
        chroma_h,
        &mut cr_mcu,
        mcu_chroma_w,
        mcu_chroma_h,
    );

    println!("\nAfter MCU expansion ({}x{}):", mcu_chroma_w, mcu_chroma_h);

    // Check if MCU expansion changed anything
    let cb_changed = cb_mcu.iter().zip(cb_sub.iter()).any(|(a, b)| a != b);
    let cr_changed = cr_mcu.iter().zip(cr_sub.iter()).any(|(a, b)| a != b);
    println!("  Cb MCU differs from Cb_sub: {}", cb_changed);
    println!("  Cr MCU differs from Cr_sub: {}", cr_changed);

    // Check specific MCU blocks (each block is 8x8)
    println!("\nChroma block DC values (8x8 block averages):");
    for by in 0..4 {
        print!("  ");
        for bx in 0..4 {
            let mut cb_sum = 0u32;
            for y in 0..8 {
                for x in 0..8 {
                    let px = bx * 8 + x;
                    let py = by * 8 + y;
                    let idx = py * mcu_chroma_w + px;
                    cb_sum += cb_mcu[idx] as u32;
                }
            }
            let cb_avg = cb_sum / 64;
            print!("Cb{},{}: {:3}  ", bx, by, cb_avg);
        }
        println!();
    }

    println!("\n=== Test complete ===");
}
