//! Test progressive encoding with refinement scans.

use mozjpeg_oxide::Encoder;

fn main() {
    let width = 8u32;
    let height = 8u32;
    let mut pixels = vec![0u8; (width * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            pixels[y * width as usize + x] = ((x * 32 + y * 16) % 256) as u8;
        }
    }

    // Test with progressive + refinement
    let jpeg = Encoder::new()
        .quality(90)
        .progressive(true)
        .optimize_huffman(false) // Use standard tables
        .encode_gray(&pixels, width, height)
        .expect("encode");

    println!("mozjpeg-oxide progressive: {} bytes", jpeg.len());

    // Save and test with djpeg
    std::fs::write("/tmp/mozjpeg_test.jpg", &jpeg).ok();
}
