//! Compare Rust vs C quantization methods
//!
//! Rust does: (dct_value + 4) >> 3, then (val + q/2) / q
//! C does: (dct_value + correction) * reciprocal >> shift
//!
//! These should give same results, but let's verify.

use mozjpeg_rs::quant::{create_quant_table, get_luminance_quant_table, quantize_coef};
use mozjpeg_rs::QuantTableIdx;

fn main() {
    println!("=== Quantization Method Comparison ===\n");

    // Get qtable for Q85
    let base_table = get_luminance_quant_table(QuantTableIdx::ImageMagick);
    let qtable = create_quant_table(base_table, 85, true);

    let mut differences = 0;
    let mut total_tests = 0;

    // Test all possible DCT values (-32768..32767) for first few quant values
    for qi in 0..8 {
        let q = qtable.values[qi];
        let q_with_scale = q * 8; // C includes 8x scale factor

        println!("Testing quant[{}] = {} (scaled = {})", qi, q, q_with_scale);

        let mut diff_count = 0;

        // Test a range of DCT values
        for dct_raw in -2000i32..=2000 {
            // Rust method: descale then quantize
            let descaled = (dct_raw + 4) >> 3;
            let rust_result = quantize_coef(descaled, q);

            // C method: quantize raw value with scaled quant
            // (dct_raw + q_with_scale/2) / q_with_scale
            let c_correction = (q_with_scale as i32) / 2;
            let c_result = if dct_raw >= 0 {
                ((dct_raw + c_correction) / q_with_scale as i32) as i16
            } else {
                (-(-dct_raw + c_correction) / q_with_scale as i32) as i16
            };

            if rust_result != c_result {
                if diff_count < 5 {
                    println!(
                        "  DIFF at dct={}: Rust={} (descaled={}), C_approx={}",
                        dct_raw, rust_result, descaled, c_result
                    );
                }
                diff_count += 1;
                differences += 1;
            }
            total_tests += 1;
        }

        if diff_count > 0 {
            println!("  {} differences for this quant value", diff_count);
        } else {
            println!("  ✓ All values match");
        }
    }

    println!(
        "\nTotal: {} differences out of {} tests",
        differences, total_tests
    );
}
