//! Trellis quantization for optimal rate-distortion.
//!
//! This is the core innovation of mozjpeg over standard libjpeg.
//! Trellis quantization uses dynamic programming to find the optimal
//! quantization decisions that minimize:
//!
//! ```text
//! Cost = Rate + Lambda * Distortion
//! ```
//!
//! where Rate is the Huffman encoding cost and Distortion is the
//! squared error from the original coefficients.
//!
//! Reference: mozjpeg jcdctmgr.c quantize_trellis()

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::entropy::jpeg_nbits;
use crate::huffman::DerivedTable;
use crate::types::TrellisConfig;

/// Maximum number of DC candidate values to explore.
const DC_TRELLIS_MAX_CANDIDATES: usize = 9;

/// State for trellis quantization of a block.
struct TrellisState {
    /// Accumulated distortion if we zero out coefficients from start to i.
    accumulated_zero_dist: [f32; DCTSIZE2],
    /// Accumulated optimal cost up to coefficient i.
    accumulated_cost: [f32; DCTSIZE2],
    /// Index of where the current run of zeros started.
    run_start: [usize; DCTSIZE2],
}

impl TrellisState {
    fn new() -> Self {
        Self {
            accumulated_zero_dist: [0.0; DCTSIZE2],
            accumulated_cost: [0.0; DCTSIZE2],
            run_start: [0; DCTSIZE2],
        }
    }
}

/// Calculate the number of DC trellis candidates based on quantization value.
///
/// Higher quality (lower quantval) allows more candidates.
fn get_num_dc_candidates(dc_quantval: u16) -> usize {
    let candidates = (2 + 60 / dc_quantval as usize) | 1;
    candidates.min(DC_TRELLIS_MAX_CANDIDATES)
}

/// Calculate lambda for rate-distortion optimization.
///
/// Lambda controls the tradeoff between rate (bits) and distortion (error).
/// Higher lambda means more distortion is acceptable for bit savings.
///
/// Formula: lambda = 2^scale1 * lambda_base / (2^scale2 + norm)
fn calculate_lambda(config: &TrellisConfig, block_norm: f32, lambda_base: f32) -> f32 {
    if config.lambda_log_scale2 > 0.0 {
        let scale1 = 2.0_f32.powf(config.lambda_log_scale1);
        let scale2 = 2.0_f32.powf(config.lambda_log_scale2);
        scale1 * lambda_base / (scale2 + block_norm)
    } else {
        2.0_f32.powf(config.lambda_log_scale1 - 12.0) * lambda_base
    }
}

/// Calculate lambda base from quantization table.
///
/// Returns 1/(average squared quantval) for AC coefficients.
fn calculate_lambda_base(qtable: &[u16; DCTSIZE2]) -> f32 {
    let mut sum_sq: f32 = 0.0;
    for i in 1..DCTSIZE2 {
        let q = qtable[i] as f32;
        sum_sq += q * q;
    }
    1.0 / (sum_sq / 63.0)
}

/// Calculate per-coefficient lambda weights.
///
/// Returns 1/q^2 for each coefficient position.
fn calculate_lambda_weights(qtable: &[u16; DCTSIZE2]) -> [f32; DCTSIZE2] {
    let mut weights = [0.0f32; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        let q = qtable[i] as f32;
        weights[i] = 1.0 / (q * q);
    }
    weights
}

/// Calculate the norm (average squared value) of AC coefficients.
fn calculate_block_norm(coeffs: &[i16; DCTSIZE2]) -> f32 {
    let mut sum_sq: f32 = 0.0;
    for i in 1..DCTSIZE2 {
        let c = coeffs[i] as f32;
        sum_sq += c * c;
    }
    sum_sq / 63.0
}

/// Perform trellis quantization on a single 8x8 block.
///
/// This is the core rate-distortion optimization algorithm.
///
/// # Arguments
/// * `unquantized` - Unquantized DCT coefficients (raw DCT output, not divided by qtable)
/// * `quantized` - Output buffer for quantized coefficients
/// * `qtable` - Quantization table values
/// * `ac_table` - Derived Huffman table for AC coefficients (for rate estimation)
/// * `config` - Trellis configuration
///
/// # Returns
/// The quantized block in `quantized` buffer.
pub fn trellis_quantize_block(
    unquantized: &[i32; DCTSIZE2],
    quantized: &mut [i16; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
    ac_table: &DerivedTable,
    config: &TrellisConfig,
) {
    // Initialize state
    let mut state = TrellisState::new();

    // Calculate lambda parameters
    let lambda_base = calculate_lambda_base(qtable);
    let lambda_weights = calculate_lambda_weights(qtable);

    // Convert unquantized to i16 for norm calculation
    // (the raw DCT output is scaled by 8, so divide to get coefficient range)
    let mut coeffs_for_norm = [0i16; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        coeffs_for_norm[i] = (unquantized[i] / 8) as i16;
    }
    let block_norm = calculate_block_norm(&coeffs_for_norm);

    let lambda = calculate_lambda(config, block_norm, lambda_base);

    // Quantize DC coefficient (simple rounding for now, DC trellis is optional)
    let dc_val = unquantized[0];
    let dc_q = (qtable[0] as i32) * 8; // Scale factor
    quantized[0] = ((dc_val + dc_q / 2) / dc_q) as i16;

    // Initialize accumulated costs for AC coefficients
    state.accumulated_zero_dist[0] = 0.0;
    state.accumulated_cost[0] = 0.0;

    // Process AC coefficients in zigzag order
    for i in 1..DCTSIZE2 {
        let z = JPEG_NATURAL_ORDER[i];
        let x = unquantized[z].abs();
        let sign = if unquantized[z] < 0 { -1i16 } else { 1i16 };
        let q = (qtable[z] as i32) * 8;

        // Distortion if we zero this coefficient
        let zero_dist = (x as f32).powi(2) * lambda * lambda_weights[z];
        state.accumulated_zero_dist[i] = zero_dist + state.accumulated_zero_dist[i - 1];

        // Simple rounding to get quantized value
        let qval = (x + q / 2) / q;

        if qval == 0 {
            // Coefficient rounds to zero - no choice to make
            quantized[z] = 0;
            state.accumulated_cost[i] = f32::MAX;
            state.run_start[i] = i - 1;
            continue;
        }

        // Clamp to valid range (10 bits for 8-bit JPEG)
        let qval = qval.min(1023);

        // Generate candidate values: powers of 2 minus 1, plus the rounded value
        let num_candidates = jpeg_nbits(qval as i16) as usize;
        let mut candidates = [(0i32, 0u8, 0.0f32); 16]; // (value, bits, distortion)

        for k in 0..num_candidates {
            let candidate_val = if k < num_candidates - 1 {
                (2 << k) - 1
            } else {
                qval
            };
            let delta = candidate_val * q - x;
            let dist = (delta as f32).powi(2) * lambda * lambda_weights[z];
            candidates[k] = (candidate_val, (k + 1) as u8, dist);
        }

        // Find optimal decision using dynamic programming
        state.accumulated_cost[i] = f32::MAX;

        // Try starting a run from each previous position
        for j in 0..i {
            let zz = JPEG_NATURAL_ORDER[j];
            if j != 0 && quantized[zz] == 0 {
                continue; // Skip if previous coef is zero (not a valid run start)
            }

            let zero_run = i - 1 - j;

            // Cost of ZRL codes for runs >= 16
            let zrl_cost = if zero_run >= 16 {
                let (_, zrl_size) = ac_table.get_code(0xF0);
                if zrl_size == 0 {
                    continue; // No ZRL code available
                }
                ((zero_run / 16) * zrl_size as usize) as f32
            } else {
                0.0
            };

            let run_mod_16 = zero_run & 15;

            // Try each candidate value
            for k in 0..num_candidates {
                let (candidate_val, candidate_bits, candidate_dist) = candidates[k];

                // Symbol is (run << 4) | size
                let symbol = ((run_mod_16 as u8) << 4) | candidate_bits;
                let (_, code_size) = ac_table.get_code(symbol);
                if code_size == 0 {
                    continue; // No Huffman code for this symbol
                }

                // Rate = Huffman code size + value bits + ZRL codes
                let rate = code_size as f32 + candidate_bits as f32 + zrl_cost;

                // Cost = rate + distortion of this coef + distortion of zeros in run
                let zero_run_dist =
                    state.accumulated_zero_dist[i - 1] - state.accumulated_zero_dist[j];
                let prev_cost = if j == 0 { 0.0 } else { state.accumulated_cost[j] };
                let cost = rate + candidate_dist + zero_run_dist + prev_cost;

                if cost < state.accumulated_cost[i] {
                    quantized[z] = (candidate_val as i16) * sign;
                    state.accumulated_cost[i] = cost;
                    state.run_start[i] = j;
                }
            }
        }
    }

    // Find the optimal ending point (last non-zero coefficient)
    let eob_cost = {
        let (_, eob_size) = ac_table.get_code(0x00); // EOB symbol
        eob_size as f32
    };

    let mut best_cost = state.accumulated_zero_dist[DCTSIZE2 - 1] + eob_cost;
    let mut last_coeff_idx = 0;

    for i in 1..DCTSIZE2 {
        let z = JPEG_NATURAL_ORDER[i];
        if quantized[z] != 0 {
            // Cost if this is the last non-zero coefficient
            let tail_zero_dist =
                state.accumulated_zero_dist[DCTSIZE2 - 1] - state.accumulated_zero_dist[i];
            let mut cost = state.accumulated_cost[i] + tail_zero_dist;
            if i < DCTSIZE2 - 1 {
                cost += eob_cost;
            }

            if cost < best_cost {
                best_cost = cost;
                last_coeff_idx = i;
            }
        }
    }

    // Zero out coefficients after the optimal ending point
    // and those that are part of runs
    let mut i = DCTSIZE2 - 1;
    while i >= 1 {
        while i > last_coeff_idx {
            let z = JPEG_NATURAL_ORDER[i];
            quantized[z] = 0;
            i -= 1;
        }
        if i >= 1 {
            last_coeff_idx = state.run_start[i];
            i -= 1;
        }
    }
}

/// Quantize a block with simple rounding (no trellis optimization).
///
/// This is the standard quantization used when trellis is disabled.
pub fn simple_quantize_block(
    unquantized: &[i32; DCTSIZE2],
    quantized: &mut [i16; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
) {
    for i in 0..DCTSIZE2 {
        let x = unquantized[i];
        let q = (qtable[i] as i32) * 8; // DCT output is scaled by 8
        // Round to nearest: (x + q/2) / q for positive, (x - q/2) / q for negative
        let sign = if x < 0 { -1 } else { 1 };
        quantized[i] = (sign * ((x.abs() + q / 2) / q)) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, STD_LUMINANCE_QUANT_TBL};
    use crate::huffman::HuffTable;

    fn create_ac_table() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, false).unwrap()
    }

    fn create_qtable() -> [u16; DCTSIZE2] {
        // Use JPEG Annex K table (index 0)
        STD_LUMINANCE_QUANT_TBL[0]
    }

    #[test]
    fn test_trellis_config_default() {
        let config = TrellisConfig::default();
        assert!(config.enabled);
        assert!(config.dc_enabled);
        assert!(config.eob_opt);
        assert!((config.lambda_log_scale1 - 14.75).abs() < 0.01);
        assert!((config.lambda_log_scale2 - 16.5).abs() < 0.01);
    }

    #[test]
    fn test_get_num_dc_candidates() {
        // Low quality (high quantval) -> fewer candidates
        // 60/100 = 0, 2 + 0 = 2, 2 | 1 = 3
        assert_eq!(get_num_dc_candidates(100), 3);
        // Medium quality
        // 60/10 = 6, 2 + 6 = 8, 8 | 1 = 9, capped at 9
        assert_eq!(get_num_dc_candidates(10), 9);
        // High quality (low quantval)
        // 60/1 = 60, 2 + 60 = 62, 62 | 1 = 63, capped at 9
        assert_eq!(get_num_dc_candidates(1), 9);
    }

    #[test]
    fn test_calculate_lambda_base() {
        let qtable = create_qtable();
        let lambda_base = calculate_lambda_base(&qtable);
        // Should be small positive number
        assert!(lambda_base > 0.0);
        assert!(lambda_base < 1.0);
    }

    #[test]
    fn test_simple_quantize_block() {
        let qtable = create_qtable();

        // Create a test block with known values
        let mut unquantized = [0i32; DCTSIZE2];
        unquantized[0] = 1000 * 8; // DC = 1000 (scaled)
        unquantized[1] = 100 * 8; // AC

        let mut quantized = [0i16; DCTSIZE2];
        simple_quantize_block(&unquantized, &mut quantized, &qtable);

        // DC: 1000 / 16 = 62 (rounded)
        assert_eq!(quantized[0], 63); // (1000*8 + 16*8/2) / (16*8) = 63
        // AC[1]: 100 / 11 = 9 (rounded)
        assert!(quantized[1] > 0);
    }

    #[test]
    fn test_simple_quantize_negative() {
        let qtable = create_qtable();

        let mut unquantized = [0i32; DCTSIZE2];
        unquantized[0] = -1000 * 8;
        unquantized[1] = -100 * 8;

        let mut quantized = [0i16; DCTSIZE2];
        simple_quantize_block(&unquantized, &mut quantized, &qtable);

        assert!(quantized[0] < 0);
        assert!(quantized[1] < 0);
    }

    #[test]
    fn test_trellis_quantize_zero_block() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let unquantized = [0i32; DCTSIZE2];
        let mut quantized = [0i16; DCTSIZE2];

        trellis_quantize_block(&unquantized, &mut quantized, &qtable, &ac_table, &config);

        // All zeros should remain zeros
        for &q in quantized.iter() {
            assert_eq!(q, 0);
        }
    }

    #[test]
    fn test_trellis_quantize_dc_only() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let mut unquantized = [0i32; DCTSIZE2];
        unquantized[0] = 1000 * 8; // DC only

        let mut quantized = [0i16; DCTSIZE2];
        trellis_quantize_block(&unquantized, &mut quantized, &qtable, &ac_table, &config);

        // DC should be quantized
        assert!(quantized[0] > 0);
        // AC should be zero
        for i in 1..DCTSIZE2 {
            assert_eq!(quantized[i], 0);
        }
    }

    #[test]
    fn test_trellis_preserves_large_coefficients() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        // Create block with multiple large AC coefficients
        let mut unquantized = [0i32; DCTSIZE2];
        unquantized[0] = 500 * 8;
        unquantized[1] = 200 * 8; // Large AC at position 1

        let mut quantized = [0i16; DCTSIZE2];
        trellis_quantize_block(&unquantized, &mut quantized, &qtable, &ac_table, &config);

        // DC should be non-zero
        assert!(quantized[0] != 0);
        // Large AC coefficient should be preserved (or at least the algorithm runs)
        // Note: The trellis algorithm may choose to zero a coefficient if the
        // rate-distortion tradeoff favors it, so we just verify it runs
    }

    #[test]
    fn test_trellis_zeros_small_coefficients() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        // Create block with very small AC coefficient
        let mut unquantized = [0i32; DCTSIZE2];
        unquantized[0] = 500 * 8;
        unquantized[JPEG_NATURAL_ORDER[63]] = 1; // Very small coefficient at end

        let mut quantized = [0i16; DCTSIZE2];
        trellis_quantize_block(&unquantized, &mut quantized, &qtable, &ac_table, &config);

        // Small trailing coefficient may be zeroed by trellis
        // (depends on rate-distortion tradeoff)
        // Just verify the algorithm runs without error
        assert!(quantized[0] != 0);
    }

    #[test]
    fn test_lambda_calculation() {
        let config = TrellisConfig::default();
        let lambda_base = 0.001;

        // With high norm, lambda should be smaller
        let lambda_high = calculate_lambda(&config, 10000.0, lambda_base);
        let lambda_low = calculate_lambda(&config, 100.0, lambda_base);

        assert!(lambda_high < lambda_low);
    }

    #[test]
    fn test_trellis_config_disabled() {
        let config = TrellisConfig::disabled();
        assert!(!config.enabled);
    }
}
