//! Bit-exact port of C mozjpeg's trellis quantization pipeline.
//!
//! This module reproduces, exactly, the C mozjpeg trellis code paths:
//!
//! - `quantize_trellis()` (jcdctmgr.c): per-block-row Viterbi AC search,
//!   candidate generation at Huffman category boundaries, per-block adaptive
//!   lambda (computed in f64 like `pow(2.0, scale)`), row-batched DC dynamic
//!   programming, EOB placement, and optional cross-block EOB optimization.
//! - `compress_trellis_pass()` (jccoefct.c): per-component iMCU-row driving
//!   with `lastDC` reset per iMCU row and dummy-block synthesis (zero AC,
//!   DC copied from the last real block).
//! - The C pass schedule (jcmaster.c `select_scan_parameters` /
//!   `prepare_for_pass`): each component is gathered and re-quantized
//!   separately; the rate tables used by trellis are the optimal Huffman
//!   tables derived from the component's *normally quantized* coefficients
//!   (single-component non-interleaved scan, real blocks only). The DC
//!   slot is gathered the same way for baseline output, while progressive
//!   trellis passes only build AC tables (the DC scan keeps the standard
//!   table).
//!
//! Everything here exists solely for [`crate::TrellisMode::MozjpegExact`];
//! the default [`crate::TrellisMode::Optimized`] path is in `trellis.rs`.

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::encode::helpers::block_to_mcu_index;
use crate::entropy::{ProgressiveSymbolCounter, jpeg_nbits};
use crate::error::Result;
use crate::huffman::{DerivedTable, FrequencyCounter, HuffTable};

/// Maximum magnitude of a quantized coefficient for 8-bit JPEG
/// (`(1 << (data_precision + 2)) - 1` = 1023).
const MAX_COEF_VALUE: i32 = 1023;

/// C `COST_INFINITY` (1e38f).
const COST_INFINITY: f32 = 1e38;

/// C `DC_TRELLIS_MAX_CANDIDATES`.
const DC_TRELLIS_MAX_CANDIDATES: usize = 9;

/// C `get_num_dc_trellis_candidates()`: `MIN(9, (2 + 60 / q0) | 1)`.
#[inline]
fn num_dc_candidates(dc_quantval: u16) -> usize {
    ((2 + 60 / dc_quantval as usize) | 1).min(DC_TRELLIS_MAX_CANDIDATES)
}

/// C `compute_dc_huffman_bits()`: `nbits + dctbl->ehufsi[nbits]`.
///
/// Unlike the optimized path there is no fallback when the table lacks the
/// symbol: C returns plain `nbits` (ehufsi == 0 contributes nothing).
#[inline]
fn dc_huffman_bits(dc_delta: i32, dctbl: &DerivedTable) -> i32 {
    let nbits = jpeg_nbits(dc_delta as i16) as i32;
    let (_, code_size) = dctbl.get_code(nbits as u8);
    nbits + code_size as i32
}

/// C `compute_block_lambda()` — computed in f64 and narrowed to f32.
///
/// ```text
/// scale2 > 0: lambda = pow(2, scale1) * lambda_base / (pow(2, scale2) + norm)
/// else:       lambda = pow(2, scale1 - 12) * lambda_base
/// ```
#[inline]
fn compute_block_lambda(lambda_base: f32, norm: f32, scale1: f32, scale2: f32) -> f32 {
    if scale2 > 0.0 {
        (2.0f64.powf(scale1 as f64) * lambda_base as f64
            / (2.0f64.powf(scale2 as f64) + norm as f64)) as f32
    } else {
        (2.0f64.powf(scale1 as f64 - 12.0) * lambda_base as f64) as f32
    }
}

/// Per-component context for a C `quantize_trellis()` row call.
pub(crate) struct ExactTrellisParams<'a> {
    /// Derived DC Huffman table (standard table in C's pass schedule).
    pub dctbl: &'a DerivedTable,
    /// Derived AC Huffman table (optimal table from normally-quantized coefs).
    pub actbl: &'a DerivedTable,
    /// Quantization table (natural order).
    pub qtbl: &'a [u16; DCTSIZE2],
    /// C `trellis_quant_dc`.
    pub dc_enabled: bool,
    /// C `trellis_delta_dc_weight` (0.0 disables vertical DC gradient term).
    pub delta_dc_weight: f32,
    /// C `trellis_speed_level` (0 disables search limiting).
    pub speed_level: u8,
    /// C `lambda_log_scale1`.
    pub lambda_scale1: f32,
    /// C `lambda_log_scale2`.
    pub lambda_scale2: f32,
    /// C `trellis_eob_opt` (cross-block EOBRUN optimization).
    pub eob_opt: bool,
}

/// Port of C `quantize_trellis()` for one row of real blocks.
///
/// Exactly reproduces the C algorithm including arithmetic order:
/// - AC Viterbi search with Huffman-category-boundary candidates
/// - Row-batched DC dynamic programming (`last_dc` seeds `bi == 0`, then the
///   chosen DC chains across the row; updated to the last block's DC)
/// - EOB placement via `find_block_eob_position`
/// - Optional per-row cross-block EOB optimization (`trellis_eob_opt`)
///
/// `src`/`out` contain exactly the real blocks of one block row; padding is
/// handled by the caller like C's `compress_trellis_pass`.
/// Quantized + raw blocks of the row above (for `trellis_delta_dc_weight`).
type RowAbove<'a> = (&'a [[i16; DCTSIZE2]], &'a [[i32; DCTSIZE2]]);

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn quantize_trellis_row_exact(
    src: &[[i32; DCTSIZE2]],
    out: &mut [[i16; DCTSIZE2]],
    params: &ExactTrellisParams,
    last_dc: &mut i16,
    above: Option<RowAbove<'_>>,
) {
    let num_blocks = src.len();
    debug_assert_eq!(out.len(), num_blocks);
    if num_blocks == 0 {
        return;
    }

    let dctbl = params.dctbl;
    let actbl = params.actbl;
    let qtbl = params.qtbl;
    let dc_cands = num_dc_candidates(qtbl[0]);

    // C mode 1: lambda_table[i] = 1 / q[i]^2, lambda_base = 1.0
    let mut lambda_tbl = [0.0f32; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        let q = qtbl[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }
    let lambda_base = 1.0f32;

    // DC DP storage: accumulated_dc_cost[k][bi], dc_cost_backtrack[k][bi],
    // dc_candidate[k][bi]
    let mut accumulated_dc_cost = vec![vec![0.0f32; num_blocks]; dc_cands];
    let mut dc_cost_backtrack = vec![vec![0usize; num_blocks]; dc_cands];
    let mut dc_candidate = vec![vec![0i16; num_blocks]; dc_cands];

    // Cross-block EOB state (C trellis_eob_opt), allocated per row call
    let mut accumulated_zero_block_cost: Vec<f32> = Vec::new();
    let mut accumulated_block_cost: Vec<f32> = Vec::new();
    let mut block_run_start: Vec<usize> = Vec::new();
    let mut requires_eob: Vec<u8> = Vec::new();
    if params.eob_opt {
        accumulated_zero_block_cost = vec![0.0; num_blocks + 1];
        accumulated_block_cost = vec![0.0; num_blocks + 1];
        block_run_start = vec![0; num_blocks];
        requires_eob = vec![0; num_blocks + 1];
    }

    for bi in 0..num_blocks {
        // Adaptive lambda from block AC energy
        // (C accumulates int products into a float)
        let mut norm = 0.0f32;
        for i in 1..DCTSIZE2 {
            norm += src[bi][i].wrapping_mul(src[bi][i]) as f32;
        }
        let norm = norm / 63.0;
        let lambda = compute_block_lambda(
            lambda_base,
            norm,
            params.lambda_scale1,
            params.lambda_scale2,
        );
        let lambda_dc = lambda * lambda_tbl[0];

        let mut accumulated_zero_dist = [0.0f32; DCTSIZE2];
        let mut accumulated_cost = [0.0f32; DCTSIZE2];
        let mut run_start = [0usize; DCTSIZE2];

        // ===== DC coefficient processing (DPCM dynamic programming) =====
        if params.dc_enabled {
            let sign = src[bi][0] >> 31;
            let x = src[bi][0].abs();
            let q = 8 * qtbl[0] as i32;
            let qval = (x + q / 2) / q;

            for k in 0..dc_cands {
                let mut cand = qval - (dc_cands as i32) / 2 + k as i32;
                cand = cand.clamp(-MAX_COEF_VALUE, MAX_COEF_VALUE);

                let delta = cand * q - x;
                let mut candidate_dist = delta.wrapping_mul(delta) as f32 * lambda_dc;
                cand *= 1 + 2 * sign;

                // Vertical DC gradient term (C trellis_delta_dc_weight)
                if params.delta_dc_weight > 0.0
                    && let Some((out_above, src_above)) = above
                {
                    let dc_above_orig = src_above[bi][0];
                    let dc_above_recon = out_above[bi][0] as i32 * q;
                    let dc_orig = src[bi][0];
                    let dc_recon = cand as i32 * q;
                    let vdelta = (dc_above_orig - dc_orig) - (dc_above_recon - dc_recon);
                    let vertical_dist = vdelta.wrapping_mul(vdelta) as f32 * lambda_dc;
                    candidate_dist += params.delta_dc_weight * (vertical_dist - candidate_dist);
                }

                dc_candidate[k][bi] = cand as i16;

                if bi == 0 {
                    let dc_delta = cand - *last_dc as i32;
                    accumulated_dc_cost[k][0] =
                        dc_huffman_bits(dc_delta, dctbl) as f32 + candidate_dist;
                } else {
                    for l in 0..dc_cands {
                        let dc_delta = cand - dc_candidate[l][bi - 1] as i32;
                        let cost = dc_huffman_bits(dc_delta, dctbl) as f32
                            + candidate_dist
                            + accumulated_dc_cost[l][bi - 1];
                        if l == 0 || cost < accumulated_dc_cost[k][bi] {
                            accumulated_dc_cost[k][bi] = cost;
                            dc_cost_backtrack[k][bi] = l;
                        }
                    }
                }
            }
        }

        // ===== Speed limiting (C trellis_speed_level formula) =====
        let mut max_lookback = 63usize;
        let mut max_ac_candidates = 16usize;
        let speed_level = params.speed_level as i32;
        if speed_level > 0 {
            let mut nonzero_count = 0i32;
            for i in 1..DCTSIZE2 {
                let z = JPEG_NATURAL_ORDER[i];
                let x = src[bi][z].abs();
                let q = 8 * qtbl[z] as i32;
                if (x + q / 2) / q > 0 {
                    nonzero_count += 1;
                }
            }
            let threshold = 61 - speed_level * 3;
            if nonzero_count > threshold {
                max_lookback = (26 - speed_level * 2).max(4) as usize;
                max_ac_candidates = (9 - (speed_level + 1) / 2).max(2) as usize;
            }
        }

        // ===== AC coefficient processing (Viterbi search) =====
        for i in 1..DCTSIZE2 {
            let z = JPEG_NATURAL_ORDER[i];
            let sign = src[bi][z] >> 31;
            let x = src[bi][z].abs();
            let q = 8 * qtbl[z] as i32;

            accumulated_zero_dist[i] =
                x.wrapping_mul(x) as f32 * lambda * lambda_tbl[z] + accumulated_zero_dist[i - 1];

            let mut qval = (x + q / 2) / q;
            if qval == 0 {
                out[bi][z] = 0;
                accumulated_cost[i] = COST_INFINITY;
                continue;
            }
            if qval > MAX_COEF_VALUE {
                qval = MAX_COEF_VALUE;
            }

            // Candidates: Huffman category boundaries, then qval
            let num_candidates = (jpeg_nbits(qval as i16) as usize).min(max_ac_candidates);
            let mut candidate = [0i32; 16];
            let mut candidate_bits = [0u8; 16];
            let mut candidate_dist = [0.0f32; 16];
            for k in 0..num_candidates {
                candidate[k] = if k < num_candidates - 1 {
                    (2 << k) - 1
                } else {
                    qval
                };
                let delta = candidate[k] * q - x;
                candidate_bits[k] = (k + 1) as u8;
                candidate_dist[k] = delta.wrapping_mul(delta) as f32 * lambda * lambda_tbl[z];
            }

            accumulated_cost[i] = COST_INFINITY;

            let j_start = i.saturating_sub(max_lookback);
            for j in j_start..i {
                let zz = JPEG_NATURAL_ORDER[j];
                if j != 0 && out[bi][zz] == 0 {
                    continue;
                }

                let zero_run = i - 1 - j;
                let (_, zrl_size) = actbl.get_code(0xF0);
                if (zero_run >> 4) != 0 && zrl_size == 0 {
                    continue;
                }
                let run_bits = (zero_run >> 4) as i32 * zrl_size as i32;
                let zero_run_mod = zero_run & 15;

                for k in 0..num_candidates {
                    let (_, coef_bits) =
                        actbl.get_code((16 * zero_run_mod) as u8 | candidate_bits[k]);
                    if coef_bits == 0 {
                        continue;
                    }
                    let rate = coef_bits as i32 + candidate_bits[k] as i32 + run_bits;
                    let cost = rate as f32
                        + candidate_dist[k]
                        + (accumulated_zero_dist[i - 1] - accumulated_zero_dist[j]
                            + accumulated_cost[j]);
                    if cost < accumulated_cost[i] {
                        out[bi][z] = ((candidate[k] ^ sign) - sign) as i16;
                        accumulated_cost[i] = cost;
                        run_start[i] = j;
                    }
                }
            }
        }

        // ===== EOB placement (C find_block_eob_position) =====
        let eob_cost = actbl.get_code(0x00).1 as f32;
        let se = DCTSIZE2 - 1;

        let mut last_coeff_idx = 0usize; // Ss - 1 == 0 for Ss == 1
        let cost_all_zeros = accumulated_zero_dist[se];
        let mut best_cost = cost_all_zeros + eob_cost;
        let mut best_cost_skip = cost_all_zeros;

        for i in 1..DCTSIZE2 {
            let z = JPEG_NATURAL_ORDER[i];
            if out[bi][z] != 0 {
                let cost =
                    accumulated_cost[i] + accumulated_zero_dist[se] - accumulated_zero_dist[i];
                let cost_wo_eob = cost;
                let cost = if i < se { cost + eob_cost } else { cost };
                if cost < best_cost {
                    best_cost = cost;
                    best_cost_skip = cost_wo_eob;
                    last_coeff_idx = i;
                }
            }
        }
        let has_eob: u8 = ((last_coeff_idx < se) as u8) + ((last_coeff_idx == 0) as u8);

        // C zero_trailing_coefficients
        let mut i = se;
        while i >= 1 {
            while i > last_coeff_idx {
                let z = JPEG_NATURAL_ORDER[i];
                out[bi][z] = 0;
                i -= 1;
            }
            if i >= 1 {
                last_coeff_idx = run_start[i];
                i -= 1;
            }
        }

        // ===== Cross-block EOB optimization state update =====
        if params.eob_opt {
            accumulated_zero_block_cost[bi + 1] = accumulated_zero_block_cost[bi] + cost_all_zeros;
            requires_eob[bi + 1] = has_eob;

            let mut best = COST_INFINITY;
            if has_eob != 2 {
                for i in 0..=bi {
                    if requires_eob[i] == 2 {
                        continue;
                    }
                    let zero_block_run = bi - i + requires_eob[i] as usize;
                    let nbits = jpeg_nbits(zero_block_run as i16) as i32;
                    let (_, eobn_bits) = actbl.get_code((16 * nbits) as u8);
                    let cost = best_cost_skip + accumulated_zero_block_cost[bi]
                        - accumulated_zero_block_cost[i]
                        + accumulated_block_cost[i]
                        + (eobn_bits as i32 + nbits) as f32;
                    if cost < best {
                        block_run_start[bi] = i;
                        best = cost;
                        accumulated_block_cost[bi + 1] = cost;
                    }
                }
            }
        }
    }

    // ===== Cross-block EOB finalization =====
    if params.eob_opt {
        let mut last_block = num_blocks;
        let mut best = COST_INFINITY;
        for i in 0..=num_blocks {
            if requires_eob[i] == 2 {
                continue;
            }
            let zero_block_run = num_blocks - i + requires_eob[i] as usize;
            let nbits = jpeg_nbits(zero_block_run as i16) as i32;
            let (_, eobn_bits) = actbl.get_code((16 * nbits) as u8);
            let cost = accumulated_zero_block_cost[num_blocks] - accumulated_zero_block_cost[i]
                + (eobn_bits as i32 + nbits) as f32;
            if cost < best {
                best = cost;
                last_block = i;
            }
        }

        let mut last_block = last_block as i64 - 1;
        let mut bi = num_blocks as i64 - 1;
        while bi >= 0 {
            while bi > last_block {
                for j in 1..DCTSIZE2 {
                    let z = JPEG_NATURAL_ORDER[j];
                    out[bi as usize][z] = 0;
                }
                bi -= 1;
            }
            if bi >= 0 {
                last_block = block_run_start[bi as usize] as i64 - 1;
                bi -= 1;
            }
        }
    }

    // ===== DC backtrack =====
    if params.dc_enabled {
        let mut j = 0usize;
        for i in 1..dc_cands {
            if accumulated_dc_cost[i][num_blocks - 1] < accumulated_dc_cost[j][num_blocks - 1] {
                j = i;
            }
        }
        for bi in (0..num_blocks).rev() {
            out[bi][0] = dc_candidate[j][bi];
            j = dc_cost_backtrack[j][bi];
        }
        *last_dc = out[num_blocks - 1][0];
    }
}

// ============================================================================
// Per-component driver (C compress_trellis_pass)
// ============================================================================

/// Geometry of one component's padded block grid in MCU-order storage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentGrid {
    /// Real block columns (`width_in_blocks`).
    pub width_in_blocks: usize,
    /// Real block rows (`height_in_blocks`).
    pub height_in_blocks: usize,
    /// Blocks per MCU, horizontally (`h_samp_factor`).
    pub h_samp: usize,
    /// Blocks per MCU, vertically (`v_samp_factor`).
    pub v_samp: usize,
    /// MCUs per image row (global MCU column count).
    pub mcu_cols: usize,
    /// MCU rows in the image (`total_iMCU_rows`).
    pub mcu_rows: usize,
}

impl ComponentGrid {
    /// Padded block columns (`round_up(width_in_blocks, h_samp)`).
    #[inline]
    pub fn padded_cols(&self) -> usize {
        self.mcu_cols * self.h_samp
    }

    /// Padded block rows (`round_up(height_in_blocks, v_samp)`).
    #[inline]
    pub fn padded_rows(&self) -> usize {
        self.mcu_rows * self.v_samp
    }

    /// Storage index of a padded-grid position.
    #[inline]
    pub fn index(&self, block_row: usize, block_col: usize) -> usize {
        block_to_mcu_index(
            block_row,
            block_col,
            self.mcu_cols,
            self.h_samp,
            self.v_samp,
        )
    }

    /// Number of real block rows inside the given iMCU row
    /// (`block_rows` in C's compress_trellis_pass).
    #[inline]
    fn real_rows_of_imcu(&self, imcu_row: usize) -> usize {
        if imcu_row < self.mcu_rows - 1 {
            self.v_samp
        } else {
            let rem = self.height_in_blocks % self.v_samp;
            if rem == 0 { self.v_samp } else { rem }
        }
    }
}

/// Run the C `compress_trellis_pass` logic for one component.
///
/// For each iMCU row: `lastDC` resets to 0, each real block row is processed
/// by [`quantize_trellis_row_exact`], and right-edge/bottom dummy blocks are
/// synthesized exactly like C (all-zero AC, DC copied from the last real
/// block — for bottom rows, per-MCU from the row above).
///
/// `raw` and `blocks` hold the whole padded grid in MCU order. Only real
/// blocks are re-quantized; pad positions are overwritten with dummy content.
pub(crate) fn run_component_exact_trellis(
    raw: &[[i32; DCTSIZE2]],
    blocks: &mut [[i16; DCTSIZE2]],
    grid: &ComponentGrid,
    params: &ExactTrellisParams,
) {
    let w = grid.width_in_blocks;
    let padded_cols = grid.padded_cols();
    let mut row_src: Vec<[i32; DCTSIZE2]> = vec![[0; DCTSIZE2]; w];
    let mut row_out: Vec<[i16; DCTSIZE2]> = vec![[0; DCTSIZE2]; w];
    let mut above_out: Vec<[i16; DCTSIZE2]> = vec![[0; DCTSIZE2]; w];
    let mut above_src: Vec<[i32; DCTSIZE2]> = vec![[0; DCTSIZE2]; w];

    for imcu_row in 0..grid.mcu_rows {
        let block_rows = grid.real_rows_of_imcu(imcu_row);
        let mut last_dc = 0i16;

        for block_row in 0..block_rows {
            let r = imcu_row * grid.v_samp + block_row;

            for (bi, c) in (0..w).enumerate() {
                let idx = grid.index(r, c);
                row_src[bi] = raw[idx];
                row_out[bi] = blocks[idx];
            }
            let above = if block_row > 0 && params.delta_dc_weight > 0.0 {
                let ra = r - 1;
                for (bi, c) in (0..w).enumerate() {
                    let idx = grid.index(ra, c);
                    above_out[bi] = blocks[idx];
                    above_src[bi] = raw[idx];
                }
                Some((&above_out[..], &above_src[..]))
            } else {
                None
            };

            quantize_trellis_row_exact(
                &row_src[..w],
                &mut row_out[..w],
                params,
                &mut last_dc,
                above,
            );

            // Write back the row and fill right-edge dummy blocks
            for (bi, c) in (0..w).enumerate() {
                blocks[grid.index(r, c)] = row_out[bi];
            }
            let last_real_dc = row_out[w - 1][0];
            for c in w..padded_cols {
                let idx = grid.index(r, c);
                blocks[idx] = [0i16; DCTSIZE2];
                blocks[idx][0] = last_real_dc;
            }
            last_dc = last_real_dc;
        }

        // Bottom dummy rows at the last iMCU row (per-MCU DC from row above)
        if imcu_row == grid.mcu_rows - 1 {
            for r in grid.height_in_blocks..grid.padded_rows() {
                for mcu in 0..grid.mcu_cols {
                    let above_last = grid.index(r - 1, mcu * grid.h_samp + grid.h_samp - 1);
                    let last_dc = blocks[above_last][0];
                    for h in 0..grid.h_samp {
                        let idx = grid.index(r, mcu * grid.h_samp + h);
                        blocks[idx] = [0i16; DCTSIZE2];
                        blocks[idx][0] = last_dc;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Per-pass statistics gather (C encode_mcu_gather + finish_pass_gather)
// ============================================================================

/// Which entropy counting model a pass's gather uses. C picks this from
/// `cinfo->progressive_mode`, independent of the pass type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatherModel {
    /// Baseline/sequential counting (`htest_one_block`): DC diff nbits +
    /// all 63 AC (run,size) symbols + ZRL + EOB. Ignores Ss/Se entirely.
    Sequential,
    /// Progressive first-scan counting (Al=0): band-limited AC symbols with
    /// EOBRUN accounting, flushed at restarts and at pass end. DC slots are
    /// never touched by AC-band gathers.
    Progressive,
}

/// Gather one component's statistics exactly like a C single-component
/// "scan" (`comps_in_scan = 1`): `per_scan_setup` gives
/// `width_in_blocks × height_in_blocks` MCUs of one block each, covering
/// real blocks only — pads/dummies are never counted. DC prediction chains
/// across the whole scan from 0 and resets every `restart_interval` blocks
/// (restarts count this scan's MCUs = real blocks).
fn gather_comp_pass(
    blocks: &[[i16; DCTSIZE2]],
    grid: &ComponentGrid,
    model: GatherModel,
    restart_interval: usize,
    dc_freq: &mut FrequencyCounter,
    ac_freq: &mut FrequencyCounter,
) {
    let w = grid.width_in_blocks;
    let h = grid.height_in_blocks;

    match model {
        GatherModel::Sequential => {
            let mut last_dc = 0i16;
            let mut n = 0usize;
            for r in 0..h {
                for c in 0..w {
                    if restart_interval > 0 && n > 0 && n.is_multiple_of(restart_interval) {
                        last_dc = 0;
                    }
                    let block = &blocks[grid.index(r, c)];
                    let diff = block[0].wrapping_sub(last_dc);
                    last_dc = block[0];
                    dc_freq.count(jpeg_nbits(diff));
                    count_block_ac_symbols(block, ac_freq);
                    n += 1;
                }
            }
        }
        GatherModel::Progressive => {
            // C jcphuff.c start_pass_phuff: during trellis-region passes the
            // count table is pre-seeded so every symbol gets a code length
            // ("make sure that all codewords have an assigned length").
            for i in 0..16usize {
                for j in 0..12usize {
                    ac_freq.counts[16 * i + j] = 1;
                }
            }
            let mut counter = ProgressiveSymbolCounter::new();
            let mut n = 0usize;
            for r in 0..h {
                for c in 0..w {
                    if restart_interval > 0 && n > 0 && n.is_multiple_of(restart_interval) {
                        // C emit_restart: flush pending EOBRUN
                        counter.finish_scan(Some(ac_freq));
                    }
                    counter.count_ac_first(&blocks[grid.index(r, c)], 1, 63, 0, ac_freq);
                    n += 1;
                }
            }
            counter.finish_scan(Some(ac_freq));
        }
    }
}

/// Sequential AC symbol counting for one block (`htest_one_block`).
fn count_block_ac_symbols(block: &[i16; DCTSIZE2], freq: &mut FrequencyCounter) {
    let mut run = 0u8;
    for i in 1..DCTSIZE2 {
        let coef = block[JPEG_NATURAL_ORDER[i]];
        if coef == 0 {
            run += 1;
        } else {
            while run >= 16 {
                freq.count(0xF0);
                run -= 16;
            }
            let nbits = jpeg_nbits(coef);
            freq.count((run << 4) | nbits);
            run = 0;
        }
    }
    if run > 0 {
        freq.count(0x00);
    }
}

// ============================================================================
// C pass-schedule orchestration (jcmaster.c)
// ============================================================================

use crate::types::TrellisConfig;

/// One component's inputs for the exact trellis pass sequence.
pub(crate) struct ExactComponent<'a> {
    /// Quantized coefficients, whole padded grid in MCU order.
    pub blocks: &'a mut [[i16; DCTSIZE2]],
    /// Raw (unquantized ×8) DCT coefficients, same layout.
    pub raw: &'a [[i32; DCTSIZE2]],
    /// Component geometry.
    pub grid: ComponentGrid,
    /// Quantization table (natural order).
    pub qtbl: &'a [u16; DCTSIZE2],
    /// DC Huffman table index used by this component (0=luma, 1=chroma).
    pub dc_tbl_no: usize,
    /// AC Huffman table index used by this component (0=luma, 1=chroma).
    pub ac_tbl_no: usize,
    /// Blocks per interleaved MCU (h_samp × v_samp); used by the
    /// `optimize_coding` final real-scan gather which iterates the
    /// interleaved scan.
    pub blocks_per_mcu: usize,
}

/// Standard (Annex K) Huffman tables, referenced for slots C initializes
/// with default tables.
pub(crate) struct StdHuffTables<'a> {
    pub dc_luma: &'a HuffTable,
    pub dc_chroma: &'a HuffTable,
    pub ac_luma: &'a HuffTable,
    pub ac_chroma: &'a HuffTable,
}

/// Huffman tables left in C's table slots after the trellis pass sequence.
///
/// Only meaningful for `optimize_coding = false` output: the scan header
/// then emits whatever the last gathers produced. With
/// `optimize_coding = true` the real scans re-gather everything anyway.
#[derive(Default)]
pub(crate) struct ExactTablesOut {
    /// Final DC table per Huffman slot (baseline gathers only).
    pub dc_huff: [Option<HuffTable>; 4],
    /// Final AC table per Huffman slot.
    pub ac_huff: [Option<HuffTable>; 4],
}

/// Fill padding blocks exactly like C's `compress_first_pass`:
/// right-edge dummies on each real block row (zero AC, DC = last real
/// block's DC) and bottom dummy rows in the last iMCU row (per-MCU DC from
/// the row above).
///
/// Must run while `blocks` still holds *normally quantized* coefficients,
/// before the exact trellis passes re-quantize real blocks.
pub(crate) fn fill_main_pass_dummies(blocks: &mut [[i16; DCTSIZE2]], grid: &ComponentGrid) {
    let w = grid.width_in_blocks;
    let padded_cols = grid.padded_cols();
    if padded_cols == w && grid.padded_rows() == grid.height_in_blocks {
        return;
    }

    for imcu_row in 0..grid.mcu_rows {
        let block_rows = grid.real_rows_of_imcu(imcu_row);
        for block_row in 0..block_rows {
            let r = imcu_row * grid.v_samp + block_row;
            let last_dc = blocks[grid.index(r, w - 1)][0];
            for c in w..padded_cols {
                let idx = grid.index(r, c);
                blocks[idx] = [0i16; DCTSIZE2];
                blocks[idx][0] = last_dc;
            }
        }
        if imcu_row == grid.mcu_rows - 1 {
            for r in grid.height_in_blocks..grid.padded_rows() {
                for mcu in 0..grid.mcu_cols {
                    let above_last = grid.index(r - 1, mcu * grid.h_samp + grid.h_samp - 1);
                    let last_dc = blocks[above_last][0];
                    for h in 0..grid.h_samp {
                        let idx = grid.index(r, mcu * grid.h_samp + h);
                        blocks[idx] = [0i16; DCTSIZE2];
                        blocks[idx][0] = last_dc;
                    }
                }
            }
        }
    }
}

/// C mozjpeg restart configuration: `restart_interval` (absolute MCUs,
/// `-restart NB`) or `restart_in_rows` (`-restart N`), converted to an
/// absolute interval *per scan* via that scan's `MCUs_per_row`
/// (per_scan_setup). Single-component scans count the component's real
/// blocks; interleaved scans count iMCUs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RestartSpec {
    /// Absolute MCUs per restart, in each scan's MCU units (`-restart NB`).
    pub interval: usize,
    /// MCU rows per restart, or 0 (`-restart N`) — converted per scan.
    pub rows: usize,
}

impl RestartSpec {
    /// Effective restart interval for a scan with `mcus_per_row` MCUs per
    /// row, capped at 65535 like C (`MIN(nominal, 65535)`).
    #[inline]
    pub fn for_scan(&self, mcus_per_row: usize) -> usize {
        if self.rows > 0 {
            (self.rows * mcus_per_row).min(65535)
        } else {
            self.interval
        }
    }
}

/// Faithful port of C `jpeg_gen_optimal_table` (jchuff.c).
///
/// Differs from the shared [`crate::huffman::generate_optimal_table`] only
/// when Huffman depth-limiting occurs (any code > 16 bits): C computes
/// `bit_pos` from the *pre-limiting* length distribution and places symbols
/// by their original codesize, while the shared version re-sorts into the
/// limited lengths. Results are identical in the common (no-limiting) case;
/// exact mode uses this port so even the extreme case is byte-exact.
#[allow(clippy::needless_range_loop)]
fn gen_optimal_table_c(freq: &mut [i64; 257]) -> HuffTable {
    const MAX_CLEN: usize = 32;
    let mut htbl = HuffTable::default();
    let mut bits = [0u8; MAX_CLEN + 1];
    let mut bit_pos = [0usize; MAX_CLEN + 1];
    let mut codesize = [0usize; 257];
    let mut nz_index = [0usize; 257];
    let mut others = [-1i32; 257];

    freq[256] = 1; // pseudo-symbol

    let mut num_nz = 0usize;
    for i in 0..257 {
        if freq[i] > 0 {
            nz_index[num_nz] = i;
            freq[num_nz] = freq[i];
            num_nz += 1;
        }
    }

    loop {
        let mut c1: i32 = -1;
        let mut c2: i32 = -1;
        let mut v1: i64 = 1_000_000_000;
        let mut v2: i64 = 1_000_000_000;
        for i in 0..num_nz {
            if freq[i] <= v2 {
                if freq[i] <= v1 {
                    c2 = c1;
                    v2 = v1;
                    v1 = freq[i];
                    c1 = i as i32;
                } else {
                    v2 = freq[i];
                    c2 = i as i32;
                }
            }
        }
        if c2 < 0 {
            break;
        }
        let c1u = c1 as usize;
        let c2u = c2 as usize;
        freq[c1u] += freq[c2u];
        freq[c2u] = 1_000_000_001;

        codesize[c1u] += 1;
        let mut node = c1u;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }
        others[node] = c2;

        codesize[c2u] += 1;
        let mut node = c2u;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }
    }

    for i in 0..num_nz {
        debug_assert!(codesize[i] <= MAX_CLEN);
        bits[codesize[i].min(MAX_CLEN)] += 1;
    }

    // C computes bit_pos BEFORE depth limiting.
    let mut p = 0usize;
    for i in 1..=MAX_CLEN {
        bit_pos[i] = p;
        p += bits[i] as usize;
    }

    // Depth limiting (JPEG K.2 / T.81 Annex K)
    for i in (17..=MAX_CLEN).rev() {
        while bits[i] > 0 {
            let mut j = i - 2;
            while j > 0 && bits[j] == 0 {
                j -= 1;
            }
            if j == 0 {
                // C would run off the table; unreachable for valid Huffman
                // inputs but never produce a corrupt table in release.
                debug_assert!(false, "jpeg_gen_optimal_table: no shorter length");
                break;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }

    // Remove pseudo-symbol from the largest codelength
    let mut i = MAX_CLEN;
    while bits[i] == 0 {
        i -= 1;
    }
    bits[i] -= 1;

    htbl.bits[0] = 0;
    htbl.bits[1..=16].copy_from_slice(&bits[1..=16]);

    // huffval ordered by ORIGINAL codesize (C quirk, see note above)
    for i in 0..num_nz.saturating_sub(1) {
        let cs = codesize[i].min(MAX_CLEN);
        let pos = bit_pos[cs];
        if pos < 256 {
            htbl.huffval[pos] = nz_index[i] as u8;
        }
        bit_pos[cs] += 1;
    }

    htbl
}

/// A pass's `finish_pass` gather: count the component's single-comp scan
/// and rewrite its Huffman table slots (`jpeg_gen_optimal_table` writes
/// into `dc/ac_huff_tbl_ptrs`).
fn gather_into_slots(
    comp: &ExactComponent<'_>,
    model: GatherModel,
    restart_interval: usize,
    dc_huff: &mut [Option<HuffTable>; 4],
    ac_huff: &mut [Option<HuffTable>; 4],
) -> Result<()> {
    let mut dc_freq = FrequencyCounter::new();
    let mut ac_freq = FrequencyCounter::new();
    gather_comp_pass(
        comp.blocks,
        &comp.grid,
        model,
        restart_interval,
        &mut dc_freq,
        &mut ac_freq,
    );
    if model == GatherModel::Sequential {
        dc_huff[comp.dc_tbl_no] = Some(gen_optimal_table_c(&mut dc_freq.counts));
    }
    ac_huff[comp.ac_tbl_no] = Some(gen_optimal_table_c(&mut ac_freq.counts));
    Ok(())
}

/// The `optimize_coding` final real-scan gather: C's last huff_opt pass
/// (`pass_number == scan_opt_base`) processes the real interleaved scan
/// (`comps_in_scan = num_components`) — every MCU in scan order, including
/// right-edge and bottom dummy blocks — and its `finish_pass_gather`
/// counts the symbols into the shared per-`tbl_no` count arrays, then runs
/// `jpeg_gen_optimal_table` once per slot (guarded by `did_dc`/`did_ac`).
/// Components that share a table slot therefore get a table generated from
/// their *joint* statistics.
///
/// DC prediction chains per component across the whole scan in MCU order;
/// `restart_interval` resets count MCUs of the real scan, not blocks.
fn gather_main_scan_into_slots(
    comps: &[ExactComponent<'_>],
    restart_interval: usize,
    dc_huff: &mut [Option<HuffTable>; 4],
    ac_huff: &mut [Option<HuffTable>; 4],
) -> Result<()> {
    let (mut dc_freq, mut ac_freq) = count_main_scan(comps, restart_interval);
    let (used_dc, used_ac) = used_slots(comps);
    for t in 0..4 {
        if used_dc[t] {
            dc_huff[t] = Some(gen_optimal_table_c(&mut dc_freq[t].counts));
        }
        if used_ac[t] {
            ac_huff[t] = Some(gen_optimal_table_c(&mut ac_freq[t].counts));
        }
    }
    Ok(())
}

/// Per-slot symbol counts of the real interleaved baseline scan over the
/// final coefficients: every MCU in scan order, including right-edge and
/// bottom dummy blocks. DC prediction chains per component across the
/// whole scan; `restart_interval` resets count MCUs of the real scan.
fn count_main_scan(
    comps: &[ExactComponent<'_>],
    restart_interval: usize,
) -> ([FrequencyCounter; 4], [FrequencyCounter; 4]) {
    let mut dc_freq: [FrequencyCounter; 4] = std::array::from_fn(|_| FrequencyCounter::new());
    let mut ac_freq: [FrequencyCounter; 4] = std::array::from_fn(|_| FrequencyCounter::new());

    let mcu_rows = comps[0].grid.mcu_rows;
    let mcu_cols = comps[0].grid.mcu_cols;
    let mut last_dc = [0i16; 4];
    let mut mcu_count = 0usize;
    for my in 0..mcu_rows {
        for mx in 0..mcu_cols {
            if restart_interval > 0 && mcu_count > 0 && mcu_count.is_multiple_of(restart_interval) {
                last_dc = [0; 4];
            }
            for (ci, comp) in comps.iter().enumerate() {
                for bi in 0..comp.blocks_per_mcu {
                    let r = my * comp.grid.v_samp + bi / comp.grid.h_samp;
                    let c = mx * comp.grid.h_samp + bi % comp.grid.h_samp;
                    let block = &comp.blocks[comp.grid.index(r, c)];
                    let diff = block[0].wrapping_sub(last_dc[ci]);
                    last_dc[ci] = block[0];
                    dc_freq[comp.dc_tbl_no].count(jpeg_nbits(diff));
                    count_block_ac_symbols(block, &mut ac_freq[comp.ac_tbl_no]);
                }
            }
            mcu_count += 1;
        }
    }
    (dc_freq, ac_freq)
}

/// Which DC and AC Huffman slots the components reference.
fn used_slots(comps: &[ExactComponent<'_>]) -> ([bool; 4], [bool; 4]) {
    let mut used_dc = [false; 4];
    let mut used_ac = [false; 4];
    for comp in comps {
        used_dc[comp.dc_tbl_no] = true;
        used_ac[comp.ac_tbl_no] = true;
    }
    (used_dc, used_ac)
}

/// Whether `table` has a code for every symbol `freq` counts.
fn covers(table: &HuffTable, freq: &FrequencyCounter) -> bool {
    let n: usize = table.bits[1..=16].iter().map(|&b| b as usize).sum();
    let mut coded = [false; 256];
    for &sym in &table.huffval[..n] {
        coded[sym as usize] = true;
    }
    freq.counts[..256]
        .iter()
        .zip(coded)
        .all(|(&count, has_code)| count == 0 || has_code)
}

/// DIVERGENCE from C (see DIVERGENCES.md): with `optimize_coding = FALSE`
/// C emits the slot tables its trellis passes gathered from component 0's
/// single-component scans. The real scan can need symbols those tables
/// never saw — dummy blocks and MCU-order DC differences under
/// subsampling, or G/B sharing R's slot under JCS_RGB — and C then writes
/// codes of length 0, an undecodable file. Replace any slot table that
/// cannot code the real scan with the optimal table for that scan (what
/// `optimize_coding = TRUE` would emit for it). Output C can decode is
/// left byte-identical.
fn cover_main_scan(
    comps: &[ExactComponent<'_>],
    restart_interval: usize,
    std: &StdHuffTables<'_>,
    dc_huff: &mut [Option<HuffTable>; 4],
    ac_huff: &mut [Option<HuffTable>; 4],
) {
    let (mut dc_freq, mut ac_freq) = count_main_scan(comps, restart_interval);
    let (used_dc, used_ac) = used_slots(comps);
    for t in 0..4 {
        if used_dc[t] {
            let table =
                dc_huff[t]
                    .as_ref()
                    .unwrap_or(if t == 0 { std.dc_luma } else { std.dc_chroma });
            if !covers(table, &dc_freq[t]) {
                dc_huff[t] = Some(gen_optimal_table_c(&mut dc_freq[t].counts));
            }
        }
        if used_ac[t] {
            let table =
                ac_huff[t]
                    .as_ref()
                    .unwrap_or(if t == 0 { std.ac_luma } else { std.ac_chroma });
            if !covers(table, &ac_freq[t]) {
                ac_huff[t] = Some(gen_optimal_table_c(&mut ac_freq[t].counts));
            }
        }
    }
}

/// Run the exact C trellis pass sequence over all components.
///
/// Reproduces the `jcmaster.c` schedule for the fork's defaults
/// (`use_scans_in_trellis = FALSE`, `trellis_num_loops = 1`,
/// `trellis_speed_level = 7`, `trellis_eob_opt = FALSE`):
///
/// - Pass 0 ("main pass", `compress_first_pass`): all components are
///   normally quantized (done by the caller); its statistics gather is a
///   single-component scan over component 0's real blocks, installing
///   comp0's optimal tables (baseline also gathers DC; progressive AC-band
///   gathers never touch DC slots — they stay standard).
/// - `optimize_coding`: passes alternate `trellis`/`huff_opt`
///   single-component scans. Comp0 is trellised with the post-main table;
///   each later component is gathered (its still-normal coefficients) then
///   trellised with its freshly-gathered table. Every pass's gather
///   rewrites that component's table slots.
/// - `!optimize_coding`: C's `cur_comp_info` is never re-selected for
///   trellis passes, so component 0 is re-trellised `num_components` times
///   against its evolving table — components 1..N are **not** trellised
///   (a faithful C quirk) and emit standard tables.
///
/// Returns the Huffman slot state for `optimize_coding = false` output DHTs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_exact_trellis(
    comps: &mut [ExactComponent<'_>],
    progressive: bool,
    optimize_coding: bool,
    restart: RestartSpec,
    std: &StdHuffTables<'_>,
    trellis: &TrellisConfig,
) -> Result<ExactTablesOut> {
    let model = if progressive {
        GatherModel::Progressive
    } else {
        GatherModel::Sequential
    };
    // C forces `optimize_coding = TRUE` for progressive mode
    // (jcmaster.c: "assume default tables no good for progressive mode"),
    // so `!optimize_coding` only ever occurs for baseline scans.
    let optimize_coding = optimize_coding || progressive;

    // The main pass creates dummy blocks with normally-quantized DCs.
    for comp in comps.iter_mut() {
        fill_main_pass_dummies(comp.blocks, &comp.grid);
    }

    // Huffman table slots — C's `dc/ac_huff_tbl_ptrs`. `None` means the
    // standard table is installed (slots start at the defaults).
    let mut dc_huff: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut ac_huff: [Option<HuffTable>; 4] = [None, None, None, None];

    // Pass 0 (main pass) gather. For both modes C's select_scan_parameters
    // takes the `pass_number < scan_opt_base` branch at pass 0: a
    // single-component scan over comp0's real blocks.
    gather_into_slots(
        &comps[0],
        model,
        restart.for_scan(comps[0].grid.width_in_blocks),
        &mut dc_huff,
        &mut ac_huff,
    )?;

    let n = comps.len();

    // Trellis pass order (C pass numbers 1..pass_number_scan_opt_base):
    //   optimize_coding: each component once, in order — the trellis pass
    //     inherits comps_in_scan=1 from the preceding gather pass's
    //     select_scan_parameters, so compress_trellis_pass sees exactly one
    //     component per pass.
    //   !optimize_coding: select_scan_parameters never runs again, so all
    //     num_components trellis passes keep trellising *comp0* (a C quirk);
    //     each pass's finish gather rewrites only comp0's slot. Components
    //     1..N are never trellised and emit the standard tables installed
    //     by jpeg_set_defaults (their slot stays None here).
    for pass in 0..n {
        // Component this pass trellises: the pass's own for opt, comp0 for !opt.
        let ci = if optimize_coding { pass } else { 0 };
        if optimize_coding && pass > 0 {
            gather_into_slots(
                &comps[ci],
                model,
                restart.for_scan(comps[ci].grid.width_in_blocks),
                &mut dc_huff,
                &mut ac_huff,
            )?;
        }
        let dc_tbl_no = comps[ci].dc_tbl_no;
        let ac_tbl_no = comps[ci].ac_tbl_no;
        let dc_tbl = dc_huff[dc_tbl_no].as_ref().unwrap_or(if dc_tbl_no == 0 {
            std.dc_luma
        } else {
            std.dc_chroma
        });
        let ac_tbl = ac_huff[ac_tbl_no].as_ref().unwrap_or(if ac_tbl_no == 0 {
            std.ac_luma
        } else {
            std.ac_chroma
        });
        let dctbl = DerivedTable::from_huff_table(dc_tbl, true)?;
        let actbl = DerivedTable::from_huff_table(ac_tbl, false)?;

        let params = ExactTrellisParams {
            dctbl: &dctbl,
            actbl: &actbl,
            qtbl: comps[ci].qtbl,
            dc_enabled: trellis.dc_enabled,
            delta_dc_weight: trellis.delta_dc_weight,
            speed_level: trellis.mode.c_speed_level().unwrap_or(0),
            lambda_scale1: trellis.lambda_log_scale1,
            lambda_scale2: trellis.lambda_log_scale2,
            eob_opt: trellis.eob_opt,
        };
        let comp = &mut comps[ci];
        run_component_exact_trellis(comp.raw, comp.blocks, &comp.grid, &params);

        // For `!optimize_coding` every trellis pass's finish_pass re-gathers
        // comp0's just-trellised output — feeding the next pass's tables
        // and, at the last pass, the emitted slot-0 tables.
        if !optimize_coding {
            gather_into_slots(
                &comps[ci],
                model,
                restart.for_scan(comps[ci].grid.width_in_blocks),
                &mut dc_huff,
                &mut ac_huff,
            )?;
        }
    }

    // With `optimize_coding` the last pass before output is the real
    // scan's huff_opt gather: a joint interleaved count over the final
    // (post-trellis) coefficients, including dummies and restart resets —
    // this produces the emitted tables for a single-scan baseline image.
    if optimize_coding && !progressive {
        gather_main_scan_into_slots(
            comps,
            restart.for_scan(comps[0].grid.mcu_cols),
            &mut dc_huff,
            &mut ac_huff,
        )?;
    } else if !optimize_coding {
        cover_main_scan(
            comps,
            restart.for_scan(comps[0].grid.mcu_cols),
            std,
            &mut dc_huff,
            &mut ac_huff,
        );
    }

    Ok(ExactTablesOut { dc_huff, ac_huff })
}
