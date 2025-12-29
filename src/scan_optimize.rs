//! Scan optimization for progressive JPEG (matches C mozjpeg's optimize_scans).
//!
//! This implements the scan search algorithm from mozjpeg's jcmaster.c.
//! It generates 64 candidate scans for YCbCr images and selects the optimal
//! subset based on encoded size.

use crate::types::ScanInfo;

/// Enable debug instrumentation for scan optimization
const DEBUG_SCAN_OPT: bool = false;

/// Configuration for scan optimization search.
#[derive(Debug, Clone)]
pub struct ScanSearchConfig {
    /// Maximum successive approximation level for luma (default: 3)
    pub al_max_luma: u8,
    /// Maximum successive approximation level for chroma (default: 2)
    pub al_max_chroma: u8,
    /// Frequency split points to test (default: [2, 8, 5, 12, 18])
    pub frequency_splits: Vec<u8>,
    /// DC scan optimization mode (0=interleaved, 1=separate, 2=luma+chroma pair)
    pub dc_scan_opt_mode: u8,
}

impl Default for ScanSearchConfig {
    fn default() -> Self {
        Self {
            // Match C mozjpeg's defaults for optimize_scans
            al_max_luma: 3,
            al_max_chroma: 2,
            frequency_splits: vec![2, 8, 5, 12, 18],
            dc_scan_opt_mode: 0,
        }
    }
}

/// Generate the full set of 64 scan candidates for YCbCr images.
/// This matches C mozjpeg's jpeg_search_progression() EXACTLY.
///
/// Scan layout for YCbCr (64 scans with default config):
///
/// LUMA (23 scans):
///   0: DC scan (all components or just luma depending on dc_scan_opt_mode)
///   1: Y AC 1-8 at Al=0
///   2: Y AC 9-63 at Al=0
///   For Al in 0..Al_max_luma (3 iterations, 3 scans each = 9 scans):
///     3+3*Al+0: Y refinement 1-63 (Ah=Al+1, Al=Al)
///     3+3*Al+1: Y AC 1-8 at Al+1
///     3+3*Al+2: Y AC 9-63 at Al+1
///   12: Y full AC 1-63 at Al=0
///   13-22: Frequency splits (5 pairs)
///
/// CHROMA (41 scans):
///   23: Cb+Cr combined DC
///   24: Cb DC alone
///   25: Cr DC alone
///   26-29: Cb/Cr base AC (1-8, 9-63 for each)
///   For Al in 0..Al_max_chroma (2 iterations, 6 scans each = 12 scans):
///     30+6*Al+0: Cb refinement 1-63
///     30+6*Al+1: Cr refinement 1-63
///     30+6*Al+2: Cb 1-8 at Al+1
///     30+6*Al+3: Cb 9-63 at Al+1
///     30+6*Al+4: Cr 1-8 at Al+1
///     30+6*Al+5: Cr 9-63 at Al+1
///   42: Cb full 1-63 at Al=0
///   43: Cr full 1-63 at Al=0
///   44-63: Chroma frequency splits (5 pairs × 2 components)
pub fn generate_search_scans(num_components: u8, config: &ScanSearchConfig) -> Vec<ScanInfo> {
    if num_components == 1 {
        return generate_grayscale_search_scans(config);
    }

    let mut scans = Vec::with_capacity(64);

    // === LUMA SCANS (23 scans) ===
    // Matches jcparam.c:790-810

    // Scan 0: DC scan for all components (or just luma depending on mode)
    if config.dc_scan_opt_mode == 0 {
        scans.push(ScanInfo::dc_scan(num_components));
    } else {
        scans.push(ScanInfo::dc_scan(1)); // Just luma
    }

    // Scans 1-2: Base AC scans at Al=0
    scans.push(ScanInfo::ac_scan(0, 1, 8, 0, 0));
    scans.push(ScanInfo::ac_scan(0, 9, 63, 0, 0));

    // Scans 3-11: Successive approximation test scans for luma (3 scans per Al level)
    // C code: for (Al = 0; Al < Al_max_luma; Al++) {
    //   refinement, 1-8 at Al+1, 9-63 at Al+1
    // }
    for al in 0..config.al_max_luma {
        scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al)); // Refinement 1-63
        scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1)); // 1-8 at higher Al
        scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1)); // 9-63 at higher Al
    }

    // Scan 12: Full luma AC scan (no successive approx) - for frequency split baseline
    scans.push(ScanInfo::ac_scan(0, 1, 63, 0, 0));

    // Scans 13-22: Frequency split test scans for luma (5 pairs)
    for &split in &config.frequency_splits {
        scans.push(ScanInfo::ac_scan(0, 1, split, 0, 0));
        scans.push(ScanInfo::ac_scan(0, split + 1, 63, 0, 0));
    }

    // === CHROMA SCANS (41 scans for 3-component) ===
    // Matches jcparam.c:820-848

    if num_components >= 3 {
        // Scans 23-25: DC scan variants for chroma
        scans.push(ScanInfo::dc_scan_pair(1, 2)); // Combined Cb+Cr DC
        scans.push(ScanInfo::dc_scan_single(1)); // Cb DC alone
        scans.push(ScanInfo::dc_scan_single(2)); // Cr DC alone

        // Scans 26-29: Base AC scans for Cb and Cr at Al=0
        scans.push(ScanInfo::ac_scan(1, 1, 8, 0, 0));
        scans.push(ScanInfo::ac_scan(1, 9, 63, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 1, 8, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 9, 63, 0, 0));

        // Scans 30-41: Successive approximation test scans for chroma (6 scans per Al level)
        // C code: for (Al = 0; Al < Al_max_chroma; Al++) {
        //   Cb refine, Cr refine, Cb bands, Cr bands
        // }
        for al in 0..config.al_max_chroma {
            scans.push(ScanInfo::ac_scan(1, 1, 63, al + 1, al)); // Cb refinement
            scans.push(ScanInfo::ac_scan(2, 1, 63, al + 1, al)); // Cr refinement
            scans.push(ScanInfo::ac_scan(1, 1, 8, 0, al + 1)); // Cb 1-8 at Al+1
            scans.push(ScanInfo::ac_scan(1, 9, 63, 0, al + 1)); // Cb 9-63 at Al+1
            scans.push(ScanInfo::ac_scan(2, 1, 8, 0, al + 1)); // Cr 1-8 at Al+1
            scans.push(ScanInfo::ac_scan(2, 9, 63, 0, al + 1)); // Cr 9-63 at Al+1
        }

        // Scans 42-43: Full chroma AC scans at Al=0
        scans.push(ScanInfo::ac_scan(1, 1, 63, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 1, 63, 0, 0));

        // Scans 44-63: Frequency split test scans for chroma (5 pairs × 2 components)
        for &split in &config.frequency_splits {
            scans.push(ScanInfo::ac_scan(1, 1, split, 0, 0));
            scans.push(ScanInfo::ac_scan(1, split + 1, 63, 0, 0));
            scans.push(ScanInfo::ac_scan(2, 1, split, 0, 0));
            scans.push(ScanInfo::ac_scan(2, split + 1, 63, 0, 0));
        }
    }

    scans
}

/// Generate search scans for grayscale (23 scans).
/// Matches C mozjpeg exactly - same layout as luma scans.
fn generate_grayscale_search_scans(config: &ScanSearchConfig) -> Vec<ScanInfo> {
    let mut scans = Vec::with_capacity(23);

    // Scan 0: DC scan
    scans.push(ScanInfo::dc_scan(1));

    // Scans 1-2: Base AC scans at Al=0
    scans.push(ScanInfo::ac_scan(0, 1, 8, 0, 0));
    scans.push(ScanInfo::ac_scan(0, 9, 63, 0, 0));

    // Scans 3-11: Successive approximation test scans (3 scans per Al level)
    for al in 0..config.al_max_luma {
        scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al)); // Refinement
        scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1)); // 1-8 at Al+1
        scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1)); // 9-63 at Al+1
    }

    // Scan 12: Full AC scan at Al=0
    scans.push(ScanInfo::ac_scan(0, 1, 63, 0, 0));

    // Scans 13-22: Frequency split test scans (5 pairs)
    for &split in &config.frequency_splits {
        scans.push(ScanInfo::ac_scan(0, 1, split, 0, 0));
        scans.push(ScanInfo::ac_scan(0, split + 1, 63, 0, 0));
    }

    scans
}

/// Results from scan optimization search.
#[derive(Debug, Clone)]
pub struct ScanSearchResult {
    /// Best successive approximation level for luma
    pub best_al_luma: u8,
    /// Best successive approximation level for chroma
    pub best_al_chroma: u8,
    /// Best frequency split index for luma (index into frequency_splits)
    pub best_freq_split_luma: usize,
    /// Best frequency split index for chroma
    pub best_freq_split_chroma: usize,
    /// Whether to interleave chroma DC scans
    pub interleave_chroma_dc: bool,
    /// Sizes of each scan (for analysis)
    pub scan_sizes: Vec<usize>,
}

impl ScanSearchResult {
    /// Build the final optimized scan script from search results.
    ///
    /// When best_freq_split is 0, we use full 1-63 AC scans (most efficient for simple images).
    /// When best_freq_split > 0, we use the selected frequency split.
    ///
    /// When successive approximation (Al > 0) is selected, we also apply it to DC:
    /// - Initial DC scan with point transform (al=1)
    /// - DC refinement scan at the end (ah=1, al=0)
    pub fn build_final_scans(
        &self,
        num_components: u8,
        config: &ScanSearchConfig,
    ) -> Vec<ScanInfo> {
        let mut scans = Vec::new();

        // Determine if we should use DC successive approximation
        // Use it when either luma or chroma Al is > 0 (using successive approximation)
        let use_dc_succ_approx = self.best_al_luma > 0 || self.best_al_chroma > 0;

        // DC scan - use point transform if doing successive approximation
        if config.dc_scan_opt_mode == 0 {
            let mut dc_scan = ScanInfo::dc_scan(num_components);
            if use_dc_succ_approx {
                dc_scan.al = 1; // Point transform for DC
            }
            scans.push(dc_scan);
        } else {
            let mut dc_scan = ScanInfo::dc_scan(1);
            if use_dc_succ_approx {
                dc_scan.al = 1;
            }
            scans.push(dc_scan);
        }

        // Luma AC scans based on best Al and frequency split
        let al = self.best_al_luma;
        if self.best_freq_split_luma == 0 {
            // Use full 1-63 range (no frequency split) - most efficient for simple images
            scans.push(ScanInfo::ac_scan(0, 1, 63, 0, al));
        } else {
            let split = config.frequency_splits[self.best_freq_split_luma - 1];
            scans.push(ScanInfo::ac_scan(0, 1, split, 0, al));
            scans.push(ScanInfo::ac_scan(0, split + 1, 63, 0, al));
        }

        // Luma refinement scans if Al > 0
        for refine_al in (0..al).rev() {
            scans.push(ScanInfo::ac_scan(0, 1, 63, refine_al + 1, refine_al));
        }

        if num_components >= 3 {
            // Chroma DC - only add if DC wasn't already included for all components
            // When dc_scan_opt_mode == 0, DC for all components is in the first scan
            if config.dc_scan_opt_mode != 0 {
                if self.interleave_chroma_dc {
                    scans.push(ScanInfo::dc_scan_pair(1, 2));
                } else {
                    scans.push(ScanInfo::dc_scan_single(1));
                    scans.push(ScanInfo::dc_scan_single(2));
                }
            }

            // Chroma AC scans
            let al_c = self.best_al_chroma;
            for comp in 1..=2u8 {
                if self.best_freq_split_chroma == 0 {
                    // Use full 1-63 range (no frequency split)
                    scans.push(ScanInfo::ac_scan(comp, 1, 63, 0, al_c));
                } else {
                    let split = config.frequency_splits[self.best_freq_split_chroma - 1];
                    scans.push(ScanInfo::ac_scan(comp, 1, split, 0, al_c));
                    scans.push(ScanInfo::ac_scan(comp, split + 1, 63, 0, al_c));
                }
            }

            // Chroma refinement
            for refine_al in (0..al_c).rev() {
                scans.push(ScanInfo::ac_scan(1, 1, 63, refine_al + 1, refine_al));
                scans.push(ScanInfo::ac_scan(2, 1, 63, refine_al + 1, refine_al));
            }
        }

        // DC refinement scan (if using successive approximation)
        if use_dc_succ_approx {
            let mut dc_refine = if config.dc_scan_opt_mode == 0 {
                ScanInfo::dc_scan(num_components)
            } else {
                ScanInfo::dc_scan(1)
            };
            dc_refine.ah = 1;
            dc_refine.al = 0;
            scans.push(dc_refine);
        }

        scans
    }
}

/// Scan optimizer that selects the best scan configuration.
///
/// This implements the selection algorithm from C mozjpeg's jcmaster.c select_scans().
/// It processes scan sizes after trial encoding and determines:
/// - Best successive approximation level (Al) for luma and chroma
/// - Best frequency split point for luma and chroma
/// - Whether to interleave chroma DC scans
///
/// The algorithm matches C mozjpeg EXACTLY:
/// - Luma Al selection compares band costs (not full 1-63 scans)
/// - Early termination when higher Al doesn't improve
/// - Frequency split selection with early termination heuristics
#[derive(Debug)]
pub struct ScanSelector {
    config: ScanSearchConfig,
    num_components: u8,

    // Luma scan indices (matching C: jcparam.c:775-780)
    num_scans_luma_dc: usize,
    luma_freq_split_scan_start: usize,
    num_scans_luma: usize,

    // Chroma scan indices
    num_scans_chroma_dc: usize,
    chroma_freq_split_scan_start: usize,
}

impl ScanSelector {
    /// Create a new scan selector for the given configuration.
    pub fn new(num_components: u8, config: ScanSearchConfig) -> Self {
        let al_max_luma = config.al_max_luma as usize;
        let al_max_chroma = config.al_max_chroma as usize;
        let num_freq_splits = config.frequency_splits.len();

        // Calculate scan indices - MUST match jcparam.c:775-780
        // num_scans_luma = num_scans_luma_dc + (3 * Al_max_luma + 2) + (2 * num_frequency_splits + 1)
        let num_scans_luma_dc = 1;
        let luma_freq_split_scan_start = num_scans_luma_dc + 3 * al_max_luma + 2;
        let num_scans_luma = luma_freq_split_scan_start + 2 * num_freq_splits + 1;

        let num_scans_chroma_dc = if num_components >= 3 { 3 } else { 0 };
        // Chroma freq splits start after: luma + chroma_dc + base(4) + SA scans(6*al_max) + full(2)
        let chroma_freq_split_scan_start = if num_components >= 3 {
            num_scans_luma + num_scans_chroma_dc + 4 + 6 * al_max_chroma + 2
        } else {
            0
        };

        Self {
            config,
            num_components,
            num_scans_luma_dc,
            luma_freq_split_scan_start,
            num_scans_luma,
            num_scans_chroma_dc,
            chroma_freq_split_scan_start,
        }
    }

    /// Process scan sizes and determine the best configuration.
    ///
    /// Takes the sizes of all 64 (or 23 for grayscale) trial-encoded scans
    /// and returns the optimal scan selection.
    pub fn select_best(&self, scan_sizes: &[usize]) -> ScanSearchResult {
        let (best_al_luma, best_freq_split_luma) = self.select_luma_params(scan_sizes);

        let (best_al_chroma, best_freq_split_chroma, interleave_chroma_dc) =
            if self.num_components >= 3 {
                self.select_chroma_params(scan_sizes)
            } else {
                (0, 0, false)
            };

        if DEBUG_SCAN_OPT {
            eprintln!(
                "[SCAN_OPT] Selection: al_luma={}, al_chroma={}, freq_luma={}, freq_chroma={}, interleave={}",
                best_al_luma, best_al_chroma, best_freq_split_luma, best_freq_split_chroma, interleave_chroma_dc
            );
        }

        ScanSearchResult {
            best_al_luma,
            best_al_chroma,
            best_freq_split_luma,
            best_freq_split_chroma,
            interleave_chroma_dc,
            scan_sizes: scan_sizes.to_vec(),
        }
    }

    /// Select best Al and frequency split for luma.
    /// Matches C mozjpeg's jcmaster.c:786-830 exactly.
    fn select_luma_params(&self, scan_sizes: &[usize]) -> (u8, usize) {
        let al_max = self.config.al_max_luma as usize;

        // Scan layout (3 scans per Al level):
        //   0: DC
        //   1: 1-8 at Al=0
        //   2: 9-63 at Al=0
        //   3: refinement 1-63 (Ah=1, Al=0)
        //   4: 1-8 at Al=1
        //   5: 9-63 at Al=1
        //   6: refinement 1-63 (Ah=2, Al=1)
        //   7: 1-8 at Al=2
        //   8: 9-63 at Al=2
        //   9: refinement 1-63 (Ah=3, Al=2)
        //   10: 1-8 at Al=3
        //   11: 9-63 at Al=3
        //   12: full 1-63 at Al=0
        //   13+: frequency splits

        let mut best_al = 0u8;
        let mut best_cost = usize::MAX;

        // C algorithm: compare band costs, not full 1-63 costs
        // Al=0: cost = scan[1] + scan[2] (base bands at Al=0)
        // Al=k: cost = scan[3k+1] + scan[3k+2] + sum of refinements
        for al in 0..=al_max {
            let cost = if al == 0 {
                // Cost = base 1-8 + base 9-63 at Al=0
                scan_sizes.get(1).copied().unwrap_or(usize::MAX)
                    + scan_sizes.get(2).copied().unwrap_or(0)
            } else {
                // Bands at this Al level
                let band1_idx = 3 * al + 1; // 1-8 at Al
                let band2_idx = 3 * al + 2; // 9-63 at Al
                let mut c = scan_sizes.get(band1_idx).copied().unwrap_or(0)
                    + scan_sizes.get(band2_idx).copied().unwrap_or(0);

                // Add all refinement costs from this Al down to Al=0
                // Refinement at index 3 + 3*i for i in 0..al
                for i in 0..al {
                    let refine_idx = 3 + 3 * i;
                    c += scan_sizes.get(refine_idx).copied().unwrap_or(0);
                }
                c
            };

            if DEBUG_SCAN_OPT {
                if al == 0 {
                    eprintln!(
                        "[SCAN_OPT] Luma Al={}: cost={} (sizes[1]={}, sizes[2]={})",
                        al,
                        cost,
                        scan_sizes.get(1).copied().unwrap_or(0),
                        scan_sizes.get(2).copied().unwrap_or(0)
                    );
                } else {
                    let refine_costs: Vec<usize> = (0..al)
                        .map(|i| scan_sizes.get(3 + 3 * i).copied().unwrap_or(0))
                        .collect();
                    eprintln!(
                        "[SCAN_OPT] Luma Al={}: cost={} (sizes[{}]={}, sizes[{}]={}, refine={:?})",
                        al,
                        cost,
                        3 * al + 1,
                        scan_sizes.get(3 * al + 1).copied().unwrap_or(0),
                        3 * al + 2,
                        scan_sizes.get(3 * al + 2).copied().unwrap_or(0),
                        refine_costs
                    );
                }
            }

            if al == 0 || cost < best_cost {
                best_cost = cost;
                best_al = al as u8;
            } else {
                // C mozjpeg early termination: if this Al is worse, skip remaining
                break;
            }
        }

        // Find best frequency split
        // Baseline is full 1-63 (scan 12) or the best Al band combination
        // For frequency split, we compare scan[12] vs split pairs

        let full_1_63_idx = self.luma_freq_split_scan_start; // Index 12
        let mut best_freq_split = 0usize; // 0 means use full 1-63 (no split)
        let mut best_freq_cost = scan_sizes.get(full_1_63_idx).copied().unwrap_or(usize::MAX);

        // Frequency split scans start at index 13
        let freq_start = full_1_63_idx + 1;
        for (i, _split) in self.config.frequency_splits.iter().enumerate() {
            let idx = freq_start + 2 * i;
            let cost = scan_sizes.get(idx).copied().unwrap_or(0)
                + scan_sizes.get(idx + 1).copied().unwrap_or(0);

            if cost < best_freq_cost {
                best_freq_cost = cost;
                best_freq_split = i + 1; // 1-indexed, 0 means no split
            }

            // C mozjpeg early termination heuristics (jcmaster.c:823-829)
            // If after testing first 3 splits, no split is best, stop searching
            if i == 2 && best_freq_split == 0 {
                break;
            }
            // Additional heuristics from C code
            if i == 3 && best_freq_split != 2 {
                break;
            }
            if i == 4 && best_freq_split != 4 {
                break;
            }
        }

        (best_al, best_freq_split)
    }

    /// Select best Al, frequency split, and DC interleaving for chroma.
    /// Matches C mozjpeg's jcmaster.c:832-896 exactly.
    fn select_chroma_params(&self, scan_sizes: &[usize]) -> (u8, usize, bool) {
        let base = self.num_scans_luma;
        let al_max = self.config.al_max_chroma as usize;

        // Chroma scan layout (starting at base=23):
        //   23: Cb+Cr combined DC
        //   24: Cb DC
        //   25: Cr DC
        //   26: Cb 1-8 at Al=0
        //   27: Cb 9-63 at Al=0
        //   28: Cr 1-8 at Al=0
        //   29: Cr 9-63 at Al=0
        //   30: Cb refine (Ah=1,Al=0)
        //   31: Cr refine (Ah=1,Al=0)
        //   32: Cb 1-8 at Al=1
        //   33: Cb 9-63 at Al=1
        //   34: Cr 1-8 at Al=1
        //   35: Cr 9-63 at Al=1
        //   36: Cb refine (Ah=2,Al=1)
        //   37: Cr refine (Ah=2,Al=1)
        //   38: Cb 1-8 at Al=2
        //   39: Cb 9-63 at Al=2
        //   40: Cr 1-8 at Al=2
        //   41: Cr 9-63 at Al=2
        //   42: Cb full 1-63 at Al=0
        //   43: Cr full 1-63 at Al=0
        //   44+: frequency splits

        // Check if interleaved DC is better (jcmaster.c:838)
        let combined_dc = scan_sizes.get(base).copied().unwrap_or(0);
        let separate_dc = scan_sizes.get(base + 1).copied().unwrap_or(0)
            + scan_sizes.get(base + 2).copied().unwrap_or(0);
        let interleave_chroma_dc = combined_dc <= separate_dc;

        // Find best Al for chroma
        let dc_offset = self.num_scans_chroma_dc; // 3

        let mut best_al = 0u8;
        let mut best_cost = usize::MAX;

        for al in 0..=al_max {
            let cost = if al == 0 {
                // Base scans for Cb and Cr at Al=0
                let cb_base = base + dc_offset; // 26
                let cr_base = base + dc_offset + 2; // 28
                scan_sizes.get(cb_base).copied().unwrap_or(0)
                    + scan_sizes.get(cb_base + 1).copied().unwrap_or(0)
                    + scan_sizes.get(cr_base).copied().unwrap_or(0)
                    + scan_sizes.get(cr_base + 1).copied().unwrap_or(0)
            } else {
                // Band scans at this Al (6 scans per Al level: 2 refine + 4 bands)
                // Bands for Al=k are at: base + dc_offset + 4 + 6*(k-1) + 2..6
                let band_base = base + dc_offset + 4 + 6 * (al - 1) + 2;
                let mut c = scan_sizes.get(band_base).copied().unwrap_or(0) // Cb 1-8
                    + scan_sizes.get(band_base + 1).copied().unwrap_or(0) // Cb 9-63
                    + scan_sizes.get(band_base + 2).copied().unwrap_or(0) // Cr 1-8
                    + scan_sizes.get(band_base + 3).copied().unwrap_or(0); // Cr 9-63

                // Add refinement costs
                for i in 0..al {
                    let refine_base = base + dc_offset + 4 + 6 * i;
                    c += scan_sizes.get(refine_base).copied().unwrap_or(0); // Cb refine
                    c += scan_sizes.get(refine_base + 1).copied().unwrap_or(0); // Cr refine
                }
                c
            };

            if al == 0 || cost < best_cost {
                best_cost = cost;
                best_al = al as u8;
            } else {
                // Early termination
                break;
            }
        }

        // Find best frequency split for chroma
        // Full scans are at base + dc_offset + 4 + 6*al_max (indices 42, 43)
        let chroma_full_base = base + dc_offset + 4 + 6 * al_max;
        let mut best_freq_split = 0usize;
        let mut best_freq_cost = scan_sizes.get(chroma_full_base).copied().unwrap_or(0)
            + scan_sizes.get(chroma_full_base + 1).copied().unwrap_or(0);

        // Frequency splits start at chroma_freq_split_scan_start
        let freq_base = self.chroma_freq_split_scan_start;
        for (i, _split) in self.config.frequency_splits.iter().enumerate() {
            let idx = freq_base + 4 * i;
            let cost = scan_sizes.get(idx).copied().unwrap_or(0)
                + scan_sizes.get(idx + 1).copied().unwrap_or(0)
                + scan_sizes.get(idx + 2).copied().unwrap_or(0)
                + scan_sizes.get(idx + 3).copied().unwrap_or(0);

            if cost < best_freq_cost {
                best_freq_cost = cost;
                best_freq_split = i + 1;
            }

            // Early termination
            if i == 2 && best_freq_split == 0 {
                break;
            }
        }

        (best_al, best_freq_split, interleave_chroma_dc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Config matching C mozjpeg's full SA search (for testing internal algorithm)
    fn c_mozjpeg_config() -> ScanSearchConfig {
        ScanSearchConfig {
            al_max_luma: 3,
            al_max_chroma: 2,
            frequency_splits: vec![2, 8, 5, 12, 18],
            dc_scan_opt_mode: 0,
        }
    }

    #[test]
    fn test_generate_search_scans_ycbcr() {
        let config = c_mozjpeg_config();
        let scans = generate_search_scans(3, &config);
        // Layout (matching C mozjpeg exactly):
        // Luma: DC(1) + base(2) + 3*al_max_luma(9) + full(1) + 2*num_splits(10) = 23
        // Chroma: DC(3) + base(4) + 6*al_max_chroma(12) + full(2) + 4*num_splits(20) = 41
        // Total = 23 + 41 = 64
        assert_eq!(
            scans.len(),
            64,
            "YCbCr should generate 64 scans (matching C mozjpeg)"
        );
    }

    #[test]
    fn test_generate_search_scans_grayscale() {
        let config = c_mozjpeg_config();
        let scans = generate_search_scans(1, &config);
        assert_eq!(scans.len(), 23, "Grayscale should generate 23 scans");
    }

    #[test]
    fn test_scan_selector_defaults() {
        let config = c_mozjpeg_config();
        let selector = ScanSelector::new(3, config.clone());

        // With all equal scan sizes, should pick Al=0 and no frequency split
        let scan_sizes = vec![100usize; 64];
        let result = selector.select_best(&scan_sizes);

        // Result should have valid values
        assert!(result.best_al_luma <= config.al_max_luma);
        assert!(result.best_al_chroma <= config.al_max_chroma);
    }

    #[test]
    fn test_scan_selector_prefers_smaller() {
        let config = c_mozjpeg_config();
        let selector = ScanSelector::new(3, config);

        // Test the Al selection algorithm (matching C mozjpeg's greedy search)
        // Scan layout (3 scans per Al level, matching C mozjpeg):
        //   0: DC
        //   1: 1-8 at Al=0
        //   2: 9-63 at Al=0
        //   3: refinement (Ah=1, Al=0)
        //   4: 1-8 at Al=1
        //   5: 9-63 at Al=1
        //   6: refinement (Ah=2, Al=1)
        //   7: 1-8 at Al=2
        //   8: 9-63 at Al=2
        //   9: refinement (Ah=3, Al=2)
        //   10: 1-8 at Al=3
        //   11: 9-63 at Al=3
        //   12: full 1-63 at Al=0
        let mut scan_sizes = vec![1000usize; 64];

        // Make each successive Al level cheaper (to avoid early termination)
        // Al=0 cost = scan[1] + scan[2] = 500 + 500 = 1000
        scan_sizes[1] = 500;
        scan_sizes[2] = 500;

        // Al=1 cost = scan[4] + scan[5] + scan[3] = 300 + 300 + 100 = 700
        scan_sizes[4] = 300;
        scan_sizes[5] = 300;
        scan_sizes[3] = 100;

        // Al=2 cost = scan[7] + scan[8] + scan[3] + scan[6] = 150 + 150 + 100 + 50 = 450
        scan_sizes[7] = 150;
        scan_sizes[8] = 150;
        scan_sizes[6] = 50;

        // Al=3 cost = scan[10] + scan[11] + scan[3] + scan[6] + scan[9]
        //           = 1000 + 1000 + 100 + 50 + 500 = 2650 (worse)

        let result = selector.select_best(&scan_sizes);

        // Should pick Al=2 since each level is strictly better until Al=3
        assert_eq!(
            result.best_al_luma, 2,
            "Should pick Al=2 (cost 450 < 700 < 1000)"
        );
    }

    #[test]
    fn test_build_final_scans() {
        let config = c_mozjpeg_config();
        let result = ScanSearchResult {
            best_al_luma: 1,
            best_al_chroma: 0,
            best_freq_split_luma: 0,
            best_freq_split_chroma: 0,
            interleave_chroma_dc: true,
            scan_sizes: vec![],
        };

        let scans = result.build_final_scans(3, &config);

        // With dc_scan_opt_mode=0 (default), DC for all components is in first scan.
        // With best_freq_split=0, we use full 1-63 AC scans.
        // Scans: DC all + Y AC 1-63 + Y refine + Cb AC 1-63 + Cr AC 1-63 = 5 scans
        assert!(scans.len() >= 5, "Should have at least 5 scans");

        // First scan should be DC
        assert!(scans[0].is_dc_scan());
    }

    #[test]
    fn test_chroma_dc_interleaving_preference() {
        let config = c_mozjpeg_config();
        let selector = ScanSelector::new(3, config);

        // Make interleaved DC cheaper
        let mut scan_sizes = vec![100usize; 64];
        // num_scans_luma = 1 + 2 + 3*3 + 1 + 2*5 = 23 (matching C mozjpeg)
        let base = 23;
        scan_sizes[base] = 50; // Combined Cb+Cr DC
        scan_sizes[base + 1] = 30; // Cb DC
        scan_sizes[base + 2] = 30; // Cr DC

        let result = selector.select_best(&scan_sizes);
        assert!(
            result.interleave_chroma_dc,
            "Should interleave when combined DC is smaller"
        );

        // Make separate DC cheaper
        scan_sizes[base] = 100;
        scan_sizes[base + 1] = 20;
        scan_sizes[base + 2] = 20;

        let result = selector.select_best(&scan_sizes);
        assert!(
            !result.interleave_chroma_dc,
            "Should not interleave when separate DC is smaller"
        );
    }

    #[test]
    fn test_frequency_split_selection() {
        let config = c_mozjpeg_config();
        let selector = ScanSelector::new(3, config.clone());

        // New layout (matching C mozjpeg):
        // Full 1-63 is at index 12
        // Frequency splits start at index 13:
        //   13-14: split at 2 (1-2, 3-63)
        //   15-16: split at 8 (1-8, 9-63)
        //   17-18: split at 5 (1-5, 6-63)
        //   19-20: split at 12 (1-12, 13-63)
        //   21-22: split at 18 (1-18, 19-63)
        let mut scan_sizes = vec![1000usize; 64];

        // Make the split at freq=5 (index 2 in splits array, scans 17-18) much cheaper
        scan_sizes[17] = 20; // 1-5
        scan_sizes[18] = 20; // 6-63

        // Full 1-63 scan (index 12) should be more expensive
        scan_sizes[12] = 200;

        let result = selector.select_best(&scan_sizes);
        // best_freq_split_luma: 0 = full 1-63, 1-5 = split indices (1-indexed)
        assert!(
            result.best_freq_split_luma > 0,
            "Should pick a frequency split when cheaper"
        );
    }

    #[test]
    fn test_grayscale_scan_generation() {
        let config = c_mozjpeg_config();
        let scans = generate_search_scans(1, &config);

        // Verify expected structure for grayscale
        assert_eq!(scans.len(), 23, "Grayscale should have 23 scans");

        // First scan is DC
        assert!(scans[0].is_dc_scan());
        assert_eq!(scans[0].comps_in_scan, 1);

        // All scans should be for component 0
        for scan in &scans {
            for i in 0..scan.comps_in_scan as usize {
                assert_eq!(
                    scan.component_index[i], 0,
                    "Grayscale scans should only use component 0"
                );
            }
        }
    }

    #[test]
    fn test_build_final_scans_with_refinement() {
        let config = c_mozjpeg_config();
        let result = ScanSearchResult {
            best_al_luma: 2, // Use Al=2, which needs refinement scans
            best_al_chroma: 1,
            best_freq_split_luma: 0,
            best_freq_split_chroma: 0,
            interleave_chroma_dc: true,
            scan_sizes: vec![],
        };

        let scans = result.build_final_scans(3, &config);

        // Count AC refinement scans for luma (ah > 0, ss > 0)
        let luma_ac_refinements: Vec<_> = scans
            .iter()
            .filter(|s| s.component_index[0] == 0 && s.ah > 0 && s.ss > 0)
            .collect();

        // Should have AC refinements from Al=2 to Al=1 and Al=1 to Al=0
        assert_eq!(
            luma_ac_refinements.len(),
            2,
            "Should have 2 luma AC refinement scans for Al=2"
        );

        // Also check for DC refinement scan
        let dc_refinements: Vec<_> = scans
            .iter()
            .filter(|s| s.is_dc_scan() && s.ah > 0)
            .collect();
        assert_eq!(dc_refinements.len(), 1, "Should have 1 DC refinement scan");
    }

    #[test]
    fn test_scan_layout_structure() {
        // Verify the scan layout structure matches C mozjpeg exactly
        let config = c_mozjpeg_config();
        let scans = generate_search_scans(3, &config);

        // Scan 0: DC all components
        assert_eq!(scans[0].comps_in_scan, 3);
        assert!(scans[0].is_dc_scan());

        // Scans 1-2: Y base AC (1-8, 9-63) at Al=0
        assert_eq!(scans[1].component_index[0], 0);
        assert_eq!((scans[1].ss, scans[1].se), (1, 8));
        assert_eq!((scans[1].ah, scans[1].al), (0, 0));
        assert_eq!(scans[2].component_index[0], 0);
        assert_eq!((scans[2].ss, scans[2].se), (9, 63));
        assert_eq!((scans[2].ah, scans[2].al), (0, 0));

        // New layout: 3 scans per Al level (matching C mozjpeg)
        // Scan 3: refinement 1-63 (Ah=1, Al=0)
        assert_eq!((scans[3].ss, scans[3].se), (1, 63));
        assert_eq!((scans[3].ah, scans[3].al), (1, 0));

        // Scan 4: 1-8 at Al=1
        assert_eq!((scans[4].ss, scans[4].se), (1, 8));
        assert_eq!((scans[4].ah, scans[4].al), (0, 1));

        // Scan 5: 9-63 at Al=1
        assert_eq!((scans[5].ss, scans[5].se), (9, 63));
        assert_eq!((scans[5].ah, scans[5].al), (0, 1));

        // Scan 12 (3 + 3*3): Y full AC 1-63 at Al=0
        assert_eq!(scans[12].component_index[0], 0);
        assert_eq!((scans[12].ss, scans[12].se), (1, 63));
        assert_eq!((scans[12].ah, scans[12].al), (0, 0));

        // Chroma scans start at index 23 (1 + 2 + 9 + 1 + 10)
        // Scan 23: Cb+Cr combined DC
        assert_eq!(scans[23].comps_in_scan, 2);
        assert!(scans[23].is_dc_scan());

        // Scan 24: Cb DC alone
        assert_eq!(scans[24].comps_in_scan, 1);
        assert_eq!(scans[24].component_index[0], 1);
        assert!(scans[24].is_dc_scan());

        // Scan 25: Cr DC alone
        assert_eq!(scans[25].comps_in_scan, 1);
        assert_eq!(scans[25].component_index[0], 2);
        assert!(scans[25].is_dc_scan());
    }
}
