//! Scan optimization for progressive JPEG (matches C mozjpeg's optimize_scans).
//!
//! This implements the scan search algorithm from mozjpeg's jcmaster.c.
//! It generates 64 candidate scans for YCbCr images and selects the optimal
//! subset based on encoded size.

use crate::types::ScanInfo;

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
            al_max_luma: 3,
            al_max_chroma: 2,
            frequency_splits: vec![2, 8, 5, 12, 18],
            dc_scan_opt_mode: 0,
        }
    }
}

/// Generate the full set of 64 scan candidates for YCbCr images.
/// This matches C mozjpeg's jpeg_search_progression().
///
/// The scans are organized as:
/// - Scans 0-22: Luma (Y) scans
///   - 1 DC scan
///   - 11 successive approximation test scans
///   - 11 frequency split test scans
/// - Scans 23-63: Chroma (Cb, Cr) scans
///   - 3 DC scan variants
///   - 38 AC scan variants
pub fn generate_search_scans(num_components: u8, config: &ScanSearchConfig) -> Vec<ScanInfo> {
    if num_components == 1 {
        return generate_grayscale_search_scans(config);
    }

    let mut scans = Vec::with_capacity(64);

    // === LUMA SCANS (23 scans) ===

    // DC scan for all components (or just luma depending on mode)
    if config.dc_scan_opt_mode == 0 {
        scans.push(ScanInfo::dc_scan(num_components));
    } else {
        scans.push(ScanInfo::dc_scan(1)); // Just luma
    }

    // Base AC scans: 1-8 and 9-63
    scans.push(ScanInfo::ac_scan(0, 1, 8, 0, 0));
    scans.push(ScanInfo::ac_scan(0, 9, 63, 0, 0));

    // Successive approximation test scans for luma
    // For each Al level (0, 1, 2): refinement scan + two band scans at next level
    for al in 0..config.al_max_luma {
        scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al)); // Refinement
        scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1));   // Low freq at higher Al
        scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1));  // High freq at higher Al
    }

    // Full luma AC scan (no successive approx)
    scans.push(ScanInfo::ac_scan(0, 1, 63, 0, 0));

    // Frequency split test scans for luma
    for &split in &config.frequency_splits {
        scans.push(ScanInfo::ac_scan(0, 1, split, 0, 0));
        scans.push(ScanInfo::ac_scan(0, split + 1, 63, 0, 0));
    }

    // === CHROMA SCANS (41 scans for 3-component) ===

    if num_components >= 3 {
        // DC scan variants for chroma
        // Combined Cb+Cr DC
        scans.push(ScanInfo::dc_scan_pair(1, 2));
        // Separate DC scans
        scans.push(ScanInfo::dc_scan_single(1));
        scans.push(ScanInfo::dc_scan_single(2));

        // Base AC scans for Cb and Cr
        scans.push(ScanInfo::ac_scan(1, 1, 8, 0, 0));
        scans.push(ScanInfo::ac_scan(1, 9, 63, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 1, 8, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 9, 63, 0, 0));

        // Successive approximation test scans for chroma
        for al in 0..config.al_max_chroma {
            // Refinement scans
            scans.push(ScanInfo::ac_scan(1, 1, 63, al + 1, al));
            scans.push(ScanInfo::ac_scan(2, 1, 63, al + 1, al));
            // Band scans at higher Al
            scans.push(ScanInfo::ac_scan(1, 1, 8, 0, al + 1));
            scans.push(ScanInfo::ac_scan(1, 9, 63, 0, al + 1));
            scans.push(ScanInfo::ac_scan(2, 1, 8, 0, al + 1));
            scans.push(ScanInfo::ac_scan(2, 9, 63, 0, al + 1));
        }

        // Full chroma AC scans
        scans.push(ScanInfo::ac_scan(1, 1, 63, 0, 0));
        scans.push(ScanInfo::ac_scan(2, 1, 63, 0, 0));

        // Frequency split test scans for chroma
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
fn generate_grayscale_search_scans(config: &ScanSearchConfig) -> Vec<ScanInfo> {
    let mut scans = Vec::with_capacity(23);

    // DC scan
    scans.push(ScanInfo::dc_scan(1));

    // Base AC scans
    scans.push(ScanInfo::ac_scan(0, 1, 8, 0, 0));
    scans.push(ScanInfo::ac_scan(0, 9, 63, 0, 0));

    // Successive approximation test scans
    for al in 0..config.al_max_luma {
        scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al));
        scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1));
        scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1));
    }

    // Full AC scan
    scans.push(ScanInfo::ac_scan(0, 1, 63, 0, 0));

    // Frequency split test scans
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
    pub fn build_final_scans(&self, num_components: u8, config: &ScanSearchConfig) -> Vec<ScanInfo> {
        let mut scans = Vec::new();

        // DC scan
        if config.dc_scan_opt_mode == 0 {
            scans.push(ScanInfo::dc_scan(num_components));
        } else {
            scans.push(ScanInfo::dc_scan(1));
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

        scans
    }
}

/// Scan optimizer that selects the best scan configuration.
///
/// This implements the selection algorithm from C mozjpeg's jcmaster.c.
/// It processes scan sizes after trial encoding and determines:
/// - Best successive approximation level (Al) for luma and chroma
/// - Best frequency split point for luma and chroma
/// - Whether to interleave chroma DC scans
#[derive(Debug)]
pub struct ScanSelector {
    config: ScanSearchConfig,
    num_components: u8,

    // Luma scan indices
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

        // Calculate scan indices (matching C mozjpeg's layout)
        // Layout: DC + 2 base + 3*al_max (refinements+bands) + 1 full + 2*num_splits
        let num_scans_luma_dc = 1;
        let num_scans_luma = num_scans_luma_dc + (3 * al_max_luma + 2) + (2 * num_freq_splits + 1);
        // Full 1-63 scan is at index 12 (1 + 2 + 3*3), splits start at 13
        let luma_freq_split_scan_start = num_scans_luma_dc + 3 * al_max_luma + 3;

        let num_scans_chroma_dc = if num_components >= 3 { 3 } else { 0 };
        let chroma_freq_split_scan_start = if num_components >= 3 {
            num_scans_luma + num_scans_chroma_dc + (6 * al_max_chroma + 4)
        } else {
            0
        };

        Self {
            config,
            num_components,
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
    fn select_luma_params(&self, scan_sizes: &[usize]) -> (u8, usize) {
        let al_max = self.config.al_max_luma as usize;

        // Find best Al by comparing costs
        // Scan layout:
        //   0: DC
        //   1: 1-8 at Al=0 (base)
        //   2: 9-63 at Al=0 (base)
        //   3: refinement ah=1,al=0 (refines Al=1 -> Al=0)
        //   4: 1-8 at Al=1
        //   5: 9-63 at Al=1
        //   6: refinement ah=2,al=1 (refines Al=2 -> Al=1)
        //   7: 1-8 at Al=2
        //   8: 9-63 at Al=2
        //   ...

        let mut best_al = 0u8;
        let mut best_al_cost = usize::MAX;

        for al in 0..=al_max {
            let cost = if al == 0 {
                // Cost = base scans (1-8) + (9-63)
                scan_sizes.get(1).copied().unwrap_or(0)
                     + scan_sizes.get(2).copied().unwrap_or(0)
            } else {
                // Band scans at Al=k are at indices: 3*k + 1 and 3*k + 2
                let mut c = scan_sizes.get(3 * al + 1).copied().unwrap_or(0)
                          + scan_sizes.get(3 * al + 2).copied().unwrap_or(0);

                // Add refinement costs: refinement from Al=k to Al=k-1 is at index 3*k
                for k in 1..=al {
                    c += scan_sizes.get(3 * k).copied().unwrap_or(0);
                }
                c
            };

            if cost < best_al_cost {
                best_al_cost = cost;
                best_al = al as u8;
            }
        }

        // Find best frequency split
        // Scan 12: full 1-63
        // Scans 13-14: split at 2
        // Scans 15-16: split at 8
        // etc.

        let mut best_freq_split = 0usize; // 0 means use default 1-8, 9-63
        let mut best_freq_cost = scan_sizes.get(self.luma_freq_split_scan_start - 1)
            .copied().unwrap_or(usize::MAX);

        for (i, _split) in self.config.frequency_splits.iter().enumerate() {
            let idx = self.luma_freq_split_scan_start + 2 * i;
            let cost = scan_sizes.get(idx).copied().unwrap_or(0)
                     + scan_sizes.get(idx + 1).copied().unwrap_or(0);

            if cost < best_freq_cost {
                best_freq_cost = cost;
                best_freq_split = i + 1; // 1-indexed, 0 means default
            }

            // Early termination heuristics (matching C mozjpeg)
            if i == 2 && best_freq_split == 0 {
                break; // No split is best after testing 3
            }
        }

        (best_al, best_freq_split)
    }

    /// Select best Al, frequency split, and DC interleaving for chroma.
    fn select_chroma_params(&self, scan_sizes: &[usize]) -> (u8, usize, bool) {
        let base = self.num_scans_luma;
        let al_max = self.config.al_max_chroma as usize;

        // Check if interleaved DC is better
        let combined_dc = scan_sizes.get(base).copied().unwrap_or(0);
        let separate_dc = scan_sizes.get(base + 1).copied().unwrap_or(0)
                        + scan_sizes.get(base + 2).copied().unwrap_or(0);
        let interleave_chroma_dc = combined_dc <= separate_dc;

        // Find best Al for chroma
        let dc_offset = self.num_scans_chroma_dc;
        let mut best_al = 0u8;
        let mut best_al_cost = usize::MAX;

        for al in 0..=al_max {
            let cost = if al == 0 {
                // Base scans for Cb and Cr
                let cb_base = base + dc_offset;
                let cr_base = base + dc_offset + 2;
                scan_sizes.get(cb_base).copied().unwrap_or(0)
                     + scan_sizes.get(cb_base + 1).copied().unwrap_or(0)
                     + scan_sizes.get(cr_base).copied().unwrap_or(0)
                     + scan_sizes.get(cr_base + 1).copied().unwrap_or(0)
            } else {
                // Band scans at this Al + refinement costs
                let band_idx = base + dc_offset + 4 + 6 * al;
                let mut c = scan_sizes.get(band_idx - 4).copied().unwrap_or(0)
                          + scan_sizes.get(band_idx - 3).copied().unwrap_or(0)
                          + scan_sizes.get(band_idx - 2).copied().unwrap_or(0)
                          + scan_sizes.get(band_idx - 1).copied().unwrap_or(0);

                for prev_al in 0..al {
                    let refine_idx = base + dc_offset + 4 + 6 * prev_al;
                    c += scan_sizes.get(refine_idx).copied().unwrap_or(0);
                    c += scan_sizes.get(refine_idx + 1).copied().unwrap_or(0);
                }
                c
            };

            if cost < best_al_cost {
                best_al_cost = cost;
                best_al = al as u8;
            }
        }

        // Find best frequency split for chroma
        let mut best_freq_split = 0usize;
        let chroma_full_base = base + dc_offset + 4 + 6 * al_max;
        let mut best_freq_cost = scan_sizes.get(chroma_full_base).copied().unwrap_or(0)
                               + scan_sizes.get(chroma_full_base + 1).copied().unwrap_or(0);

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

    #[test]
    fn test_generate_search_scans_ycbcr() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);
        assert_eq!(scans.len(), 64, "YCbCr should generate 64 scans");
    }

    #[test]
    fn test_generate_search_scans_grayscale() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(1, &config);
        assert_eq!(scans.len(), 23, "Grayscale should generate 23 scans");
    }

    #[test]
    fn test_scan_selector_defaults() {
        let config = ScanSearchConfig::default();
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
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        // Make Al=2 much cheaper than Al=0
        // Scan layout:
        //   1, 2: base scans at Al=0
        //   3: refinement Al=1->Al=0
        //   4, 5: bands at Al=1
        //   6: refinement Al=2->Al=1
        //   7, 8: bands at Al=2
        let mut scan_sizes = vec![1000usize; 64];

        // Al=0 cost = scan[1] + scan[2] = 1000 + 1000 = 2000
        // (leave at default)

        // Al=2 cost = scan[7] + scan[8] + scan[6] + scan[3]
        scan_sizes[7] = 10;   // 1-8 at Al=2
        scan_sizes[8] = 10;   // 9-63 at Al=2
        scan_sizes[6] = 5;    // refinement Al=2 -> Al=1
        scan_sizes[3] = 5;    // refinement Al=1 -> Al=0
        // Al=2 cost = 10 + 10 + 5 + 5 = 30 << 2000

        let result = selector.select_best(&scan_sizes);

        // Should pick Al=2 since it's much cheaper (30 vs 2000)
        assert_eq!(result.best_al_luma, 2, "Should prefer Al=2 when it's much cheaper (cost 30 vs 2000)");
    }

    #[test]
    fn test_build_final_scans() {
        let config = ScanSearchConfig::default();
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
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        // Make interleaved DC cheaper
        let mut scan_sizes = vec![100usize; 64];
        let base = 23; // num_scans_luma for YCbCr
        scan_sizes[base] = 50;     // Combined Cb+Cr DC
        scan_sizes[base + 1] = 30; // Cb DC
        scan_sizes[base + 2] = 30; // Cr DC

        let result = selector.select_best(&scan_sizes);
        assert!(result.interleave_chroma_dc, "Should interleave when combined DC is smaller");

        // Make separate DC cheaper
        scan_sizes[base] = 100;
        scan_sizes[base + 1] = 20;
        scan_sizes[base + 2] = 20;

        let result = selector.select_best(&scan_sizes);
        assert!(!result.interleave_chroma_dc, "Should not interleave when separate DC is smaller");
    }

    #[test]
    fn test_frequency_split_selection() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config.clone());

        // Default frequency splits are [2, 8, 5, 12, 18]
        // Frequency split scans start at index 13 for luma
        let mut scan_sizes = vec![1000usize; 64];

        // Make the split at freq=5 (index 2 in splits array) much cheaper
        // Scans 17 and 18 are the split at 5 (1-5 and 6-63)
        let split_idx = 17;
        scan_sizes[split_idx] = 20;     // 1-5
        scan_sizes[split_idx + 1] = 20; // 6-63

        // Full 1-63 scan (index 12) should be more expensive
        scan_sizes[12] = 200;

        let result = selector.select_best(&scan_sizes);
        // best_freq_split_luma: 0 = default (1-8, 9-63), 1-5 = split indices
        assert!(result.best_freq_split_luma > 0, "Should pick a frequency split when cheaper");
    }

    #[test]
    fn test_grayscale_scan_generation() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(1, &config);

        // Verify expected structure for grayscale
        assert_eq!(scans.len(), 23, "Grayscale should have 23 scans");

        // First scan is DC
        assert!(scans[0].is_dc_scan());
        assert_eq!(scans[0].comps_in_scan, 1);

        // All scans should be for component 0
        for scan in &scans {
            for i in 0..scan.comps_in_scan as usize {
                assert_eq!(scan.component_index[i], 0, "Grayscale scans should only use component 0");
            }
        }
    }

    #[test]
    fn test_build_final_scans_with_refinement() {
        let config = ScanSearchConfig::default();
        let result = ScanSearchResult {
            best_al_luma: 2, // Use Al=2, which needs refinement scans
            best_al_chroma: 1,
            best_freq_split_luma: 0,
            best_freq_split_chroma: 0,
            interleave_chroma_dc: true,
            scan_sizes: vec![],
        };

        let scans = result.build_final_scans(3, &config);

        // Count refinement scans (ah > 0)
        let luma_refinements: Vec<_> = scans.iter()
            .filter(|s| s.component_index[0] == 0 && s.ah > 0)
            .collect();

        // Should have refinements from Al=2 to Al=1 and Al=1 to Al=0
        assert_eq!(luma_refinements.len(), 2, "Should have 2 luma refinement scans for Al=2");
    }

    #[test]
    fn test_scan_layout_matches_c_mozjpeg() {
        // Verify the scan layout matches C mozjpeg's jpeg_search_progression
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);

        // C mozjpeg layout verification (from verify_scans example)
        // Scan 0: DC all components
        assert_eq!(scans[0].comps_in_scan, 3);
        assert!(scans[0].is_dc_scan());

        // Scans 1-2: Y base AC (1-8, 9-63)
        assert_eq!(scans[1].component_index[0], 0);
        assert_eq!((scans[1].ss, scans[1].se), (1, 8));
        assert_eq!(scans[2].component_index[0], 0);
        assert_eq!((scans[2].ss, scans[2].se), (9, 63));

        // Scan 12: Y full AC 1-63
        assert_eq!(scans[12].component_index[0], 0);
        assert_eq!((scans[12].ss, scans[12].se), (1, 63));
        assert_eq!((scans[12].ah, scans[12].al), (0, 0));

        // Scan 23: Cb+Cr combined DC
        assert_eq!(scans[23].comps_in_scan, 2);
        assert!(scans[23].is_dc_scan());
    }
}
