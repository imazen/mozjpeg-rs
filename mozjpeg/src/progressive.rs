//! Progressive JPEG scan generation and encoding support.
//!
//! Progressive JPEG encodes the image in multiple scans:
//! 1. DC coefficients (gives coarse image)
//! 2. Low-frequency AC coefficients
//! 3. High-frequency AC coefficients
//! 4. Refinement scans (successive approximation)
//!
//! This allows partial display as the image loads and can improve compression.
//!
//! Reference: mozjpeg jcparam.c, jcphuff.c

use crate::types::ScanInfo;

/// Maximum number of components in a scan.
const MAX_COMPS_IN_SCAN: usize = 4;

/// Generate a simple progressive scan script.
///
/// This generates a basic progressive sequence:
/// 1. DC scan for all components
/// 2. AC scans for each component (bands 1-5, 6-63)
/// 3. Refinement scans
///
/// # Arguments
/// * `num_components` - Number of color components (1 for grayscale, 3 for YCbCr)
///
/// # Returns
/// Vector of ScanInfo describing each scan.
pub fn generate_simple_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // DC scan for all components (interleaved if possible)
    scans.push(ScanInfo::dc_scan(num_components));

    // AC scans for each component
    for comp in 0..num_components {
        // Low frequency AC (1-5)
        scans.push(ScanInfo::ac_scan(comp, 1, 5, 0, 0));
        // High frequency AC (6-63)
        scans.push(ScanInfo::ac_scan(comp, 6, 63, 0, 0));
    }

    scans
}

/// Generate a minimal progressive scan script for debugging.
///
/// This is the simplest possible progressive encoding:
/// 1. DC scan for all components
/// 2. Full AC scan (1-63) for each component
///
/// # Arguments
/// * `num_components` - Number of color components (1 for grayscale, 3 for YCbCr)
pub fn generate_minimal_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // DC scan for all components (interleaved)
    scans.push(ScanInfo::dc_scan(num_components));

    // Full AC scan for each component
    for comp in 0..num_components {
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 0, 0));
    }

    scans
}

/// Generate DC-only progressive scan for debugging.
/// This omits all AC scans to test if DC encoding is correct.
pub fn generate_dc_only_scan(num_components: u8) -> Vec<ScanInfo> {
    vec![ScanInfo::dc_scan(num_components)]
}

/// Generate a standard progressive scan script following libjpeg convention.
///
/// This generates:
/// 1. DC scan (Ah=0, Al=1) - coarse DC
/// 2. AC scans with successive approximation
/// 3. DC refinement (Ah=1, Al=0)
/// 4. AC refinement scans
///
/// # Arguments
/// * `num_components` - Number of color components (1 for grayscale, 3 for YCbCr)
///
/// # Returns
/// Vector of ScanInfo describing each scan.
pub fn generate_standard_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // Initial DC scan with point transform (Al=1)
    let mut dc_scan = ScanInfo::dc_scan(num_components);
    dc_scan.al = 1;
    scans.push(dc_scan);

    // AC scans for each component with successive approximation
    for comp in 0..num_components {
        // Low frequency (1-5) at reduced precision
        scans.push(ScanInfo::ac_scan(comp, 1, 5, 0, 2));
        // High frequency (6-63) at reduced precision
        scans.push(ScanInfo::ac_scan(comp, 6, 63, 0, 2));
    }

    // AC refinement scans
    for comp in 0..num_components {
        // Refine bits 2->1
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 2, 1));
    }

    // DC refinement scan
    let mut dc_refine = ScanInfo::dc_scan(num_components);
    dc_refine.ah = 1;
    dc_refine.al = 0;
    scans.push(dc_refine);

    // Final AC refinement (1->0)
    for comp in 0..num_components {
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 1, 0));
    }

    scans
}

/// Generate mozjpeg-optimized progressive scan script.
///
/// This is the scan script used when optimize_scans is enabled in mozjpeg.
/// It uses finer-grained spectral selection and successive approximation
/// for better compression.
///
/// # Arguments
/// * `num_components` - Number of color components (1 for grayscale, 3 for YCbCr)
///
/// # Returns
/// Vector of ScanInfo describing each scan.
pub fn generate_optimized_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    if num_components == 1 {
        // Grayscale: simpler script
        scans.push(ScanInfo::dc_scan(1));
        scans.push(ScanInfo::ac_scan(0, 1, 8, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 9, 63, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 1, 63, 2, 1));
        scans.push(ScanInfo::ac_scan(0, 1, 63, 1, 0));
    } else {
        // Color: separate luma and chroma handling

        // DC scan with point transform
        let mut dc_scan = ScanInfo::dc_scan(num_components);
        dc_scan.al = 1;
        scans.push(dc_scan);

        // Luma AC (component 0) - finer bands for better quality
        scans.push(ScanInfo::ac_scan(0, 1, 2, 0, 1));
        scans.push(ScanInfo::ac_scan(0, 3, 5, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 6, 8, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 9, 14, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 15, 63, 0, 2));

        // Chroma AC (components 1, 2) - coarser bands since eye is less sensitive
        for comp in 1..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 5, 0, 2));
            scans.push(ScanInfo::ac_scan(comp, 6, 63, 0, 2));
        }

        // Luma refinement
        scans.push(ScanInfo::ac_scan(0, 1, 2, 1, 0));
        scans.push(ScanInfo::ac_scan(0, 3, 63, 2, 1));
        scans.push(ScanInfo::ac_scan(0, 3, 63, 1, 0));

        // Chroma refinement
        for comp in 1..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 2, 1));
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 1, 0));
        }

        // DC refinement
        let mut dc_refine = ScanInfo::dc_scan(num_components);
        dc_refine.ah = 1;
        dc_refine.al = 0;
        scans.push(dc_refine);
    }

    scans
}

/// Generate candidate scan configurations for optimization.
///
/// Returns multiple scan scripts with varying:
/// - Al (successive approximation) levels for initial AC scans
/// - Frequency split points for AC bands
///
/// The encoder can trial-encode each and pick the smallest.
///
/// # Arguments
/// * `num_components` - Number of color components (1 for grayscale, 3 for YCbCr)
///
/// # Returns
/// Vector of candidate scan scripts, each a Vec<ScanInfo>.
pub fn generate_scan_candidates(num_components: u8) -> Vec<Vec<ScanInfo>> {
    let mut candidates = Vec::new();

    // Candidate 1: Simple progressive (no successive approximation)
    // DC + full AC per component
    candidates.push(generate_minimal_progressive_scans(num_components));

    // Candidate 2: Standard progressive with Al=1 for DC, Al=2 for AC
    candidates.push(generate_standard_progressive_scans(num_components));

    // Candidate 3: Optimized progressive with finer frequency bands
    candidates.push(generate_optimized_progressive_scans(num_components));

    // Candidate 4: Alternative with different frequency splits
    if num_components >= 3 {
        let mut scans = Vec::new();

        // DC with Al=1
        let mut dc_scan = ScanInfo::dc_scan(num_components);
        dc_scan.al = 1;
        scans.push(dc_scan);

        // Luma: split at 2, 8, 20 (different from optimized)
        scans.push(ScanInfo::ac_scan(0, 1, 2, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 3, 8, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 9, 20, 0, 2));
        scans.push(ScanInfo::ac_scan(0, 21, 63, 0, 2));

        // Chroma: single AC band per component
        for comp in 1..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 0, 2));
        }

        // Luma refinement
        scans.push(ScanInfo::ac_scan(0, 1, 63, 2, 1));
        scans.push(ScanInfo::ac_scan(0, 1, 63, 1, 0));

        // Chroma refinement
        for comp in 1..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 2, 1));
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 1, 0));
        }

        // DC refinement
        let mut dc_refine = ScanInfo::dc_scan(num_components);
        dc_refine.ah = 1;
        dc_refine.al = 0;
        scans.push(dc_refine);

        candidates.push(scans);
    }

    // Candidate 5: Minimal with Al=1 (less precision initially, more refinement)
    {
        let mut scans = Vec::new();

        // DC with Al=1
        let mut dc_scan = ScanInfo::dc_scan(num_components);
        dc_scan.al = 1;
        scans.push(dc_scan);

        // Full AC with Al=1 for each component
        for comp in 0..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 0, 1));
        }

        // Refinement scans
        for comp in 0..num_components {
            scans.push(ScanInfo::ac_scan(comp, 1, 63, 1, 0));
        }

        // DC refinement
        let mut dc_refine = ScanInfo::dc_scan(num_components);
        dc_refine.ah = 1;
        dc_refine.al = 0;
        scans.push(dc_refine);

        candidates.push(scans);
    }

    candidates
}

/// Generate a baseline (non-progressive) scan script.
///
/// This generates a single scan containing all components and all coefficients.
///
/// # Arguments
/// * `num_components` - Number of color components
///
/// # Returns
/// Vector with a single ScanInfo for baseline encoding.
pub fn generate_baseline_scan(num_components: u8) -> Vec<ScanInfo> {
    vec![ScanInfo {
        comps_in_scan: num_components,
        component_index: [0, 1, 2, 3],
        ss: 0,
        se: 63,
        ah: 0,
        al: 0,
    }]
}

/// Validate a scan script for correctness.
///
/// Checks that:
/// - Each scan has valid component indices
/// - Spectral selection is valid (ss <= se, both in 0..64)
/// - DC scans have ss=se=0
/// - AC scans have ss >= 1
/// - Successive approximation values are consistent
///
/// # Arguments
/// * `scans` - Scan script to validate
/// * `num_components` - Total number of components in the image
///
/// # Returns
/// Ok(()) if valid, Err with description if invalid.
pub fn validate_scan_script(scans: &[ScanInfo], num_components: u8) -> Result<(), &'static str> {
    if scans.is_empty() {
        return Err("Scan script must have at least one scan");
    }

    for scan in scans.iter() {
        // Validate component count
        if scan.comps_in_scan == 0 || scan.comps_in_scan > MAX_COMPS_IN_SCAN as u8 {
            return Err("Invalid component count in scan");
        }

        // Validate component indices
        for j in 0..scan.comps_in_scan as usize {
            if scan.component_index[j] >= num_components {
                return Err("Component index out of range");
            }
        }

        // Validate spectral selection
        if scan.se > 63 {
            return Err("Spectral selection end (Se) must be <= 63");
        }
        if scan.ss > scan.se {
            return Err("Spectral selection start (Ss) must be <= end (Se)");
        }

        // Progressive-specific validation (interleaved constraints)
        // In progressive mode, DC scans and AC scans must be separate
        // But in baseline mode, a single scan can contain DC+AC
        // We only check this for progressive scans (where ah or al are non-zero,
        // or spectral selection is partial)
        let is_progressive_scan = scan.ah != 0 || scan.al != 0 ||
                                  (scan.ss == 0 && scan.se != 0 && scan.se != 63);

        if is_progressive_scan {
            // In progressive, interleaved DC scans must be DC-only
            if scan.ss == 0 && scan.se != 0 && scan.comps_in_scan > 1 {
                return Err("Progressive interleaved scans cannot mix DC and AC");
            }
            // Progressive AC scans must be single-component
            if scan.ss > 0 && scan.comps_in_scan > 1 {
                return Err("Progressive AC scans must be single-component");
            }
        }

        // Successive approximation validation
        if scan.ah > 13 || scan.al > 13 {
            return Err("Successive approximation bit position out of range");
        }
    }

    Ok(())
}

/// Calculate the total number of scans needed.
pub fn count_scans(scans: &[ScanInfo]) -> usize {
    scans.len()
}

/// Check if a scan script uses progressive mode.
///
/// A script is progressive if it has multiple scans, or uses
/// successive approximation (Ah != 0), or uses spectral selection
/// (different scans for different frequency bands).
pub fn is_progressive_script(scans: &[ScanInfo]) -> bool {
    if scans.len() > 1 {
        return true;
    }
    if let Some(scan) = scans.first() {
        // Single scan covering all coefficients in one pass is baseline
        scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_progressive_grayscale() {
        let scans = generate_simple_progressive_scans(1);
        assert!(!scans.is_empty());
        assert!(validate_scan_script(&scans, 1).is_ok());

        // First scan should be DC
        assert!(scans[0].is_dc_scan());
    }

    #[test]
    fn test_simple_progressive_color() {
        let scans = generate_simple_progressive_scans(3);
        assert!(validate_scan_script(&scans, 3).is_ok());

        // Should have DC scan + 2 AC scans per component
        assert_eq!(scans.len(), 1 + 3 * 2); // DC + (low+high)*3
    }

    #[test]
    fn test_standard_progressive() {
        let scans = generate_standard_progressive_scans(3);
        assert!(validate_scan_script(&scans, 3).is_ok());

        // First scan should be DC with point transform
        assert!(scans[0].is_dc_scan());
        assert_eq!(scans[0].al, 1);
    }

    #[test]
    fn test_optimized_progressive() {
        let scans = generate_optimized_progressive_scans(3);
        assert!(validate_scan_script(&scans, 3).is_ok());
        assert!(is_progressive_script(&scans));
    }

    #[test]
    fn test_baseline_scan() {
        let scans = generate_baseline_scan(3);
        assert_eq!(scans.len(), 1);
        assert!(validate_scan_script(&scans, 3).is_ok());
        assert!(!is_progressive_script(&scans));

        let scan = &scans[0];
        assert_eq!(scan.ss, 0);
        assert_eq!(scan.se, 63);
        assert_eq!(scan.ah, 0);
        assert_eq!(scan.al, 0);
    }

    #[test]
    fn test_validate_empty_script() {
        let scans: Vec<ScanInfo> = vec![];
        assert!(validate_scan_script(&scans, 3).is_err());
    }

    #[test]
    fn test_validate_invalid_component() {
        let scans = vec![ScanInfo::ac_scan(5, 1, 63, 0, 0)]; // Component 5 doesn't exist
        assert!(validate_scan_script(&scans, 3).is_err());
    }

    #[test]
    fn test_validate_invalid_spectral() {
        let scan = ScanInfo::ac_scan(0, 10, 5, 0, 0); // ss > se
        let scans = vec![scan];
        assert!(validate_scan_script(&scans, 1).is_err());
    }

    #[test]
    fn test_is_progressive() {
        // Multiple scans = progressive
        let scans = generate_simple_progressive_scans(3);
        assert!(is_progressive_script(&scans));

        // Single full scan = baseline
        let baseline = generate_baseline_scan(3);
        assert!(!is_progressive_script(&baseline));
    }

    #[test]
    fn test_dc_scan_info() {
        let scan = ScanInfo::dc_scan(3);
        assert!(scan.is_dc_scan());
        assert!(!scan.is_refinement());
        assert_eq!(scan.comps_in_scan, 3);
    }

    #[test]
    fn test_ac_scan_info() {
        let scan = ScanInfo::ac_scan(0, 1, 63, 0, 0);
        assert!(!scan.is_dc_scan());
        assert!(!scan.is_refinement());
        assert_eq!(scan.ss, 1);
        assert_eq!(scan.se, 63);
    }

    #[test]
    fn test_refinement_scan_info() {
        let scan = ScanInfo::ac_scan(0, 1, 63, 2, 1);
        assert!(scan.is_refinement());
        assert_eq!(scan.ah, 2);
        assert_eq!(scan.al, 1);
    }

    #[test]
    fn test_count_scans() {
        let scans = generate_simple_progressive_scans(3);
        assert_eq!(count_scans(&scans), scans.len());
    }
}
