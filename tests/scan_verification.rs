//! Verify Rust scan generation structure.
//!
//! Note: Rust now generates 67 scans (vs C mozjpeg's 64) because we added
//! full 1-63 scans at each Al level for better successive approximation
//! cost comparison. This improves compression at low-mid quality levels.

use mozjpeg_oxide::scan_optimize::{generate_search_scans, ScanSearchConfig};

/// Test that Rust generates the expected number of progressive scans.
///
/// Layout (67 scans for YCbCr with al_max=3, 5 frequency splits):
/// - 1 DC scan for all components
/// - 2 base luma AC scans (1-8, 9-63)
/// - 12 luma successive approximation scans (4 per Al level)
/// - 1 luma full 1-63 at Al=0
/// - 10 luma frequency split scans
/// - 3 chroma DC variants
/// - 4 chroma base AC scans
/// - 12 chroma successive approximation scans
/// - 2 chroma full 1-63
/// - 20 chroma frequency split scans
#[test]
fn test_scan_generation_structure() {
    let config = ScanSearchConfig::default();
    let rust_scans = generate_search_scans(3, &config);

    // Layout: 1 + 2 + 4*3 + 1 + 2*5 + 3 + 4 + 6*2 + 2 + 4*5 = 67
    assert_eq!(
        rust_scans.len(),
        67,
        "Expected 67 scans for YCbCr with default config"
    );

    // Verify key scan types are present
    // DC scan for all components
    assert!(rust_scans[0].is_dc_scan());
    assert_eq!(rust_scans[0].comps_in_scan, 3);

    // Luma base AC scans
    assert_eq!(rust_scans[1].component_index[0], 0);
    assert_eq!((rust_scans[1].ss, rust_scans[1].se), (1, 8));
    assert_eq!(rust_scans[2].component_index[0], 0);
    assert_eq!((rust_scans[2].ss, rust_scans[2].se), (9, 63));

    // Verify we have refinement scans (ah > 0)
    let refinement_count = rust_scans.iter().filter(|s| s.ah > 0).count();
    assert!(
        refinement_count >= 6,
        "Should have at least 6 refinement scans"
    );

    // Verify we have full 1-63 scans at various Al levels
    let full_scans: Vec<_> = rust_scans
        .iter()
        .filter(|s| s.ss == 1 && s.se == 63 && s.ah == 0)
        .collect();
    assert!(
        full_scans.len() >= 4,
        "Should have full 1-63 scans at multiple Al levels"
    );
}

/// C scan data captured from mozjpeg.
/// Format: (comps_in_scan, component_indices, Ss, Se, Ah, Al)
fn get_c_reference_scans() -> Vec<(u8, [u8; 4], u8, u8, u8, u8)> {
    vec![
        (3, [0, 1, 2, 0], 0, 0, 0, 0),   // 0: DC all
        (1, [0, 0, 0, 0], 1, 8, 0, 0),   // 1: Y AC 1-8
        (1, [0, 0, 0, 0], 9, 63, 0, 0),  // 2: Y AC 9-63
        (1, [0, 0, 0, 0], 1, 63, 1, 0),  // 3: Y refine
        (1, [0, 0, 0, 0], 1, 8, 0, 1),   // 4
        (1, [0, 0, 0, 0], 9, 63, 0, 1),  // 5
        (1, [0, 0, 0, 0], 1, 63, 2, 1),  // 6
        (1, [0, 0, 0, 0], 1, 8, 0, 2),   // 7
        (1, [0, 0, 0, 0], 9, 63, 0, 2),  // 8
        (1, [0, 0, 0, 0], 1, 63, 3, 2),  // 9
        (1, [0, 0, 0, 0], 1, 8, 0, 3),   // 10
        (1, [0, 0, 0, 0], 9, 63, 0, 3),  // 11
        (1, [0, 0, 0, 0], 1, 63, 0, 0),  // 12: Y full
        (1, [0, 0, 0, 0], 1, 2, 0, 0),   // 13: freq split 2
        (1, [0, 0, 0, 0], 3, 63, 0, 0),  // 14
        (1, [0, 0, 0, 0], 1, 8, 0, 0),   // 15: freq split 8
        (1, [0, 0, 0, 0], 9, 63, 0, 0),  // 16
        (1, [0, 0, 0, 0], 1, 5, 0, 0),   // 17: freq split 5
        (1, [0, 0, 0, 0], 6, 63, 0, 0),  // 18
        (1, [0, 0, 0, 0], 1, 12, 0, 0),  // 19: freq split 12
        (1, [0, 0, 0, 0], 13, 63, 0, 0), // 20
        (1, [0, 0, 0, 0], 1, 18, 0, 0),  // 21: freq split 18
        (1, [0, 0, 0, 0], 19, 63, 0, 0), // 22
        (2, [1, 2, 0, 0], 0, 0, 0, 0),   // 23: Cb+Cr DC
        (1, [1, 0, 0, 0], 0, 0, 0, 0),   // 24: Cb DC
        (1, [2, 0, 0, 0], 0, 0, 0, 0),   // 25: Cr DC
        (1, [1, 0, 0, 0], 1, 8, 0, 0),   // 26
        (1, [1, 0, 0, 0], 9, 63, 0, 0),  // 27
        (1, [2, 0, 0, 0], 1, 8, 0, 0),   // 28
        (1, [2, 0, 0, 0], 9, 63, 0, 0),  // 29
        (1, [1, 0, 0, 0], 1, 63, 1, 0),  // 30
        (1, [2, 0, 0, 0], 1, 63, 1, 0),  // 31
        (1, [1, 0, 0, 0], 1, 8, 0, 1),   // 32
        (1, [1, 0, 0, 0], 9, 63, 0, 1),  // 33
        (1, [2, 0, 0, 0], 1, 8, 0, 1),   // 34
        (1, [2, 0, 0, 0], 9, 63, 0, 1),  // 35
        (1, [1, 0, 0, 0], 1, 63, 2, 1),  // 36
        (1, [2, 0, 0, 0], 1, 63, 2, 1),  // 37
        (1, [1, 0, 0, 0], 1, 8, 0, 2),   // 38
        (1, [1, 0, 0, 0], 9, 63, 0, 2),  // 39
        (1, [2, 0, 0, 0], 1, 8, 0, 2),   // 40
        (1, [2, 0, 0, 0], 9, 63, 0, 2),  // 41
        (1, [1, 0, 0, 0], 1, 63, 0, 0),  // 42
        (1, [2, 0, 0, 0], 1, 63, 0, 0),  // 43
        (1, [1, 0, 0, 0], 1, 2, 0, 0),   // 44
        (1, [1, 0, 0, 0], 3, 63, 0, 0),  // 45
        (1, [2, 0, 0, 0], 1, 2, 0, 0),   // 46
        (1, [2, 0, 0, 0], 3, 63, 0, 0),  // 47
        (1, [1, 0, 0, 0], 1, 8, 0, 0),   // 48
        (1, [1, 0, 0, 0], 9, 63, 0, 0),  // 49
        (1, [2, 0, 0, 0], 1, 8, 0, 0),   // 50
        (1, [2, 0, 0, 0], 9, 63, 0, 0),  // 51
        (1, [1, 0, 0, 0], 1, 5, 0, 0),   // 52
        (1, [1, 0, 0, 0], 6, 63, 0, 0),  // 53
        (1, [2, 0, 0, 0], 1, 5, 0, 0),   // 54
        (1, [2, 0, 0, 0], 6, 63, 0, 0),  // 55
        (1, [1, 0, 0, 0], 1, 12, 0, 0),  // 56
        (1, [1, 0, 0, 0], 13, 63, 0, 0), // 57
        (1, [2, 0, 0, 0], 1, 12, 0, 0),  // 58
        (1, [2, 0, 0, 0], 13, 63, 0, 0), // 59
        (1, [1, 0, 0, 0], 1, 18, 0, 0),  // 60
        (1, [1, 0, 0, 0], 19, 63, 0, 0), // 61
        (1, [2, 0, 0, 0], 1, 18, 0, 0),  // 62
        (1, [2, 0, 0, 0], 19, 63, 0, 0), // 63
    ]
}
