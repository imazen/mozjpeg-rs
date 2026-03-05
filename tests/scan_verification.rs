//! Verify Rust scan generation structure matches C mozjpeg exactly.
//!
//! Rust now generates exactly 64 scans for YCbCr, matching C mozjpeg's
//! jpeg_search_progression() in jcparam.c.

use mozjpeg_rs::scan_optimize::{ScanSearchConfig, generate_search_scans};

/// Test that Rust generates the expected number of progressive scans.
///
/// Layout (64 scans for YCbCr with al_max_luma=3, al_max_chroma=2, 5 frequency splits):
///
/// LUMA (23 scans):
/// - 1 DC scan for all components
/// - 2 base luma AC scans (1-8, 9-63 at Al=0)
/// - 9 luma successive approximation scans (3 per Al level: refine, 1-8, 9-63)
/// - 1 luma full 1-63 at Al=0
/// - 10 luma frequency split scans (5 pairs)
///
/// CHROMA (41 scans):
/// - 3 chroma DC variants (combined, Cb alone, Cr alone)
/// - 4 chroma base AC scans (1-8, 9-63 for Cb and Cr)
/// - 12 chroma successive approximation scans (6 per Al level)
/// - 2 chroma full 1-63 (Cb and Cr)
/// - 20 chroma frequency split scans (5 pairs × 2 components)
#[test]
fn test_scan_generation_structure() {
    // Use explicit config matching C mozjpeg's defaults (with SA enabled)
    // Note: ScanSearchConfig::default() currently has al_max=0 to avoid
    // refinement scan bugs. This test verifies the SA scan generation logic.
    let config = ScanSearchConfig {
        al_max_luma: 3,
        al_max_chroma: 2,
        frequency_splits: vec![2, 8, 5, 12, 18],
        dc_scan_opt_mode: 0,
    };
    let rust_scans = generate_search_scans(3, &config);

    // Layout (matching C mozjpeg exactly):
    // Luma: DC(1) + base(2) + 3*al_max_luma(9) + full(1) + 2*num_splits(10) = 23
    // Chroma: DC(3) + base(4) + 6*al_max_chroma(12) + full(2) + 4*num_splits(20) = 41
    // Total = 23 + 41 = 64
    assert_eq!(
        rust_scans.len(),
        64,
        "Expected 64 scans for YCbCr with default config (matching C mozjpeg)"
    );

    // Verify against C reference scans
    let c_scans = get_c_reference_scans();
    assert_eq!(
        rust_scans.len(),
        c_scans.len(),
        "Rust should generate same number of scans as C mozjpeg"
    );

    // Verify each scan matches the C reference
    for (i, (rust, c)) in rust_scans.iter().zip(c_scans.iter()).enumerate() {
        let (c_comps, c_indices, c_ss, c_se, c_ah, c_al) = c;

        assert_eq!(
            rust.comps_in_scan, *c_comps,
            "Scan {}: comps_in_scan mismatch (Rust={}, C={})",
            i, rust.comps_in_scan, c_comps
        );
        assert_eq!(
            rust.ss, *c_ss,
            "Scan {}: Ss mismatch (Rust={}, C={})",
            i, rust.ss, c_ss
        );
        assert_eq!(
            rust.se, *c_se,
            "Scan {}: Se mismatch (Rust={}, C={})",
            i, rust.se, c_se
        );
        assert_eq!(
            rust.ah, *c_ah,
            "Scan {}: Ah mismatch (Rust={}, C={})",
            i, rust.ah, c_ah
        );
        assert_eq!(
            rust.al, *c_al,
            "Scan {}: Al mismatch (Rust={}, C={})",
            i, rust.al, c_al
        );

        // Verify component indices
        for j in 0..(*c_comps as usize) {
            assert_eq!(
                rust.component_index[j], c_indices[j],
                "Scan {}: component_index[{}] mismatch (Rust={}, C={})",
                i, j, rust.component_index[j], c_indices[j]
            );
        }
    }

    // Verify key scan types are present
    // DC scan for all components
    assert!(rust_scans[0].is_dc_scan());
    assert_eq!(rust_scans[0].comps_in_scan, 3);

    // Luma base AC scans at Al=0
    assert_eq!(rust_scans[1].component_index[0], 0);
    assert_eq!((rust_scans[1].ss, rust_scans[1].se), (1, 8));
    assert_eq!((rust_scans[1].ah, rust_scans[1].al), (0, 0));
    assert_eq!(rust_scans[2].component_index[0], 0);
    assert_eq!((rust_scans[2].ss, rust_scans[2].se), (9, 63));
    assert_eq!((rust_scans[2].ah, rust_scans[2].al), (0, 0));

    // Verify we have 5 refinement scans (3 luma + 2×2 chroma = but only 5 total with Ah>0)
    // Luma refinements: indices 3, 6, 9 (Ah=1,2,3)
    // Chroma refinements: indices 30, 31, 36, 37 (Ah=1,2 for Cb and Cr)
    let refinement_count = rust_scans.iter().filter(|s| s.ah > 0).count();
    assert_eq!(
        refinement_count, 7,
        "Should have exactly 7 refinement scans (3 luma + 4 chroma)"
    );

    // Verify we have full 1-63 scans at Al=0 (luma at index 12, chroma at 42, 43)
    assert_eq!((rust_scans[12].ss, rust_scans[12].se), (1, 63));
    assert_eq!((rust_scans[12].ah, rust_scans[12].al), (0, 0));
    assert_eq!((rust_scans[42].ss, rust_scans[42].se), (1, 63));
    assert_eq!((rust_scans[43].ss, rust_scans[43].se), (1, 63));
}

/// C scan data captured from mozjpeg's jpeg_search_progression().
/// Format: (comps_in_scan, component_indices, Ss, Se, Ah, Al)
///
/// This exactly matches the output of jcparam.c:733-852 with:
/// - Al_max_luma = 3
/// - Al_max_chroma = 2
/// - frequency_splits = [2, 8, 5, 12, 18]
/// - dc_scan_opt_mode = 0 (interleaved DC for all components)
fn get_c_reference_scans() -> Vec<(u8, [u8; 4], u8, u8, u8, u8)> {
    vec![
        // LUMA (23 scans, indices 0-22)
        (3, [0, 1, 2, 0], 0, 0, 0, 0),  // 0: DC all components
        (1, [0, 0, 0, 0], 1, 8, 0, 0),  // 1: Y AC 1-8 at Al=0
        (1, [0, 0, 0, 0], 9, 63, 0, 0), // 2: Y AC 9-63 at Al=0
        // For Al=0 in loop:
        (1, [0, 0, 0, 0], 1, 63, 1, 0), // 3: Y refine 1-63 (Ah=1, Al=0)
        (1, [0, 0, 0, 0], 1, 8, 0, 1),  // 4: Y 1-8 at Al=1
        (1, [0, 0, 0, 0], 9, 63, 0, 1), // 5: Y 9-63 at Al=1
        // For Al=1 in loop:
        (1, [0, 0, 0, 0], 1, 63, 2, 1), // 6: Y refine 1-63 (Ah=2, Al=1)
        (1, [0, 0, 0, 0], 1, 8, 0, 2),  // 7: Y 1-8 at Al=2
        (1, [0, 0, 0, 0], 9, 63, 0, 2), // 8: Y 9-63 at Al=2
        // For Al=2 in loop:
        (1, [0, 0, 0, 0], 1, 63, 3, 2), // 9: Y refine 1-63 (Ah=3, Al=2)
        (1, [0, 0, 0, 0], 1, 8, 0, 3),  // 10: Y 1-8 at Al=3
        (1, [0, 0, 0, 0], 9, 63, 0, 3), // 11: Y 9-63 at Al=3
        // Full 1-63 and frequency splits:
        (1, [0, 0, 0, 0], 1, 63, 0, 0),  // 12: Y full 1-63 at Al=0
        (1, [0, 0, 0, 0], 1, 2, 0, 0),   // 13: freq split at 2
        (1, [0, 0, 0, 0], 3, 63, 0, 0),  // 14
        (1, [0, 0, 0, 0], 1, 8, 0, 0),   // 15: freq split at 8
        (1, [0, 0, 0, 0], 9, 63, 0, 0),  // 16
        (1, [0, 0, 0, 0], 1, 5, 0, 0),   // 17: freq split at 5
        (1, [0, 0, 0, 0], 6, 63, 0, 0),  // 18
        (1, [0, 0, 0, 0], 1, 12, 0, 0),  // 19: freq split at 12
        (1, [0, 0, 0, 0], 13, 63, 0, 0), // 20
        (1, [0, 0, 0, 0], 1, 18, 0, 0),  // 21: freq split at 18
        (1, [0, 0, 0, 0], 19, 63, 0, 0), // 22
        // CHROMA (41 scans, indices 23-63)
        (2, [1, 2, 0, 0], 0, 0, 0, 0),  // 23: Cb+Cr DC combined
        (1, [1, 0, 0, 0], 0, 0, 0, 0),  // 24: Cb DC alone
        (1, [2, 0, 0, 0], 0, 0, 0, 0),  // 25: Cr DC alone
        (1, [1, 0, 0, 0], 1, 8, 0, 0),  // 26: Cb 1-8 at Al=0
        (1, [1, 0, 0, 0], 9, 63, 0, 0), // 27: Cb 9-63 at Al=0
        (1, [2, 0, 0, 0], 1, 8, 0, 0),  // 28: Cr 1-8 at Al=0
        (1, [2, 0, 0, 0], 9, 63, 0, 0), // 29: Cr 9-63 at Al=0
        // For Al=0 in chroma loop:
        (1, [1, 0, 0, 0], 1, 63, 1, 0), // 30: Cb refine (Ah=1, Al=0)
        (1, [2, 0, 0, 0], 1, 63, 1, 0), // 31: Cr refine (Ah=1, Al=0)
        (1, [1, 0, 0, 0], 1, 8, 0, 1),  // 32: Cb 1-8 at Al=1
        (1, [1, 0, 0, 0], 9, 63, 0, 1), // 33: Cb 9-63 at Al=1
        (1, [2, 0, 0, 0], 1, 8, 0, 1),  // 34: Cr 1-8 at Al=1
        (1, [2, 0, 0, 0], 9, 63, 0, 1), // 35: Cr 9-63 at Al=1
        // For Al=1 in chroma loop:
        (1, [1, 0, 0, 0], 1, 63, 2, 1), // 36: Cb refine (Ah=2, Al=1)
        (1, [2, 0, 0, 0], 1, 63, 2, 1), // 37: Cr refine (Ah=2, Al=1)
        (1, [1, 0, 0, 0], 1, 8, 0, 2),  // 38: Cb 1-8 at Al=2
        (1, [1, 0, 0, 0], 9, 63, 0, 2), // 39: Cb 9-63 at Al=2
        (1, [2, 0, 0, 0], 1, 8, 0, 2),  // 40: Cr 1-8 at Al=2
        (1, [2, 0, 0, 0], 9, 63, 0, 2), // 41: Cr 9-63 at Al=2
        // Full 1-63 and frequency splits:
        (1, [1, 0, 0, 0], 1, 63, 0, 0),  // 42: Cb full 1-63 at Al=0
        (1, [2, 0, 0, 0], 1, 63, 0, 0),  // 43: Cr full 1-63 at Al=0
        (1, [1, 0, 0, 0], 1, 2, 0, 0),   // 44: Cb freq split at 2
        (1, [1, 0, 0, 0], 3, 63, 0, 0),  // 45
        (1, [2, 0, 0, 0], 1, 2, 0, 0),   // 46: Cr freq split at 2
        (1, [2, 0, 0, 0], 3, 63, 0, 0),  // 47
        (1, [1, 0, 0, 0], 1, 8, 0, 0),   // 48: Cb freq split at 8
        (1, [1, 0, 0, 0], 9, 63, 0, 0),  // 49
        (1, [2, 0, 0, 0], 1, 8, 0, 0),   // 50: Cr freq split at 8
        (1, [2, 0, 0, 0], 9, 63, 0, 0),  // 51
        (1, [1, 0, 0, 0], 1, 5, 0, 0),   // 52: Cb freq split at 5
        (1, [1, 0, 0, 0], 6, 63, 0, 0),  // 53
        (1, [2, 0, 0, 0], 1, 5, 0, 0),   // 54: Cr freq split at 5
        (1, [2, 0, 0, 0], 6, 63, 0, 0),  // 55
        (1, [1, 0, 0, 0], 1, 12, 0, 0),  // 56: Cb freq split at 12
        (1, [1, 0, 0, 0], 13, 63, 0, 0), // 57
        (1, [2, 0, 0, 0], 1, 12, 0, 0),  // 58: Cr freq split at 12
        (1, [2, 0, 0, 0], 13, 63, 0, 0), // 59
        (1, [1, 0, 0, 0], 1, 18, 0, 0),  // 60: Cb freq split at 18
        (1, [1, 0, 0, 0], 19, 63, 0, 0), // 61
        (1, [2, 0, 0, 0], 1, 18, 0, 0),  // 62: Cr freq split at 18
        (1, [2, 0, 0, 0], 19, 63, 0, 0), // 63
    ]
}
