//! Verify Rust generates same 64 scans as C mozjpeg
//!
//! Run with: cargo run --release --example verify_scans

fn main() {
    use mozjpeg::scan_optimize::{generate_search_scans, ScanSearchConfig};

    let config = ScanSearchConfig::default();
    let rust_scans = generate_search_scans(3, &config);

    println!("Rust progressive scans: {}", rust_scans.len());
    for (i, scan) in rust_scans.iter().enumerate() {
        let comps: Vec<_> = (0..scan.comps_in_scan).map(|j| scan.component_index[j as usize]).collect();
        println!("  Scan {:2}: components={:?}, Ss={:2}, Se={:2}, Ah={}, Al={}",
            i, comps, scan.ss, scan.se, scan.ah, scan.al);
    }

    println!("\nComparing with C mozjpeg...");
    let c_scans = get_c_scans();
    println!("C progressive scans: {}", c_scans.len());

    // Compare
    let mut matches = 0;
    let mut mismatches = Vec::new();
    for i in 0..rust_scans.len().min(c_scans.len()) {
        let r = &rust_scans[i];
        let c = &c_scans[i];
        if r.comps_in_scan == c.0 && r.ss == c.2 && r.se == c.3 && r.ah == c.4 && r.al == c.5 {
            matches += 1;
        } else {
            mismatches.push(i);
        }
    }

    println!("\nMatches: {}/{}", matches, rust_scans.len().min(c_scans.len()));
    if !mismatches.is_empty() {
        println!("Mismatches at indices: {:?}", mismatches);
        for &i in mismatches.iter().take(5) {
            let r = &rust_scans[i];
            let c = &c_scans[i];
            println!("  Scan {}: Rust=({}, {:?}, {}, {}, {}, {}) vs C=({}, {:?}, {}, {}, {}, {})",
                i, r.comps_in_scan, &r.component_index[..r.comps_in_scan as usize], r.ss, r.se, r.ah, r.al,
                c.0, &c.1[..c.0 as usize], c.2, c.3, c.4, c.5);
        }
    }
}

// C scan data from earlier check_scans output
fn get_c_scans() -> Vec<(u8, [u8; 4], u8, u8, u8, u8)> {
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
