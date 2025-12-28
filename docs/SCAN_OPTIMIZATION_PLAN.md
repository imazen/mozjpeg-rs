# Scan Optimization Implementation Plan

## Executive Summary

Current state (Dec 2024):
- Q50-Q85: Within 0.6% of C mozjpeg (Rust can be smaller at Q50!)
- Q90-Q97: 1.7-4.8% larger than C mozjpeg

Root cause: Cost calculation method differs - we use full 1-63 scans while C uses band-split scans.

---

## Phase 1: Fix Cost Calculation (Highest Priority)

### Problem

Current Rust code in `select_luma_params()`:
```rust
// We compare full 1-63 scans
let cost = if al == 0 {
    scan_sizes[full_al0_idx]  // Full 1-63 at Al=0
} else {
    scan_sizes[full_1_63_at_al_k] + sum_of_refinements
};
```

C mozjpeg compares band-split scans:
```c
// C compares band-split scans
cost = scan_size[next_scan_number-2] + scan_size[next_scan_number-1];  // 1-8 + 9-63
for (i = 0; i < Al; i++)
    cost += scan_size[3 + 3*i];  // refinements
```

### Solution

Change `select_luma_params()` to compare band costs:

```rust
fn select_luma_params(&self, scan_sizes: &[usize]) -> (u8, usize) {
    let al_max = self.config.al_max_luma as usize;

    // C mozjpeg scan layout (3 scans per Al level):
    //   0: DC
    //   1, 2: base bands (1-8, 9-63) at Al=0
    //   3: refinement 1-63 (ah=1, al=0)
    //   4, 5: bands at Al=1
    //   6: refinement (ah=2, al=1)
    //   7, 8: bands at Al=2
    //   9: refinement (ah=3, al=2)
    //   10, 11: bands at Al=3
    //   12: full 1-63 (for freq split baseline)
    //   13+: frequency splits

    let mut best_al = 0u8;
    let mut best_al_cost = usize::MAX;

    for al in 0..=al_max {
        // Cost = band scans at this Al + all previous refinements
        let band_low_idx = 1 + 3 * al;  // 1-8 at this Al
        let band_high_idx = 2 + 3 * al; // 9-63 at this Al

        let mut cost = scan_sizes.get(band_low_idx).copied().unwrap_or(0)
                     + scan_sizes.get(band_high_idx).copied().unwrap_or(0);

        // Add refinement costs for levels 0..(al-1)
        for prev_al in 0..al {
            let refine_idx = 3 + 3 * prev_al;  // Refinement at prev_al
            cost += scan_sizes.get(refine_idx).copied().unwrap_or(0);
        }

        if cost < best_al_cost {
            best_al_cost = cost;
            best_al = al as u8;
        }
    }

    // ... frequency split selection (unchanged)
}
```

### Files to Modify

1. `src/scan_optimize.rs`:
   - Revert `generate_search_scans()` to 64-scan layout (3 per Al level)
   - Update `ScanSelector::new()` indices to match 64-scan layout
   - Rewrite `select_luma_params()` to use band costs
   - Rewrite `select_chroma_params()` similarly

### Verification

```bash
# Run progressive comparison at all quality levels
cargo test --test progressive_comparison -- --nocapture
```

Expected result: Q90+ gap should reduce from 4.8% to ~1-2%.

---

## Phase 2: DC Successive Approximation (Medium Priority)

### Problem

C mozjpeg with `JCP_MAX_COMPRESSION` does NOT use DC successive approximation:
```c
// jcparam.c:936 - DC scan at Al=0
scanptr = fill_dc_scans(scanptr, ncomps, 0, 0);
```

Our Rust applies DC SA when AC SA is used, which differs from C behavior.

### Solution

In `build_final_scans()`, remove DC SA for max compression mode:

```rust
pub fn build_final_scans(&self, num_components: u8, config: &ScanSearchConfig) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // DC scan - NO point transform for max compression (matching C mozjpeg)
    if config.dc_scan_opt_mode == 0 {
        scans.push(ScanInfo::dc_scan(num_components));  // al=0
    } else {
        scans.push(ScanInfo::dc_scan(1));
    }

    // ... rest unchanged, but remove DC refinement scan at end
}
```

### Files to Modify

1. `src/scan_optimize.rs`: Remove `use_dc_succ_approx` logic

---

## Phase 3: Scan Layout Cleanup (Low Priority)

### Problem

Our 67-scan layout adds extra full 1-63 scans that C mozjpeg doesn't have.
These aren't needed if we compare band costs.

### Solution

Revert to 64-scan layout matching C exactly:

```rust
// For each Al level, generate 3 scans (not 4):
for al in 0..config.al_max_luma {
    scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al));  // Refinement
    scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1));    // Low band
    scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1));   // High band
}
```

### Index Mapping (C mozjpeg)

```
Luma (23 scans):
  0:      DC all
  1-2:    Base bands (1-8, 9-63)
  3:      Refine Al=0→none (ah=1, al=0)
  4-5:    Bands at Al=1
  6:      Refine Al=1→0 (ah=2, al=1)
  7-8:    Bands at Al=2
  9:      Refine Al=2→1 (ah=3, al=2)
  10-11:  Bands at Al=3
  12:     Full 1-63 (freq split baseline)
  13-22:  Frequency splits

Chroma (41 scans, starting at 23):
  23:     Cb+Cr DC interleaved
  24-25:  Cb DC, Cr DC
  26-29:  Base bands for Cb and Cr
  30-41:  SA scans (6 per Al level × 2)
  42-43:  Full 1-63 for Cb and Cr
  44-63:  Frequency splits
```

---

## Phase 4: Refinement Order (Low Priority)

### Problem

C mozjpeg interleaves refinements differently for progressive display.

### Current Rust Order

```
1. DC
2. Y AC at Al=2
3. Y refine 2→1
4. Y refine 1→0
5. Cb AC at Al=1
6. Cr AC at Al=1
7. Cb refine 1→0
8. Cr refine 1→0
```

### C mozjpeg Order

```
1. DC
2. Y AC at Al=2
3. Cb AC at Al=1
4. Cr AC at Al=1
5. Y refine 2→1  ← Interleaved!
6. Cb refine 1→0
7. Cr refine 1→0
8. Y refine 1→0
```

This doesn't affect file size, only progressive display order.

---

## Testing Matrix

| Phase | Test | Expected Result |
|-------|------|-----------------|
| 1 | Q95 ratio | < 2% larger than C |
| 1 | Q97 ratio | < 3% larger than C |
| 2 | Scan count at Q95 | Match C (likely 7-9 scans) |
| 3 | Total scan candidates | 64 (not 67) |
| 4 | Progressive display | Smoother (subjective) |

---

## Code Changes Summary

### `src/scan_optimize.rs`

```diff
// generate_search_scans - revert to 64-scan layout
-    // 4 scans per Al level (refine + full + bands)
+    // 3 scans per Al level (refine + bands)
     for al in 0..config.al_max_luma {
         scans.push(ScanInfo::ac_scan(0, 1, 63, al + 1, al));
-        scans.push(ScanInfo::ac_scan(0, 1, 63, 0, al + 1));
         scans.push(ScanInfo::ac_scan(0, 1, 8, 0, al + 1));
         scans.push(ScanInfo::ac_scan(0, 9, 63, 0, al + 1));
     }

// select_luma_params - use band costs
-    let cost = scan_sizes[full_1_63_idx] + refinements;
+    let cost = scan_sizes[band_low_idx] + scan_sizes[band_high_idx] + refinements;

// build_final_scans - remove DC SA
-    if use_dc_succ_approx { dc_scan.al = 1; }
+    // DC always at al=0 for max compression
```

### `tests/scan_verification.rs`

```diff
-    assert_eq!(rust_scans.len(), 67, "...");
+    assert_eq!(rust_scans.len(), 64, "...");
```

---

## Timeline Estimate

| Phase | Complexity | Impact |
|-------|------------|--------|
| 1 | Medium (2-3 hours) | High - fixes main gap |
| 2 | Low (30 min) | Medium - matches C exactly |
| 3 | Low (1 hour) | Low - cleanup |
| 4 | Medium (1-2 hours) | Very Low - display only |

---

## Rollback Plan

If changes cause regressions:
1. `git checkout successive-approximation -- src/scan_optimize.rs`
2. Keep current 67-scan layout
3. Focus on other optimization areas
