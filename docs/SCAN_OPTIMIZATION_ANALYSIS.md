# C mozjpeg Scan Optimization: Complete Analysis

## Overview

This document provides a comprehensive analysis of C mozjpeg's progressive scan
optimization algorithm (`optimize_scans`) and how it compares to our Rust implementation.

**Key Files in C mozjpeg:**
- `jcparam.c:733-852` - `jpeg_search_progression()` - Scan script generation
- `jcmaster.c:773-962` - `select_scans()` - Scan selection algorithm
- `jcmaster.h:24-57` - `my_comp_master` - Master state structure

---

## 1. Scan Script Generation

### C mozjpeg (`jpeg_search_progression`)

For YCbCr images, C mozjpeg generates **64 scans** organized as:

```
LUMA SCANS (23 total):
├── 1 DC scan (all components)
├── 2 base AC scans (1-8, 9-63 at Al=0)
├── 9 successive approximation scans (3 per Al level × 3 levels)
│   For each Al in [0, 1, 2]:
│   ├── Refinement: (1-63, ah=Al+1, al=Al)
│   ├── Band low:   (1-8, ah=0, al=Al+1)
│   └── Band high:  (9-63, ah=0, al=Al+1)
├── 1 full AC scan (1-63 at Al=0)
└── 10 frequency split scans (5 splits × 2 scans each)
    Splits at: [2, 8, 5, 12, 18]

CHROMA SCANS (41 total):
├── 3 DC variants
│   ├── Combined Cb+Cr
│   ├── Separate Cb
│   └── Separate Cr
├── 4 base AC scans (Cb 1-8, Cb 9-63, Cr 1-8, Cr 9-63)
├── 12 successive approximation scans (6 per Al level × 2 levels)
│   For each Al in [0, 1]:
│   ├── Cb refinement + Cr refinement
│   └── Cb bands + Cr bands at Al+1
├── 2 full AC scans (Cb 1-63, Cr 1-63 at Al=0)
└── 20 frequency split scans (5 splits × 2 components × 2 scans)
```

**Configuration Parameters:**
```c
Al_max_luma = 3           // Max successive approximation for luma
Al_max_chroma = 2         // Max successive approximation for chroma
num_frequency_splits = 5  // Number of split points to test
frequency_splits = [2, 8, 5, 12, 18]
```

### Rust Implementation (Current)

We generate **67 scans** with a slightly different layout:

```
LUMA SCANS (26 total):
├── 1 DC scan
├── 2 base AC scans (1-8, 9-63 at Al=0)
├── 12 successive approximation scans (4 per Al level × 3 levels)
│   For each Al in [0, 1, 2]:
│   ├── Refinement: (1-63, ah=Al+1, al=Al)
│   ├── Full:       (1-63, ah=0, al=Al+1)     ← EXTRA
│   ├── Band low:   (1-8, ah=0, al=Al+1)
│   └── Band high:  (9-63, ah=0, al=Al+1)
├── 1 full AC scan (1-63 at Al=0)
└── 10 frequency split scans
```

**Key Difference:** We added full 1-63 scans at each Al level for proper cost comparison.

---

## 2. Scan Selection Algorithm

### C mozjpeg (`select_scans` in jcmaster.c:773-962)

The selection happens in multiple phases:

#### Phase 1: Luma Successive Approximation (lines 786-802)

```c
// Every 3 scans (refine + 2 bands), calculate cost
if ((next_scan_number - 1) % 3 == 2) {
    int Al = (next_scan_number - 1) / 3;

    // Cost = band scans at this Al + all previous refinements
    cost = scan_size[next_scan_number-2];    // Low band
    cost += scan_size[next_scan_number-1];   // High band
    for (i = 0; i < Al; i++)
        cost += scan_size[3 + 3*i];          // Refinement scans

    if (Al == 0 || cost < best_cost) {
        best_cost = cost;
        best_Al_luma = Al;
    } else {
        // Skip to frequency split scans (early termination)
        scan_number = luma_freq_split_scan_start - 1;
    }
}
```

**Key Insight:** C mozjpeg compares using **band-split scans** (1-8 + 9-63), not full 1-63 scans.
This is because the final output uses band splits, not full range scans.

#### Phase 2: Luma Frequency Split (lines 805-829)

```c
// First test full 1-63 (no split)
best_freq_split_idx_luma = 0;
best_cost = scan_size[luma_freq_split_scan_start - 1];  // Full 1-63

// Test each split point
for (idx = 1..5) {
    cost = scan_size[split_low] + scan_size[split_high];
    if (cost < best_cost) {
        best_cost = cost;
        best_freq_split_idx_luma = idx;
    }

    // Early termination heuristics:
    // If after 3 splits, no-split is still best, stop
    if (idx == 2 && best == 0) break;
    // If idx=3, best must be idx=2 to continue
    if (idx == 3 && best != 2) break;
    // If idx=4, best must be idx=4 to continue
    if (idx == 4 && best != 4) break;
}
```

#### Phase 3: Chroma DC Interleaving (lines 832-839)

```c
// Compare interleaved (Cb+Cr together) vs separate
interleave_chroma_dc = scan_size[combined] <= scan_size[cb] + scan_size[cr];
```

#### Phase 4: Chroma Successive Approximation (lines 840-865)

Similar to luma, but with 2 Al levels and 2 components per level.

#### Phase 5: Chroma Frequency Split (lines 867-895)

Similar to luma, but tests both Cb and Cr together.

#### Phase 6: Final Assembly (lines 898-961)

```c
// Output order (CRITICAL for progressive display):
1. DC scan(s)
2. Luma AC (full or split at best_Al_luma)
3. Luma refinements (from best_Al_luma-1 down to min_Al)
4. Chroma AC (full or split for both Cb and Cr at best_Al_chroma)
5. Chroma refinements (interleaved by Al level)
6. Joint refinements at lowest Al levels
```

---

## 3. Critical Differences Found

### 3.1 Cost Calculation Method

| Aspect | C mozjpeg | Rust (current) |
|--------|-----------|----------------|
| Compare using | Band-split scans (1-8 + 9-63) | Full 1-63 scans |
| Refinement cost | Sum of previous refinements | Sum of previous refinements |

**Impact:** At high quality (Q95+), C mozjpeg may choose higher Al because band-split
costs can differ from full-range costs.

### 3.2 DC Successive Approximation

| Mode | C mozjpeg | Rust |
|------|-----------|------|
| JCP_MAX_COMPRESSION | **NO** DC SA (Al=0) | Conditional (based on AC SA) |
| Standard | DC SA (Al=1, then refine) | Same |

Looking at `jcparam.c:931-958`, with `JCP_MAX_COMPRESSION`:
```c
// DC scan at Al=0 (no point transform)
scanptr = fill_dc_scans(scanptr, ncomps, 0, 0);  // ah=0, al=0
```

But our Rust uses DC SA when AC SA is used - this may cause differences.

### 3.3 Final Scan Count

Observed at Q95/Q97:
- **C mozjpeg:** 9 scans
- **Rust:** 6 scans

C's 9 scans suggest it's using higher Al levels with more refinement passes.

### 3.4 Scan Layout Indices

C mozjpeg has 3 scans per Al level; we have 4 (added full 1-63 for comparison).
This changes all index calculations.

---

## 4. Root Cause of Q90+ Gap

The remaining gap at high quality is due to:

1. **Different Al Selection:** C mozjpeg compares band-split costs; we compare full 1-63 costs.
   At high quality, these can lead to different optimal Al choices.

2. **Higher Al Levels:** C's 9 scans vs our 6 indicates C is using more SA levels.
   With `Al_max_luma=3`, C can use Al=1,2,3 while outputting at Al=3 needs 2 refinement
   passes.

3. **Interleaved Refinements:** C interleaves luma and chroma refinements at each Al level;
   we output all luma refinements, then all chroma refinements.

---

## 5. Implementation Plan for Parity

### Phase 1: Cost Calculation Fix (HIGH PRIORITY)

Change cost calculation to use band-split scans instead of full 1-63:

```rust
// Current (incorrect for C parity):
let cost_al_k = full_1_63_at_al_k + sum(refinements);

// Correct (matching C):
let cost_al_k = band_1_8_at_al_k + band_9_63_at_al_k + sum(refinements);
```

**Files:** `src/scan_optimize.rs:select_luma_params()`

### Phase 2: DC Successive Approximation (MEDIUM PRIORITY)

Match C mozjpeg's JCP_MAX_COMPRESSION behavior:
- Don't use DC SA by default
- Only use DC SA with standard progressive (non-MAX_COMPRESSION)

**Files:** `src/scan_optimize.rs:build_final_scans()`

### Phase 3: Scan Layout Alignment (LOW PRIORITY)

Consider reverting to 64-scan layout (3 per Al level) to match C exactly.
The extra full 1-63 scans aren't needed if we compare band-split costs.

### Phase 4: Interleaved Refinement Order (LOW PRIORITY)

Change refinement output order to match C:
```rust
// Current:
Y refine Al=2->1, Y refine Al=1->0, Cb refine Al=1->0, Cr refine Al=1->0

// C mozjpeg:
Y refine Al=2->1, Cb refine Al=1->0, Cr refine Al=1->0, Y refine Al=1->0
```

This affects progressive display order but not final file size.

---

## 6. Test Vectors from C mozjpeg

### Scan indices for YCbCr (64 scans):

```
Luma (23):
  [0] DC all (ah=0, al=0)
  [1-2] Y 1-8, 9-63 (ah=0, al=0)
  [3] Y refine 1-63 (ah=1, al=0)
  [4-5] Y 1-8, 9-63 (ah=0, al=1)
  [6] Y refine 1-63 (ah=2, al=1)
  [7-8] Y 1-8, 9-63 (ah=0, al=2)
  [9] Y refine 1-63 (ah=3, al=2)
  [10-11] Y 1-8, 9-63 (ah=0, al=3)
  [12] Y full 1-63 (ah=0, al=0)
  [13-22] Frequency splits

Chroma (41):
  [23] Cb+Cr DC
  [24-25] Cb DC, Cr DC
  [26-29] Cb 1-8, Cb 9-63, Cr 1-8, Cr 9-63
  [30-31] Cb refine, Cr refine (ah=1, al=0)
  [32-35] Cb/Cr bands at Al=1
  [36-37] Cb refine, Cr refine (ah=2, al=1)
  [38-41] Cb/Cr bands at Al=2
  [42-43] Cb full, Cr full
  [44-63] Frequency splits
```

### Key Indices for Selection:

```c
luma_freq_split_scan_start = 13  // (1 + 11 + 1)
num_scans_luma = 23
chroma_freq_split_scan_start = 44  // (23 + 3 + 16 + 2)
```

---

## 7. Verification Steps

1. **Add debug output to C mozjpeg** matching `select_scans()` decisions
2. **Compare scan-by-scan costs** between Rust and C for same image
3. **Verify Al selection** matches at each quality level
4. **Verify final scan count** matches C output

---

## 8. References

- `mozjpeg/jcparam.c:733-852` - Scan generation
- `mozjpeg/jcmaster.c:773-962` - Scan selection
- `mozjpeg/jcmaster.h` - Master structure
- `mozjpeg/jpegint.h:120-128` - Config fields
