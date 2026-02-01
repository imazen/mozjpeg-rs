# optimize_scans Parity with C mozjpeg

## Goal
Match C mozjpeg's `optimize_scans` output exactly (0% file size difference).

## Status: FIXED (Dec 2025)

**Result:** Max Compression mode (progressive + optimize_scans + trellis) matches
C mozjpeg within ±0.15% at all quality levels. At Q75 and below, Rust beats C.

**Note (Feb 2025):** The previous ±2.2% claim was inflated because C test wrappers
didn't explicitly disable `optimize_scans`. C mozjpeg's default `JCP_MAX_COMPRESSION`
profile enables `optimize_scans=TRUE`, so C was using an optimized ~12-scan script
while Rust used the fixed 9-scan script. All C wrappers now properly control this.

## What Was Fixed

### Problem (identified Dec 2025-12-28)
Trial encoding for refinement scans was done independently (no state between scans),
producing garbage sizes for Ah>0 scans. The optimizer always picked Al=0 (no successive
approximation), losing 2-4% compression.

### Fix (commits 70cf3e2, 0f22f3b, 09bb1e5 on 2025-12-28)

1. **`ScanTrialEncoder` (`src/scan_trial.rs`)** — Sequential trial encoding with state
   tracking between scans. Maintains `BlockState` per block (DC coded status, AC state).
   Refinement scans now produce correct sizes because they execute after their first scans.

2. **Per-scan Huffman tables** — Each trial-encoded AC scan builds its own optimal
   Huffman table via two-pass encoding (count frequencies → build table → encode).
   This matches C mozjpeg's behavior when `optimize_scans=true`.

3. **Re-encoding for output** — After selection, the chosen scan configuration is
   re-encoded from scratch (unlike C which copies pre-encoded buffers). This produces
   equivalent results since the Huffman tables are rebuilt from the same data.

## Architecture: C mozjpeg vs Rust

### C mozjpeg (jcmaster.c)
1. Encode all 64 candidate scans **in sequence**, storing bytes in buffers
2. `select_scans()` called after each scan, uses early termination
3. `copy_buffer()` copies selected pre-encoded scan bytes to output

### Rust (encode.rs + scan_trial.rs)
1. `ScanTrialEncoder::encode_all_scans()` encodes 64 scans **in sequence** with state
2. `ScanSelector::select_best()` processes all sizes, matching C's algorithm exactly
3. `build_final_scans()` generates optimal scan script, encoder re-encodes from scratch

### Remaining difference from C
- Rust re-encodes selected scans instead of copying pre-encoded buffers
- This is functionally equivalent but slightly less efficient at encode time
- Could be optimized in the future by using stored scan buffers from trial encoding

## Key Code References

### Rust implementation
- `src/scan_trial.rs` — `ScanTrialEncoder` (sequential trial encoding with state)
- `src/scan_optimize.rs` — `ScanSelector`, `ScanSearchConfig`, `generate_search_scans()`
- `src/encode.rs:optimize_progressive_scans()` — Wires it all together

### C mozjpeg references
- `jcmaster.c:select_scans()` (lines 773-962) — Selection with early termination
- `jcmaster.c:copy_buffer()` (lines ~902-956) — Buffer-based output assembly
- `jcparam.c:jpeg_search_progression()` (lines 733-852) — 64 candidate scan generation

## Test Commands

```bash
# Run benchmark comparison
cargo test --release --test benchmark_runner -- --nocapture

# Enable debug output in scan_optimize.rs
# const DEBUG_SCAN_OPT: bool = true;
```
