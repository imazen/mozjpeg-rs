# Investigation: optimize_scans divergence at low quality

## Problem

The `Max Compression` config (`optimize_scans: true`) shows increasing file size gap
between Rust and C at low quality levels. At Q40, it exceeds our 1% average / 3%
per-image thresholds:

| Q  | Avg Delta | Max Dev | Worst Image |
|----|-----------|---------|-------------|
| 40 | +1.11%    | 3.37%   | kodim23, kodim09 |
| 50 | +0.77%    | 3.13%   | kodim23 |
| 55 | +0.75%    | 2.82%   | kodim23 |
| 65 | +0.70%    | 2.74%   | |
| 75 | +0.59%    | 2.12%   | |
| 85 | +0.41%    | 1.25%   | |
| 90 | +0.28%    | 0.59%   | |
| 95 | +0.40%    | 0.81%   | |

Without `optimize_scans`, all configs are within ±0.7% average even at Q40.
The gap is strictly in the scan optimization search.

## Context

`optimize_scans` tries multiple progressive scan configurations and picks the
smallest. Both Rust and C implement this, but their scan search heuristics may
differ. At low quality, more coefficients are quantized to zero, giving the
optimizer a larger search space where different heuristics produce different
local optima.

## What to investigate

1. **Map the full curve.** Run Max Compression at Q10, Q20, Q25, Q30, Q35, Q40,
   Q45, Q50 on the Kodak corpus. Add a temporary `#[test]` or `#[ignore]` test
   to `parity_benchmark.rs` that only runs Max Compression across these qualities
   and prints per-image detail for each. Determine where the gap plateaus.

2. **Per-image scan counts.** For the worst images (kodim23, kodim09), compare
   the number of scans chosen by Rust vs C at Q40. Use `count_scans()` (pattern
   in `corpus_comparison.rs`). If scan counts differ, the search is finding
   fundamentally different scan scripts.

3. **Compare scan scripts directly.** Parse the SOS markers from both outputs
   and print `(Ns, comps, Ss, Se, Ah, Al)` for each scan. Pattern is in
   `corpus_comparison.rs::print_scan_details()`. Identify which scans differ.

4. **Trace the scan trial encoder.** The Rust implementation is in
   `src/scan_trial.rs`. The C implementation calls `jpeg_search_progression()`
   in `jcmaster.c`. Compare:
   - How many candidate scans are evaluated
   - The cost function (file size estimation)
   - The greedy selection order
   - Whether the trial encoder's Huffman table estimation matches C's

5. **Check if C uses `trellis_freq_split` during scan search.** C mozjpeg has
   `trellis_freq_split = 8` which splits AC trellis into low/high frequency
   passes. If C's scan optimizer accounts for this split during trial encoding
   but Rust doesn't, that could explain the gap at low quality where the split
   matters more.

6. **Kodim23 specifically.** This image consistently has the worst deviation.
   It's a landscape with lots of sky gradient + sharp foreground detail.
   Encode it standalone at Q40 with both, diff the scan scripts, and check
   if one finds genuinely smaller output or if it's a Huffman table estimation
   error in the trial encoder.

## Key files

- `src/scan_trial.rs` — Rust scan trial encoder
- `src/progressive.rs` — Rust progressive scan generation
- `tests/parity_benchmark.rs` — benchmark test (add exploration tests here)
- `tests/corpus_comparison.rs` — has `count_scans()` and `print_scan_details()`
- C: `jcmaster.c` → `jpeg_search_progression()`
- C: `jcphuff.c` → trial encoding for scan cost estimation

## Acceptance criteria

- Understand whether the gap is from different scan scripts or different
  file sizes for the same scan script
- If different scripts: determine if Rust's choice is suboptimal or just different
- If same scripts: the gap is in entropy coding, not scan search — investigate
  per-scan Huffman table differences
- Document findings, decide whether to fix or accept and adjust thresholds
