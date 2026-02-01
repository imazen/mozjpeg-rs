# User Feedback Log

## 2026-02-01
- Request: Wire up `delta_dc_weight` in DC trellis, update doc comments for unimplemented TrellisConfig fields, add tests.
- Note: Linter/cargo check was stashing changes when intermediate edits left files in inconsistent compilation state. Used Python script to apply all changes atomically.
- Question: "any reason we are still 2% larger than c mozjpeg?" → Discovered the entire 2-5% gap was a measurement bug: C encoder wrappers didn't disable `optimize_scans`, so C used an optimized ~12-scan script while Rust used the fixed 9-scan script. With same settings, Rust matches or beats C (-0.15% to +0.15% across Q75-Q95).
